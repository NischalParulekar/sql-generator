"""System #3: a Claude tool-use agent that answers by writing SQL.

The loop is written by hand rather than with the SDK's tool runner because the
benchmark needs things the runner does not hand over directly: a per-step trace with
token usage, a hard step and token budget per question, and a tool call
(`final_answer`) that ends the loop.

Each step is one model call. The model's tool calls are executed against the sandbox
and their results (or errors, so it can repair a failed query) go back in one message.
The loop ends when the model calls `final_answer`, or when a budget runs out.

Needs the optional dependencies and a key: `uv sync --extra agent`, and
ANTHROPIC_API_KEY in the environment or in a git-ignored `.env` file.
"""

import time

from tableqa.execution.sandbox import TABLE_NAME, Sandbox
from tableqa.interface import Prediction
from tableqa.systems.llm_agent.tools import FINAL_ANSWER, TOOL_DEFINITIONS, TableTools
from tableqa.table import Table

# USD per million tokens (input, output), for prompts up to 100K tokens.
# Cache reads cost 0.1x the input price and 5-minute cache writes 1.25x.
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
}
CACHE_READ, CACHE_WRITE = 0.1, 1.25

SYSTEM_PROMPT = f"""\
You answer questions about one table. The table is loaded into a DuckDB database as \
`{TABLE_NAME}`, and the tools are your only access to it.

How to work
- Look at the table first: call get_schema and sample_rows, which you can do in the same \
turn. Column names have been normalised to snake_case; get_schema also gives each \
column's original header text, which is often what the question refers to.
- Work the answer out with run_sql. The sample is only the first few rows, so an answer \
read off the sample is usually wrong. If a query fails, read the error and fix the query.
- The tables come from Wikipedia and are messy. A column is typed as a number only when \
every cell in it is a plain number. Otherwise it is text, and cells such as "1,234", \
"3rd", "12 (4)" or "2004–05" need string functions (replace, regexp_extract, try_cast) \
before they can be compared or added. Some tables end with a totals row that should not \
be counted as data.
- `rowid` is each row's position on the page, starting at 0. Use it for questions about \
order, such as "first", "last", "next", "previous" or "above".

Giving the answer
- Call final_answer when you have the answer. Text you write is not read, so anything \
that is not in a final_answer call is lost.
- Answer with the value itself, not a sentence: a name or date as its cell shows it, a \
count or calculation as a bare number.
- If the question has several answers, pass each as its own item.
- You have at most {{max_steps}} turns for a question, so answer once you are confident \
rather than re-checking.
"""

REMINDER = (
    f"You have not called {FINAL_ANSWER} yet. Call it now with your answer; "
    "text outside a tool call is not read."
)


class LLMAgentSystem:
    def __init__(
        self,
        name: str = "agent_haiku",
        model: str = "claude-haiku-5-5",
        *,
        client=None,
        effort: str = "medium",
        max_steps: int = 6,
        max_output_tokens: int = 4096,
        max_question_tokens: int = 60_000,
    ):
        if client is None:
            import anthropic  # imported here so tests with a fake client need no SDK

            client = anthropic.Anthropic()
        self.name = name
        self.model = model
        self._client = client
        self.effort = effort
        self.max_steps = max_steps
        self.max_output_tokens = max_output_tokens
        self.max_question_tokens = max_question_tokens
        self._system = SYSTEM_PROMPT.format(max_steps=max_steps)

    @property
    def config(self) -> dict:
        """Everything that defines this variant, recorded next to its results."""
        return {
            "model": self.model,
            "effort": self.effort,
            "max_steps": self.max_steps,
            "max_output_tokens": self.max_output_tokens,
            "max_question_tokens": self.max_question_tokens,
            "system_prompt": self._system,
            "tools": TOOL_DEFINITIONS,
        }

    def answer(self, question: str, table: Table) -> Prediction:
        start = time.perf_counter()
        sandbox = Sandbox(table)
        try:
            answer, error, trace, tokens, cost = self._loop(question, TableTools(table, sandbox))
        finally:
            sandbox.close()
        return Prediction(
            answer=answer,
            sql_or_trace=trace,
            latency_ms=(time.perf_counter() - start) * 1000,
            cost_usd=cost,
            tokens=tokens,
            error=error,
        )

    def _loop(self, question: str, tools: TableTools):
        messages = [{"role": "user", "content": f"Question: {question}"}]
        trace, tokens, cost = [], 0, 0.0
        for step in range(1, self.max_steps + 1):
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_output_tokens,
                system=self._system,
                tools=TOOL_DEFINITIONS,
                messages=messages,
                output_config={"effort": self.effort},
                cache_control={"type": "ephemeral"},
            )
            usage = response.usage
            tokens += _billed_tokens(usage)
            cost += self._cost(usage)
            record = {
                "step": step,
                "stop_reason": response.stop_reason,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cache_read_tokens": _cached(usage, "cache_read_input_tokens"),
                "cache_write_tokens": _cached(usage, "cache_creation_input_tokens"),
                "text": [b.text for b in response.content if b.type == "text"],
                "tool_calls": [],
            }
            trace.append(record)

            if response.stop_reason in ("refusal", "max_tokens"):
                return [], f"model stopped: {response.stop_reason}", trace, tokens, cost

            # The whole content goes back unchanged: it can hold thinking blocks,
            # which the API requires to be returned as they came.
            messages.append({"role": "assistant", "content": response.content})
            calls = [b for b in response.content if b.type == "tool_use"]
            results = []
            for call in calls:
                if call.name == FINAL_ANSWER:
                    record["tool_calls"].append({"name": call.name, "input": call.input})
                    return list(call.input["values"]), None, trace, tokens, cost
                result, is_error = tools.execute(call.name, call.input)
                record["tool_calls"].append(
                    {"name": call.name, "input": call.input, "result": result, "error": is_error}
                )
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": call.id,
                        "content": result,
                        "is_error": is_error,
                    }
                )
            # All results of one turn go back in a single message.
            messages.append({"role": "user", "content": results or REMINDER})

            if tokens > self.max_question_tokens:
                return [], "token budget exhausted", trace, tokens, cost
        return [], "step budget exhausted", trace, tokens, cost

    def _cost(self, usage) -> float:
        price_in, price_out = PRICES[self.model]
        input_cost = (
            usage.input_tokens
            + _cached(usage, "cache_read_input_tokens") * CACHE_READ
            + _cached(usage, "cache_creation_input_tokens") * CACHE_WRITE
        ) * price_in
        return (input_cost + usage.output_tokens * price_out) / 1_000_000


def _cached(usage, field: str) -> int:
    return getattr(usage, field, None) or 0


def _billed_tokens(usage) -> int:
    return (
        usage.input_tokens
        + usage.output_tokens
        + _cached(usage, "cache_read_input_tokens")
        + _cached(usage, "cache_creation_input_tokens")
    )
