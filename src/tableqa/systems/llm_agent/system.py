"""System #3: a Claude tool-use agent that answers by writing SQL.

The loop is written by hand rather than with the SDK's tool runner because the
benchmark needs things the runner does not hand over directly: a per-step trace with
token usage, a hard step and token budget per question, and a tool call
(`final_answer`) that ends the loop.

Each step is one model call. The model's tool calls are executed against the sandbox
and their results (or errors, so it can repair a failed query) go back in one message.
The loop ends when the model calls `final_answer`, or when a budget runs out.

Context-engineering variants each change one thing from that default:

- samples=False: no sample_rows tool; the model sees only the schema.
- profile=True: the schema also profiles each column (filled cells, distinct values,
  most frequent values, numeric range).
- check_answer=True: the first final_answer is not accepted; the model is asked to check
  it and answer again.
- single_shot=True: no loop. The model is shown the schema and the first rows and writes
  one query; the first column of its result is the answer, with no chance to repair it.

Needs the optional dependencies and a key: `uv sync --extra agent`, and
ANTHROPIC_API_KEY in the environment or in a git-ignored `.env` file.
"""

import time

from tableqa.execution.sandbox import TABLE_NAME, Sandbox
from tableqa.interface import Prediction
from tableqa.systems.llm_agent.tools import (
    FINAL_ANSWER,
    MAX_SAMPLE_ROWS,
    SUBMIT_SQL,
    SUBMIT_SQL_DEFINITION,
    TableTools,
    tool_definitions,
)
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

# The two notes about the data, taken from the default prompt so every variant gets
# exactly the same wording.
DATA_NOTES = SYSTEM_PROMPT[
    SYSTEM_PROMPT.index("- The tables come from Wikipedia") : SYSTEM_PROMPT.index(
        "\nGiving the answer"
    )
]

# Without sample_rows, the two sentences that mention the sample go.
NO_SAMPLE_EDITS = [
    ("call get_schema and sample_rows, which you can do in the same turn.", "call get_schema."),
    (
        (
            " The sample is only the first few rows, so an answer read off the sample is "
            "usually wrong."
        ),
        "",
    ),
]

SINGLE_SHOT_PROMPT = (
    "You answer questions about one table by writing one DuckDB SQL query. The table is "
    f"loaded into DuckDB as `{TABLE_NAME}`. You are shown its schema and its first rows; you "
    "cannot see the rest of it, and you get one attempt: your query is run once, and the "
    "values in the first column of its result are taken as the answer.\n"
    "\n"
    "About the data\n"
    "- Column names have been normalised to snake_case; the schema also gives each column's "
    "original header text, which is often what the question refers to.\n"
    f"{DATA_NOTES}"
    "\n"
    "Giving the answer\n"
    f"- Call {SUBMIT_SQL} with a single SELECT statement. Text you write is not read.\n"
    "- The first column of the result must be the answer itself: a name or date as its cell "
    "shows it, a count or calculation as a bare number, and one row per item if the question "
    "has several answers.\n"
)

REMINDER = (
    f"You have not called {FINAL_ANSWER} yet. Call it now with your answer; "
    "text outside a tool call is not read."
)

CHECK = (
    "Before this answer is accepted, check it. Re-read the question and the results you "
    "have, and make sure the answer is what was asked, in the form asked for, and worked out "
    "from the whole table rather than the sample. Then call final_answer again: with the "
    "same values if they are right, or with corrected values if not. You can use run_sql "
    "first if you need to."
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
        samples: bool = True,
        profile: bool = False,
        check_answer: bool = False,
        single_shot: bool = False,
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
        self.samples = samples
        self.profile = profile
        self.check_answer = check_answer
        self.single_shot = single_shot

        if single_shot:
            self._system = SINGLE_SHOT_PROMPT
            self._tools = [SUBMIT_SQL_DEFINITION]
        else:
            system = SYSTEM_PROMPT.format(max_steps=max_steps)
            if not samples:
                for old, new in NO_SAMPLE_EDITS:
                    assert old in system, old
                    system = system.replace(old, new)
            self._system = system
            self._tools = tool_definitions(samples=samples, profile=profile)

    @property
    def config(self) -> dict:
        """Everything that defines this variant, recorded next to its results."""
        return {
            "model": self.model,
            "effort": self.effort,
            "max_steps": self.max_steps,
            "max_output_tokens": self.max_output_tokens,
            "max_question_tokens": self.max_question_tokens,
            "samples": self.samples,
            "profile": self.profile,
            "check_answer": self.check_answer,
            "single_shot": self.single_shot,
            "system_prompt": self._system,
            "tools": self._tools,
        }

    def answer(self, question: str, table: Table) -> Prediction:
        start = time.perf_counter()
        sandbox = Sandbox(table)
        run = _Run()
        try:
            tools = TableTools(table, sandbox, profile=self.profile)
            if self.single_shot:
                answer, error = self._single_shot(question, tools, run)
            else:
                answer, error = self._loop(question, tools, run)
        finally:
            sandbox.close()
        return Prediction(
            answer=answer,
            sql_or_trace=run.trace,
            latency_ms=(time.perf_counter() - start) * 1000,
            cost_usd=run.cost,
            tokens=run.tokens,
            error=error,
        )

    def _call(self, messages: list, run: "_Run"):
        """One model call; its usage and a new trace step are added to the run."""
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_output_tokens,
            system=self._system,
            tools=self._tools,
            messages=messages,
            output_config={"effort": self.effort},
            cache_control={"type": "ephemeral"},
        )
        usage = response.usage
        run.tokens += _billed_tokens(usage)
        run.cost += self._cost(usage)
        record = {
            "step": len(run.trace) + 1,
            "stop_reason": response.stop_reason,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cache_read_tokens": _cached(usage, "cache_read_input_tokens"),
            "cache_write_tokens": _cached(usage, "cache_creation_input_tokens"),
            "text": [b.text for b in response.content if b.type == "text"],
            "tool_calls": [],
        }
        run.trace.append(record)
        return response, record

    def _loop(self, question: str, tools: TableTools, run: "_Run"):
        messages = [{"role": "user", "content": f"Question: {question}"}]
        provisional = None  # with check_answer: the first answer, until it is confirmed
        for _ in range(self.max_steps):
            response, record = self._call(messages, run)

            if response.stop_reason in ("refusal", "max_tokens"):
                return [], f"model stopped: {response.stop_reason}"

            # The whole content goes back unchanged: it can hold thinking blocks,
            # which the API requires to be returned as they came.
            messages.append({"role": "assistant", "content": response.content})
            calls = [b for b in response.content if b.type == "tool_use"]
            results = []
            for call in calls:
                if call.name == FINAL_ANSWER:
                    values = list(call.input["values"])
                    if self.check_answer and provisional is None:
                        provisional = values
                        record["tool_calls"].append(
                            {
                                "name": call.name,
                                "input": call.input,
                                "result": CHECK,
                                "error": False,
                            }
                        )
                        results.append(_result(call.id, CHECK, False))
                        continue
                    record["tool_calls"].append({"name": call.name, "input": call.input})
                    return values, None
                result, is_error = tools.execute(call.name, call.input)
                record["tool_calls"].append(
                    {"name": call.name, "input": call.input, "result": result, "error": is_error}
                )
                results.append(_result(call.id, result, is_error))
            # All results of one turn go back in a single message.
            messages.append({"role": "user", "content": results or REMINDER})

            if run.tokens > self.max_question_tokens:
                return _out_of_budget(provisional, "token budget exhausted")
        return _out_of_budget(provisional, "step budget exhausted")

    def _single_shot(self, question: str, tools: TableTools, run: "_Run"):
        schema, _ = tools.execute("get_schema", {})
        sample, _ = tools.execute("sample_rows", {"n": MAX_SAMPLE_ROWS})
        content = f"Schema:\n{schema}\n\nFirst rows:\n{sample}\n\nQuestion: {question}"
        response, record = self._call([{"role": "user", "content": content}], run)

        if response.stop_reason in ("refusal", "max_tokens"):
            return [], f"model stopped: {response.stop_reason}"
        calls = [b for b in response.content if b.type == "tool_use" and b.name == SUBMIT_SQL]
        if not calls:
            return [], "no query submitted"
        query = calls[0].input["query"]
        answer, result, is_error = tools.answer_with_sql(query)
        record["tool_calls"].append(
            {"name": SUBMIT_SQL, "input": {"query": query}, "result": result, "error": is_error}
        )
        return answer, (result if is_error else None)

    def _cost(self, usage) -> float:
        price_in, price_out = PRICES[self.model]
        input_cost = (
            usage.input_tokens
            + _cached(usage, "cache_read_input_tokens") * CACHE_READ
            + _cached(usage, "cache_creation_input_tokens") * CACHE_WRITE
        ) * price_in
        return (input_cost + usage.output_tokens * price_out) / 1_000_000


class _Run:
    """What one question has cost so far, and its trace."""

    def __init__(self):
        self.trace: list[dict] = []
        self.tokens = 0
        self.cost = 0.0


def _result(tool_use_id: str, content: str, is_error: bool) -> dict:
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": content,
        "is_error": is_error,
    }


def _out_of_budget(provisional, reason: str):
    """A budget ran out: keep an unconfirmed answer if there is one, else fail."""
    if provisional is not None:
        return provisional, None
    return [], reason


def _cached(usage, field: str) -> int:
    return getattr(usage, field, None) or 0


def _billed_tokens(usage) -> int:
    return (
        usage.input_tokens
        + usage.output_tokens
        + _cached(usage, "cache_read_input_tokens")
        + _cached(usage, "cache_creation_input_tokens")
    )
