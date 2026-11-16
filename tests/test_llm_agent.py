"""The agent loop, driven by a scripted stand-in for the Claude API (no network, no key)."""

import copy
import json
from types import SimpleNamespace

import pytest

from tableqa.systems.llm_agent.system import REMINDER, LLMAgentSystem
from tableqa.systems.llm_agent.tools import TOOL_DEFINITIONS

TABLE_ROWS = [["Ann", "10", "Brazil"], ["Bob", "7", "Chile"], ["Cy", "1,200", "Brazil"]]


@pytest.fixture
def table():
    from tableqa.table import Table

    return Table.from_strings("players", ["Player", "Total Goals", "Country"], TABLE_ROWS)


def call(id, name, **arguments):
    return SimpleNamespace(type="tool_use", id=id, name=name, input=arguments)


def text(value):
    return SimpleNamespace(type="text", text=value)


def thinking():
    return SimpleNamespace(type="thinking", thinking="", signature="sig")


def reply(*content, stop_reason="tool_use", tokens=(1000, 100), cache=(0, 0)):
    usage = SimpleNamespace(
        input_tokens=tokens[0],
        output_tokens=tokens[1],
        cache_read_input_tokens=cache[0],
        cache_creation_input_tokens=cache[1],
    )
    return SimpleNamespace(content=list(content), stop_reason=stop_reason, usage=usage)


class ScriptedClient:
    """Returns scripted replies in order and records what it was sent."""

    def __init__(self, *replies):
        self._replies = list(replies)
        self.requests = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **request):
        self.requests.append(copy.deepcopy(request))
        return self._replies.pop(0)


def tool_results(request):
    """The tool_result blocks in the last message of a request."""
    return request["messages"][-1]["content"]


def test_inspects_queries_and_answers(table):
    client = ScriptedClient(
        reply(thinking(), call("t1", "get_schema"), call("t2", "sample_rows", n=2)),
        reply(call("t3", "run_sql", query="SELECT COUNT(*) FROM t WHERE country = 'Brazil'")),
        reply(call("t4", "final_answer", values=["2"])),
    )
    prediction = LLMAgentSystem(client=client).answer("how many players are from brazil?", table)

    assert prediction.answer == ["2"]
    assert prediction.error is None
    assert prediction.tokens == 3300
    assert prediction.cost_usd == pytest.approx((3000 * 0.10 + 300 * 0.50) / 1e6)
    assert prediction.latency_ms > 0

    first, second, third = client.requests
    assert first["model"] == "claude-haiku-5-5"
    assert first["tools"] == TOOL_DEFINITIONS
    assert first["output_config"] == {"effort": "medium"}
    assert first["cache_control"] == {"type": "ephemeral"}
    assert "temperature" not in first
    assert first["messages"] == [
        {"role": "user", "content": "Question: how many players are from brazil?"}
    ]

    # Both results of the first turn come back together, after the untouched assistant turn.
    assert [b.type for b in second["messages"][1]["content"]] == [
        "thinking",
        "tool_use",
        "tool_use",
    ]
    schema_result, sample_result = tool_results(second)
    assert (schema_result["tool_use_id"], sample_result["tool_use_id"]) == ("t1", "t2")
    assert json.loads(schema_result["content"]) == {
        "table": "t",
        "row_count": 3,
        "columns": [
            {"name": "player", "type": "VARCHAR", "header": "Player"},
            {"name": "total_goals", "type": "BIGINT", "header": "Total Goals"},
            {"name": "country", "type": "VARCHAR", "header": "Country"},
        ],
    }
    assert json.loads(sample_result["content"])["rows"] == [
        ["Ann", 10, "Brazil"],
        ["Bob", 7, "Chile"],
    ]
    (sql_result,) = tool_results(third)
    assert json.loads(sql_result["content"]) == {
        "columns": ["count_star()"],
        "rows": [[2]],
        "truncated": False,
    }
    assert sql_result["is_error"] is False

    steps = prediction.sql_or_trace
    assert [s["step"] for s in steps] == [1, 2, 3]
    assert [c["name"] for c in steps[0]["tool_calls"]] == ["get_schema", "sample_rows"]
    assert steps[1]["tool_calls"][0]["input"] == {
        "query": "SELECT COUNT(*) FROM t WHERE country = 'Brazil'"
    }
    assert steps[2]["tool_calls"] == [{"name": "final_answer", "input": {"values": ["2"]}}]


def test_a_failed_query_is_fed_back_so_the_model_can_repair_it(table):
    client = ScriptedClient(
        reply(call("t1", "run_sql", query="SELECT goals FROM t")),
        reply(call("t2", "run_sql", query="SELECT SUM(total_goals) FROM t")),
        reply(call("t3", "final_answer", values=["1217"])),
    )
    prediction = LLMAgentSystem(client=client).answer("total goals?", table)

    (failed,) = tool_results(client.requests[1])
    assert failed["is_error"] is True
    assert failed["content"].startswith("QueryFailed:")
    assert "goals" in failed["content"]
    assert prediction.answer == ["1217"]
    assert prediction.sql_or_trace[0]["tool_calls"][0]["error"] is True


@pytest.mark.parametrize(
    "query",
    ["DROP TABLE t", "DELETE FROM t", "SELECT 1; DROP TABLE t", "COPY t TO 'out.csv'"],
)
def test_the_model_cannot_write_through_run_sql(table, query):
    client = ScriptedClient(
        reply(call("t1", "run_sql", query=query)),
        reply(call("t2", "run_sql", query="SELECT COUNT(*) FROM t")),
        reply(call("t3", "final_answer", values=["3"])),
    )
    LLMAgentSystem(client=client).answer("how many rows?", table)

    (rejected,) = tool_results(client.requests[1])
    assert rejected["is_error"] is True
    assert rejected["content"].startswith("NotReadOnly:")
    (after,) = tool_results(client.requests[2])
    assert json.loads(after["content"])["rows"] == [[3]]


def test_long_results_are_cut_and_flagged(table):
    from tableqa.table import Table

    big = Table.from_strings("big", ["n"], [[str(i)] for i in range(80)])
    client = ScriptedClient(
        reply(call("t1", "run_sql", query="SELECT n FROM t ORDER BY n")),
        reply(call("t2", "final_answer", values=["80"])),
    )
    LLMAgentSystem(client=client).answer("how many?", big)

    result = json.loads(tool_results(client.requests[1])[0]["content"])
    assert len(result["rows"]) == 30
    assert result["truncated"] is True


def test_sample_rows_is_capped(table):
    client = ScriptedClient(
        reply(call("t1", "sample_rows", n=500)),
        reply(call("t2", "final_answer", values=["x"])),
    )
    LLMAgentSystem(client=client).answer("q", table)
    assert len(json.loads(tool_results(client.requests[1])[0]["content"])["rows"]) == 3


def test_an_unknown_tool_is_an_error_result_not_a_crash(table):
    client = ScriptedClient(
        reply(call("t1", "delete_everything")),
        reply(call("t2", "final_answer", values=["x"])),
    )
    prediction = LLMAgentSystem(client=client).answer("q", table)
    (result,) = tool_results(client.requests[1])
    assert (result["is_error"], result["content"]) == (True, "Unknown tool: delete_everything")
    assert prediction.answer == ["x"]


def test_answering_in_text_gets_a_reminder_to_use_final_answer(table):
    client = ScriptedClient(
        reply(text("The answer is 3."), stop_reason="end_turn"),
        reply(call("t1", "final_answer", values=["3"])),
    )
    prediction = LLMAgentSystem(client=client).answer("how many rows?", table)

    assert client.requests[1]["messages"][-1] == {"role": "user", "content": REMINDER}
    assert prediction.answer == ["3"]
    assert prediction.sql_or_trace[0]["text"] == ["The answer is 3."]


def test_step_budget_ends_the_question_without_an_answer(table):
    client = ScriptedClient(*[reply(call(f"t{i}", "run_sql", query="SELECT 1")) for i in range(3)])
    prediction = LLMAgentSystem(client=client, max_steps=3).answer("q", table)

    assert prediction.answer == []
    assert prediction.error == "step budget exhausted"
    assert len(client.requests) == 3
    assert "at most 3 turns" in client.requests[0]["system"]


def test_token_budget_ends_the_question_without_an_answer(table):
    client = ScriptedClient(
        reply(call("t1", "get_schema"), tokens=(40_000, 500)),
        reply(call("t2", "get_schema"), tokens=(40_000, 500)),
        reply(call("t3", "final_answer", values=["never reached"])),
    )
    prediction = LLMAgentSystem(client=client, max_question_tokens=60_000).answer("q", table)

    assert prediction.answer == []
    assert prediction.error == "token budget exhausted"
    assert len(client.requests) == 2
    assert prediction.tokens == 81_000


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
def test_a_refusal_or_cut_off_reply_is_recorded_as_an_error(table, stop_reason):
    client = ScriptedClient(reply(stop_reason=stop_reason))
    prediction = LLMAgentSystem(client=client).answer("q", table)

    assert prediction.answer == []
    assert prediction.error == f"model stopped: {stop_reason}"
    assert len(client.requests) == 1


def test_cost_counts_cache_reads_and_writes_at_their_own_rates(table):
    client = ScriptedClient(
        reply(call("t1", "get_schema"), tokens=(200, 50), cache=(0, 800)),
        reply(call("t2", "final_answer", values=["x"]), tokens=(300, 20), cache=(1000, 0)),
    )
    prediction = LLMAgentSystem(client=client).answer("q", table)

    input_units = (200 + 800 * 1.25) + (300 + 1000 * 0.1)
    assert prediction.cost_usd == pytest.approx((input_units * 0.10 + 70 * 0.50) / 1e6)
    assert prediction.tokens == 200 + 50 + 800 + 300 + 20 + 1000
    assert prediction.sql_or_trace[1]["cache_read_tokens"] == 1000


def test_an_api_failure_still_closes_the_sandbox_and_propagates(table):
    class Failing:
        def __init__(self):
            self.messages = SimpleNamespace(create=self._create)

        def _create(self, **request):
            raise RuntimeError("401 invalid x-api-key")

    with pytest.raises(RuntimeError, match="401"):
        LLMAgentSystem(client=Failing()).answer("q", table)


def test_config_records_what_defines_the_variant():
    config = LLMAgentSystem(client=ScriptedClient(), effort="low", max_steps=4).config
    assert config["model"] == "claude-haiku-5-5"
    assert (config["effort"], config["max_steps"]) == ("low", 4)
    assert "at most 4 turns" in config["system_prompt"]
    assert [t["name"] for t in config["tools"]] == [
        "get_schema",
        "sample_rows",
        "run_sql",
        "final_answer",
    ]
    assert all(
        t["strict"] and not t["input_schema"]["additionalProperties"] for t in config["tools"]
    )
