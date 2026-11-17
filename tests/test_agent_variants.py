"""The context-engineering variants of the agent, driven by the scripted stand-in model."""

import json
from pathlib import Path

import pytest

from tableqa.systems.llm_agent.system import (
    CHECK,
    DATA_NOTES,
    SYSTEM_PROMPT,
    LLMAgentSystem,
)
from test_llm_agent import ScriptedClient, call, reply, text, tool_results


@pytest.fixture
def table():
    from tableqa.table import Table

    return Table.from_strings(
        "players",
        ["Player", "Total Goals", "Country"],
        [["Ann", "10", "Brazil"], ["Bob", "7", "Chile"], ["Cy", "1,200", "Brazil"]],
    )


def test_the_default_agent_is_exactly_the_one_behind_the_committed_dev_result():
    summary = Path("results/wtq-dev-n500-seed42/agent_haiku.summary.json")
    committed = json.loads(summary.read_text(encoding="utf-8"))["config"]
    current = LLMAgentSystem(client=ScriptedClient()).config
    for key in committed:
        assert current[key] == committed[key], key


def test_without_samples_there_is_no_sample_tool_and_no_mention_of_it():
    agent = LLMAgentSystem(client=ScriptedClient(), samples=False)
    names = [t["name"] for t in agent.config["tools"]]
    assert names == ["get_schema", "run_sql", "final_answer"]
    assert "sample" not in agent.config["system_prompt"]
    assert "call get_schema. Column names" in agent.config["system_prompt"]


def test_profile_describes_the_values_in_each_column(table):
    client = ScriptedClient(
        reply(call("t1", "get_schema")),
        reply(call("t2", "final_answer", values=["x"])),
    )
    agent = LLMAgentSystem(client=client, profile=True)
    agent.answer("q", table)

    columns = json.loads(tool_results(client.requests[1])[0]["content"])["columns"]
    assert columns[0]["profile"] == {
        "filled": 3,
        "distinct": 3,
        "most_frequent": ["Ann", "Bob", "Cy"],
    }
    assert columns[1]["profile"] == {
        "filled": 3,
        "distinct": 3,
        "most_frequent": ["10", "7", "1200"],
        "min": 7,
        "max": 1200,
    }
    assert columns[2]["profile"]["most_frequent"][0] == "Brazil"
    schema_tool = next(t for t in agent.config["tools"] if t["name"] == "get_schema")
    assert "profile" in schema_tool["description"]


def test_check_answer_asks_once_and_accepts_the_second_answer(table):
    client = ScriptedClient(
        reply(call("t1", "final_answer", values=["5"])),
        reply(call("t2", "run_sql", query="SELECT COUNT(*) FROM t")),
        reply(call("t3", "final_answer", values=["3"])),
    )
    prediction = LLMAgentSystem(client=client, check_answer=True).answer("how many?", table)

    (check,) = tool_results(client.requests[1])
    assert (check["tool_use_id"], check["content"]) == ("t1", CHECK)
    assert prediction.answer == ["3"]
    assert len(client.requests) == 3
    assert prediction.sql_or_trace[0]["tool_calls"][0]["result"] == CHECK


def test_check_answer_keeps_the_first_answer_if_the_budget_runs_out(table):
    client = ScriptedClient(
        reply(call("t1", "final_answer", values=["5"])),
        reply(call("t2", "run_sql", query="SELECT 1")),
    )
    agent = LLMAgentSystem(client=client, check_answer=True, max_steps=2)
    prediction = agent.answer("q", table)
    assert (prediction.answer, prediction.error) == (["5"], None)


def test_single_shot_runs_the_one_query_and_reads_its_first_column(table):
    query = "SELECT player FROM t WHERE total_goals > 8"
    client = ScriptedClient(reply(call("t1", "submit_sql", query=query)))
    agent = LLMAgentSystem(client=client, single_shot=True)
    prediction = agent.answer("who scored more than 8?", table)

    assert prediction.answer == ["Ann", "Cy"]
    assert prediction.error is None
    (request,) = client.requests
    assert [t["name"] for t in request["tools"]] == ["submit_sql"]
    (message,) = request["messages"]
    assert message["content"].startswith("Schema:\n")
    assert '"header": "Total Goals"' in message["content"]
    assert '["Ann", 10, "Brazil"]' in message["content"]
    assert message["content"].endswith("Question: who scored more than 8?")
    (step,) = prediction.sql_or_trace
    assert step["tool_calls"][0]["input"] == {"query": query}


def test_single_shot_gets_no_second_chance(table):
    client = ScriptedClient(reply(call("t1", "submit_sql", query="SELECT goals FROM t")))
    prediction = LLMAgentSystem(client=client, single_shot=True).answer("q", table)
    assert prediction.answer == []
    assert prediction.error.startswith("QueryFailed:")
    assert len(client.requests) == 1

    client = ScriptedClient(reply(text("I think it is 3."), stop_reason="end_turn"))
    prediction = LLMAgentSystem(client=client, single_shot=True).answer("q", table)
    assert (prediction.answer, prediction.error) == ([], "no query submitted")


def test_single_shot_prompt_shares_the_data_notes():
    agent = LLMAgentSystem(client=ScriptedClient(), single_shot=True)
    assert DATA_NOTES.startswith("- The tables come from Wikipedia")
    assert DATA_NOTES in SYSTEM_PROMPT
    assert DATA_NOTES in agent.config["system_prompt"]
