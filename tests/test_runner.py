import pytest

from tableqa.data.wtq import Example
from tableqa.eval.runner import run, summarize
from tableqa.interface import Prediction
from tableqa.systems.rule_based.system import RuleBasedSystem
from tableqa.table import Table

TABLE = Table.from_strings(
    "players",
    ["Player", "Goals"],
    [["Ann", "10"], ["Bob", "7"], ["Cy", "12"]],
)


class ScriptedSystem:
    """Answers from a script; raises for questions it has no line for."""

    name = "scripted"

    def __init__(self, script):
        self.script = script

    def answer(self, question, table):
        answer, latency = self.script[question]
        return Prediction(answer, "trace", latency, cost_usd=0.002, tokens=50)


def test_run_scores_each_question_and_survives_a_crashing_system():
    examples = [
        Example("q1", "top scorer?", "players", ("Cy",)),
        Example("q2", "how many players?", "players", ("3",)),
        Example("q3", "unscripted", "players", ("x",)),
    ]
    system = ScriptedSystem({"top scorer?": (["cy"], 10.0), "how many players?": (["4"], 30.0)})

    records = run(system, examples, {"players": TABLE}.__getitem__)

    assert [r["correct"] for r in records] == [True, False, False]
    assert records[0] == {
        "id": "q1",
        "question": "top scorer?",
        "table_id": "players",
        "gold": ["Cy"],
        "correct": True,
        "answer": ["cy"],
        "sql_or_trace": "trace",
        "latency_ms": 10.0,
        "cost_usd": 0.002,
        "tokens": 50,
        "error": None,
    }
    assert records[2]["answer"] == []
    assert records[2]["error"] == "KeyError: 'unscripted'"


def test_summarize():
    base = {"cost_usd": 0.002, "tokens": 50, "error": None}
    records = [
        {**base, "correct": True, "answer": ["a"], "latency_ms": 10.0},
        {**base, "correct": False, "answer": ["b"], "latency_ms": 20.0},
        {**base, "correct": False, "answer": [], "latency_ms": 30.0, "error": "QueryFailed: x"},
        {**base, "correct": True, "answer": ["d"], "latency_ms": 40.0},
    ]
    assert summarize(records) == {
        "n": 4,
        "correct": 2,
        "accuracy": 0.5,
        "errors": 1,
        "no_answer": 1,
        "latency_ms_p50": 25.0,
        "latency_ms_p95": pytest.approx(38.5),
        "cost_usd_per_1k": pytest.approx(2.0),
        "tokens": 200,
    }


def test_rule_based_system_end_to_end_through_the_runner():
    examples = [
        Example("q1", "how many players are there", "players", ("3",)),
        Example("q2", "what is the average goals", "players", ("9.67",)),  # rounded: no match
    ]
    records = run(RuleBasedSystem(), examples, {"players": TABLE}.__getitem__)
    assert [r["correct"] for r in records] == [True, False]
    assert records[0]["answer"] == ["3"]
