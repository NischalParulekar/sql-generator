import json

import pytest

from tableqa.data.wtq import Example
from tableqa.eval.runner import run, run_and_save, summarize
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

    records = list(run(system, examples, {"players": TABLE}.__getitem__))

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
    records = list(run(RuleBasedSystem(), examples, {"players": TABLE}.__getitem__))
    assert [r["correct"] for r in records] == [True, False]
    assert records[0]["answer"] == ["3"]


def _paid_examples(n):
    system = ScriptedSystem({f"q{i}": (["a"], 5.0) for i in range(n)})  # $0.002 per question
    examples = [Example(f"id{i}", f"q{i}", "players", ("a",)) for i in range(n)]
    return system, examples


def test_predictions_are_written_one_by_one(tmp_path):
    system, examples = _paid_examples(3)
    path = tmp_path / "predictions.jsonl"

    records, stopped = run_and_save(system, examples, {"players": TABLE}.__getitem__, path)

    assert stopped is False
    saved = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert saved == records
    assert [r["id"] for r in saved] == ["id0", "id1", "id2"]


def test_a_run_stops_once_it_has_spent_the_cost_cap(tmp_path):
    system, examples = _paid_examples(10)
    path = tmp_path / "predictions.jsonl"

    records, stopped = run_and_save(
        system, examples, {"players": TABLE}.__getitem__, path, max_cost=0.005
    )

    assert stopped is True
    assert len(records) == 3  # 0.002 + 0.002 + 0.002 crosses 0.005
    assert len(path.read_text(encoding="utf-8").splitlines()) == 3


def test_a_cap_reached_on_the_last_question_is_not_an_early_stop(tmp_path):
    system, examples = _paid_examples(3)
    records, stopped = run_and_save(
        system, examples, {"players": TABLE}.__getitem__, tmp_path / "p.jsonl", max_cost=0.005
    )
    assert (len(records), stopped) == (3, False)


def test_work_already_done_survives_a_crash_mid_run(tmp_path):
    class Dying:
        name = "dying"

        def __init__(self):
            self.calls = 0

        def answer(self, question, table):
            self.calls += 1
            if self.calls == 3:
                raise KeyboardInterrupt
            return Prediction(["a"], None, 1.0)

    _, examples = _paid_examples(5)
    path = tmp_path / "predictions.jsonl"
    with pytest.raises(KeyboardInterrupt):
        run_and_save(Dying(), examples, {"players": TABLE}.__getitem__, path)
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2


def test_a_system_that_keeps_crashing_aborts_the_run(tmp_path):
    system = ScriptedSystem({})  # raises KeyError for every question, like a missing API key
    examples = [Example(f"id{i}", f"q{i}", "players", ("a",)) for i in range(20)]
    path = tmp_path / "predictions.jsonl"

    with pytest.raises(SystemExit, match="crashed on 5 questions in a row"):
        run_and_save(system, examples, {"players": TABLE}.__getitem__, path)
    assert len(path.read_text(encoding="utf-8").splitlines()) == 5


def test_scattered_crashes_and_ordinary_errors_do_not_abort_the_run(tmp_path):
    # Every third question crashes; the rule-based system's failed queries are errors
    # with SQL attached, which are results, not crashes.
    script = {f"q{i}": (["a"], 1.0) for i in range(12) if i % 3}
    examples = [Example(f"id{i}", f"q{i}", "players", ("a",)) for i in range(12)]
    records, _ = run_and_save(
        ScriptedSystem(script), examples, {"players": TABLE}.__getitem__, tmp_path / "a.jsonl"
    )
    assert sum(r["error"] is not None for r in records) == 4

    failing_sql = [
        Example(f"id{i}", "players from last 30 days", "players", ("a",)) for i in range(8)
    ]
    records, _ = run_and_save(
        RuleBasedSystem(), failing_sql, {"players": TABLE}.__getitem__, tmp_path / "b.jsonl"
    )
    assert len(records) == 8 and all(r["error"].startswith("QueryFailed") for r in records)
