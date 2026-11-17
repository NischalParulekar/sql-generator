"""The recorded demo bundle the static site replays (built by scripts/build_demo.py)."""

import json
from pathlib import Path

BUNDLE = Path("frontend/public/demo.json")
RESULTS = Path("results")


def test_every_example_question_has_a_recorded_answer_from_every_system():
    bundle = json.loads(BUNDLE.read_text(encoding="utf-8"))
    systems = {s["system"] for s in bundle["systems"]}
    assert systems == {"rule_based", "tapas", "agent_haiku"}
    assert bundle["tables"].keys() == bundle["answers"].keys()
    for table_id, table in bundle["tables"].items():
        assert table["example_questions"]
        assert table["name"] and not table["name"].startswith(",")
        for question in table["example_questions"]:
            entry = bundle["answers"][table_id][question]
            assert entry["systems"].keys() == systems
            assert entry["gold"]


def test_recorded_answers_match_the_committed_runs():
    bundle = json.loads(BUNDLE.read_text(encoding="utf-8"))
    runs = {}
    for path in RESULTS.glob("*/agent_haiku.predictions.jsonl"):
        with open(path, encoding="utf-8") as f:
            for record in map(json.loads, f):
                runs[(path.parent.name, record["question"])] = record
    checked = 0
    for table_id, questions in bundle["answers"].items():
        run = "synthetic-rows100000-seed42" if table_id == "orders-100k" else "wtq-dev-n500-seed42"
        for question, entry in questions.items():
            record = runs[(run, question)]
            assert entry["systems"]["agent_haiku"]["answer"] == record["answer"]
            assert entry["systems"]["agent_haiku"]["correct"] == record["correct"]
            checked += 1
    assert checked >= 20
