import json
from collections import Counter
from pathlib import Path

import pytest

from tableqa.eval import failures as f


def records(n_wrong, n_right=3):
    wrong = [{"id": f"w{i}", "correct": False} for i in range(n_wrong)]
    right = [{"id": f"r{i}", "correct": True} for i in range(n_right)]
    return right + wrong


def test_failures_samples_reproducibly_and_keeps_order():
    first = f.failures(records(80), 50)
    assert len(first) == 50 and all(not r["correct"] for r in first)
    assert first == f.failures(records(80), 50)
    ids = [int(r["id"][1:]) for r in first]
    assert ids == sorted(ids)


def test_failures_takes_everything_when_asked_or_when_there_are_few():
    assert len(f.failures(records(20), None)) == 20
    assert len(f.failures(records(20), 50)) == 20


def test_gold_questionable_is_checked_first():
    assert next(iter(f.CATEGORIES)) == "gold_questionable"


def test_tally_counts_per_system_and_rejects_unknown_labels():
    labels = {
        ("agent_haiku", "a"): ("answer_form", ""),
        ("agent_haiku", "b"): ("answer_form", ""),
        ("tapas", "c"): ("wrong_rows", ""),
    }
    assert f.tally(labels) == {
        "agent_haiku": Counter({"answer_form": 2}),
        "tapas": Counter({"wrong_rows": 1}),
    }
    with pytest.raises(ValueError, match="unknown label"):
        f.tally({("tapas", "x"): ("typo_label", "")})


def test_committed_first_pass_covers_exactly_the_sample():
    labels = f.read_labels(Path("results/failures/first_pass.csv"))
    assert all(label in f.CATEGORIES for label, _ in labels.values())
    for system, n in f.SAMPLE.items():
        sampled = {r["id"] for r in f.failures(f.load(system), n)}
        labelled = {i for s, i in labels if s == system}
        assert labelled == sampled, system


def test_committed_review_file_matches_the_first_pass():
    with open("results/failures/review.jsonl", encoding="utf-8") as file:
        rows = [json.loads(line) for line in file]
    labels = f.read_labels(Path("results/failures/first_pass.csv"))
    assert len(rows) == len(labels) == 169
    assert all(labels[(r["system"], r["id"])][0] == r["first_pass"] for r in rows)
