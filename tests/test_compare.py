import json

import pytest

from tableqa.eval.compare import compare, load, sign_test


def test_sign_test():
    assert sign_test(0, 0) == 1.0
    assert sign_test(5, 5) == 1.0
    assert sign_test(10, 0) == pytest.approx(2 / 1024)
    assert sign_test(0, 10) == sign_test(10, 0)
    assert sign_test(16, 10) == pytest.approx(0.3269, abs=1e-4)
    assert sign_test(20, 5) == pytest.approx(0.0041, abs=1e-4)


def test_compare_counts_fixed_and_broken_questions():
    baseline = {"a": True, "b": True, "c": False, "d": False}
    variant = {"a": True, "b": False, "c": True, "d": True}
    assert compare(baseline, variant) == {
        "n": 4,
        "baseline_correct": 2,
        "variant_correct": 3,
        "fixed": 2,
        "broken": 1,
        "p_value": 1.0,
    }


def test_compare_refuses_runs_on_different_questions():
    with pytest.raises(ValueError, match="same questions"):
        compare({"a": True}, {"b": True})


def test_load_reads_predictions(tmp_path):
    path = tmp_path / "p.jsonl"
    rows = [{"id": "a", "correct": True}, {"id": "b", "correct": False}]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    assert load(path) == {"a": True, "b": False}
