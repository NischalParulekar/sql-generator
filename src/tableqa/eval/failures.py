"""Failure taxonomy: which failures to label, the categories, and the tally.

    python -m tableqa.eval.failures export      # write the review files
    python -m tableqa.eval.failures tally       # count the labels

Every agent failure on the test subset is labelled, and a seeded sample of 50 failures
for the rule-based system and for TAPAS (truncated). Each failure gets one primary
category. The categories were fixed before labelling began.

A first-pass label is proposed for every failure; a person then reviews each one and
either confirms it or replaces it. Only reviewed labels are tallied.
"""

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

RESULTS = Path("results/wtq-test-n500-seed42")
OUT = Path("results/failures")
SAMPLE = {"rule_based": 50, "tapas": 50, "agent_haiku": None}  # None: every failure
SEED = 42

# Checked in this order: the first category that applies is the label. A broken item
# comes first: if the gold answer is wrong, nothing the system did is the cause.
CATEGORIES = {
    "gold_questionable": "The gold answer is wrong, depends on the date the question was "
    "written, or needs information the table does not contain; or the question is "
    "ambiguous enough that the prediction is a defensible reading. Say why in the notes.",
    "execution_error": "The system produced a query that failed to run, or ran out of "
    "budget, and gave no answer.",
    "declined": "The system said the question cannot be answered from the table.",
    "not_understood": "The output does not engage with the question at all, such as the "
    "rule engine's SELECT * fallback, or a cell where the answer must be yes or no.",
    "missing_context": "The answer depends on rows or columns the system never saw, such as "
    "rows TAPAS dropped to fit its 512-token window.",
    "answer_form": "The right value or entity in the wrong form: missing or extra units, "
    "a shorter or longer name, extra explanation, different rounding.",
    "misread_question": "The system answered a different question: the wrong target or "
    "entity, a condition ignored, or the wrong thing counted or compared.",
    "wrong_column": "The right rows, but the answer was read from the wrong column.",
    "wrong_rows": "The wrong rows were selected: a wrong filter, an off-by-one in order or "
    "position, items missing from or added to a list, a totals row counted.",
    "wrong_calculation": "The right rows and column, but the arithmetic was wrong: count "
    "instead of sum, a date or duration worked out wrongly, a messy number misparsed.",
    "hallucinated_value": "The answer is neither in the table nor derivable from it.",
}

FIELDS = ["system", "id", "question", "gold", "answer", "evidence", "first_pass", "notes"]


def load(system: str) -> list[dict]:
    with open(RESULTS / f"{system}.predictions.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def failures(records: list[dict], n: int | None, seed: int = SEED) -> list[dict]:
    """All failures, or a seeded sample of n of them, in their original order."""
    wrong = [r for r in records if not r["correct"]]
    if n is None or n >= len(wrong):
        return wrong
    keep = set(random.Random(seed).sample(range(len(wrong)), n))
    return [r for i, r in enumerate(wrong) if i in keep]


def evidence(system: str, record: dict) -> str:
    """What the system did, in a few lines a reviewer can read."""
    trace = record["sql_or_trace"]
    if record["error"] and not trace:
        return f"error: {record['error']}"
    if system == "rule_based":
        sql = " ".join(trace.split())
        return f"SQL: {sql}" + (f"\nerror: {record['error']}" if record["error"] else "")
    if system.startswith("tapas"):
        step = trace[0]
        return (
            f"operator: {step['operator']}; cells (row, column): {step['cells']}; "
            f"rows dropped to fit: {step['rows_dropped']}"
        )
    lines = []
    for step in trace:
        for call in step["tool_calls"]:
            if call["name"] == "run_sql":
                result = str(call.get("result", ""))[:300]
                lines.append(f"SQL: {' '.join(call['input']['query'].split())}\n -> {result}")
    if record["error"]:
        lines.append(f"error: {record['error']}")
    return "\n".join(lines[-3:]) or "(no query run)"


def export(first_pass: dict[tuple[str, str], tuple[str, str]]) -> list[dict]:
    """The rows to review: one per failure, with any first-pass label and notes."""
    rows = []
    for system, n in SAMPLE.items():
        for record in failures(load(system), n):
            label, notes = first_pass.get((system, record["id"]), ("", ""))
            rows.append(
                {
                    "system": system,
                    "id": record["id"],
                    "question": record["question"],
                    "gold": " | ".join(record["gold"]),
                    "answer": " | ".join(record["answer"]),
                    "evidence": evidence(system, record),
                    "first_pass": label,
                    "notes": notes,
                }
            )
    return rows


def read_labels(path: Path) -> dict[tuple[str, str], tuple[str, str]]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8", newline="") as f:
        return {(r["system"], r["id"]): (r["label"], r["notes"]) for r in csv.DictReader(f)}


def tally(labels: dict[tuple[str, str], tuple[str, str]]) -> dict[str, Counter]:
    """Label counts per system; unknown labels are an error, not a silent extra column."""
    counts: dict[str, Counter] = {}
    for (system, _), (label, _) in labels.items():
        if label not in CATEGORIES:
            raise ValueError(f"unknown label {label!r}")
        counts.setdefault(system, Counter())[label] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["export", "tally"])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.command == "export":
        rows = export(read_labels(OUT / "first_pass.csv"))
        with open(OUT / "review.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        print(f"{len(rows)} failures written to {OUT / 'review.jsonl'}")
    else:
        for system, counts in tally(read_labels(OUT / "reviewed.csv")).items():
            print(system, dict(counts.most_common()))


if __name__ == "__main__":
    main()
