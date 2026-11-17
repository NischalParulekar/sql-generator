"""Build the recorded demo for the static site: sample tables plus the answers and
traces each system gave in the committed benchmark runs.

    uv run python scripts/build_demo.py

Writes frontend/public/demo.json. The questions are a seeded pick from the WTQ dev
subset (dev, not test) and one of each question type from the 100,000-row synthetic
table. Every answer shown is exactly what the run recorded, right or wrong.
"""

import json
import random
from collections import defaultdict
from pathlib import Path

from tableqa.data import synthetic
from tableqa.data.wtq import WTQ, download

OUT = Path("frontend/public/demo.json")
SYSTEMS = [
    {"system": "rule_based", "name": "Rule-based", "paid": False},
    {"system": "tapas", "name": "TAPAS", "paid": False},
    {"system": "agent_haiku", "name": "Claude agent", "paid": True},
]
FIRST_ROWS = 8
WTQ_TABLES, PER_TABLE, SEED = 8, 3, 3


def predictions(run: str, system: str) -> dict[str, dict]:
    with open(f"results/{run}/{system}.predictions.jsonl", encoding="utf-8") as f:
        return {r["id"]: r for r in map(json.loads, f)}


def describe(table_id: str, name: str, table, questions: list[str]) -> dict:
    return {
        "table_id": table_id,
        "name": name,
        "row_count": len(table.rows),
        "columns": [
            {"name": c, "type": t, "header": h}
            for c, t, h in zip(table.columns, table.types, table.headers)
        ],
        "first_rows": [list(row) for row in table.raw_rows[:FIRST_ROWS]],
        "example_questions": questions,
    }


def recorded(record: dict) -> dict:
    return {
        "answer": record["answer"],
        "correct": record["correct"],
        "error": record["error"],
        "latency_ms": record["latency_ms"],
        "cost_usd": record["cost_usd"],
        "tokens": record["tokens"],
        "trace": record["sql_or_trace"],
    }


def add(bundle: dict, table_id: str, records: dict[str, dict]) -> None:
    first = next(iter(records.values()))
    bundle["answers"].setdefault(table_id, {})[first["question"]] = {
        "gold": first["gold"],
        "systems": {system: recorded(r) for system, r in records.items()},
    }


def main() -> None:
    bundle = {"systems": SYSTEMS, "tables": {}, "answers": {}}
    runs = {s["system"]: predictions("wtq-dev-n500-seed42", s["system"]) for s in SYSTEMS}

    by_table = defaultdict(list)
    for qid, record in runs["agent_haiku"].items():
        by_table[record["table_id"]].append(qid)
    tables = sorted(t for t, ids in by_table.items() if len(ids) >= 2)
    dataset = WTQ(download())
    for path in random.Random(SEED).sample(tables, WTQ_TABLES):
        table = dataset.table(path)
        table_id = "wtq-" + path.removeprefix("csv/").removesuffix(".csv").replace("/", "-")
        ids = sorted(by_table[path])[:PER_TABLE]
        named = [h for h in table.headers if h.strip()]
        name = ", ".join(named[:3]) + (" …" if len(named) > 3 else "")
        questions = [runs["agent_haiku"][q]["question"] for q in ids]
        bundle["tables"][table_id] = describe(table_id, name, table, questions)
        for qid in ids:
            add(bundle, table_id, {s: runs[s][qid] for s in runs})

    rows = 100_000
    table, _ = synthetic.generate(rows)
    run = f"synthetic-rows{rows}-seed42"
    syn = {s["system"]: predictions(run, s["system"]) for s in SYSTEMS}
    picks = [qid for qid in syn["agent_haiku"] if qid.endswith("-1")]
    table_id = "orders-100k"
    questions = [syn["agent_haiku"][q]["question"] for q in picks]
    bundle["tables"][table_id] = describe(
        table_id, "Orders (100,000 rows, generated)", table, questions
    )
    for qid in picks:
        add(bundle, table_id, {s: syn[s][qid] for s in syn})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(bundle, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    answered = sum(len(q) for q in bundle["answers"].values())
    print(f"{len(bundle['tables'])} tables, {answered} questions, {OUT.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
