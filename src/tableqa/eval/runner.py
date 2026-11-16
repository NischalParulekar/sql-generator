"""Run one system over a seeded WikiTableQuestions subset.

    python -m tableqa.eval.runner --system rule_based --split dev --n 500 --seed 42

Writes, under results/wtq-<split>-n<n>-seed<seed>/:
- <system>.predictions.jsonl  one line per question with the raw prediction, so anyone
                              can re-score it;
- <system>.summary.json       accuracy, error counts, latency p50/p95, cost per 1k.
"""

import argparse
import functools
import json
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from tableqa.data.wtq import WTQ, Example, download, sample
from tableqa.eval.metrics import denotation_match, percentile
from tableqa.interface import Prediction, System
from tableqa.systems.rule_based.system import RuleBasedSystem
from tableqa.table import Table

SYSTEMS: dict[str, Callable[[], System]] = {
    "rule_based": RuleBasedSystem,
}


def run(system: System, examples: list[Example], get_table: Callable[[str], Table]) -> list[dict]:
    records = []
    for example in examples:
        table = get_table(example.table_id)
        start = time.perf_counter()
        try:
            prediction = system.answer(example.question, table)
        except Exception as e:  # noqa: BLE001 - one broken question must not lose a paid run
            prediction = Prediction(
                answer=[],
                sql_or_trace=None,
                latency_ms=(time.perf_counter() - start) * 1000,
                error=f"{type(e).__name__}: {e}",
            )
        records.append(
            {
                "id": example.id,
                "question": example.question,
                "table_id": example.table_id,
                "gold": list(example.answer),
                "correct": denotation_match(prediction.answer, list(example.answer)),
                **asdict(prediction),
            }
        )
    return records


def summarize(records: list[dict]) -> dict:
    n = len(records)
    latencies = [r["latency_ms"] for r in records]
    correct = sum(r["correct"] for r in records)
    return {
        "n": n,
        "correct": correct,
        "accuracy": correct / n,
        "errors": sum(r["error"] is not None for r in records),
        "no_answer": sum(not r["answer"] for r in records),
        "latency_ms_p50": percentile(latencies, 50),
        "latency_ms_p95": percentile(latencies, 95),
        "cost_usd_per_1k": sum(r["cost_usd"] for r in records) / n * 1000,
        "tokens": sum(r["tokens"] for r in records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--system", choices=sorted(SYSTEMS), required=True)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    dataset = WTQ(download())
    examples = sample(dataset.examples(args.split), args.n, args.seed)
    system = SYSTEMS[args.system]()
    records = run(system, examples, functools.cache(dataset.table))

    summary = {
        "system": system.name,
        "dataset": "wtq",
        "split": args.split,
        "seed": args.seed,
        **summarize(records),
    }
    out_dir = args.out / f"wtq-{args.split}-n{args.n}-seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{system.name}.predictions.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    with open(out_dir / f"{system.name}.summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
