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
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict
from pathlib import Path

from tableqa.data.wtq import WTQ, Example, download, sample
from tableqa.eval.metrics import denotation_match, percentile
from tableqa.interface import Prediction, System
from tableqa.systems.rule_based.system import RuleBasedSystem
from tableqa.table import Table


def _tapas(variant: str) -> System:
    # Imported on demand: TAPAS needs the optional torch/transformers dependencies.
    from tableqa.systems.tapas.system import TapasSystem

    return TapasSystem(variant)


def _agent(name: str, model: str) -> System:
    # Imported on demand: the agent needs the optional anthropic dependency and an API
    # key, which is read from the environment or from a git-ignored .env file.
    from dotenv import load_dotenv

    from tableqa.systems.llm_agent.system import LLMAgentSystem

    load_dotenv()
    return LLMAgentSystem(name, model)


SYSTEMS: dict[str, Callable[[], System]] = {
    "rule_based": RuleBasedSystem,
    "tapas": functools.partial(_tapas, "tapas"),
    "tapas_chunked": functools.partial(_tapas, "tapas_chunked"),
    "tapas_chunked_routed": functools.partial(_tapas, "tapas_chunked_routed"),
    "agent_haiku": functools.partial(_agent, "agent_haiku", "claude-haiku-5-5"),
}


def run(
    system: System, examples: list[Example], get_table: Callable[[str], Table]
) -> Iterator[dict]:
    """Answer and score each question, yielding one record at a time."""
    for done, example in enumerate(examples):
        if done and done % 50 == 0:
            print(f"{done}/{len(examples)} questions", file=sys.stderr)
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
        yield {
            "id": example.id,
            "question": example.question,
            "table_id": example.table_id,
            "gold": list(example.answer),
            "correct": denotation_match(prediction.answer, list(example.answer)),
            **asdict(prediction),
        }


MAX_CRASHES_IN_A_ROW = 5


def run_and_save(
    system: System,
    examples: list[Example],
    get_table: Callable[[str], Table],
    path: Path,
    max_cost: float | None = None,
) -> tuple[list[dict], bool]:
    """Run the system, writing each prediction as it arrives.

    An interrupted paid run keeps the work it has paid for. With max_cost, the run stops
    once it has spent that many dollars. Returns (records, stopped early at the cap).

    A system that crashes on several questions in a row is broken, not unlucky (a missing
    or invalid API key, no credit left, a missing dependency), so the run is aborted
    rather than allowed to finish with a meaningless score.
    """
    records, spent, crashes_in_a_row = [], 0.0, 0
    with open(path, "w", encoding="utf-8") as f:
        for record in run(system, examples, get_table):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            records.append(record)
            spent += record["cost_usd"]
            # run() records a crash as an error with no SQL or trace at all.
            crashed = record["error"] is not None and record["sql_or_trace"] is None
            crashes_in_a_row = crashes_in_a_row + 1 if crashed else 0
            if crashes_in_a_row == MAX_CRASHES_IN_A_ROW:
                raise SystemExit(
                    f"Stopped: {system.name} crashed on {MAX_CRASHES_IN_A_ROW} questions in a "
                    f"row. Last error: {record['error']}"
                )
            if max_cost is not None and spent >= max_cost and len(records) < len(examples):
                print(f"cost cap: ${spent:.4f} spent, stopping", file=sys.stderr)
                return records, True
    return records, False


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
    parser.add_argument(
        "--max-cost",
        type=float,
        help="stop once the run has spent this many US dollars (paid systems only)",
    )
    args = parser.parse_args()

    dataset = WTQ(download())
    examples = sample(dataset.examples(args.split), args.n, args.seed)
    system = SYSTEMS[args.system]()

    out_dir = args.out / f"wtq-{args.split}-n{args.n}-seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    records, stopped_at_cost_cap = run_and_save(
        system,
        examples,
        functools.cache(dataset.table),
        out_dir / f"{system.name}.predictions.jsonl",
        args.max_cost,
    )

    summary = {
        "system": system.name,
        "dataset": "wtq",
        "split": args.split,
        "seed": args.seed,
        **summarize(records),
        "cost_usd": sum(r["cost_usd"] for r in records),
        "stopped_at_cost_cap": stopped_at_cost_cap,
    }
    if hasattr(system, "config"):
        summary["config"] = system.config
    with open(out_dir / f"{system.name}.summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2))


if __name__ == "__main__":
    main()
