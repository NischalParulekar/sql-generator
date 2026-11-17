"""Compare two runs on the same questions, question by question.

    python -m tableqa.eval.compare BASELINE.predictions.jsonl VARIANT.predictions.jsonl

Two runs on the same questions differ only where one is right and the other wrong.
Counting those questions (fixed and broken) is more sensitive than comparing two
accuracies, and the exact sign test on them says how often a split at least that uneven
would happen if the two runs were really equally good.
"""

import argparse
import json
from math import comb
from pathlib import Path


def load(path: Path) -> dict[str, bool]:
    """Question id -> whether the run answered it correctly."""
    with open(path, encoding="utf-8") as f:
        return {r["id"]: r["correct"] for r in map(json.loads, f)}


def sign_test(fixed: int, broken: int) -> float:
    """Two-sided exact sign test: p-value for a fixed/broken split under no difference."""
    n = fixed + broken
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(fixed, broken) + 1)) / 2**n
    return min(1.0, 2 * tail)


def compare(baseline: dict[str, bool], variant: dict[str, bool]) -> dict:
    if baseline.keys() != variant.keys():
        raise ValueError("the two runs did not answer the same questions")
    fixed = sum(variant[q] and not baseline[q] for q in baseline)
    broken = sum(baseline[q] and not variant[q] for q in baseline)
    return {
        "n": len(baseline),
        "baseline_correct": sum(baseline.values()),
        "variant_correct": sum(variant.values()),
        "fixed": fixed,
        "broken": broken,
        "p_value": sign_test(fixed, broken),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("baseline", type=Path)
    parser.add_argument("variant", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(load(args.baseline), load(args.variant)), indent=2))


if __name__ == "__main__":
    main()
