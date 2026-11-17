"""Sample tables the demo offers: a seeded handful of WTQ dev tables, each with example
questions from the dataset, and one generated 10,000-row order table."""

import hashlib
import json
import random
from dataclasses import dataclass, field

from tableqa.data import synthetic
from tableqa.data.wtq import WTQ
from tableqa.table import Table


@dataclass
class Sample:
    id: str
    name: str
    table: Table
    questions: list[str] = field(default_factory=list)


def fingerprint(table: Table) -> str:
    """A content hash, so cached answers follow the data rather than the name."""
    body = json.dumps([table.headers, table.raw_rows], ensure_ascii=False)
    return hashlib.sha256(body.encode()).hexdigest()


def wtq_samples(dataset: WTQ, n: int = 10, seed: int = 7) -> list[Sample]:
    """n dev tables of 8 to 200 rows, picked with a fixed seed, with up to 3 of their
    questions. Dev, not test: the demo must not show the held-out questions."""
    by_table: dict[str, list[str]] = {}
    for example in dataset.examples("dev"):
        by_table.setdefault(example.table_id, []).append(example.question)
    candidates = sorted(by_table)
    random.Random(seed).shuffle(candidates)
    samples = []
    for table_id in candidates:
        table = dataset.table(table_id)
        if not 8 <= len(table.rows) <= 200:
            continue
        sample_id = "wtq-" + table_id.removeprefix("csv/").removesuffix(".csv").replace("/", "-")
        name = ", ".join(table.headers[:3]) + (" …" if len(table.headers) > 3 else "")
        samples.append(Sample(sample_id, name, table, by_table[table_id][:3]))
        if len(samples) == n:
            break
    return samples


def synthetic_sample(rows: int = 10_000) -> Sample:
    table, examples = synthetic.generate(rows)
    picks = [e.question for e in examples if e.type in ("count", "top_n", "multi_step")][::5][:3]
    return Sample(f"orders-{rows // 1000}k", f"Orders ({rows:,} rows, generated)", table, picks)
