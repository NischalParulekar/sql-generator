"""WikiTableQuestions loader.

Dataset: Pasupat & Liang, "Compositional Semantic Parsing on Semi-Structured Tables"
(ACL 2015), https://github.com/ppasupat/WikiTableQuestions, licensed CC BY-SA 4.0.
The release zip is downloaded once and read in place; it is never committed.

Splits:
- test:  pristine-unseen-tables (4,344 questions). Reported numbers come from here.
- dev:   random-split-1-dev, carved out of the training data. Use it for development
         and prompt tuning. TAPAS's WTQ checkpoint was fine-tuned on the training data,
         so TAPAS scores on dev are inflated and must not be compared across systems.
- train: training (14,152 questions).
"""

import argparse
import hashlib
import random
import re
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from tableqa.table import Table

URL = (
    "https://github.com/ppasupat/WikiTableQuestions/releases/download/"
    "v1.0.2/WikiTableQuestions-1.0.2-compact.zip"
)
SHA256 = "7c9ca7cc1ccd75fe4be0255b44be63f7b566761005f4ee6ce67e51c129d8b085"
DEFAULT_ZIP = Path("data/wtq/WikiTableQuestions-1.0.2-compact.zip")
ROOT = "WikiTableQuestions/"
SPLITS = {
    "test": "data/pristine-unseen-tables.tsv",
    "dev": "data/random-split-1-dev.tsv",
    "train": "data/training.tsv",
}

_ESCAPES = {"n": "\n", "p": "|", "\\": "\\"}


@dataclass(frozen=True)
class Example:
    id: str
    question: str
    table_id: str  # path of the table inside the dataset, e.g. csv/204-csv/590.csv
    answer: tuple[str, ...]


def download(dest: Path = DEFAULT_ZIP) -> Path:
    """Fetch the release zip (about 29 MB) unless it is already there, and verify it."""
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL) as response:
            dest.write_bytes(response.read())
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    if digest != SHA256:
        raise ValueError(f"{dest} has SHA-256 {digest}, expected {SHA256}")
    return dest


def _unescape(text: str) -> str:
    return re.sub(r"\\(.)", lambda m: _ESCAPES.get(m.group(1), m.group(0)), text)


class WTQ:
    def __init__(self, zip_path: Path = DEFAULT_ZIP):
        self._zip = zipfile.ZipFile(zip_path)

    def _lines(self, member: str) -> list[str]:
        # Split on "\n" only: cells may contain other characters str.splitlines() breaks on.
        return self._zip.read(ROOT + member).decode("utf-8").rstrip("\n").split("\n")

    def examples(self, split: str) -> list[Example]:
        rows = [line.split("\t") for line in self._lines(SPLITS[split])[1:]]
        return [
            Example(
                id=id,
                question=_unescape(question),
                table_id=table_id,
                answer=tuple(_unescape(value) for value in answer.split("|")),
            )
            for id, question, table_id, answer in rows
        ]

    def table(self, table_id: str) -> Table:
        lines = self._lines(table_id.removesuffix(".csv") + ".tsv")
        header, *rows = [[_unescape(cell) for cell in line.split("\t")] for line in lines]
        return Table.from_strings(table_id, header, rows)


def sample(examples: list[Example], n: int, seed: int) -> list[Example]:
    """A reproducible random subset, independent of the order the examples arrive in."""
    ordered = sorted(examples, key=lambda e: e.id)
    return random.Random(seed).sample(ordered, n)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download WikiTableQuestions")
    parser.add_argument("--dest", type=Path, default=DEFAULT_ZIP)
    print(download(parser.parse_args().dest))
