"""Denotation accuracy: is the predicted answer the same set of values as the gold answer?

The rules follow the official WikiTableQuestions evaluator (evaluator.py, v1.0.2), with
one deliberate difference. The official evaluator gets canonical numbers and dates for
gold answers from CoreNLP annotations; this module only sees raw strings, so it parses
them itself: plain numbers, numbers with thousands separators ("12,467"), and
yyyy-mm-dd dates. A gold answer such as "June 5, 2004" therefore matches only as text.

A prediction is correct when, after removing duplicates,
1. it has as many values as the gold answer, and
2. every gold value matches some predicted value.

Two values match when their normalised text is identical, or both are numbers within
1e-6 of each other, or both are dates with identical year, month and day.
"""

import math
import re
import unicodedata
from dataclasses import dataclass

_GROUPED_NUMBER = re.compile(r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?")


def normalize(text: str) -> str:
    """Strip the formatting noise that should not decide whether an answer is right."""
    text = "".join(
        c for c in unicodedata.normalize("NFKD", text) if unicodedata.category(c) != "Mn"
    )
    text = re.sub(r"[‘’´`]", "'", text)
    text = re.sub(r"[“”]", '"', text)
    text = re.sub(r"[‐‑‒–—−]", "-", text)
    while True:
        before = text
        # Trailing citation marks: [1], [note 2], or footnote symbols.
        text = re.sub(r"((?<!^)\[[^\]]*\]|\[\d+\]|[•♦†‡*#+])*$", "", text.strip())
        # Trailing details in parentheses.
        text = re.sub(r"(?<!^)( \([^)]*\))*$", "", text.strip())
        # Outermost quotation marks.
        text = re.sub(r'^"([^"]*)"$', r"\1", text.strip())
        if text == before:
            break
    text = text.removesuffix(".")
    return re.sub(r"\s+", " ", text).lower().strip()


@dataclass(frozen=True)
class Value:
    normalized: str
    number: float | None = None
    date: tuple[int, int, int] | None = None  # (year, month, day); -1 for an unknown field

    def matches(self, other: "Value") -> bool:
        if self.normalized == other.normalized:
            return True
        if self.number is not None and other.number is not None:
            return abs(self.number - other.number) < 1e-6
        if self.date is not None and other.date is not None:
            return self.date == other.date
        return False

    @property
    def _identity(self):
        if self.number is not None:
            return ("number", round(self.number, 6))
        if self.date is not None:
            return ("date", self.date)
        return ("text", self.normalized)


def _parse_number(text: str) -> float | None:
    text = text.strip()
    if _GROUPED_NUMBER.fullmatch(text):
        text = text.replace(",", "")
    if "_" in text:  # float() accepts 1_000; an answer never means that
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return None if math.isnan(number) or math.isinf(number) else number


def _parse_date(text: str) -> tuple[int, int, int] | None:
    parts = text.strip().lower().split("-")
    if len(parts) != 3:
        return None
    try:
        year, month, day = (-1 if set(p) == {"x"} else int(p) for p in parts)
    except ValueError:
        return None
    if year == month == day == -1:
        return None
    if not (month == -1 or 1 <= month <= 12) or not (day == -1 or 1 <= day <= 31):
        return None
    return year, month, day


def to_value(text: str) -> Value:
    normalized = normalize(text)
    number = _parse_number(text)
    if number is not None:
        return Value(normalized, number=number)
    date = _parse_date(text)
    if date is not None:
        if date[1] == date[2] == -1:  # only the year is known: treat as a number
            return Value(normalized, number=float(date[0]))
        return Value(normalized, date=date)
    return Value(normalized)


def _unique_values(texts) -> list[Value]:
    return list({v._identity: v for v in map(to_value, texts)}.values())


def denotation_match(predicted: list[str], gold: list[str]) -> bool:
    predicted_values, gold_values = _unique_values(predicted), _unique_values(gold)
    if len(predicted_values) != len(gold_values):
        return False
    return all(any(g.matches(p) for p in predicted_values) for g in gold_values)


def percentile(values: list[float], q: float) -> float:
    """The q-th percentile (0-100) by linear interpolation between closest ranks."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    low = math.floor(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)
