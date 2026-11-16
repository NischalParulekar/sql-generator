"""The table every system is asked about.

Raw tables (WTQ, uploads) arrive as strings under free-text headers. SQL systems need
identifiers they can write unquoted and columns they can do arithmetic on, so a Table
carries SQL-safe column names and inferred types next to the original headers.
"""

import re
import unicodedata
from dataclasses import dataclass
from functools import cache

import duckdb

_INT64_MAX = 2**63 - 1
_NUMBER = re.compile(r"-?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?")


@dataclass(frozen=True)
class Table:
    id: str
    headers: tuple[str, ...]  # original header text
    columns: tuple[str, ...]  # unique SQL-safe names, same order as headers
    types: tuple[str, ...]  # DuckDB type per column: BIGINT, DOUBLE or VARCHAR
    rows: tuple[tuple, ...]  # typed cell values; None for an empty cell
    raw_rows: tuple[tuple[str, ...], ...]  # the same cells as their original text

    @classmethod
    def from_strings(cls, id: str, headers: list[str], rows: list[list[str]]) -> "Table":
        """Build a table from string cells, inferring column names and types."""
        width = len(headers)
        cells = [list(row[:width]) + [""] * (width - len(row)) for row in rows]
        types, typed_columns = [], []
        for i in range(width):
            dtype, values = _infer_column([row[i].strip() for row in cells])
            types.append(dtype)
            typed_columns.append(values)
        return cls(
            id=id,
            headers=tuple(headers),
            columns=tuple(_sql_names(headers)),
            types=tuple(types),
            rows=tuple(zip(*typed_columns)) if width else (),
            raw_rows=tuple(tuple(row) for row in cells),
        )


def parse_number(text: str) -> int | float | None:
    """Parse '42', '-3.5' or '1,234,567' as a number; anything else is None.

    Digit strings with a leading zero ('007', '02134') are codes, not numbers.
    """
    if not _NUMBER.fullmatch(text):
        return None
    digits = text.replace(",", "")
    whole = digits.lstrip("-").split(".")[0]
    if len(whole) > 1 and whole.startswith("0"):
        return None
    if "." in digits:
        return float(digits)
    value = int(digits)
    return value if abs(value) <= _INT64_MAX else None


def _infer_column(cells: list[str]) -> tuple[str, list]:
    """A column is numeric only if every non-empty cell parses as a number."""
    parsed = [parse_number(c) if c else None for c in cells]
    filled = [c for c in cells if c]
    if filled and all(p is not None for c, p in zip(cells, parsed) if c):
        if any(isinstance(p, float) for p in parsed):
            return "DOUBLE", [None if p is None else float(p) for p in parsed]
        return "BIGINT", parsed
    return "VARCHAR", [c or None for c in cells]


@cache
def _unsafe_identifiers() -> frozenset[str]:
    """DuckDB keywords that cannot be used as a bare column name."""
    rows = duckdb.sql(
        "SELECT keyword_name FROM duckdb_keywords() "
        "WHERE keyword_category IN ('reserved', 'type_function')"
    ).fetchall()
    return frozenset(name for (name,) in rows)


def _sql_names(headers: list[str]) -> list[str]:
    names: list[str] = []
    for i, header in enumerate(headers):
        ascii_header = unicodedata.normalize("NFKD", header).encode("ascii", "ignore").decode()
        name = re.sub(r"[^a-z0-9]+", "_", ascii_header.lower()).strip("_")
        if not name:
            name = f"col_{i + 1}"
        if name[0].isdigit():
            name = f"c_{name}"
        if name in _unsafe_identifiers():
            name += "_"
        base, n = name, 2
        while name in names:
            name = f"{base}_{n}"
            n += 1
        names.append(name)
    return names
