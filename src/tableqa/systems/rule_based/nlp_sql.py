"""
Rule-based NLP → SQL converter.
No AI or ML models — pure pattern matching and heuristics.
"""

import re
from dataclasses import dataclass, field

# ─── Schema ────────────────────────────────────────────────────────────────────

@dataclass
class Column:
    name: str
    dtype: str = "TEXT"

    @property
    def is_numeric(self) -> bool:
        return any(t in self.dtype.upper() for t in
                   ["INT", "FLOAT", "DOUBLE", "DECIMAL", "NUMERIC", "REAL", "BIGINT", "SMALLINT"])

    @property
    def is_date(self) -> bool:
        return any(t in self.dtype.upper() for t in ["DATE", "TIME", "TIMESTAMP"])


@dataclass
class Table:
    name: str
    columns: list[Column] = field(default_factory=list)

    def find_col(self, keyword: str) -> Column | None:
        kw = keyword.lower().replace(" ", "_").replace("-", "_")
        cmap = {c.name.lower(): c for c in self.columns}
        if kw in cmap:
            return cmap[kw]
        for name, col in cmap.items():
            if kw in name or name in kw:
                return col
        return None

    @property
    def col_names(self) -> list[str]:
        return [c.name for c in self.columns]


def parse_schema(sql: str) -> list[Table]:
    """Extract tables + columns from CREATE TABLE statements."""
    tables: list[Table] = []
    pattern = re.compile(
        r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[`\"\[]?(\w+)[`\"\]]?\s*\(([^;]+)\)",
        re.IGNORECASE | re.DOTALL,
    )
    for m in pattern.finditer(sql):
        tname = m.group(1)
        body = m.group(2)
        cols: list[Column] = []
        for line in re.split(r",(?:\s*\n?)", body):
            line = line.strip()
            if re.match(r"(PRIMARY|FOREIGN|UNIQUE|INDEX|KEY|CONSTRAINT|CHECK)\b", line, re.IGNORECASE):
                continue
            cm = re.match(r"[`\"\[]?(\w+)[`\"\]]?\s+(\w+)", line)
            if cm and cm.group(1).upper() not in ("PRIMARY", "FOREIGN", "UNIQUE", "INDEX", "KEY"):
                cols.append(Column(cm.group(1), cm.group(2)))
        if cols:
            tables.append(Table(tname, cols))
    return tables


# ─── Helpers ───────────────────────────────────────────────────────────────────

STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "it", "its", "of", "in",
    "on", "at", "to", "for", "with", "about", "by", "from", "into",
    "show", "get", "find", "list", "select", "fetch", "retrieve", "give",
    "display", "return", "all", "every", "each", "any", "some", "no",
    "not", "and", "or", "but", "if", "then", "else", "when", "where",
    "which", "who", "whom", "that", "this", "these", "those", "me",
    "my", "we", "our", "you", "your", "he", "she", "they", "their",
    "what", "how", "many", "much", "more", "most", "other", "another",
    "records", "rows", "entries", "data", "results", "items", "table",
    "database", "column", "field", "value", "number",
}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def _singularize(w: str) -> str:
    if w.endswith("ies"):
        return w[:-3] + "y"
    if w.endswith(("ses", "xes", "zes")):
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def _pluralize(w: str) -> str:
    if w.endswith("y") and len(w) > 1 and w[-2] not in "aeiou":
        return w[:-1] + "ies"
    if w.endswith(("s", "x", "z", "ch", "sh")):
        return w + "es"
    return w + "s"


def _resolve_col(word: str, table: Table | None) -> str:
    if table:
        col = table.find_col(word)
        return col.name if col else word
    return word


def _guess_date_col(table: Table | None) -> str:
    if table:
        for col in table.columns:
            if col.is_date or any(k in col.name.lower() for k in ["created", "date", "time", "_at"]):
                return col.name
    return "created_at"


# ─── Component extractors ──────────────────────────────────────────────────────

def _find_table(prompt: str, tables: list[Table]) -> Table | None:
    candidates = sorted(tables, key=lambda t: len(t.name), reverse=True)
    for tbl in candidates:
        for variant in {tbl.name.lower(), _singularize(tbl.name.lower()), _pluralize(tbl.name.lower())}:
            if re.search(r"\b" + re.escape(variant) + r"\b", prompt):
                return tbl
    return None


def _infer_table_name(prompt: str) -> str:
    # "from/in/of TABLE"
    m = re.search(r"(?:from|in|of|on)\s+(?:the\s+)?(\w+)(?:\s+table)?", prompt)
    if m and m.group(1) not in STOP_WORDS:
        return m.group(1)
    # "[VERB] all TABLE"
    m = re.search(
        r"(?:show|get|find|list|select|fetch|retrieve|display)\s+(?:all\s+|me\s+|the\s+)?(\w+)", prompt
    )
    if m and m.group(1) not in STOP_WORDS:
        return m.group(1)
    # Last resort: first plural noun
    for word in prompt.split():
        word = re.sub(r"\W", "", word)
        if word.endswith("s") and word not in STOP_WORDS and len(word) > 3:
            return word
    return "records"


def _extract_aggregate(prompt: str, table: Table | None) -> tuple[str | None, str | None]:
    patterns = [
        (r"(?:how many|number of|count of)", "COUNT", "*"),
        (r"(?:count)\b", "COUNT", "*"),
        (r"(?:total|sum of|sum)\s+(?:the\s+)?(\w+)", "SUM", None),
        (r"(?:average|avg|mean)(?:\s+of)?\s+(?:the\s+)?(\w+)", "AVG", None),
        (r"(?:maximum|max)(?:\s+of)?\s+(?:the\s+)?(\w+)", "MAX", None),
        (r"(?:minimum|min)(?:\s+of)?\s+(?:the\s+)?(\w+)", "MIN", None),
    ]
    for pat, func, default_col in patterns:
        m = re.search(pat, prompt)
        if m:
            if func == "COUNT":
                return "COUNT", "*"
            col_word = m.group(1) if m.lastindex else None
            if col_word and col_word not in STOP_WORDS:
                return func, _resolve_col(col_word, table)
            return func, default_col
    return None, None


def _extract_select_cols(prompt: str, table: Table | None) -> list[str]:
    # "[verb] col1, col2 and col3 from/where/in"
    m = re.search(
        r"(?:select|show|get|retrieve|fetch|display|list)\s+(?:me\s+)?(?:the\s+)?(.+?)"
        r"\s+(?:from|where|in\s+the\b|grouped|ordered|sorted|with\b|that\b)",
        prompt,
    )
    if m:
        raw = m.group(1).strip()
        if raw in ("all", "everything", "*", "records", "rows", "data", "entries", "results", "me"):
            return ["*"]
        parts = re.split(r",\s*|\s+and\s+", raw)
        cols = []
        for part in parts:
            part = part.strip()
            if not part or part in STOP_WORDS:
                continue
            cols.append(_resolve_col(part, table))
        if cols:
            return cols
    return ["*"]


def _extract_where_segment(prompt: str) -> str | None:
    """Isolate the WHERE portion of the prompt for condition parsing."""
    m = re.search(
        r"\b(?:where|with|that have|having|whose)\b(.+?)(?:\border\b|\bgroup\b|\blimit\b|\bsort\b|$)",
        prompt,
    )
    return m.group(1).strip() if m else None


def _extract_conditions(prompt: str, table: Table | None) -> list[str]:
    conditions: list[str] = []
    segment = _extract_where_segment(prompt) or prompt

    # ── Date helpers ──────────────────────────────────────────────────────────
    m = re.search(r"(?:last|past|recent)\s+(\d+)\s+(day|week|month|year)s?", segment)
    if m:
        n, unit = m.group(1), m.group(2).upper()
        dcol = _guess_date_col(table)
        conditions.append(f"{dcol} >= NOW() - INTERVAL '{n} {unit}S'")

    if re.search(r"\bthis year\b", segment):
        conditions.append(f"YEAR({_guess_date_col(table)}) = YEAR(NOW())")
    if re.search(r"\bthis month\b", segment):
        dcol = _guess_date_col(table)
        conditions.append(f"MONTH({dcol}) = MONTH(NOW()) AND YEAR({dcol}) = YEAR(NOW())")
    if re.search(r"\btoday\b", segment):
        conditions.append(f"DATE({_guess_date_col(table)}) = CURDATE()")

    # ── Numeric comparisons ───────────────────────────────────────────────────
    numeric_patterns = [
        (r"(\w+)\s+(?:greater than|more than|above|over|>)\s+(\d+(?:\.\d+)?)", ">"),
        (r"(\w+)\s+(?:less than|below|under|<)\s+(\d+(?:\.\d+)?)", "<"),
        (r"(\w+)\s+(?:at least|>=)\s+(\d+(?:\.\d+)?)", ">="),
        (r"(\w+)\s+(?:at most|no more than|<=)\s+(\d+(?:\.\d+)?)", "<="),
        (r"(\w+)\s+(?:not equal to|!=|<>)\s+(\d+(?:\.\d+)?)", "!="),
    ]
    for pat, op in numeric_patterns:
        for m in re.finditer(pat, segment):
            col_word = m.group(1)
            if col_word in STOP_WORDS:
                continue
            conditions.append(f"{_resolve_col(col_word, table)} {op} {m.group(2)}")

    # BETWEEN
    for m in re.finditer(r"(\w+)\s+between\s+(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)", segment):
        col_word = m.group(1)
        if col_word not in STOP_WORDS:
            conditions.append(
                f"{_resolve_col(col_word, table)} BETWEEN {m.group(2)} AND {m.group(3)}"
            )

    # ── String matching ───────────────────────────────────────────────────────
    str_patterns = [
        (r"(\w+)\s+contains\s+['\"]?(\w+)['\"]?", "LIKE '%{}%'"),
        (r"(\w+)\s+starts with\s+['\"]?(\w+)['\"]?", "LIKE '{}%'"),
        (r"(\w+)\s+ends with\s+['\"]?(\w+)['\"]?", "LIKE '%{}'"),
        (r"(\w+)\s+like\s+['\"]?([^'\"]+)['\"]?", "LIKE '{}'"),
    ]
    for pat, tmpl in str_patterns:
        for m in re.finditer(pat, segment):
            col_word = m.group(1)
            if col_word in STOP_WORDS:
                continue
            conditions.append(f"{_resolve_col(col_word, table)} {tmpl.format(m.group(2))}")

    # ── Equality ──────────────────────────────────────────────────────────────
    # col is/= value (exclude comparative phrases to avoid double-matching)
    for m in re.finditer(
        r"(\w+)\s+(?:is|=|equals?)\s+"
        r"(?!greater|less|more|not|null|between|at\s)(true|false|\d+(?:\.\d+)?|['\"]?[\w\s]+['\"]?)"
        r"(?=\s|$)",
        segment,
    ):
        col_word, val = m.group(1), m.group(2).strip().strip("'\"")
        if col_word in STOP_WORDS:
            continue
        col_name = _resolve_col(col_word, table)
        if re.match(r"^\d+(?:\.\d+)?$", val) or val.lower() in ("true", "false"):
            conditions.append(f"{col_name} = {val}")
        else:
            conditions.append(f"{col_name} = '{val}'")

    # ── NULL checks ───────────────────────────────────────────────────────────
    for m in re.finditer(r"(\w+)\s+(is null|is not null)", segment):
        col_word = m.group(1)
        if col_word not in STOP_WORDS:
            conditions.append(f"{_resolve_col(col_word, table)} {m.group(2).upper()}")

    # Deduplicate while preserving order
    seen, unique = set(), []
    for c in conditions:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


def _extract_group_by(prompt: str, table: Table | None) -> list[str]:
    m = re.search(r"(?:group(?:ed)?\s+by|per|by\s+each)\s+(?:the\s+)?(\w+(?:\s*,\s*\w+)*)", prompt)
    if m:
        parts = re.split(r",\s*|\s+and\s+", m.group(1))
        return [_resolve_col(p.strip(), table) for p in parts if p.strip()]
    return []


def _extract_order_by(prompt: str, table: Table | None) -> tuple[str | None, str]:
    # "ordered/sorted by col [asc|desc]"
    m = re.search(
        r"(?:order(?:ed)?|sort(?:ed)?)\s+by\s+(?:the\s+)?(\w+)"
        r"(?:\s+(asc|desc|ascending|descending))?",
        prompt,
    )
    if m:
        col = _resolve_col(m.group(1), table)
        direction = "DESC" if (m.group(2) or "").startswith("desc") else "ASC"
        return col, direction

    # "top N … by col"
    m = re.search(r"(?:top|highest|most|largest|greatest)\s+(?:\d+\s+)?(?:\w+\s+)?by\s+(?:the\s+)?(\w+)", prompt)
    if m:
        return _resolve_col(m.group(1), table), "DESC"

    # "with the highest/lowest col"
    m = re.search(r"with\s+(?:the\s+)?(?:highest|most|largest)\s+(\w+)", prompt)
    if m and m.group(1) not in STOP_WORDS:
        return _resolve_col(m.group(1), table), "DESC"
    m = re.search(r"with\s+(?:the\s+)?(?:lowest|least|smallest)\s+(\w+)", prompt)
    if m and m.group(1) not in STOP_WORDS:
        return _resolve_col(m.group(1), table), "ASC"

    # "highest/lowest col" without "with"
    m = re.search(r"(?:highest|most|largest|best)\s+(\w+)", prompt)
    if m and m.group(1) not in STOP_WORDS:
        return _resolve_col(m.group(1), table), "DESC"
    m = re.search(r"(?:lowest|least|smallest|worst)\s+(\w+)", prompt)
    if m and m.group(1) not in STOP_WORDS:
        return _resolve_col(m.group(1), table), "ASC"

    return None, "ASC"


def _extract_limit(prompt: str) -> int | None:
    m = re.search(r"(?:top|first|limit|only)\s+(\d+)", prompt)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)\s+(?:records?|rows?|results?|entries|items)", prompt)
    if m:
        return int(m.group(1))
    return None


def _detect_distinct(prompt: str) -> bool:
    return bool(re.search(r"\b(?:distinct|unique|different)\b", prompt))


# ─── Public API ────────────────────────────────────────────────────────────────

def generate_sql(prompt: str, schema: str = "") -> str:
    tables = parse_schema(schema) if schema.strip() else []
    p = _norm(prompt)

    # Resolve table
    table = _find_table(p, tables) if tables else None
    if table is None and len(tables) == 1:
        # Single-table schema: the question is about that table even if it never names it.
        table = tables[0]
    table_name = table.name if table else _infer_table_name(p)

    # Aggregation
    agg_func, agg_col = _extract_aggregate(p, table)

    # GROUP BY
    group_by = _extract_group_by(p, table)

    # ORDER BY
    order_col, order_dir = _extract_order_by(p, table)

    # LIMIT
    limit = _extract_limit(p)

    # DISTINCT
    distinct = _detect_distinct(p)

    # WHERE
    conditions = _extract_conditions(p, table)

    # SELECT clause
    if agg_func:
        agg_expr = f"{agg_func}({agg_col})" if agg_col else f"{agg_func}(*)"
        if group_by:
            select_cols = group_by + [agg_expr]
        else:
            select_cols = [agg_expr]
    else:
        select_cols = _extract_select_cols(p, table)

    distinct_kw = "DISTINCT " if distinct and not agg_func else ""

    # Assemble
    lines = [f"SELECT {distinct_kw}{', '.join(select_cols)}"]
    lines.append(f"FROM {table_name}")
    if conditions:
        lines.append(f"WHERE {' AND '.join(conditions)}")
    if group_by:
        lines.append(f"GROUP BY {', '.join(group_by)}")
    if order_col:
        lines.append(f"ORDER BY {order_col} {order_dir}")
    if limit:
        lines.append(f"LIMIT {limit}")

    return "\n".join(lines) + ";"
