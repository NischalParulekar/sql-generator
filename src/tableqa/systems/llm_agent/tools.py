"""The tools the agent can call, and what they do to one table.

The model never sees the table directly. It learns the schema, looks at a few rows, and
pushes all computation into SQL that runs in the read-only sandbox. That keeps the
context small however large the table is.
"""

import json
from collections import Counter

from tableqa.execution.sandbox import TABLE_NAME, Sandbox, SandboxError
from tableqa.table import Table

MAX_SAMPLE_ROWS = 5
MAX_RESULT_ROWS = 30  # rows of a query result shown to the model
MAX_ERROR_CHARS = 600

PROFILE_EXAMPLES = 3  # most frequent values shown per column in a profile
MAX_VALUE_CHARS = 40

FINAL_ANSWER = "final_answer"
SUBMIT_SQL = "submit_sql"


def _tool(name: str, description: str, properties: dict) -> dict:
    return {
        "name": name,
        "description": description,
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        },
    }


_SCHEMA_DESCRIPTION = (
    f"Describe the table `{TABLE_NAME}`: its row count and, for each column, the SQL "
    "name, the DuckDB type, and the original header text from the page. Call this first."
)
_PROFILE_DESCRIPTION = (
    " Each column also has a profile of its values: how many cells are filled, how many "
    f"distinct values there are, the {PROFILE_EXAMPLES} most frequent values, and the "
    "minimum and maximum of a numeric column."
)

TOOL_DEFINITIONS = [
    _tool("get_schema", _SCHEMA_DESCRIPTION, {}),
    _tool(
        "sample_rows",
        f"Return the first rows of `{TABLE_NAME}` so you can see what the values look like. "
        f"At most {MAX_SAMPLE_ROWS} rows are returned.",
        {"n": {"type": "integer", "description": f"How many rows, 1 to {MAX_SAMPLE_ROWS}."}},
    ),
    _tool(
        "run_sql",
        f"Run one read-only DuckDB SELECT statement against `{TABLE_NAME}` and return the "
        f"result. At most {MAX_RESULT_ROWS} rows come back; `truncated` says whether there "
        "were more. If the query fails, the error message is returned so you can fix it.",
        {"query": {"type": "string", "description": "A single SELECT statement."}},
    ),
    _tool(
        FINAL_ANSWER,
        "Give the final answer and finish. This is the only way to answer.",
        {
            "values": {
                "type": "array",
                "items": {"type": "string"},
                "description": "The answer: one item, or several items for a list answer.",
            }
        },
    ),
]


SUBMIT_SQL_DEFINITION = _tool(
    SUBMIT_SQL,
    "Submit the one SELECT statement that answers the question. It is run once against "
    f"`{TABLE_NAME}`, and the values in the first column of its result are the answer.",
    {"query": {"type": "string", "description": "A single SELECT statement."}},
)


def tool_definitions(samples: bool = True, profile: bool = False) -> list[dict]:
    """The agent's tools: the defaults, optionally without sample_rows or with profiles."""
    tools = [t for t in TOOL_DEFINITIONS if samples or t["name"] != "sample_rows"]
    if profile:
        schema = _tool("get_schema", _SCHEMA_DESCRIPTION + _PROFILE_DESCRIPTION, {})
        tools = [schema if t["name"] == "get_schema" else t for t in tools]
    return tools


class TableTools:
    """Executes the non-final tools against one table."""

    def __init__(self, table: Table, sandbox: Sandbox, profile: bool = False):
        self._table = table
        self._sandbox = sandbox
        self._profile = profile

    def execute(self, name: str, arguments: dict) -> tuple[str, bool]:
        """Run a tool. Returns (result text for the model, whether it is an error)."""
        if name == "get_schema":
            return self._get_schema(), False
        if name == "sample_rows":
            return self._sample_rows(arguments["n"]), False
        if name == "run_sql":
            return self._run_sql(arguments["query"])
        return f"Unknown tool: {name}", True

    def _get_schema(self) -> str:
        table = self._table
        return _json(
            {
                "table": TABLE_NAME,
                "row_count": len(table.rows),
                "columns": [
                    {
                        "name": name,
                        "type": dtype,
                        "header": header,
                        **({"profile": self._column_profile(i)} if self._profile else {}),
                    }
                    for i, (name, dtype, header) in enumerate(
                        zip(table.columns, table.types, table.headers)
                    )
                ],
            }
        )

    def _column_profile(self, i: int) -> dict:
        values = [row[i] for row in self._table.rows if row[i] is not None]
        counts = Counter(values)
        profile = {
            "filled": len(values),
            "distinct": len(counts),
            "most_frequent": [
                str(v)[:MAX_VALUE_CHARS] for v, _ in counts.most_common(PROFILE_EXAMPLES)
            ],
        }
        if values and self._table.types[i] != "VARCHAR":
            profile["min"], profile["max"] = min(values), max(values)
        return profile

    def answer_with_sql(self, query: str) -> tuple[list[str], str, bool]:
        """Run a query as the whole answer: (first-column values, result text, is error)."""
        try:
            result = self._sandbox.run(query)
        except SandboxError as e:
            return [], f"{type(e).__name__}: {e}"[:MAX_ERROR_CHARS], True
        shown = _json({"columns": list(result.columns), "rows": result.rows[:MAX_RESULT_ROWS]})
        return result.first_column(), shown, False

    def _sample_rows(self, n: int) -> str:
        n = max(1, min(n, MAX_SAMPLE_ROWS))
        return _json({"columns": list(self._table.columns), "rows": self._table.rows[:n]})

    def _run_sql(self, query: str) -> tuple[str, bool]:
        try:
            result = self._sandbox.run(query)
        except SandboxError as e:
            return f"{type(e).__name__}: {e}"[:MAX_ERROR_CHARS], True
        shown = result.rows[:MAX_RESULT_ROWS]
        return _json(
            {
                "columns": list(result.columns),
                "rows": shown,
                "truncated": result.truncated or len(result.rows) > len(shown),
            }
        ), False


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)
