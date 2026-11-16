"""The tools the agent can call, and what they do to one table.

The model never sees the table directly. It learns the schema, looks at a few rows, and
pushes all computation into SQL that runs in the read-only sandbox. That keeps the
context small however large the table is.
"""

import json

from tableqa.execution.sandbox import TABLE_NAME, Sandbox, SandboxError
from tableqa.table import Table

MAX_SAMPLE_ROWS = 5
MAX_RESULT_ROWS = 30  # rows of a query result shown to the model
MAX_ERROR_CHARS = 600

FINAL_ANSWER = "final_answer"


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


TOOL_DEFINITIONS = [
    _tool(
        "get_schema",
        f"Describe the table `{TABLE_NAME}`: its row count and, for each column, the SQL "
        "name, the DuckDB type, and the original header text from the page. Call this first.",
        {},
    ),
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


class TableTools:
    """Executes the non-final tools against one table."""

    def __init__(self, table: Table, sandbox: Sandbox):
        self._table = table
        self._sandbox = sandbox

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
                    {"name": name, "type": dtype, "header": header}
                    for name, dtype, header in zip(table.columns, table.types, table.headers)
                ],
            }
        )

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
