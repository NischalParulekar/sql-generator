"""Run untrusted SQL against one table, and nothing else.

Generated SQL comes from a regex engine or an LLM, so it is treated as hostile:

- the table lives in a private in-memory DuckDB database;
- only a single SELECT statement is accepted (no writes, DDL, COPY, ATTACH, SET, ...);
- file and network access are switched off and the configuration is then locked;
- every query has a wall-clock timeout and a cap on returned rows.
"""

import threading
from dataclasses import dataclass

import duckdb

from tableqa.table import Table

TABLE_NAME = "t"


class SandboxError(Exception):
    """Base class: the query did not produce a result."""


class NotReadOnly(SandboxError):
    """The SQL was not exactly one SELECT statement."""


class QueryTimeout(SandboxError):
    """The query ran longer than the sandbox allows."""


class QueryFailed(SandboxError):
    """DuckDB rejected or failed the query (syntax, unknown column, blocked access, ...)."""


@dataclass(frozen=True)
class QueryResult:
    columns: tuple[str, ...]
    rows: tuple[tuple, ...]
    truncated: bool  # True if the query produced more than max_rows rows

    def first_column(self) -> list[str]:
        """The result read as an answer: the non-NULL values of its first column."""
        return [str(row[0]) for row in self.rows if row[0] is not None]


def schema_ddl(table: Table) -> str:
    """The CREATE TABLE statement for a table as it exists inside the sandbox."""
    columns = ", ".join(f"{name} {dtype}" for name, dtype in zip(table.columns, table.types))
    return f"CREATE TABLE {TABLE_NAME} ({columns})"


class Sandbox:
    def __init__(
        self,
        table: Table,
        *,
        timeout_s: float = 5.0,
        max_rows: int = 100,
        memory_limit: str = "512MB",
    ):
        self.timeout_s = timeout_s
        self.max_rows = max_rows
        self._con = duckdb.connect(":memory:")
        self._con.execute(schema_ddl(table))
        if table.rows:
            # One list parameter per column, unnested side by side. Inserting row by row
            # takes over a minute for 100,000 rows; this takes about a second.
            columns = ", ".join(f"unnest(?::{dtype}[])" for dtype in table.types)
            self._con.execute(
                f"INSERT INTO {TABLE_NAME} SELECT {columns}",
                [list(column) for column in zip(*table.rows)],
            )
        self._con.execute(f"SET memory_limit = '{memory_limit}'")
        self._con.execute("SET enable_external_access = false")
        self._con.execute("SET lock_configuration = true")

    def run(self, sql: str) -> QueryResult:
        try:
            statements = self._con.extract_statements(sql)
        except duckdb.Error as e:
            raise QueryFailed(str(e)) from e
        if len(statements) != 1 or statements[0].type != duckdb.StatementType.SELECT:
            raise NotReadOnly("only a single SELECT statement is allowed")

        timed_out = threading.Event()

        def cancel():
            timed_out.set()
            self._con.interrupt()

        timer = threading.Timer(self.timeout_s, cancel)
        timer.start()
        try:
            cursor = self._con.execute(sql)
            rows = cursor.fetchmany(self.max_rows + 1)
            columns = tuple(d[0] for d in cursor.description)
        except duckdb.Error as e:
            if timed_out.is_set():
                raise QueryTimeout(f"query exceeded {self.timeout_s}s") from e
            raise QueryFailed(str(e)) from e
        finally:
            timer.cancel()
        return QueryResult(
            columns=columns,
            rows=tuple(rows[: self.max_rows]),
            truncated=len(rows) > self.max_rows,
        )

    def close(self) -> None:
        self._con.close()
