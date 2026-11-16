import time

import pytest

from tableqa.execution.sandbox import (
    NotReadOnly,
    QueryFailed,
    QueryTimeout,
    Sandbox,
    schema_ddl,
)
from tableqa.table import Table


@pytest.fixture
def table():
    return Table.from_strings(
        "players",
        ["Player", "Goals", "Country"],
        [["Ann", "10", "Brazil"], ["Bob", "7", "Chile"], ["Cy", "12", "Brazil"]],
    )


@pytest.fixture
def sandbox(table):
    sb = Sandbox(table)
    yield sb
    sb.close()


def test_schema_ddl(table):
    assert schema_ddl(table) == "CREATE TABLE t (player VARCHAR, goals BIGINT, country VARCHAR)"


def test_select_returns_typed_rows(sandbox):
    result = sandbox.run("SELECT player, goals FROM t WHERE country = 'Brazil' ORDER BY goals DESC")
    assert result.columns == ("player", "goals")
    assert result.rows == (("Cy", 12), ("Ann", 10))
    assert result.truncated is False


def test_trailing_semicolon_and_cte_are_allowed(sandbox):
    result = sandbox.run("WITH b AS (SELECT * FROM t WHERE goals > 8) SELECT COUNT(*) FROM b;")
    assert result.rows == ((2,),)


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO t VALUES ('Di', 1, 'Peru')",
        "UPDATE t SET goals = 0",
        "DELETE FROM t",
        "DROP TABLE t",
        "CREATE TABLE u AS SELECT * FROM t",
        "ALTER TABLE t ADD COLUMN x INT",
        "COPY t TO 'leak.csv'",
        "ATTACH 'other.db'",
        "SET enable_external_access = true",
        "PRAGMA database_list",
        "EXPLAIN SELECT * FROM t",
        "SELECT 1; DROP TABLE t",
        "SELECT 1; SELECT 2",
        "",
    ],
)
def test_anything_but_a_single_select_is_rejected(sandbox, sql):
    with pytest.raises(NotReadOnly):
        sandbox.run(sql)
    assert sandbox.run("SELECT COUNT(*) FROM t").rows == ((3,),)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM read_csv('pyproject.toml')",
        "SELECT * FROM read_text('pyproject.toml')",
        "SELECT * FROM glob('*')",
        "SELECT * FROM read_csv('https://example.com/x.csv')",
    ],
)
def test_select_cannot_reach_files_or_network(sandbox, sql):
    with pytest.raises(QueryFailed, match="disabled by configuration"):
        sandbox.run(sql)


def test_a_file_path_is_not_a_table(sandbox):
    # DuckDB normally lets you SELECT straight from a file name; here it is just an unknown table.
    with pytest.raises(QueryFailed, match="does not exist"):
        sandbox.run("SELECT * FROM 'pyproject.toml'")


def test_bad_sql_is_reported_not_raised_raw(sandbox):
    with pytest.raises(QueryFailed, match="syntax error"):
        sandbox.run("SELEC * FROM t")
    with pytest.raises(QueryFailed, match="no_such_column"):
        sandbox.run("SELECT no_such_column FROM t")


def test_results_are_capped(table):
    sandbox = Sandbox(table, max_rows=2)
    result = sandbox.run("SELECT player FROM t ORDER BY player")
    assert result.rows == (("Ann",), ("Bob",))
    assert result.truncated is True
    exact = sandbox.run("SELECT player FROM t ORDER BY player LIMIT 2")
    assert exact.truncated is False


def test_slow_query_times_out_and_sandbox_stays_usable(table):
    sandbox = Sandbox(table, timeout_s=0.3)
    start = time.perf_counter()
    with pytest.raises(QueryTimeout):
        sandbox.run("SELECT COUNT(*) FROM range(100000000) a, range(100000000) b")
    assert time.perf_counter() - start < 3
    assert sandbox.run("SELECT COUNT(*) FROM t").rows == ((3,),)


def test_empty_table(table):
    empty = Table.from_strings("empty", ["a", "b"], [])
    assert Sandbox(empty).run("SELECT COUNT(*) FROM t").rows == ((0,),)
