"""The rule-based engine, pinned to the examples its README advertises."""

import pytest

from tableqa.systems.rule_based.nlp_sql import generate_sql, parse_schema


def one_line(sql: str) -> str:
    return " ".join(sql.split())


# Three README examples do not produce the SQL the README claims. They are kept as
# strict xfails so the gap stays visible and the test flips loudly if the engine changes.
README_EXAMPLES = [
    ("show all users", "SELECT * FROM users;"),
    ("count the number of orders", "SELECT COUNT(*) FROM orders;"),
    ("top 10 customers by revenue", "SELECT * FROM customers ORDER BY revenue DESC LIMIT 10;"),
    ("average salary of employees", "SELECT AVG(salary) FROM employees;"),
    pytest.param(
        "orders where total greater than 100",
        "SELECT * FROM orders WHERE total > 100;",
        marks=pytest.mark.xfail(strict=True, reason="'total' is read as SUM: SELECT SUM(greater)"),
    ),
    pytest.param(
        "users grouped by country",
        "SELECT country, COUNT(*) FROM users GROUP BY country;",
        marks=pytest.mark.xfail(strict=True, reason="emits SELECT * ... GROUP BY country"),
    ),
    pytest.param(
        "orders from last 30 days",
        "SELECT * FROM orders WHERE created_at >= NOW() - INTERVAL '30 DAYS';",
        marks=pytest.mark.xfail(strict=True, reason="table is inferred as 'last'"),
    ),
]


@pytest.mark.parametrize("prompt, expected", README_EXAMPLES)
def test_readme_examples(prompt, expected):
    assert one_line(generate_sql(prompt)) == expected


SCHEMA = "CREATE TABLE t (player VARCHAR, total_goals BIGINT, country VARCHAR)"


def test_parse_schema_reads_columns_and_types():
    (table,) = parse_schema(SCHEMA)
    assert table.name == "t"
    assert [(c.name, c.dtype) for c in table.columns] == [
        ("player", "VARCHAR"),
        ("total_goals", "BIGINT"),
        ("country", "VARCHAR"),
    ]


def test_single_table_schema_is_used_when_question_does_not_name_it():
    assert one_line(generate_sql("average goals", SCHEMA)) == "SELECT AVG(total_goals) FROM t;"


def test_named_table_still_wins_with_several_tables():
    schema = "CREATE TABLE users (id INT); CREATE TABLE orders (id INT, total INT);"
    assert one_line(generate_sql("count the number of orders", schema)) == (
        "SELECT COUNT(*) FROM orders;"
    )


def test_unnamed_table_is_still_guessed_with_several_tables():
    schema = "CREATE TABLE users (id INT); CREATE TABLE orders (id INT, total INT);"
    assert one_line(generate_sql("show all products", schema)) == "SELECT * FROM products;"
