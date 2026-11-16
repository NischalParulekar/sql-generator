from collections import Counter
from fractions import Fraction

import pytest

from tableqa.data.synthetic import (
    HEADERS,
    TEMPLATES,
    Order,
    _few,
    _round2,
    _top,
    _totals,
    generate,
    generate_orders,
)
from tableqa.eval.metrics import denotation_match
from tableqa.execution.sandbox import Sandbox


@pytest.fixture(scope="module")
def generated():
    return generate(2_000, seed=7)


def test_generation_is_deterministic_and_seed_dependent():
    assert generate(300, seed=1) == generate(300, seed=1)
    assert generate_orders(300, seed=1) != generate_orders(300, seed=2)


def test_a_smaller_table_is_a_prefix_of_a_larger_one():
    assert generate_orders(5_000, seed=3)[:500] == generate_orders(500, seed=3)


def test_table_shape_and_types(generated):
    table, _ = generated
    assert table.headers == tuple(HEADERS)
    assert len(table.rows) == 2_000
    assert table.columns[6:9] == ("quantity", "unit_price", "amount")
    assert table.types == ("VARCHAR",) * 6 + ("BIGINT",) * 3 + ("VARCHAR",) * 2
    assert len({row[0] for row in table.rows}) == 2_000  # order IDs are unique
    assert all(qty * price == amount for *_, qty, price, amount, _, _ in table.rows)


def test_every_type_gets_its_share_of_distinct_answerable_questions(generated):
    table, examples = generated
    assert Counter(e.type for e in examples) == dict.fromkeys(TEMPLATES, 5)
    assert len({e.question for e in examples}) == len(examples)
    assert len({e.id for e in examples}) == len(examples)
    assert all(e.table_id == table.id and e.answer and all(e.answer) for e in examples)


def test_top_refuses_to_break_ties():
    scores = {"a": 9, "b": 7, "c": 7, "d": 1}
    assert _top(scores) == ["a"]
    assert _top(scores, 2) is None  # b and c tie for second place
    assert _top(scores, 3) == ["a", "b", "c"]
    assert _top(scores, lowest=True) == ["d"]
    assert _top({"a": 5, "b": 5}) is None
    assert _top({}) is None
    assert _top({"a": 1}, 3) is None


def test_round2_refuses_rounding_boundaries():
    assert _round2(Fraction(1, 3)) == "0.33"
    assert _round2(Fraction(2, 3)) == "0.67"
    assert _round2(Fraction(21, 2)) == "10.50"
    assert _round2(Fraction(1005, 1000)) is None  # 1.005 sits exactly on the boundary
    assert _round2(Fraction(7)) == "7.00"


def test_few_keeps_one_to_five_distinct_values_in_order():
    assert _few(["b", "a", "b"]) == ["b", "a"]
    assert _few([]) is None
    assert _few(list("abcdef")) is None


def test_totals_counts_or_sums_per_key():
    def order(region, amount):
        return Order("ORD-1", "2024-01-01", "c", region, "p", "cat", 1, amount, amount, "s", "r")

    orders = [order("North", 10), order("South", 5), order("North", 7)]
    assert _totals(orders, "region") == {"North": 2, "South": 1}
    assert _totals(orders, "region", "amount") == {"North": 17, "South": 5}


# Gold answers are computed in Python. These queries compute the same things in DuckDB,
# written separately, so a mistake in either shows up as a disagreement.
SQL_FOR = {
    "Which 3 products have the highest total quantity ordered?": (
        "SELECT product FROM t GROUP BY product ORDER BY SUM(quantity) DESC LIMIT 3"
    ),
    "Which 3 customers have the highest total amount?": (
        "SELECT customer FROM t GROUP BY customer ORDER BY SUM(amount) DESC LIMIT 3"
    ),
    "Which 3 sales reps handled the most orders?": (
        "SELECT sales_rep FROM t GROUP BY sales_rep ORDER BY COUNT(*) DESC LIMIT 3"
    ),
    "Which 2 regions have the highest total amount?": (
        "SELECT region FROM t GROUP BY region ORDER BY SUM(amount) DESC LIMIT 2"
    ),
    "Which 3 products have the lowest total amount?": (
        "SELECT product FROM t GROUP BY product ORDER BY SUM(amount) LIMIT 3"
    ),
    "Which region has the most orders?": (
        "SELECT region FROM t GROUP BY region ORDER BY COUNT(*) DESC LIMIT 1"
    ),
    "Which category has the highest total amount?": (
        "SELECT category FROM t GROUP BY category ORDER BY SUM(amount) DESC LIMIT 1"
    ),
    "Which sales rep has the highest total amount?": (
        "SELECT sales_rep FROM t GROUP BY sales_rep ORDER BY SUM(amount) DESC LIMIT 1"
    ),
    "Which product has the lowest average quantity per order?": (
        "SELECT product FROM t GROUP BY product ORDER BY AVG(quantity) LIMIT 1"
    ),
    "How many orders have an amount greater than the average order amount?": (
        "SELECT COUNT(*) FROM t WHERE amount > (SELECT AVG(amount) FROM t)"
    ),
    "What is the total amount of all orders placed by the customer with the most orders?": (
        "SELECT SUM(amount) FROM t WHERE customer = "
        "(SELECT customer FROM t GROUP BY customer ORDER BY COUNT(*) DESC LIMIT 1)"
    ),
}
SQL_PATTERNS = [  # (question prefix, query built from the words of the question)
    ("What is the amount of order ", lambda w: f"SELECT amount FROM t WHERE order_id = '{w[-1]}'"),
    (
        "Which customer placed order ",
        lambda w: f"SELECT customer FROM t WHERE order_id = '{w[-1]}'",
    ),
    (
        "How many orders were placed in the ",
        lambda w: f"SELECT COUNT(*) FROM t WHERE region = '{w[-2]}'",
    ),
    (
        "How many orders have the status ",
        lambda w: f"SELECT COUNT(*) FROM t WHERE status = '{w[-1]}'",
    ),
    (
        "How many orders have a quantity greater than ",
        lambda w: f"SELECT COUNT(*) FROM t WHERE quantity > {w[-1]}",
    ),
    (
        "What is the total amount of all orders in the ",
        lambda w: f"SELECT SUM(amount) FROM t WHERE region = '{w[-2]}'",
    ),
    (
        "What is the total quantity ordered in orders placed in ",
        lambda w: f"SELECT SUM(quantity) FROM t WHERE order_date LIKE '{w[-1]}%'",
    ),
    (
        "What is the average amount of orders in the ",
        lambda w: f"SELECT ROUND(AVG(amount), 2) FROM t WHERE region = '{w[-7]}'",
    ),
    (
        "What is the average quantity of orders with the status ",
        lambda w: f"SELECT ROUND(AVG(quantity), 2) FROM t WHERE status = '{w[-6].rstrip(',')}'",
    ),
    (
        "What percentage of all orders have the status ",
        lambda w: (
            "SELECT ROUND(100.0 * COUNT(*) FILTER (WHERE status = "
            f"'{w[-6].rstrip(',')}') / COUNT(*), 2) FROM t"
        ),
    ),
]


def _independent_sql(question: str) -> str | None:
    if question in SQL_FOR:
        return SQL_FOR[question]
    words = question.rstrip("?").split()
    for prefix, build in SQL_PATTERNS:
        if question.startswith(prefix):
            return build(words)
    return None


@pytest.mark.parametrize("n_rows, seed", [(1_000, 42), (2_000, 7), (10_000, 42)])
def test_gold_answers_agree_with_independent_sql(n_rows, seed):
    table, examples = generate(n_rows, seed=seed)
    sandbox = Sandbox(table)
    checked = Counter()
    for example in examples:
        sql = _independent_sql(example.question)
        if sql is None:
            continue
        from_sql = sandbox.run(sql).first_column()
        assert denotation_match(from_sql, list(example.answer)), (example.question, from_sql)
        checked[example.type] += 1
    # The cross-check has to cover a real share of the questions, in most types.
    assert sum(checked.values()) >= 20
    assert len(checked) >= 6
