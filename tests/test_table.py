import pytest

from tableqa.table import Table, parse_number


@pytest.mark.parametrize(
    "text, expected",
    [
        ("42", 42),
        ("-7", -7),
        ("3.50", 3.5),
        ("1,234,567", 1234567),
        ("1,234.5", 1234.5),
        ("0", 0),
        ("0.5", 0.5),
        ("007", None),  # a code, not a number
        ("12,34", None),  # not a thousands grouping
        ("1st", None),
        ("$5", None),
        ("", None),
        ("99999999999999999999", None),  # does not fit in BIGINT
    ],
)
def test_parse_number(text, expected):
    assert parse_number(text) == expected


def test_types_are_inferred_per_column():
    table = Table.from_strings(
        "t1",
        ["Rank", "Score", "Name", "Notes"],
        [["1", "9.5", "Ann", ""], ["2", "8", "Bob", ""], ["", "1,200", "3", ""]],
    )
    assert table.types == ("BIGINT", "DOUBLE", "VARCHAR", "VARCHAR")
    assert table.rows == (
        (1, 9.5, "Ann", None),
        (2, 8.0, "Bob", None),
        (None, 1200.0, "3", None),
    )


def test_one_non_numeric_cell_keeps_the_column_as_text():
    table = Table.from_strings("t1", ["Year"], [["2001"], ["2002"], ["n/a"]])
    assert table.types == ("VARCHAR",)
    assert table.rows == (("2001",), ("2002",), ("n/a",))


def test_column_names_are_sql_safe_and_unique():
    table = Table.from_strings(
        "t1",
        ["Total\nGoals", "Año", "", "2010", "Order", "Left", "Name", "name", "Position"],
        [],
    )
    assert table.columns == (
        "total_goals",
        "ano",
        "col_3",
        "c_2010",
        "order_",
        "left_",
        "name",
        "name_2",
        "position",
    )
    assert table.headers[0] == "Total\nGoals"


def test_short_and_long_rows_are_squared_off():
    table = Table.from_strings("t1", ["a", "b"], [["1"], ["2", "x", "extra"]])
    assert table.rows == ((1, None), (2, "x"))
