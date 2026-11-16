import pytest

from tableqa.systems.rule_based.system import RuleBasedSystem
from tableqa.table import Table


@pytest.fixture
def table():
    return Table.from_strings(
        "players",
        ["Player", "Total Goals", "Country"],
        [["Ann", "10", "Brazil"], ["Bob", "7", "Chile"], ["Cy", "12", "Brazil"]],
    )


def test_aggregate_question_is_answered_from_the_table(table):
    prediction = RuleBasedSystem().answer("what is the average goals", table)
    assert " ".join(prediction.sql_or_trace.split()) == "SELECT AVG(total_goals) FROM t;"
    assert [float(a) for a in prediction.answer] == [pytest.approx(29 / 3)]
    assert prediction.error is None
    assert prediction.latency_ms > 0
    assert (prediction.cost_usd, prediction.tokens) == (0.0, 0)


def test_multi_row_result_answers_with_its_first_column(table):
    prediction = RuleBasedSystem().answer("show player where goals greater than 8", table)
    assert prediction.answer == ["Ann", "Cy"]


def test_string_filters_only_match_lowercase_cells(table):
    # Known engine limitation: the prompt is lowercased, so the literal becomes 'brazil'
    # and the comparison is case-sensitive. Recorded here, not fixed.
    prediction = RuleBasedSystem().answer("show player where country is Brazil", table)
    assert "country = 'brazil'" in prediction.sql_or_trace
    assert prediction.answer == []
    assert prediction.error is None


def test_unexecutable_sql_is_an_error_not_a_crash(table):
    prediction = RuleBasedSystem().answer("players from last 30 days", table)
    assert prediction.answer == []
    assert prediction.error.startswith("QueryFailed:")
    assert "SELECT" in prediction.sql_or_trace
