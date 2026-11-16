import pytest

from tableqa.eval.metrics import denotation_match, normalize, percentile, to_value


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("  New   York ", "new york"),
        ("Café Müller", "cafe muller"),
        ("‘quoted’ – dash", "'quoted' - dash"),
        ("Brazil[1]", "brazil"),
        ("Brazil[note 3]*", "brazil"),
        ("Springfield (Illinois)", "springfield"),
        ('"Hey Jude"', "hey jude"),
        ("St. Louis.", "st. louis"),
        ("(only parentheses)", "(only parentheses)"),
        ("[1]", ""),
    ],
)
def test_normalize(raw, expected):
    assert normalize(raw) == expected


@pytest.mark.parametrize(
    "predicted, gold",
    [
        (["Brazil"], ["brazil"]),
        (["  ITALY "], ["Italy"]),
        (["3"], ["3.0"]),
        (["3.0000001"], ["3"]),
        (["12467"], ["12,467"]),
        (["100000.0"], ["100,000"]),
        (["-5"], ["-5.0"]),
        (["2004-06-05"], ["2004-6-5"]),
        (["xx-01-12"], ["xx-1-12"]),
        (["2004"], ["2004-xx-xx"]),
        (["Reds", "Blues"], ["blues", "reds"]),
        (["Reds", "Reds", "Blues"], ["Blues", "Reds"]),
        (["Springfield"], ["Springfield (Illinois)"]),
    ],
)
def test_matches(predicted, gold):
    assert denotation_match(predicted, gold)


@pytest.mark.parametrize(
    "predicted, gold",
    [
        ([], ["Brazil"]),
        (["Brazil"], ["Chile"]),
        (["3.1"], ["3"]),
        (["Reds"], ["Reds", "Blues"]),
        (["Reds", "Blues", "Greens"], ["Reds", "Blues"]),
        (["4 years"], ["4"]),
        (["12,34"], ["1234"]),
        (["1990-01-12"], ["xx-01-12"]),
        (["2004-06-05"], ["June 5, 2004"]),
        (["1_000"], ["1000"]),
        (["nan"], ["nan-1"]),
    ],
)
def test_mismatches(predicted, gold):
    assert not denotation_match(predicted, gold)


def test_to_value_kinds():
    assert to_value("1,234.5").number == 1234.5
    assert to_value("2010-03-xx").date == (2010, 3, -1)
    assert to_value("2010-13-01").date is None
    assert to_value("inf").number is None
    text = to_value("Wolfe Tones")
    assert (text.number, text.date, text.normalized) == (None, None, "wolfe tones")


def test_percentile():
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert percentile(values, 50) == 30.0
    assert percentile(values, 95) == pytest.approx(48.0)
    assert percentile([7.0], 95) == 7.0
