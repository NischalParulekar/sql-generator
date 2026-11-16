import pytest

from tableqa.systems.tapas.logic import compute_answer, pack_rows


@pytest.mark.parametrize(
    "costs, budget, expected",
    [
        ([], 10, []),
        ([3, 3, 3], 10, [range(3)]),
        ([3, 3, 3], 9, [range(3)]),
        ([3, 3, 3], 8, [range(2), range(2, 3)]),
        ([4, 4, 4, 4, 4], 8, [range(2), range(2, 4), range(4, 5)]),
        ([5, 1, 5, 1], 6, [range(2), range(2, 4)]),
        ([20, 3, 3], 10, [range(1), range(1, 3)]),  # an oversized row stands alone
        ([3, 20, 3], 10, [range(1), range(1, 2), range(2, 3)]),
    ],
)
def test_pack_rows(costs, budget, expected):
    assert pack_rows(costs, budget) == expected


@pytest.mark.parametrize("costs, budget", [([7, 2, 9, 1, 1, 4, 8, 3], 10), ([1] * 50, 7)])
def test_pack_rows_covers_every_row_once_within_budget(costs, budget):
    chunks = pack_rows(costs, budget)
    assert [i for chunk in chunks for i in chunk] == list(range(len(costs)))
    assert all(sum(costs[i] for i in chunk) <= budget for chunk in chunks)


@pytest.mark.parametrize(
    "operator, cells, expected",
    [
        ("NONE", ["Wolfe Tones"], ["Wolfe Tones"]),
        ("NONE", ["Reds", "", "Blues"], ["Reds", "Blues"]),
        ("NONE", [], []),
        ("COUNT", ["a", "b", "c"], ["3"]),
        ("COUNT", [], ["0"]),
        ("SUM", ["10", "7", "1,200"], ["1217"]),
        ("SUM", ["1.5", "2"], ["3.5"]),
        ("SUM", ["12", "n/a", ""], ["12"]),
        ("SUM", ["n/a"], []),
        ("SUM", [], []),
        ("AVERAGE", ["10", "7", "12"], [str(29 / 3)]),
        ("AVERAGE", ["4", "6"], ["5"]),
        ("AVERAGE", ["4", "unknown", "6"], ["5"]),
    ],
)
def test_compute_answer(operator, cells, expected):
    assert compute_answer(operator, cells) == expected
