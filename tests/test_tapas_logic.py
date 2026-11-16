import pytest

from tableqa.systems.tapas.logic import ChunkResult, combine, compute_answer, pack_rows


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


# Operator probabilities are ordered (NONE, SUM, AVERAGE, COUNT).
LOOKUP, COUNTING = (0.9, 0.0, 0.0, 0.1), (0.1, 0.0, 0.0, 0.9)


def test_union_keeps_every_selected_cell_in_table_order():
    results = [
        ChunkResult({(12, 0): 0.8}, LOOKUP),
        ChunkResult({(3, 0): 0.99, (1, 2): 0.6}, LOOKUP),
    ]
    assert combine(results, routed=False) == ("NONE", [(1, 2), (3, 0), (12, 0)])


def test_routed_lookup_keeps_only_the_chunk_with_the_most_confident_cell():
    results = [
        ChunkResult({(0, 1): 0.7, (1, 1): 0.7}, LOOKUP),
        ChunkResult({(20, 1): 0.95, (22, 1): 0.55}, LOOKUP),
        ChunkResult({}, LOOKUP),
    ]
    assert combine(results, routed=True) == ("NONE", [(20, 1), (22, 1)])


def test_routed_aggregation_still_uses_every_chunk():
    results = [
        ChunkResult({(0, 1): 0.7}, COUNTING),
        ChunkResult({(20, 1): 0.95}, COUNTING),
    ]
    assert combine(results, routed=True) == ("COUNT", [(0, 1), (20, 1)])
    assert combine(results, routed=True) == combine(results, routed=False)


def test_operator_is_chosen_by_mean_probability_across_chunks():
    results = [
        ChunkResult({(0, 0): 0.9}, (0.6, 0.0, 0.0, 0.4)),
        ChunkResult({(9, 0): 0.9}, (0.2, 0.0, 0.0, 0.8)),
    ]
    assert combine(results, routed=False)[0] == "COUNT"


def test_single_chunk_is_the_same_under_both_rules():
    results = [ChunkResult({(2, 1): 0.9, (4, 1): 0.8}, LOOKUP)]
    assert combine(results, routed=True) == combine(results, routed=False)


def test_routed_lookup_with_nothing_selected_anywhere_is_empty():
    results = [ChunkResult({}, LOOKUP), ChunkResult({}, LOOKUP)]
    assert combine(results, routed=True) == ("NONE", [])
