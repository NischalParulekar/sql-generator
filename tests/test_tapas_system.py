"""TAPAS with the real checkpoint. Skipped unless the optional dependencies are installed
and the checkpoint is already in the local Hugging Face cache (CI has neither)."""

import copy

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

import pandas as pd
from huggingface_hub import try_to_load_from_cache

from tableqa.systems.tapas.system import MODEL, REVISION, TapasSystem
from tableqa.table import Table

pytestmark = pytest.mark.skipif(
    not isinstance(try_to_load_from_cache(MODEL, "config.json", revision=REVISION), str),
    reason="TAPAS checkpoint is not downloaded",
)

SMALL = Table.from_strings(
    "players",
    ["Player", "Goals", "Country"],
    [["Ann", "10", "Brazil"], ["Bob", "7", "Chile"], ["Cy", "1,200", "Brazil"]],
)
LONG = Table.from_strings(
    "long",
    ["Player", "Goals", "Club"],
    [[f"Player number {i}", str(i * 3), f"Football Club {i % 7}"] for i in range(300)],
)


@pytest.fixture(scope="module")
def chunked():
    return TapasSystem(chunked=True)


@pytest.fixture(scope="module")
def truncating(chunked):
    system = copy.copy(chunked)  # share the loaded model
    system.chunked, system.name = False, "tapas"
    return system


def test_cell_lookup(truncating):
    prediction = truncating.answer("how many goals did bob score?", SMALL)
    assert prediction.answer == ["7"]
    (trace,) = prediction.sql_or_trace
    assert trace == {"operator": "SUM", "cells": [[1, 1]], "chunks": 1, "rows_dropped": 0}
    assert prediction.error is None
    assert (prediction.cost_usd, prediction.tokens) == (0.0, 0)


def test_aggregation_uses_original_cell_text(truncating):
    prediction = truncating.answer("what is the total goals?", SMALL)
    assert prediction.sql_or_trace[0]["operator"] == "SUM"
    assert prediction.answer == ["1217"]


def test_a_table_that_fits_is_one_chunk_and_both_variants_agree(truncating, chunked):
    question = "how many players are there?"
    a, b = truncating.answer(question, SMALL), chunked.answer(question, SMALL)
    assert a.answer == b.answer == ["3"]
    assert a.sql_or_trace == b.sql_or_trace


def test_token_cost_estimate_matches_the_tokenizer(chunked):
    question = "which country is bob from?"
    estimate = len(chunked._tokenizer.tokenize(question)) + 2
    estimate += sum(chunked._cost(header) for header in SMALL.headers)
    estimate += sum(chunked._cost(cell) for row in SMALL.raw_rows for cell in row)
    frame = pd.DataFrame(list(SMALL.raw_rows), columns=list(SMALL.headers), dtype=str)
    encoded = chunked._tokenizer(table=frame, queries=[question], return_tensors="pt")
    assert encoded["input_ids"].shape[1] == estimate


def test_truncation_drops_rows_from_a_long_table(truncating):
    prediction = truncating.answer("how many goals did player number 290 score?", LONG)
    (trace,) = prediction.sql_or_trace
    assert trace["chunks"] == 1
    assert 0 < trace["rows_dropped"] < 300
    assert all(row < 300 - trace["rows_dropped"] for row, _ in trace["cells"])


def test_chunking_sees_every_row_of_a_long_table(chunked):
    prediction = chunked.answer("how many goals did player number 290 score?", LONG)
    (trace,) = prediction.sql_or_trace
    assert trace["chunks"] > 1
    assert trace["rows_dropped"] == 0
    assert [290, 1] in trace["cells"]
