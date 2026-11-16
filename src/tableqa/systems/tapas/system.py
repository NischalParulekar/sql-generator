"""System #2: TAPAS (Herzig et al., 2020), the checkpoint fine-tuned on WikiTableQuestions.

TAPAS reads the question and the flattened table as one sequence of at most 512 tokens,
selects cells, and predicts an aggregation operator (NONE, SUM, AVERAGE, COUNT).

A table that does not fit is handled in one of two ways:

- chunked (`tapas_chunked`): rows are packed into consecutive chunks that each fit and
  TAPAS runs on every chunk. The selected cells are the union over all chunks; the
  operator is the one with the highest mean probability across chunks.
- truncate (`tapas`): only the first chunk is used, so rows that do not fit are dropped
  from the end. This is what the tokenizer's own truncation does, except that the rest
  of the table is cut off before tokenizing. Letting the tokenizer cut it instead makes
  it rank numeric cells over the whole table, and a column with more than 255 ranked
  values then overflows the model's rank embeddings and crashes.

Chunking lets TAPAS see every row, but each chunk is answered in isolation: a question
that compares rows across chunks ("which has the highest ...") cannot be answered
correctly by construction.

Needs the optional dependencies: `uv sync --extra tapas`.
"""

import time

import pandas as pd
import torch
from transformers import TapasForQuestionAnswering, TapasTokenizer

from tableqa.interface import Prediction
from tableqa.systems.tapas.logic import OPERATORS, compute_answer, pack_rows
from tableqa.table import Table

MODEL = "google/tapas-base-finetuned-wtq"
REVISION = "e3dde1905dea877b0df1a5c057533e48327dee77"
ROW_IDS = 2  # position of the row index in TAPAS's token_type_ids


class TapasSystem:
    def __init__(self, chunked: bool = False):
        self.name = "tapas_chunked" if chunked else "tapas"
        self.chunked = chunked
        self._tokenizer = TapasTokenizer.from_pretrained(MODEL, revision=REVISION)
        self._model = TapasForQuestionAnswering.from_pretrained(MODEL, revision=REVISION).eval()

    def answer(self, question: str, table: Table) -> Prediction:
        start = time.perf_counter()
        chunks = self._chunks(question, table)
        if not self.chunked:
            chunks = chunks[:1]
        cells, operator_probs, rows_dropped = [], [], len(table.raw_rows)
        for chunk in chunks:
            rows = table.raw_rows[chunk.start : chunk.stop]
            selected, probs, rows_seen = self._run(question, table.headers, rows)
            cells += [(chunk.start + row, column) for row, column in selected]
            operator_probs.append(probs)
            rows_dropped -= rows_seen
        operator = OPERATORS[int(torch.stack(operator_probs).mean(dim=0).argmax())]
        cells.sort()
        answer = compute_answer(operator, [table.raw_rows[row][col] for row, col in cells])
        trace = {
            "operator": operator,
            "cells": [list(cell) for cell in cells],
            "chunks": len(chunks),
            "rows_dropped": rows_dropped,
        }
        return Prediction(
            answer=answer,
            sql_or_trace=[trace],
            latency_ms=(time.perf_counter() - start) * 1000,
        )

    def _run(self, question: str, headers: tuple[str, ...], rows: tuple[tuple[str, ...], ...]):
        """One forward pass: (selected (row, column) cells, operator probabilities, rows seen)."""
        frame = pd.DataFrame(list(rows), columns=list(headers), dtype=str)
        inputs = self._tokenizer(
            table=frame, queries=[question], truncation="drop_rows_to_fit", return_tensors="pt"
        )
        with torch.no_grad():
            outputs = self._model(**inputs)
        (selected,), _ = self._tokenizer.convert_logits_to_predictions(
            inputs, outputs.logits, outputs.logits_aggregation
        )
        probs = outputs.logits_aggregation[0].softmax(dim=-1)
        rows_seen = int(inputs["token_type_ids"][0, :, ROW_IDS].max())
        return selected, probs, rows_seen

    def _cost(self, text: str) -> int:
        """Tokens a cell occupies; an empty cell still takes one."""
        return max(1, len(self._tokenizer.tokenize(text)))

    def _chunks(self, question: str, table: Table) -> list[range]:
        fixed = len(self._tokenizer.tokenize(question)) + 2  # [CLS] question [SEP]
        fixed += sum(self._cost(header) for header in table.headers)
        row_costs = [sum(self._cost(cell) for cell in row) for row in table.raw_rows]
        return pack_rows(row_costs, self._tokenizer.model_max_length - fixed)
