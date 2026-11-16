"""System #2: TAPAS (Herzig et al., 2020), the checkpoint fine-tuned on WikiTableQuestions.

TAPAS reads the question and the flattened table as one sequence of at most 512 tokens,
selects cells, and predicts an aggregation operator (NONE, SUM, AVERAGE, COUNT).

Rows are packed into consecutive chunks that each fit. Three variants differ in what
they do when there is more than one chunk:

- `tapas`: only the first chunk is used, so rows that do not fit are dropped from the
  end. This is what the tokenizer's own truncation does, except that the rest of the
  table is cut off before tokenizing. Letting the tokenizer cut it instead makes it rank
  numeric cells over the whole table, and a column with more than 255 ranked values then
  overflows the model's rank embeddings and crashes.
- `tapas_chunked`: TAPAS runs on every chunk; the selected cells are the union over all
  chunks.
- `tapas_chunked_routed`: TAPAS runs on every chunk; aggregations keep the union, while
  lookups keep only the chunk with the most confident cell.

In both chunked variants the operator is the one with the highest mean probability
across chunks. See `logic.combine`.

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
from tableqa.systems.tapas.logic import ChunkResult, combine, compute_answer, pack_rows
from tableqa.table import Table

MODEL = "google/tapas-base-finetuned-wtq"
REVISION = "e3dde1905dea877b0df1a5c057533e48327dee77"


class TapasSystem:
    def __init__(self, variant: str = "tapas"):
        self.name = variant  # tapas, tapas_chunked or tapas_chunked_routed
        self._tokenizer = TapasTokenizer.from_pretrained(MODEL, revision=REVISION)
        self._model = TapasForQuestionAnswering.from_pretrained(MODEL, revision=REVISION).eval()

    def answer(self, question: str, table: Table) -> Prediction:
        start = time.perf_counter()
        chunks = self._chunks(question, table)
        if self.name == "tapas":
            chunks = chunks[:1]
        results, rows_dropped = [], len(table.raw_rows)
        for chunk in chunks:
            rows = table.raw_rows[chunk.start : chunk.stop]
            cells, operator_probs, rows_seen = self._run(question, table.headers, rows)
            results.append(
                ChunkResult(
                    cells={(chunk.start + row, col): p for (row, col), p in cells.items()},
                    operator_probs=operator_probs,
                )
            )
            rows_dropped -= rows_seen
        operator, cells = combine(results, routed=self.name == "tapas_chunked_routed")
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
        """One forward pass over one chunk.

        Returns the selected cells as {(row, column): probability} with rows counted
        within the chunk, the operator probabilities, and how many rows the model saw.
        """
        frame = pd.DataFrame(list(rows), columns=list(headers), dtype=str)
        inputs = self._tokenizer(
            table=frame, queries=[question], truncation="drop_rows_to_fit", return_tensors="pt"
        )
        with torch.no_grad():
            outputs = self._model(**inputs)
        (selected,), _ = self._tokenizer.convert_logits_to_predictions(
            inputs, outputs.logits, outputs.logits_aggregation
        )
        # A cell's probability is the mean over its tokens, as in the tokenizer's own decoding.
        token_probs = outputs.logits[0].sigmoid() * inputs["attention_mask"][0]
        segment, column_id, row_id = (inputs["token_type_ids"][0, :, i] for i in range(3))
        cells = {}
        for row, col in selected:
            tokens = (segment == 1) & (column_id == col + 1) & (row_id == row + 1)
            cells[(row, col)] = float(token_probs[tokens].mean())
        operator_probs = tuple(outputs.logits_aggregation[0].softmax(dim=-1).tolist())
        return cells, operator_probs, int(row_id.max())

    def _cost(self, text: str) -> int:
        """Tokens a cell occupies; an empty cell still takes one."""
        return max(1, len(self._tokenizer.tokenize(text)))

    def _chunks(self, question: str, table: Table) -> list[range]:
        fixed = len(self._tokenizer.tokenize(question)) + 2  # [CLS] question [SEP]
        fixed += sum(self._cost(header) for header in table.headers)
        row_costs = [sum(self._cost(cell) for cell in row) for row in table.raw_rows]
        return pack_rows(row_costs, self._tokenizer.model_max_length - fixed)
