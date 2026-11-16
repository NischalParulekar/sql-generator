"""The one interface every table-QA system implements."""

from dataclasses import dataclass
from typing import Protocol

from tableqa.table import Table


@dataclass
class Prediction:
    answer: list[str]  # one or more values; empty when the system produced no answer
    sql_or_trace: str | list | None  # generated SQL, or the agent's step-by-step trace
    latency_ms: float
    cost_usd: float = 0.0
    tokens: int = 0
    error: str | None = None  # why there is no answer, if the system failed


class System(Protocol):
    name: str

    def answer(self, question: str, table: Table) -> Prediction: ...
