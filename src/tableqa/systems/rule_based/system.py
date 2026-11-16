"""System #1: the regex/heuristic English-to-SQL engine, executed in the sandbox."""

import time

from tableqa.execution.sandbox import Sandbox, SandboxError, schema_ddl
from tableqa.interface import Prediction
from tableqa.systems.rule_based.nlp_sql import generate_sql
from tableqa.table import Table


class RuleBasedSystem:
    name = "rule_based"

    def answer(self, question: str, table: Table) -> Prediction:
        start = time.perf_counter()
        sql = generate_sql(question, schema_ddl(table))
        answer, error = [], None
        sandbox = Sandbox(table)
        try:
            answer = sandbox.run(sql).first_column()
        except SandboxError as e:
            error = f"{type(e).__name__}: {e}"
        finally:
            sandbox.close()
        return Prediction(
            answer=answer,
            sql_or_trace=sql,
            latency_ms=(time.perf_counter() - start) * 1000,
            error=error,
        )
