"""Load-test scenarios for the TableQA service. Both use only the rule-based system, so
a run costs nothing; the agent's cached answers take the same path as `CachedUser`.

- CachedUser: example questions about the sample tables, all answered beforehand, so
  every request is a Redis cache hit.
- UncachedUser: a fresh question on every request (a counter makes each one unique),
  so every request misses the cache, loads the table into a DuckDB sandbox, generates
  SQL and runs it.

A response that is not what the scenario expects (a miss in the cached scenario, a hit
in the uncached one, or any non-200) is counted as a failure.

Run through loadtest/run.py, which warms the cache and steps through user counts.
"""

import itertools
import json
import random

from locust import HttpUser, constant, task

UNCACHED_TABLE = "wtq-204-csv-1"


def hot_questions(client) -> list[tuple[str, str]]:
    tables = client.get("/tables", name="/tables").json()
    return [(t["table_id"], q) for t in tables for q in t["example_questions"]]


class CachedUser(HttpUser):
    wait_time = constant(0)

    def on_start(self):
        self.pairs = hot_questions(self.client)

    @task
    def ask_cached(self):
        table_id, question = random.choice(self.pairs)
        body = {"question": question, "table_id": table_id, "system": "rule_based"}
        with self.client.post("/ask", json=body, name="/ask cached", catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"HTTP {r.status_code}")
            elif not json.loads(r.text)["cached"]:
                r.failure("expected a cache hit")


class UncachedUser(HttpUser):
    wait_time = constant(0)
    counter = itertools.count()

    @task
    def ask_fresh(self):
        question = f"how many rows are listed (load test {next(self.counter)} {random.random()})"
        body = {"question": question, "table_id": UNCACHED_TABLE, "system": "rule_based"}
        with self.client.post("/ask", json=body, name="/ask uncached", catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"HTTP {r.status_code}")
            elif json.loads(r.text)["cached"]:
                r.failure("expected a cache miss")
