# TableQA

How well do three generations of table question answering hold up on the same public
benchmark, measured the same way?

1. **Rule-based** — a regex/heuristic English-to-SQL engine (this repo's original code).
2. **TAPAS** — a fine-tuned table transformer. *Not built yet.*
3. **LLM agent** — a Claude tool-use agent that writes and repairs SQL. *Not built yet.*

Every system answers through one interface, generated SQL runs in a locked-down DuckDB
sandbox, and answers are scored by denotation match on
[WikiTableQuestions](https://github.com/ppasupat/WikiTableQuestions) (WTQ).

**Status:** work in progress. The benchmark pipeline and the rule-based baseline are in
place; TAPAS, the agent, the large-table stress test and the service are still to come.

## Results so far

WTQ **dev** subset: 500 questions sampled with seed 42 from `random-split-1-dev`.

| System | Accuracy | Execution errors | Latency p50 / p95 | Cost per 1k questions |
|---|---|---|---|---|
| Rule-based | **3.8%** (19 / 500) | 77 | 34 ms / 73 ms | $0 |
| TAPAS | — | — | — | — |
| LLM agent | — | — | — | — |

Read the rule-based number for what it is:

- All 19 correct answers come from `SELECT COUNT(*) FROM t` on "how many ..." questions
  whose answer happens to be the table's row count. The engine did not get a single
  lookup, filter, comparison or superlative question right.
- 284 of the 500 questions fall through to `SELECT *`, which can never be a correct answer.
- 77 queries fail to execute, almost all with a binder error (a word from the question
  used as a column name that does not exist).
- Latency is measured on a laptop and includes loading the table into DuckDB per question.

Raw predictions for every question are in [`results/`](results/), so anyone can re-score
them. Numbers on the WTQ **test** split are deliberately not reported yet: test is held
back until all three systems are frozen.

## Reproduce

Requires [uv](https://docs.astral.sh/uv/) (it installs Python 3.11 and the pinned
dependencies from `uv.lock`).

```bash
uv sync
uv run pytest
uv run python -m tableqa.eval.runner --system rule_based --split dev --n 500 --seed 42
```

The first eval run downloads the WTQ v1.0.2 release (29 MB) from the official GitHub
repository into `data/wtq/` and verifies its SHA-256. The dataset is not committed.

## How it is measured

These rules were fixed before the first run.

**Splits.** `dev` is `random-split-1-dev` (2,831 questions), used for development and
any tuning. `test` is `pristine-unseen-tables` (4,344 questions). `dev` is carved out of
WTQ's training data, and the public TAPAS WTQ checkpoint was fine-tuned on that training
data, so TAPAS scores on `dev` will be inflated; cross-system comparisons will use `test`.

**Tables.** WTQ tables are strings under free-text headers. Before any SQL system sees
one, headers become unique snake_case column names (`Total Goals` → `total_goals`), and a
column is typed `BIGINT`/`DOUBLE` only if every non-empty cell parses as a number
(thousands separators allowed); otherwise it stays `VARCHAR`. The table is always named
`t`.

**Answers.** A system's answer is a list of values. For SQL systems it is the non-NULL
values of the first column of the result, capped at 100 rows.

**Accuracy.** Denotation match, following the rules of the official WTQ evaluator: text
is normalised (case, whitespace, diacritics, quotes and dashes, trailing citations and
parenthesised details), numbers match within 1e-6, `yyyy-mm-dd` dates match field by
field, and the predicted values must be the same set as the gold values. One difference:
the official evaluator takes canonical numbers and dates for gold answers from CoreNLP
annotations, while this implementation parses gold strings itself (plain numbers and
thousands-separated numbers). It is not the official evaluator and scores may differ
slightly from published ones.

**Latency and cost.** p50 and p95 of per-question wall time on the same machine; cost is
LLM tokens × price, 0 for local systems.

## The SQL sandbox

Generated SQL is untrusted. Each table is loaded into its own in-memory DuckDB database,
and a query is run only if it is exactly one `SELECT` statement. File and network access
are disabled and the configuration is locked, queries are interrupted after a timeout
(5 s by default), and results are capped. See
[`src/tableqa/execution/sandbox.py`](src/tableqa/execution/sandbox.py) and its tests.

## Layout

```
src/tableqa/
  table.py              Table: original headers, SQL-safe names, inferred types
  interface.py          answer(question, table) -> Prediction
  data/wtq.py           WTQ download, loader, seeded sampling
  execution/sandbox.py  read-only DuckDB sandbox
  systems/rule_based/   nlp_sql.py (the engine) + system.py (adapter)
  eval/                 metrics.py, runner.py
  service/app.py        FastAPI app (currently only POST /generate-sql)
frontend/               React UI for the SQL generator (Vite)
results/                committed predictions and summaries
tests/
```

## The rule-based engine

`generate_sql(prompt, schema)` turns English into SQL with regular expressions and
heuristics, with no model involved. It handles aggregates (count, sum, average, max,
min), comparisons, `LIKE`, relative date ranges, `GROUP BY`, `ORDER BY`, `LIMIT` and
`DISTINCT`.

| Prompt | Generated SQL |
|---|---|
| show all users | `SELECT * FROM users;` |
| count the number of orders | `SELECT COUNT(*) FROM orders;` |
| top 10 customers by revenue | `SELECT * FROM customers ORDER BY revenue DESC LIMIT 10;` |
| average salary of employees | `SELECT AVG(salary) FROM employees;` |

Known limitations, kept as they are so the baseline stays honest (each is pinned by a
test):

- "orders where total greater than 100" reads `total` as an aggregate and emits
  `SELECT SUM(greater) ...`.
- "users grouped by country" emits `SELECT * ... GROUP BY country`, which is not valid SQL.
- "orders from last 30 days" infers the table name `last`.
- The prompt is lowercased, so string filters only match lowercase cell values.

To run the original SQL-generator UI:

```bash
uv run --extra service uvicorn tableqa.service.app:app --port 8000
```

```bash
cd frontend && npm install && npm run dev
```

The UI runs at `http://localhost:3000` and proxies `/generate-sql` to the API on port 8000.

## Data and license

WikiTableQuestions is by Panupong Pasupat and Percy Liang
([Compositional Semantic Parsing on Semi-Structured Tables](https://arxiv.org/abs/1508.00305),
ACL 2015) and is licensed
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). The files under
`results/` quote WTQ questions and answers and are shared under the same license.
