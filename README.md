# TableQA

How well do three generations of table question answering hold up on the same public
benchmark, measured the same way?

1. **Rule-based** — a regex/heuristic English-to-SQL engine (this repo's original code).
2. **TAPAS** — a table transformer fine-tuned on WTQ, with and without chunking for
   tables over its 512-token limit.
3. **LLM agent** — a Claude tool-use agent (Haiku 5.5) that writes and repairs SQL.

Every system answers through one interface, generated SQL runs in a locked-down DuckDB
sandbox, and answers are scored by denotation match on
[WikiTableQuestions](https://github.com/ppasupat/WikiTableQuestions) (WTQ).

**Status:** work in progress. All three systems run on the dev subset. Still to come:
the held-out test split, the context-engineering variants, the large-table stress test,
the failure taxonomy and the service.

## Results so far

WTQ **dev** subset: 500 questions sampled with seed 42 from `random-split-1-dev`.

| System | Accuracy | Table fits in 512 tokens (380 q) | Table does not fit (120 q) | Latency p50 / p95 | Cost per 1k |
|---|---|---|---|---|---|
| Rule-based | 3.8% (19) | 3.9% (15) | 3.3% (4) | 34 ms / 73 ms | $0 |
| TAPAS, truncated | **47.2%** (236) | 55.3% (210) | 21.7% (26) | 547 ms / 1,148 ms | $0 |
| TAPAS, chunked (union) | 45.0% (225) | 55.3% (210) | 12.5% (15) | 532 ms / 2,715 ms | $0 |
| TAPAS, chunked (routed) | **48.4%** (242) | 55.3% (210) | 26.7% (32) | 413 ms / 2,122 ms | $0 |
| LLM agent (Haiku 5.5) | **87.0%** (435) | 88.9% (338) | 80.8% (97) | 4.0 s / 6.1 s | $0.47 |

**This table is not a fair comparison yet. Read it with three caveats.**

- *TAPAS is inflated.* The dev split is carved out of WTQ's training data, which this
  TAPAS checkpoint was fine-tuned on.
- *The agent may be inflated too.* WTQ has been public since 2015, so a current LLM may
  have seen these questions and answers in training. That cannot be ruled out on any
  split of a public benchmark; the synthetic large-table test, generated fresh, will be
  the uncontaminated check.
- *The agent's runs are not repeatable.* Current Claude models accept no sampling
  temperature. Of 20 questions answered in two separate runs, 2 changed from right to
  wrong. The 87.0% is one run, not an average.

The cross-system comparison will be made on the held-out test split.

What the numbers say:

- **The agent is far ahead on this subset, and it is cheap.** 435 of 500 correct for
  $0.23 in total. It is right on 213 questions TAPAS (truncated) gets wrong and wrong on
  14 that TAPAS gets right; 51 defeat both. It is also slower: about 4 seconds a
  question against half a second for TAPAS.
- **Large tables cost the agent 8 points, not 30.** It scores 80.8% on tables too big
  for TAPAS's window against 88.9% on the rest, because it queries the table with SQL
  and never reads it whole. TAPAS drops from 55.3% to 26.7% at best.
- **It almost never needs a second try.** 439 questions took exactly three model turns
  (look at the table, one query, answer); only 3 had a failed query to repair. Prompt
  caching carried nearly all the input, which is why the cost is under $0.50 per 1,000.
- **Where it fails (65 questions), from a first read:** 9 have the right number in the
  wrong form ("5" for "5 days", or the reverse); others are partial lists, wrong rows on
  "longest" or "least" questions, and counts done by eye instead of in SQL. 34 questions
  were answered from the sample rows with no query at all (30 of them correctly, all on
  tables of 43 rows or fewer). These are impressions, not the labelled failure taxonomy.
- **Naive chunking makes TAPAS worse, not better.** On the 120 questions whose table
  does not fit, taking the union of every chunk's selected cells scores 12.5% against
  21.7% for simply dropping the rows that do not fit. It fixed 12 questions and broke 23.
  The wins are counting questions, where seeing every row matters ("how many teams
  played altogether?": 29 truncated, 64 chunked, gold 64). The losses are lookups and
  superlatives, where each chunk volunteers its own answer and the union returns two
  values instead of one.
- **Routing by operator repairs that.** The routed rule keeps the union for sum, average
  and count but answers a lookup from the single most confident chunk. Against the union
  rule it fixes 17 questions and breaks none (12.5% → 26.7% on the 120).
- **Routed chunking is not yet shown to beat truncation.** It is ahead by 6 questions on
  the 120 (16 fixed, 10 broken), which is well within what chance would produce on a
  sample this size. The routed rule was also designed after looking at the union rule's
  failures on this same subset, so even that margin is optimistic. It was run once and
  not iterated on.
- Every variant is much weaker on tables that do not fit (26.7% at best) than on tables
  that do (55.3%), so large tables are a problem for TAPAS either way.
- The rule-based engine's 19 correct answers all come from `SELECT COUNT(*) FROM t` on
  "how many ..." questions whose answer happens to be the table's row count. It did not
  get a single lookup, filter, comparison or superlative right; 284 questions fall
  through to `SELECT *` and 77 queries fail to execute.

Latency is measured on a laptop CPU and is noisy: the two chunked variants do the same
model work, yet their timings differ by about 20% between runs. For the rule-based
system it includes loading the table into DuckDB per question; for TAPAS it excludes
loading the model; for the agent it is dominated by network round trips to the API.
The agent's cost is computed from the token usage the API reported, at list prices.

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

TAPAS needs the optional dependencies (CPU-only PyTorch, transformers, pandas) and
downloads the `google/tapas-base-finetuned-wtq` checkpoint (443 MB) on first use:

```bash
uv sync --extra tapas
uv run python -m tableqa.eval.runner --system tapas --split dev --n 500 --seed 42
uv run python -m tableqa.eval.runner --system tapas_chunked --split dev --n 500 --seed 42
uv run python -m tableqa.eval.runner --system tapas_chunked_routed --split dev --n 500 --seed 42
```

The agent needs the `agent` extra and an Anthropic API key, and each run costs money.
Copy `.env.example` to `.env` and put the key there; `.env` is git-ignored. `--max-cost`
stops a run once it has spent that many US dollars, and predictions are written as they
arrive, so a stopped run keeps what it paid for.

```bash
uv sync --extra agent
uv run python -m tableqa.eval.runner --system agent_haiku --split dev --n 20 --seed 42 --max-cost 0.25
```

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

## TAPAS and the 512-token limit

TAPAS reads the question and the whole table as one sequence of at most 512 tokens,
selects cells and predicts an operator (none, sum, average, count). The checkpoint is
`google/tapas-base-finetuned-wtq`, pinned to a revision, run on CPU.

Rows are packed greedily, in order, into chunks that each fit the limit together with
the question and the header.

- **`tapas`** uses the first chunk only: rows that do not fit are dropped from the end.
- **`tapas_chunked`** runs TAPAS on every chunk and takes the union of the selected cells.
- **`tapas_chunked_routed`** runs TAPAS on every chunk and lets the predicted operator
  decide: sum, average and count keep the union, while a lookup (no operator) keeps only
  the chunk that holds the single most confident cell.

In both chunked variants the operator is the one with the highest mean probability
across chunks.

Chunking lets TAPAS see every row, but each chunk is answered in isolation, so a
question that compares rows in different chunks ("which has the highest ...") cannot be
answered correctly by construction.

The union rule was fixed before the first run. The routed rule was designed afterwards,
from the union rule's failures on the dev subset, which is what dev is for; it has no
tuned thresholds and was run once.

## The LLM agent

The agent never sees the whole table. It has four tools and a budget of six model turns
per question:

- `get_schema` — row count, and each column's SQL name, type and original header;
- `sample_rows` — the first few rows (at most 5);
- `run_sql` — one read-only DuckDB `SELECT` in the sandbox, returning at most 30 rows,
  or the error message so the model can repair the query;
- `final_answer` — ends the question.

Because computation is pushed into SQL, the context stays small however large the table
is. The loop is written by hand so that every step's tool calls, results and token
usage are recorded; the full trace is stored with each prediction, and the model, prompt
and tool definitions are stored in the run's summary. Cost is computed from the token
usage the API reports, at the list prices in `systems/llm_agent/system.py`.

The model is `claude-haiku-5-5`. Current Claude models do not accept a sampling
temperature, so runs are not bit-for-bit repeatable; the committed predictions are the
record of what each run answered.

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
  systems/tapas/        logic.py (chunk packing, answer computation) + system.py (model)
  systems/llm_agent/    tools.py (the four tools) + system.py (prompt, loop, cost)
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
