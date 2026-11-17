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

**Status:** work in progress. All three systems have been measured on the held-out test
split, the dev subset and a synthetic large-table test; four context-engineering
variants of the agent have been compared; and 169 test failures have been labelled by
cause. Still to come: the service and demo.

## Results

### Held-out test split

WTQ **test** subset: 500 questions sampled with seed 42 from `pristine-unseen-tables`,
whose tables appear nowhere in WTQ's training data. Each system ran on it once, after
its design had been frozen on the dev subset.

| System | Accuracy (95% CI) | Table fits in 512 tokens (379 q) | Table does not fit (121 q) | Latency p50 / p95 | Cost per 1k |
|---|---|---|---|---|---|
| Rule-based | 5.0% (25), 3.4–7.3% | 4.7% | 5.8% | 22 ms / 27 ms | $0 |
| TAPAS, truncated | 45.8% (229), 41.5–50.2% | 51.2% | 28.9% | 423 ms / 846 ms | $0 |
| TAPAS, chunked (union) | 41.0% (205), 36.8–45.4% | 51.2% | 9.1% | 813 ms / 5,263 ms | $0 |
| TAPAS, chunked (routed) | 46.0% (230), 41.7–50.4% | 51.2% | 29.8% | 429 ms / 2,977 ms | $0 |
| LLM agent (Haiku 5.5) | **86.2%** (431), 82.9–88.9% | 87.6% | 81.8% | 5.7 s / 8.4 s | $0.48 |

The intervals are Wilson 95% intervals for 500 questions.

- **The agent nearly doubles TAPAS on unseen tables.** It is right on 214 questions
  that TAPAS (truncated) gets wrong and wrong on 12 that TAPAS gets right; 57 defeat
  both. The whole run cost $0.24.
- **The dev results held up.** Every system lands within 1 to 4 points of its dev score
  and the ranking is unchanged. TAPAS's fine-tuning on the data dev comes from inflated
  it less than feared: 1.4 points for the truncated variant, 2.4 for the routed one.
- **Large tables:** the agent loses 6 points on tables too big for TAPAS's window;
  TAPAS loses 21 at best.
- **Routed chunking ties truncation** (230 against 229). Its small lead on dev was noise,
  as suspected. Union chunking is still the worst variant.
- **Two caveats remain for the agent.** WTQ, test tables included, has been public since
  2015, so the model may have seen these questions in training; the synthetic test shows
  the agent does not need memorisation, but cannot rule it out here. And this is one
  run: two runs of the identical agent differ on about 20 questions of 500 (see the
  context-engineering section), roughly a point of accuracy.

### Dev subset

Every design decision was made on this subset, so it is not the headline result. TAPAS
was fine-tuned on the data it comes from.

WTQ **dev** subset: 500 questions sampled with seed 42 from `random-split-1-dev`.

| System | Accuracy | Table fits in 512 tokens (380 q) | Table does not fit (120 q) | Latency p50 / p95 | Cost per 1k |
|---|---|---|---|---|---|
| Rule-based | 3.8% (19) | 3.9% (15) | 3.3% (4) | 34 ms / 73 ms | $0 |
| TAPAS, truncated | **47.2%** (236) | 55.3% (210) | 21.7% (26) | 547 ms / 1,148 ms | $0 |
| TAPAS, chunked (union) | 45.0% (225) | 55.3% (210) | 12.5% (15) | 532 ms / 2,715 ms | $0 |
| TAPAS, chunked (routed) | **48.4%** (242) | 55.3% (210) | 26.7% (32) | 413 ms / 2,122 ms | $0 |
| LLM agent (Haiku 5.5) | **87.0%** (435) | 88.9% (338) | 80.8% (97) | 4.0 s / 6.1 s | $0.47 |

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
The agent's cost is computed from the token usage the API reported, at list prices. Its
test run was slower than its dev run (5.7 s against 4.0 s median) with the same model
and prompt, which points to network conditions rather than the questions.

Raw predictions for every question are in [`results/`](results/), so anyone can re-score
them.

## Large-table stress test

Invented order tables (11 columns) generated from a seed at 1,000, 10,000 and 100,000
rows, with 40 templated questions per size: five each of lookup, filter, count, sum,
average, top-N, group-by and multi-step. Gold answers are computed in plain Python from
the rows and cross-checked in the tests against separately written SQL. The tables did
not exist before they were generated, so no model can have seen them in training.

Correct answers out of 40, with median latency per question:

| System | 1,000 rows | 10,000 rows | 100,000 rows |
|---|---|---|---|
| Rule-based | 1 (39 ms) | 1 (114 ms) | 1 (1.0 s) |
| TAPAS, truncated | 2 (0.9 s) | 0 (0.9 s) | 1 (0.9 s) |
| TAPAS, chunked (routed) | 2 (58 s) | not run | not run |
| LLM agent (Haiku 5.5) | **40** (4.8 s) | **40** (5.0 s) | **40** (5.7 s) |

- **The agent's accuracy does not depend on table size.** It answered all 120
  questions, in every type, for $0.054 in total ($0.43-0.48 per 1,000 questions). Its
  latency rises by under a second from 1,000 to 100,000 rows, and that second is the
  time to load the table into DuckDB.
- **A perfect score also means this test is too easy to rank the agent.** The tables are
  clean and the questions are templated, unlike WTQ's messy Wikipedia tables. The result
  shows that the agent scales and is not reciting memorised answers. It does not confirm
  the 87% on WTQ, and it does not show where the agent's limits are.
- **TAPAS cannot be scaled up by chunking on this hardware.** Only 18 of these rows fit
  in its 512-token window. Truncated, it sees 18 rows of 100,000 and its few correct
  answers are luck. Chunked, a 1,000-row table takes 56 model passes and about a minute
  per question on a laptop CPU, and still scores 2 of 40, because a sum or a ranking
  over the whole table cannot be assembled from chunks answered in isolation. At about
  one second per pass, 10,000 rows would take roughly 10 minutes per question and
  100,000 rows roughly an hour and a half; those two are estimates and were not run.
- **The rule-based engine's one correct answer** at each size is a plain row count.

```bash
uv run python -m tableqa.eval.runner --system rule_based --dataset synthetic --rows 100000
uv run python -m tableqa.eval.runner --system agent_haiku --dataset synthetic --rows 100000 --max-cost 0.25
```

## Context-engineering variants

Four variants of the agent, each changing one thing from the default, run on the same
500 dev questions. The default agent was also run a second time, unchanged, to measure
how much two identical runs differ.

- **No samples** — no `sample_rows` tool; the model sees only the schema.
- **Column profiles** — the schema also gives, for each column, how many cells are
  filled, how many distinct values it has, its three most frequent values and, for a
  number column, its range. WTQ has no human-written column descriptions, so this
  automatic profile stands in for them.
- **Check step** — the first `final_answer` is not accepted; the model is asked to check
  its answer against the question and answer again.
- **Single shot** — no loop: the model sees the schema and the first five rows and
  writes one query, whose first column is the answer. No chance to repair it.

Each run is compared question by question with the default: *fixed* counts questions
the variant gets right and the default gets wrong, *broken* the reverse, and *p* is the
two-sided exact sign test on that split.

| Agent | Accuracy | vs default, first run | vs default, second run | Cost per 1k | Latency p50 |
|---|---|---|---|---|---|
| Default, first run | 87.0% (435) | — | 12 fixed / 8 broken, p = 0.50 | $0.47 | 4.0 s |
| Default, second run | 86.2% (431) | 8 / 12, p = 0.50 | — | $0.48 | 5.3 s |
| No samples | 84.6% (423) | 8 / 20, p = 0.04 | 14 / 22, p = 0.24 | $0.40 | 5.4 s |
| Column profiles | 85.8% (429) | 9 / 15, p = 0.31 | 14 / 16, p = 0.86 | $0.52 | 5.2 s |
| Check step | 86.2% (431) | 6 / 10, p = 0.45 | 10 / 10, p = 1.00 | $0.58 | 7.4 s |
| Single shot | 78.4% (392) | 10 / 53, p < 0.0001 | 16 / 55, p < 0.0001 | $0.35 | 2.8 s |

What the numbers say:

- **The noise floor is about 20 questions.** Two runs of the identical agent disagree on
  20 of the 500 (0.8 points of accuracy). Any variant has to beat that to mean anything.
- **The tool loop is what matters.** Single shot loses 8 to 9 points, far beyond the
  noise: looking at the table and repairing a query is worth more than any change to
  what the model is told. It is the cheapest and fastest variant, so where speed matters
  more than 8 points of accuracy it is a real option. 12 of its 500 queries failed to
  run, with nothing to catch them.
- **Nothing else made a measurable difference.** Column profiles and the check step are
  within the noise, and both cost more: the check step about 20% more and two seconds
  more per question. It changed only 12 of 500 answers (4 to right, 2 to wrong).
- **Dropping the samples may cost one or two points.** It is significant against one
  run of the default (p = 0.04) but not the other (p = 0.24), and with four variants
  tested at once a single p = 0.04 is weak evidence. It saves about 15% of the cost.
- **Conclusion: the default agent stays the final one.** No variant earned its change.

Latency is comparable only among the five runs of the second batch, which ran at the
same time; the default's first run ran alone and was faster. During the batch a short
network outage on the test machine crashed 4 consecutive questions in every run.
Those 20 questions were asked again with `--retry-crashed` (a crash is a failure of the
run, not an answer), and every other answer was kept as it was. The batch cost $1.16.

```bash
uv run python -m tableqa.eval.runner --system agent_haiku_single_shot --split dev --n 500 --seed 42 --max-cost 0.50
uv run python -m tableqa.eval.compare results/wtq-dev-n500-seed42/agent_haiku.predictions.jsonl results/wtq-dev-n500-seed42/agent_haiku_single_shot.predictions.jsonl
```

## Failure analysis

Why does each system get questions wrong? Failures on the held-out test subset were
labelled by cause: all 69 of the agent's, and a seeded random 50 each from the
rule-based system (of 475) and TAPAS, truncated (of 271).

Each failure gets one primary label from a fixed list of eleven causes, checked in a
fixed order; the first that applies is the label. A questionable gold answer comes
first, because if the gold answer is wrong nothing the system did is the cause.

| Cause | Claude agent (69) | TAPAS (50) | Rule-based (50) |
|---|---|---|---|
| Gold questionable: gold wrong, ill-posed, or not answerable from the table | **21** | 1 | 1 |
| Execution error: the query failed or the budget ran out | 1 | — | 4 |
| Declined: said the table cannot answer it | 2 | — | — |
| Not understood: output unrelated to the question | — | 2 | **23** |
| Missing context: needed rows or page content it never saw | 1 | 10 | — |
| Answer form: right value, wrong form | **15** | 1 | — |
| Misread question: answered a different question | 7 | 1 | **20** |
| Wrong column | 2 | 2 | — |
| Wrong rows: wrong filter, position, or list items | 11 | **24** | 1 |
| Wrong calculation: arithmetic, dates, distinct counts | 8 | 9 | 1 |
| Hallucinated value: not in the table or derivable from it | 1 | — | — |

What it shows:

- **Half of the agent's failures are not reasoning errors.** 21 are problems with the
  benchmark itself: gold answers that the table contradicts (it lists Honda 14 times;
  gold says 13), answers relative to the year the question was written ("how many
  years ago..."), and questions the table cannot answer, where the agent correctly said
  so. Another 15 have the right value in a form the strict metric rejects: "7" for
  "7 days", "Hofmann" for "Alex Hofmann", "$1.2 billion" without the dollar sign.
- **The agent's real errors are selection and arithmetic on messy text.** Wrong rows
  (11), calculations over date ranges ending in "present" or unusual score formats (8),
  and misreading what is being counted (7). It hallucinated once: it guessed players'
  nationalities, which the table does not contain.
- **TAPAS's failures are built into its design.** It selects wrong rows (24), cannot see
  rows dropped to fit its window (10), and has no operator for a difference, a maximum,
  a most-frequent value or a yes/no answer (9 calculations, 2 not understood).
- **The rule-based engine mostly never engages.** 43 of 50 failures are its
  `SELECT *` fallback or a `COUNT(*)` that ignores the question's condition.
- **The benchmark has a ceiling below 100%.** At least 21 of the 500 test answers (4%)
  are questionable, and they were found only by looking at the agent's failures; gold
  errors among answers counted as correct were not looked for. Setting those 21 aside,
  the agent answers 431 of the remaining 479 (90.0%). The 86.2% stays the headline
  because it is measured the same way for every system.

**How the labels were made.** The categories were fixed before labelling. Claude wrote a
first-pass label and a note for every failure, checking the uncertain ones against the
table itself. The author then reviewed all 169 on a review page showing the question,
gold answer, prediction and the system's SQL or selected cells, and agreed with every
first-pass label; none was changed. The labels and notes are in
[`results/failures/reviewed.csv`](results/failures/reviewed.csv), and
`python -m tableqa.eval.failures tally` recounts them.

## Reproduce

Requires [uv](https://docs.astral.sh/uv/) (it installs Python 3.11 and the pinned
dependencies from `uv.lock`) and `make`.

```bash
make install       # every optional dependency
make test
make eval-quick    # the rule-based system on every benchmark, about two minutes
make eval          # every free run, TAPAS included; about an hour on a laptop CPU
make eval-agent    # the Claude agent's runs; about $0.60, each run capped by --max-cost
```

Runs write to `results/`, so `git diff --stat results/` shows what changed. Re-running
the rule-based system reproduces every committed score exactly. The order of a long
answer list from a query without `ORDER BY` can differ between runs, because DuckDB
executes queries in parallel; answers are compared as sets, so scores do not change.
The agent's runs are not repeatable at all (see the noise floor below), and timings
always differ.

One system and benchmark at a time:

```bash
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

## The service

The three systems are also served over HTTP, so they can be compared live and load
tested. `docker compose up --build` starts the API with Redis, Prometheus and Grafana,
every port bound to this machine only.

| Endpoint | What it does |
|---|---|
| `POST /ask` | Answer `{question, table_id, system}` with one system |
| `GET /tables` | Sample tables with example questions: ten WTQ dev tables and a generated 10,000-row order table |
| `POST /tables` | Upload a CSV (2 MB, 50,000 rows and 60 columns at most), kept for a week |
| `GET /tables/{id}/schema` | Columns, types and first rows |
| `GET /systems`, `GET /healthz`, `GET /metrics` | Systems on offer, health, Prometheus metrics |

- **Answer cache.** Answers are cached in Redis for a week, keyed by system, a hash of
  the table's content and the question normalised for case, spacing and trailing
  punctuation. A cached answer is returned without running the system.
- **Rate limiting.** Each client address has a token bucket in Redis, updated
  atomically by a Lua script: 20 tokens, refilled at one every five seconds. A fresh
  answer costs its system's weight (rule-based 1, TAPAS 3, agent 5), a cached one 1.
  Over the limit the API returns 429 with `Retry-After`.
- **Spend cap.** LLM spend is added up per UTC day in Redis; at the cap
  (`DAILY_SPEND_CAP_USD`, $0.50 in the Compose file) the agent is refused with 503
  while the free systems keep working. Concurrent requests can overshoot the cap by
  at most a few in-flight questions, each costing well under a cent. Redis runs with
  `noeviction`, so memory pressure cannot reset the counter.
- **Isolation.** Systems run in worker threads with a per-system concurrency limit
  (TAPAS one at a time) and a 90-second timeout. A system that fails returns 502
  without its error details; a Redis outage returns 503.
- **Metrics.** Questions by system and outcome (answered, cached, failed, rate limited,
  spend capped), latency by system and cache, LLM tokens and cost, today's spend,
  sandbox errors by kind, and uploads. The Grafana dashboard in
  `ops/grafana/dashboards/tableqa.json` is provisioned automatically
  (http://localhost:3001).
- **Secrets.** The API key is read from the git-ignored `.env` and never leaves the
  server; TAPAS is left out of the image by default because it adds PyTorch and a
  443 MB model.

```bash
docker compose up --build
curl -s localhost:8000/ask -H 'content-type: application/json' \n  -d '{"question": "How many orders were placed in the South region?", "table_id": "orders-10k", "system": "rule_based"}'
```

### The web UI

The React UI in `frontend/` asks the service. Pick a sample table or upload a CSV, ask a
question (or click one of the table's example questions), and each selected system's
answer appears side by side with its latency, cost, whether it came from the cache, and
how it got there: the rule engine's SQL, the cells and operator TAPAS chose, or each of
the agent's turns with its queries, their results and the final answer.

```bash
docker compose up -d                        # the API on port 8000
cd frontend && npm install && npm run dev   # the UI on http://localhost:3000
```

In development the UI proxies API calls to port 8000; a deployed build reads the API's
address from `VITE_API_URL`.

### Deployment

**Recorded demo (live now, free).** GitHub Pages serves the UI at
https://nischalparulekar.github.io/tableqa/ in replay mode: it shows the answers and
traces each system gave in the committed benchmark runs, for eight WTQ dev tables and the
100,000-row generated table, right or wrong. It needs no server and no API key. The
`Pages` workflow rebuilds it whenever the frontend changes; `scripts/build_demo.py`
rebuilds the bundle it replays.

**Live API (free, one-time setup).** `render.yaml` is a Render Blueprint for the API and
a Redis-compatible Key Value instance, both on Render's free plan:

1. In Render, choose New, then Blueprint, and pick this repository. Render asks for
   `ANTHROPIC_API_KEY`; use a key made for this service, with a monthly spend limit set
   in the Anthropic Console. That limit is the hard ceiling: the free Key Value instance
   keeps no data across restarts, which resets the service's own daily counter.
2. When the deploy finishes, open `https://<service>.onrender.com/healthz`.
3. In this repository's settings, add the Actions variable `API_URL` with the service's
   address and re-run the `Pages` workflow. The site then calls the live API instead of
   replaying recordings.

The free web service sleeps after 15 minutes without traffic and takes about a minute to
wake, so the first question after a quiet spell is slow. The service's own guards still
apply: $0.50 a day of agent spend, per-address rate limits, and cached answers served
free. Behind Render's proxy the client address is read from `X-Forwarded-For` with
`PROXY_HOPS=1`, so a client cannot dodge the rate limit by writing its own header;
check one request's header after deploying in case Render adds another hop.

### Load test

[Locust](https://locust.io) drove the service with users sending questions back to back,
30 seconds per step, against the Docker Compose stack on the same laptop (8 logical
CPUs, 7.6 GB RAM, Docker given 3.9 GB). Both scenarios use the rule-based system, so
the test cost nothing:

- **Cached:** the sample tables' example questions, answered once beforehand, so every
  request is a Redis hit.
- **Uncached:** a new question on every request, so every request loads the table into
  a fresh DuckDB sandbox, generates SQL and runs it.

Every response was checked: a miss in the cached scenario or a hit in the uncached one
would count as a failure. The per-address rate limit was lifted for the run
(`docker-compose.loadtest.yml`), since all requests come from one machine.

| Scenario | Users | Requests/s | p50 | p95 | p99 | Failures |
|---|---|---|---|---|---|---|
| Cached | 1 | 157 | 5 ms | 10 ms | 25 ms | 0 of 4,426 |
| Cached | 10 | 377 | 21 ms | 40 ms | 61 ms | 0 of 10,611 |
| Cached | 25 | 422 | 48 ms | 85 ms | 120 ms | 0 of 11,889 |
| Cached | 50 | 471 | 88 ms | 150 ms | 250 ms | 0 of 13,298 |
| Uncached | 1 | 23 | 36 ms | 60 ms | 89 ms | 0 of 654 |
| Uncached | 10 | 49 | 170 ms | 300 ms | 510 ms | 0 of 1,393 |
| Uncached | 25 | 55 | 400 ms | 590 ms | 2,500 ms | 0 of 1,618 |
| Uncached | 50 | 34 | 1,200 ms | 2,700 ms | 3,000 ms | 0 of 1,005 |

- **The cache is worth about 8x in throughput.** At 25 users the service serves 422
  cached answers a second at 85 ms p95, against 55 fresh ones at 590 ms.
- **Fresh answers saturate at about 25 users.** Past that, throughput falls (34 a
  second at 50 users) and p95 reaches 2.7 s: each fresh answer builds a DuckDB database
  in a worker thread, and the threads, the load generator and Docker compete for the
  same CPUs. Raising throughput would mean more API processes, not more threads.
- **The agent was not load tested,** because every fresh answer costs money. Its
  latency is set by the model API, about 4 to 6 seconds a question, and at most 8 run at
  once, so it tops out near 1.5 fresh answers a second; a cached agent answer takes the
  same path as the cached scenario above. That ceiling is worked out from the
  concurrency limit, not measured.
- These numbers are from one laptop with the load generator on the same machine; a
  server would do better, and the ratios are the useful part.

```bash
docker compose -f docker-compose.yml -f docker-compose.loadtest.yml up -d
uv run --group load python loadtest/run.py
docker compose up -d   # restore the normal rate limit
```

## Layout

```
src/tableqa/
  table.py              Table: original headers, SQL-safe names, inferred types
  interface.py          answer(question, table) -> Prediction
  data/wtq.py           WTQ download, loader, seeded sampling
  data/synthetic.py     generated order tables and typed questions for the stress test
  execution/sandbox.py  read-only DuckDB sandbox
  systems/rule_based/   nlp_sql.py (the engine) + system.py (adapter)
  systems/tapas/        logic.py (chunk packing, answer computation) + system.py (model)
  systems/llm_agent/    tools.py (the four tools) + system.py (prompt, loop, cost)
  eval/                 metrics.py, runner.py, compare.py, failures.py
  service/              FastAPI app, Redis store, metrics, sample tables, settings
loadtest/               Locust scenarios and the step runner
ops/                    Prometheus config, Grafana provisioning and dashboard
frontend/               React + Vite UI: compare the systems side by side, with traces
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

The service keeps the engine's original `POST /generate-sql` endpoint, which returns SQL
for a prompt and an optional `CREATE TABLE` schema without running it.

## Data and license

WikiTableQuestions is by Panupong Pasupat and Percy Liang
([Compositional Semantic Parsing on Semi-Structured Tables](https://arxiv.org/abs/1508.00305),
ACL 2015) and is licensed
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). The files under
`results/` quote WTQ questions and answers and are shared under the same license.
