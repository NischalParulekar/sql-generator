# Reproduce the benchmark. Results go to results/; compare them with the committed ones
# using `git diff --stat results/` (the agent and TAPAS timings will differ).
#
#   make eval         every free run: rule-based and the three TAPAS variants on the dev
#                     and test subsets, plus rule-based and truncated TAPAS on the
#                     synthetic tables (TAPAS takes about an hour on a laptop CPU)
#   make eval-quick   the rule-based system only, in about a minute
#   make eval-agent   the Claude agent's runs; costs about $0.60 in total, needs a key
#                     in .env, and every run stops at its --max-cost
#   make failures     recount the reviewed failure labels
#   make test, make lint, make stack, make loadtest

RUN = uv run python -m tableqa.eval.runner
SPLITS = --n 500 --seed 42
ROWS = 1000 10000 100000

.PHONY: install eval eval-quick eval-agent failures test lint stack loadtest

install:
	uv sync --extra service --extra agent --extra tapas --group load

eval-quick:
	$(RUN) --system rule_based --split dev $(SPLITS)
	$(RUN) --system rule_based --split test $(SPLITS)
	for rows in $(ROWS); do $(RUN) --system rule_based --dataset synthetic --rows $$rows; done

eval: eval-quick
	for system in tapas tapas_chunked tapas_chunked_routed; do \
	  for split in dev test; do $(RUN) --system $$system --split $$split $(SPLITS); done; \
	done
	for rows in $(ROWS); do $(RUN) --system tapas --dataset synthetic --rows $$rows; done
	$(RUN) --system tapas_chunked_routed --dataset synthetic --rows 1000

eval-agent:
	$(RUN) --system agent_haiku --split dev $(SPLITS) --max-cost 0.50
	$(RUN) --system agent_haiku --split test $(SPLITS) --max-cost 0.50
	for rows in $(ROWS); do $(RUN) --system agent_haiku --dataset synthetic --rows $$rows --max-cost 0.25; done

failures:
	uv run python -m tableqa.eval.failures tally

test:
	uv run pytest

lint:
	uv run ruff check .

stack:
	docker compose up -d --build

loadtest:
	docker compose -f docker-compose.yml -f docker-compose.loadtest.yml up -d
	uv run --group load python loadtest/run.py
	docker compose up -d
