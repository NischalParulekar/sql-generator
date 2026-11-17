"""Prometheus metrics, kept in the app's own registry so tests can build many apps."""

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 40, 90)


OUTCOMES = ("answered", "cached", "failed", "rate_limited", "spend_capped")


class Metrics:
    def __init__(self):
        self.registry = CollectorRegistry()
        r = self.registry
        self.asks = Counter(
            "tableqa_ask_total",
            "Questions asked, by system and outcome "
            "(answered, cached, failed, rate_limited, spend_capped).",
            ["system", "outcome"],
            registry=r,
        )
        self.latency = Histogram(
            "tableqa_ask_latency_seconds",
            "Time to answer a question, by system and whether it came from the cache.",
            ["system", "cache"],
            buckets=LATENCY_BUCKETS,
            registry=r,
        )
        self.tokens = Counter(
            "tableqa_llm_tokens_total", "LLM tokens billed, by system.", ["system"], registry=r
        )
        self.cost = Counter(
            "tableqa_llm_cost_usd_total",
            "LLM spend in US dollars, by system.",
            ["system"],
            registry=r,
        )
        self.spend_today = Gauge(
            "tableqa_llm_spend_today_usd", "LLM spend so far today (UTC).", registry=r
        )
        self.sandbox_errors = Counter(
            "tableqa_sandbox_errors_total",
            "SQL the sandbox refused or could not run, by system and kind.",
            ["system", "kind"],
            registry=r,
        )
        self.uploads = Counter(
            "tableqa_table_uploads_total", "CSV uploads, by outcome.", ["outcome"], registry=r
        )

    def start_at_zero(self, systems) -> None:
        """Create every labelled series at zero. A series that first appears with a
        non-zero value has no starting point, so rate() and increase() miss what it
        counted before Prometheus first saw it."""
        for system in systems:
            for outcome in OUTCOMES:
                self.asks.labels(system, outcome)
            for cache in ("hit", "miss"):
                self.latency.labels(system, cache)
            self.tokens.labels(system)
            self.cost.labels(system)
        for outcome in ("stored", "rejected", "too_large"):
            self.uploads.labels(outcome)


def sandbox_error_kinds(prediction) -> list[str]:
    """Kinds of sandbox error behind a prediction: its own error, or the agent's
    failed queries along the way (each fed back to the model to repair)."""
    kinds = []
    if prediction.error and ":" in prediction.error:
        kind = prediction.error.split(":", 1)[0]
        if kind in ("QueryFailed", "NotReadOnly", "QueryTimeout"):
            kinds.append(kind)
    if isinstance(prediction.sql_or_trace, list):
        for step in prediction.sql_or_trace:
            for call in step.get("tool_calls", []) if isinstance(step, dict) else []:
                if call.get("error") and call.get("name") == "run_sql":
                    kinds.append(str(call.get("result", "")).split(":", 1)[0] or "QueryFailed")
    return kinds
