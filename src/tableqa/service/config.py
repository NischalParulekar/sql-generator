"""Service settings, read from environment variables (all optional)."""

import os
from dataclasses import dataclass, field


def _list(name: str, default: str) -> list[str]:
    return [item.strip() for item in os.environ.get(name, default).split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    redis_url: str = "redis://localhost:6379/0"
    # Systems the service offers. TAPAS needs the heavy optional dependencies.
    systems: list[str] = field(default_factory=lambda: ["rule_based", "agent_haiku"])
    cors_origins: list[str] = field(default_factory=lambda: ["http://localhost:3000"])
    # Behind reverse proxies, how many of them to trust. Each appends the address it
    # received from to X-Forwarded-For, so the client is the entry `proxy_hops` from
    # the right; entries further left were written by the client and can be forged.
    # 0 ignores the header and uses the connecting address.
    proxy_hops: int = 0

    max_question_chars: int = 500
    max_upload_bytes: int = 2_000_000
    max_upload_rows: int = 50_000
    max_upload_columns: int = 60
    table_ttl_s: int = 7 * 24 * 3600
    cache_ttl_s: int = 7 * 24 * 3600

    # Token bucket per client address: `rate_capacity` tokens, refilled at
    # `rate_refill_per_s`. A cached answer costs 1 token; a fresh one costs its
    # system's weight, so the LLM agent is the most limited.
    rate_capacity: float = 20
    rate_refill_per_s: float = 0.2
    system_weight: dict[str, float] = field(
        default_factory=lambda: {"rule_based": 1, "tapas": 3, "agent_haiku": 5}
    )

    # Hard ceiling on LLM spend per UTC day, across all clients.
    daily_spend_cap_usd: float = 1.0
    # At most this many questions per system run at once.
    concurrency: dict[str, int] = field(
        default_factory=lambda: {"rule_based": 16, "tapas": 1, "agent_haiku": 8}
    )
    ask_timeout_s: float = 90

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.environ.get
        return cls(
            redis_url=env("REDIS_URL", cls.redis_url),
            systems=_list("TABLEQA_SYSTEMS", "rule_based,agent_haiku"),
            cors_origins=_list("CORS_ORIGINS", "http://localhost:3000"),
            proxy_hops=int(env("PROXY_HOPS", "0")),
            daily_spend_cap_usd=float(env("DAILY_SPEND_CAP_USD", cls.daily_spend_cap_usd)),
            rate_capacity=float(env("RATE_CAPACITY", cls.rate_capacity)),
            rate_refill_per_s=float(env("RATE_REFILL_PER_S", cls.rate_refill_per_s)),
        )
