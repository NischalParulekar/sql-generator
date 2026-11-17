"""Everything the service keeps in Redis: uploaded tables, cached answers, rate-limit
buckets and the day's LLM spend. Keys are prefixed `tableqa:` and all expire."""

import datetime
import hashlib
import json
import re
import time

from redis.asyncio import Redis

PREFIX = "tableqa:"

# Token bucket, applied atomically in Redis. Returns {allowed, seconds until enough
# tokens} (the wait as a string, since Lua numbers come back as integers).
TOKEN_BUCKET = """
local capacity = tonumber(ARGV[1])
local refill = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local cost = tonumber(ARGV[4])
local state = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
local tokens = tonumber(state[1]) or capacity
local ts = tonumber(state[2]) or now
tokens = math.min(capacity, tokens + math.max(0, now - ts) * refill)
local allowed, wait = 0, 0
if tokens >= cost then
  tokens = tokens - cost
  allowed = 1
else
  wait = (cost - tokens) / refill
end
redis.call('HSET', KEYS[1], 'tokens', tostring(tokens), 'ts', tostring(now))
redis.call('EXPIRE', KEYS[1], math.ceil(capacity / refill) + 1)
return {allowed, tostring(wait)}
"""


def normalize_question(question: str) -> str:
    """Questions that differ only in case, spacing or a trailing ? share a cache entry."""
    return re.sub(r"\s+", " ", question).strip().rstrip("?!. ").lower()


def answer_key(system: str, table_fingerprint: str, question: str) -> str:
    digest = hashlib.sha256(
        f"{table_fingerprint}\n{normalize_question(question)}".encode()
    ).hexdigest()
    return f"{PREFIX}answer:{system}:{digest}"


class Store:
    def __init__(self, redis: Redis, table_ttl_s: int, cache_ttl_s: int):
        self.redis = redis
        self.table_ttl_s = table_ttl_s
        self.cache_ttl_s = cache_ttl_s
        self._bucket = redis.register_script(TOKEN_BUCKET)

    async def ping(self) -> bool:
        try:
            return bool(await self.redis.ping())
        except Exception:  # noqa: BLE001 - health check reports, never raises
            return False

    # -- uploaded tables ------------------------------------------------------------

    async def put_table(self, table_id: str, name: str, headers: list[str], rows: list) -> None:
        body = json.dumps({"name": name, "headers": headers, "rows": rows}, ensure_ascii=False)
        await self.redis.set(f"{PREFIX}table:{table_id}", body, ex=self.table_ttl_s)

    async def get_table(self, table_id: str) -> dict | None:
        body = await self.redis.get(f"{PREFIX}table:{table_id}")
        return json.loads(body) if body else None

    # -- answer cache ---------------------------------------------------------------

    async def get_answer(self, key: str) -> dict | None:
        body = await self.redis.get(key)
        return json.loads(body) if body else None

    async def put_answer(self, key: str, answer: dict) -> None:
        await self.redis.set(key, json.dumps(answer, ensure_ascii=False), ex=self.cache_ttl_s)

    # -- rate limiting --------------------------------------------------------------

    async def take_tokens(
        self, client: str, cost: float, capacity: float, refill_per_s: float
    ) -> tuple[bool, float]:
        """Take `cost` tokens from the client's bucket: (allowed, seconds to wait)."""
        allowed, wait = await self._bucket(
            keys=[f"{PREFIX}bucket:{client}"], args=[capacity, refill_per_s, time.time(), cost]
        )
        return bool(int(allowed)), float(wait)

    # -- daily LLM spend ------------------------------------------------------------

    @staticmethod
    def _spend_key() -> str:
        return f"{PREFIX}spend:{datetime.datetime.now(datetime.UTC).date().isoformat()}"

    async def spend_today(self) -> float:
        return float(await self.redis.get(self._spend_key()) or 0)

    async def add_spend(self, usd: float) -> float:
        key = self._spend_key()
        total = float(await self.redis.incrbyfloat(key, usd))
        await self.redis.expire(key, 2 * 24 * 3600)
        return total
