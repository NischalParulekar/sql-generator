"""The TableQA HTTP service.

    uvicorn tableqa.service.app:app --port 8000

Endpoints:
- GET  /tables               sample tables with example questions
- POST /tables               upload a CSV table (size-capped)
- GET  /tables/{id}/schema   columns, types and first rows of a table
- POST /ask                  answer a question about a table with one system
- GET  /healthz              liveness, Redis and the systems on offer
- GET  /metrics              Prometheus metrics
- POST /generate-sql         the original SQL-generator endpoint, kept for the old UI

Answers are cached in Redis by (system, table content, normalised question). Each
client address has a token bucket; a fresh answer costs its system's weight, a cached
one costs 1. LLM spend is capped per UTC day across all clients.
"""

import asyncio
import csv
import functools
import hashlib
import io
import logging
import math
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field
from redis.asyncio import Redis
from redis.exceptions import RedisError

from tableqa.service.config import Settings
from tableqa.service.metrics import Metrics, sandbox_error_kinds
from tableqa.service.samples import Sample, fingerprint
from tableqa.service.store import Store, answer_key
from tableqa.systems.rule_based.nlp_sql import generate_sql
from tableqa.table import Table

log = logging.getLogger("tableqa.service")

SYSTEM_NAMES = {"rule_based": "Rule-based", "tapas": "TAPAS", "agent_haiku": "Claude agent"}
PAID = {"agent_haiku"}
UPLOADED_TABLES_IN_MEMORY = 32


class AskRequest(BaseModel):
    question: str
    table_id: str
    system: str


class AskResponse(BaseModel):
    system: str
    table_id: str
    question: str
    answer: list[str]
    error: str | None
    cached: bool
    latency_ms: float = Field(description="Time this request took.")
    system_latency_ms: float = Field(description="Time the system took when it answered.")
    cost_usd: float
    tokens: int
    trace: Any = Field(description="The system's SQL, selected cells or agent steps.")


class QueryRequest(BaseModel):
    prompt: str
    schema: str = ""


def load_systems(names: list[str]) -> dict:
    """Build each configured system; one that cannot be built is left out and logged."""
    from tableqa.eval.runner import SYSTEMS

    systems = {}
    for name in names:
        try:
            systems[name] = SYSTEMS[name]()
        except Exception:
            log.exception("system %s is unavailable", name)
    return systems


def load_samples() -> list[Sample]:
    from tableqa.data.wtq import DEFAULT_ZIP, WTQ
    from tableqa.service.samples import synthetic_sample, wtq_samples

    samples = wtq_samples(WTQ(DEFAULT_ZIP)) if DEFAULT_ZIP.exists() else []
    return samples + [synthetic_sample()]


def create_app(
    settings: Settings | None = None,
    redis: Redis | None = None,
    systems: dict | None = None,
    samples: list[Sample] | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    metrics = Metrics()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state = app.state
        state.redis = redis or Redis.from_url(settings.redis_url, decode_responses=True)
        state.store = Store(state.redis, settings.table_ttl_s, settings.cache_ttl_s)
        state.systems = systems if systems is not None else load_systems(settings.systems)
        state.samples = {s.id: s for s in (samples if samples is not None else load_samples())}
        state.fingerprints = {s.id: fingerprint(s.table) for s in state.samples.values()}
        state.uploaded = OrderedDict()
        state.limits = {
            name: asyncio.Semaphore(settings.concurrency.get(name, 4)) for name in state.systems
        }
        metrics.start_at_zero(state.systems)
        try:
            metrics.spend_today.set(await state.store.spend_today())
        except RedisError:
            log.warning("Redis is not reachable at startup")
        yield
        if redis is None:
            await state.redis.aclose()

    app = FastAPI(title="TableQA", lifespan=lifespan)
    app.state.metrics = metrics
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.exception_handler(RedisError)
    async def redis_down(request: Request, exc: RedisError):
        log.error("Redis error: %s", exc)
        return JSONResponse(
            {"detail": "The service's store is unavailable. Try again shortly."}, status_code=503
        )

    def client_of(request: Request) -> str:
        forwarded = request.headers.get("x-forwarded-for")
        if settings.trust_proxy and forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    async def table_for(app_state, table_id: str) -> tuple[Table, str, str]:
        """(table, content fingerprint, display name), or 404."""
        sample = app_state.samples.get(table_id)
        if sample:
            return sample.table, app_state.fingerprints[table_id], sample.name
        if table_id.startswith("up-"):
            if table_id in app_state.uploaded:
                app_state.uploaded.move_to_end(table_id)
                return app_state.uploaded[table_id]
            body = await app_state.store.get_table(table_id)
            if body:
                entry = (
                    Table.from_strings(table_id, body["headers"], body["rows"]),
                    table_id,  # an uploaded table's id is already its content hash
                    body["name"],
                )
                app_state.uploaded[table_id] = entry
                if len(app_state.uploaded) > UPLOADED_TABLES_IN_MEMORY:
                    app_state.uploaded.popitem(last=False)
                return entry
        raise HTTPException(404, f"No table with id {table_id!r}. It may have expired.")

    def describe(table: Table, table_id: str, name: str, questions: list[str]) -> dict:
        return {
            "table_id": table_id,
            "name": name,
            "row_count": len(table.rows),
            "columns": [
                {"name": c, "type": t, "header": h}
                for c, t, h in zip(table.columns, table.types, table.headers)
            ],
            "first_rows": [list(row) for row in table.raw_rows[:5]],
            "example_questions": questions,
        }

    @app.get("/healthz")
    async def healthz(request: Request):
        redis_ok = await request.app.state.store.ping()
        body = {
            "status": "ok" if redis_ok else "degraded",
            "redis": redis_ok,
            "systems": sorted(request.app.state.systems),
        }
        return JSONResponse(body, status_code=200 if redis_ok else 503)

    @app.get("/metrics")
    async def prometheus_metrics():
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    @app.get("/systems")
    async def list_systems(request: Request):
        return [
            {"system": name, "name": SYSTEM_NAMES.get(name, name), "paid": name in PAID}
            for name in request.app.state.systems
        ]

    @app.get("/tables")
    async def list_tables(request: Request):
        return [
            {
                "table_id": s.id,
                "name": s.name,
                "row_count": len(s.table.rows),
                "example_questions": s.questions,
            }
            for s in request.app.state.samples.values()
        ]

    @app.get("/tables/{table_id}/schema")
    async def table_schema(table_id: str, request: Request):
        table, _, name = await table_for(request.app.state, table_id)
        sample = request.app.state.samples.get(table_id)
        return describe(table, table_id, name, sample.questions if sample else [])

    @app.post("/tables", status_code=201)
    async def upload_table(request: Request, file: UploadFile = File(...)):
        data = await file.read(settings.max_upload_bytes + 1)
        if len(data) > settings.max_upload_bytes:
            metrics.uploads.labels("too_large").inc()
            raise HTTPException(413, f"The file is over {settings.max_upload_bytes:,} bytes.")
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            metrics.uploads.labels("rejected").inc()
            raise HTTPException(400, "The file is not UTF-8 text. Save it as UTF-8 CSV.") from None
        rows = [row for row in csv.reader(io.StringIO(text)) if any(c.strip() for c in row)]
        if len(rows) < 2 or not any(h.strip() for h in rows[0]):
            metrics.uploads.labels("rejected").inc()
            raise HTTPException(400, "The CSV needs a header row and at least one data row.")
        headers, body = rows[0], rows[1:]
        if len(headers) > settings.max_upload_columns or len(body) > settings.max_upload_rows:
            metrics.uploads.labels("too_large").inc()
            raise HTTPException(
                413,
                f"Tables are limited to {settings.max_upload_columns} columns and "
                f"{settings.max_upload_rows:,} rows.",
            )
        table_id = "up-" + hashlib.sha256(text.encode()).hexdigest()[:16]
        name = (file.filename or "uploaded.csv")[:100]
        table = Table.from_strings(table_id, headers, body)
        await request.app.state.store.put_table(table_id, name, headers, body)
        metrics.uploads.labels("stored").inc()
        return describe(table, table_id, name, [])

    @app.post("/ask", response_model=AskResponse)
    async def ask(body: AskRequest, request: Request):
        state = request.app.state
        started = time.perf_counter()
        system = state.systems.get(body.system)
        if system is None:
            raise HTTPException(400, f"Unknown system. Choose one of: {', '.join(state.systems)}.")
        question = body.question.strip()
        if not question or len(question) > settings.max_question_chars:
            raise HTTPException(
                400, f"Ask a question of 1 to {settings.max_question_chars} characters."
            )
        table, table_fingerprint, _ = await table_for(state, body.table_id)

        key = answer_key(body.system, table_fingerprint, question)
        cached = await state.store.get_answer(key)
        weight = 1 if cached else settings.system_weight.get(body.system, 1)
        allowed, wait = await state.store.take_tokens(
            client_of(request), weight, settings.rate_capacity, settings.rate_refill_per_s
        )
        if not allowed:
            metrics.asks.labels(body.system, "rate_limited").inc()
            raise HTTPException(
                429,
                "Too many questions from this address. Wait a moment and try again.",
                headers={"Retry-After": str(max(1, math.ceil(wait)))},
            )

        if cached:
            elapsed = time.perf_counter() - started
            metrics.latency.labels(body.system, "hit").observe(elapsed)
            metrics.asks.labels(body.system, "cached").inc()
            # Serving from the cache costs nothing; the stored cost was paid once.
            return AskResponse(
                system=body.system,
                table_id=body.table_id,
                question=question,
                cached=True,
                latency_ms=elapsed * 1000,
                **{**cached, "cost_usd": 0.0, "tokens": 0},
            )

        if body.system in PAID:
            spent = await state.store.spend_today()
            metrics.spend_today.set(spent)
            if spent >= settings.daily_spend_cap_usd:
                metrics.asks.labels(body.system, "spend_capped").inc()
                raise HTTPException(
                    503,
                    "The Claude agent has reached today's spending limit. "
                    "Try another system, or come back tomorrow (UTC).",
                )

        async with state.limits[body.system]:
            run = functools.partial(system.answer, question, table)
            try:
                prediction = await asyncio.wait_for(
                    asyncio.to_thread(run), timeout=settings.ask_timeout_s
                )
            except TimeoutError:
                metrics.asks.labels(body.system, "failed").inc()
                raise HTTPException(504, "The system took too long to answer.") from None
            except Exception as e:
                log.exception("system %s failed", body.system)
                metrics.asks.labels(body.system, "failed").inc()
                raise HTTPException(502, f"The system failed ({type(e).__name__}).") from None

        elapsed = time.perf_counter() - started
        metrics.latency.labels(body.system, "miss").observe(elapsed)
        metrics.tokens.labels(body.system).inc(prediction.tokens)
        metrics.cost.labels(body.system).inc(prediction.cost_usd)
        for kind in sandbox_error_kinds(prediction):
            metrics.sandbox_errors.labels(body.system, kind).inc()
        if body.system in PAID and prediction.cost_usd:
            metrics.spend_today.set(await state.store.add_spend(prediction.cost_usd))

        result = {
            "answer": [str(a) for a in prediction.answer],
            "error": prediction.error,
            "system_latency_ms": prediction.latency_ms,
            "cost_usd": prediction.cost_usd,
            "tokens": prediction.tokens,
            "trace": prediction.sql_or_trace,
        }
        await state.store.put_answer(key, result)
        metrics.asks.labels(body.system, "answered").inc()
        return AskResponse(
            system=body.system,
            table_id=body.table_id,
            question=question,
            cached=False,
            latency_ms=elapsed * 1000,
            **result,
        )

    @app.post("/generate-sql")
    def generate(request: QueryRequest):
        if not request.prompt.strip():
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")
        sql = generate_sql(request.prompt, request.schema)
        return {"sql": sql}

    return app


def __getattr__(name: str):
    # `uvicorn tableqa.service.app:app` builds the app on first access, so importing
    # this module (as the tests do) has no side effects.
    if name == "app":
        globals()["app"] = create_app()
        return globals()["app"]
    raise AttributeError(name)
