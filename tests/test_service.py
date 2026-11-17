"""The HTTP service, against an in-process fake Redis and scripted systems."""

import fakeredis
import pytest
from fastapi.testclient import TestClient

from tableqa.interface import Prediction
from tableqa.service.app import create_app
from tableqa.service.config import Settings
from tableqa.service.metrics import sandbox_error_kinds
from tableqa.service.samples import Sample
from tableqa.service.store import normalize_question
from tableqa.table import Table

TABLE = Table.from_strings(
    "players", ["Player", "Goals"], [["Ann", "10"], ["Bob", "7"], ["Cy", "12"]]
)


class FakeSystem:
    def __init__(self, name, answer=("3",), cost=0.0, error=None, trace="SELECT 1", fail=False):
        self.name, self.calls = name, 0
        self._answer, self._cost, self._error, self._trace, self._fail = (
            list(answer),
            cost,
            error,
            trace,
            fail,
        )

    def answer(self, question, table):
        self.calls += 1
        if self._fail:
            raise RuntimeError("secret detail that must not leak")
        return Prediction(self._answer, self._trace, 12.0, self._cost, 100, self._error)


def make(settings=None, server=None, **systems):
    systems = systems or {"rule_based": FakeSystem("rule_based")}
    redis = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
    sample = Sample("sample-1", "Players", TABLE, ["how many players are there?"])
    app = create_app(settings or Settings(), redis=redis, systems=systems, samples=[sample])
    return TestClient(app), systems


def ask(client, question="How many players?", system="rule_based", table_id="sample-1", **kw):
    return client.post(
        "/ask", json={"question": question, "system": system, "table_id": table_id}, **kw
    )


def test_health_lists_systems_and_redis():
    client, _ = make()
    with client:
        r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "redis": True, "systems": ["rule_based"]}


def test_tables_and_schema():
    client, _ = make()
    with client:
        tables = client.get("/tables").json()
        schema = client.get("/tables/sample-1/schema").json()
        missing = client.get("/tables/nope/schema")
    assert tables == [
        {
            "table_id": "sample-1",
            "name": "Players",
            "row_count": 3,
            "example_questions": ["how many players are there?"],
        }
    ]
    assert schema["columns"] == [
        {"name": "player", "type": "VARCHAR", "header": "Player"},
        {"name": "goals", "type": "BIGINT", "header": "Goals"},
    ]
    assert schema["first_rows"] == [["Ann", "10"], ["Bob", "7"], ["Cy", "12"]]
    assert missing.status_code == 404


def test_an_answer_is_cached_by_normalised_question_and_system():
    client, systems = make(rule_based=FakeSystem("rule_based"), tapas=FakeSystem("tapas"))
    with client:
        first = ask(client, "How many players?").json()
        again = ask(client, "  how many   PLAYERS ").json()
        other = ask(client, "How many players?", system="tapas").json()
    assert first["cached"] is False and first["answer"] == ["3"] and first["trace"] == "SELECT 1"
    assert again["cached"] is True and again["answer"] == ["3"]
    assert again["system_latency_ms"] == first["system_latency_ms"]
    assert other["cached"] is False
    assert systems["rule_based"].calls == 1 and systems["tapas"].calls == 1


def test_normalize_question():
    assert normalize_question("  How many\nplayers?? ") == "how many players"


@pytest.mark.parametrize(
    "payload, status",
    [
        ({"question": "q", "system": "nope", "table_id": "sample-1"}, 400),
        ({"question": "   ", "system": "rule_based", "table_id": "sample-1"}, 400),
        ({"question": "x" * 501, "system": "rule_based", "table_id": "sample-1"}, 400),
        ({"question": "q", "system": "rule_based", "table_id": "missing"}, 404),
        ({"question": "q", "system": "rule_based"}, 422),
    ],
)
def test_bad_requests_are_refused(payload, status):
    client, systems = make()
    with client:
        assert client.post("/ask", json=payload).status_code == status
    assert systems["rule_based"].calls == 0


def test_rate_limit_charges_fresh_answers_their_weight_and_cached_ones_one():
    settings = Settings(rate_capacity=6, rate_refill_per_s=0.001)
    client, _ = make(settings, agent_haiku=FakeSystem("agent_haiku", cost=0.0001))
    with client:
        fresh = ask(client, "q1", system="agent_haiku")  # 5 tokens, 1 left
        cached = ask(client, "q1", system="agent_haiku")  # 1 token, 0 left
        limited = ask(client, "q1", system="agent_haiku")
    assert fresh.status_code == cached.status_code == 200
    assert limited.status_code == 429
    assert int(limited.headers["retry-after"]) >= 1


def test_rate_limits_are_per_client_address_behind_a_trusted_proxy():
    settings = Settings(rate_capacity=1, rate_refill_per_s=0.001, proxy_hops=1)
    client, _ = make(settings)
    with client:
        a = ask(client, "q1", headers={"x-forwarded-for": "1.1.1.1"})
        a_again = ask(client, "q2", headers={"x-forwarded-for": "1.1.1.1"})
        b = ask(client, "q2", headers={"x-forwarded-for": "2.2.2.2"})
    assert (a.status_code, a_again.status_code, b.status_code) == (200, 429, 200)


def test_a_client_cannot_dodge_the_rate_limit_by_forging_forwarded_for():
    settings = Settings(rate_capacity=1, rate_refill_per_s=0.001, proxy_hops=1)
    client, _ = make(settings)
    with client:
        # The proxy appends the real address (3.3.3.3); the forged entries come first.
        codes = [
            ask(client, f"q{i}", headers={"x-forwarded-for": f"9.9.9.{i}, 3.3.3.3"}).status_code
            for i in range(3)
        ]
    assert codes == [200, 429, 429]


def test_forwarded_for_is_ignored_unless_proxies_are_trusted():
    settings = Settings(rate_capacity=1, rate_refill_per_s=0.001)
    client, _ = make(settings)
    with client:
        a = ask(client, "q1", headers={"x-forwarded-for": "1.1.1.1"})
        b = ask(client, "q2", headers={"x-forwarded-for": "2.2.2.2"})
    assert (a.status_code, b.status_code) == (200, 429)


def test_daily_spend_cap_stops_the_agent_but_not_free_systems():
    settings = Settings(daily_spend_cap_usd=0.01)
    client, systems = make(
        settings, agent_haiku=FakeSystem("agent_haiku", cost=0.006), rule_based=FakeSystem("rb")
    )
    with client:
        statuses = [ask(client, f"q{i}", system="agent_haiku").status_code for i in range(3)]
        free = ask(client, "q9", system="rule_based")
        cached = ask(client, "q0", system="agent_haiku")
    assert statuses == [200, 200, 503]
    assert systems["agent_haiku"].calls == 2
    assert free.status_code == 200
    assert cached.status_code == 200 and cached.json()["cached"] is True


def test_a_failing_system_is_a_502_without_details_and_is_not_cached():
    client, systems = make(rule_based=FakeSystem("rule_based", fail=True))
    with client:
        first = ask(client)
        second = ask(client)
    assert first.status_code == second.status_code == 502
    assert "secret" not in first.text and "RuntimeError" in first.text
    assert systems["rule_based"].calls == 2


def test_upload_a_csv_and_ask_about_it():
    client, _ = make()
    csv_bytes = b"City,Population\nAnn Arbor,123000\nBoston,650000\n"
    with client:
        r = client.post("/tables", files={"file": ("cities.csv", csv_bytes, "text/csv")})
        same = client.post("/tables", files={"file": ("again.csv", csv_bytes, "text/csv")})
        table_id = r.json()["table_id"]
        answer = ask(client, "biggest city?", table_id=table_id)
        schema = client.get(f"/tables/{table_id}/schema").json()
    assert r.status_code == 201 and table_id.startswith("up-")
    assert same.json()["table_id"] == table_id
    assert r.json()["columns"][1] == {
        "name": "population",
        "type": "BIGINT",
        "header": "Population",
    }
    assert schema["name"] == "again.csv"  # same content, same id; the latest name is kept
    assert schema["row_count"] == 2
    assert answer.status_code == 200


def test_uploaded_tables_survive_a_restart_through_redis():
    server = fakeredis.FakeServer()
    csv_bytes = b"A,B\n1,2\n"
    client, _ = make(server=server)
    with client:
        table_id = client.post("/tables", files={"file": ("t.csv", csv_bytes)}).json()["table_id"]
    restarted, _ = make(server=server)
    with restarted:
        assert restarted.get(f"/tables/{table_id}/schema").status_code == 200


@pytest.mark.parametrize(
    "content, status",
    [
        (b"\xff\xfe\x00bad", 400),
        (b"Only,Header\n", 400),
        (b"", 400),
        (b"A\n" + b"1\n" * 60, 413),
        (b"x" * 3000, 413),
    ],
)
def test_bad_uploads_are_refused(content, status):
    settings = Settings(max_upload_rows=50, max_upload_bytes=2000)
    client, _ = make(settings)
    with client:
        r = client.post("/tables", files={"file": ("t.csv", content, "text/csv")})
    assert r.status_code == status


def test_metrics_record_outcomes_latency_tokens_and_sandbox_errors():
    failing = FakeSystem("rule_based", answer=(), error="QueryFailed: Binder Error")
    client, _ = make(rule_based=failing, agent_haiku=FakeSystem("agent_haiku", cost=0.002))
    with client:
        ask(client, "q1")
        ask(client, "q1")
        ask(client, "q2", system="agent_haiku")
        text = client.get("/metrics").text
    assert 'tableqa_ask_total{outcome="answered",system="rule_based"} 1.0' in text
    assert 'tableqa_ask_total{outcome="cached",system="rule_based"} 1.0' in text
    assert 'tableqa_sandbox_errors_total{kind="QueryFailed",system="rule_based"} 1.0' in text
    assert 'tableqa_llm_tokens_total{system="agent_haiku"} 100.0' in text
    assert 'tableqa_llm_cost_usd_total{system="agent_haiku"} 0.002' in text
    assert "tableqa_llm_spend_today_usd 0.002" in text
    assert 'tableqa_ask_latency_seconds_count{cache="hit",system="rule_based"} 1.0' in text


def test_sandbox_error_kinds_reads_errors_and_agent_traces():
    plain = Prediction([], "SELECT x", 1.0, error="QueryFailed: no column")
    agent = Prediction(
        ["3"],
        [{"tool_calls": [{"name": "run_sql", "error": True, "result": "NotReadOnly: no"}]}],
        1.0,
    )
    assert sandbox_error_kinds(plain) == ["QueryFailed"]
    assert sandbox_error_kinds(agent) == ["NotReadOnly"]
    assert sandbox_error_kinds(Prediction([], None, 1.0, error="step budget exhausted")) == []


def test_redis_outage_is_a_503_not_a_crash():
    server = fakeredis.FakeServer()
    client, systems = make(server=server)
    with client:
        server.connected = False
        health = client.get("/healthz")
        r = ask(client)
    assert health.status_code == 503 and health.json()["redis"] is False
    assert r.status_code == 503 and "unavailable" in r.json()["detail"]
    assert systems["rule_based"].calls == 0


def test_the_original_sql_generator_endpoint_still_works():
    client, _ = make()
    with client:
        r = client.post("/generate-sql", json={"prompt": "show all users"})
    assert " ".join(r.json()["sql"].split()) == "SELECT * FROM users;"


def test_a_cached_answer_costs_nothing_to_serve():
    client, _ = make(agent_haiku=FakeSystem("agent_haiku", cost=0.002))
    with client:
        fresh = ask(client, "q1", system="agent_haiku").json()
        cached = ask(client, "q1", system="agent_haiku").json()
        text = client.get("/metrics").text
    assert (fresh["cost_usd"], fresh["tokens"]) == (0.002, 100)
    assert (cached["cost_usd"], cached["tokens"]) == (0.0, 0)
    assert "tableqa_llm_spend_today_usd 0.002" in text


def test_every_series_starts_at_zero_so_rates_see_the_first_events():
    client, _ = make(rule_based=FakeSystem("rule_based"), agent_haiku=FakeSystem("agent_haiku"))
    with client:
        text = client.get("/metrics").text
    for system in ("rule_based", "agent_haiku"):
        for outcome in ("answered", "cached", "failed", "rate_limited", "spend_capped"):
            assert f'tableqa_ask_total{{outcome="{outcome}",system="{system}"}} 0.0' in text
        assert f'tableqa_ask_latency_seconds_count{{cache="hit",system="{system}"}} 0.0' in text
        assert f'tableqa_llm_tokens_total{{system="{system}"}} 0.0' in text
    assert 'tableqa_table_uploads_total{outcome="stored"} 0.0' in text
