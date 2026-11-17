"""Step the two load scenarios through several user counts and record the results.

    docker compose -f docker-compose.yml -f docker-compose.loadtest.yml up -d
    uv run --group load python loadtest/run.py

Each step runs Locust headless for a fixed time with all users started at once and no
wait between requests, so it measures how much the service can do, not a realistic
traffic mix. Writes Locust's CSVs and a summary to results/loadtest/.
"""

import argparse
import csv
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

HOST = "http://localhost:8000"
OUT = Path("results/loadtest")


def warm_cache() -> int:
    """Ask every example question once, so the cached scenario only sees hits."""
    with urllib.request.urlopen(f"{HOST}/tables") as r:
        tables = json.load(r)
    asked = 0
    for table in tables:
        for question in table["example_questions"]:
            body = json.dumps(
                {"question": question, "table_id": table["table_id"], "system": "rule_based"}
            ).encode()
            request = urllib.request.Request(
                f"{HOST}/ask", body, {"content-type": "application/json"}
            )
            with urllib.request.urlopen(request) as r:
                r.read()
            asked += 1
    return asked


def run_step(scenario: str, users: int, seconds: int) -> dict:
    prefix = OUT / f"{scenario}_u{users}"
    subprocess.run(
        [
            sys.executable, "-m", "locust", "-f", "loadtest/locustfile.py", "--headless",
            "--host", HOST, "-u", str(users), "-r", str(users), "-t", f"{seconds}s",
            "--csv", str(prefix), "--only-summary", "--loglevel", "WARNING", scenario,
        ],
        check=True,
    )
    with open(f"{prefix}_stats.csv", encoding="utf-8") as f:
        row = next(r for r in csv.DictReader(f) if r["Name"] == "Aggregated")
    requests, failures = int(row["Request Count"]), int(row["Failure Count"])
    return {
        "scenario": scenario,
        "users": users,
        "seconds": seconds,
        "requests": requests,
        "failures": failures,
        "requests_per_s": float(row["Requests/s"]),
        "p50_ms": float(row["50%"]),
        "p95_ms": float(row["95%"]),
        "p99_ms": float(row["99%"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--users", default="1,10,25,50")
    parser.add_argument("--seconds", type=int, default=30)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"warmed the cache with {warm_cache()} questions", file=sys.stderr)
    steps = []
    for scenario in ("CachedUser", "UncachedUser"):
        for users in map(int, args.users.split(",")):
            step = run_step(scenario, users, args.seconds)
            steps.append(step)
            print(json.dumps(step), file=sys.stderr)
    with open(OUT / "summary.json", "w", encoding="utf-8") as f:
        json.dump(steps, f, indent=2)
        f.write("\n")
    print("| Scenario | Users | Requests/s | p50 | p95 | p99 | Failures |")
    print("|---|---|---|---|---|---|---|")
    for s in steps:
        print(
            f"| {s['scenario']} | {s['users']} | {s['requests_per_s']:.0f} | {s['p50_ms']:.0f} ms "
            f"| {s['p95_ms']:.0f} ms | {s['p99_ms']:.0f} ms | {s['failures']} of {s['requests']} |"
        )


if __name__ == "__main__":
    main()
