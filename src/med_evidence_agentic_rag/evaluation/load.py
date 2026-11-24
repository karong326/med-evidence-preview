"""Bounded authenticated chat traffic against an explicitly selected deployment."""

import argparse
import asyncio
import json
import os
from collections import Counter
from time import perf_counter
from urllib.parse import urlsplit

import httpx
from pydantic import Field, model_validator

from .adapters import CleanupFailure, HTTPAgentAdapter
from .runner import quantile
from .schemas import Case, StrictModel, fingerprint


class LoadConfig(StrictModel):
    target: str
    allow_live: bool = False
    concurrency: int = Field(default=1, ge=1, le=32)
    duration: float = Field(default=30, gt=0, le=3600)
    requests: int = Field(default=100, ge=1, le=10000)
    max_error_rate: float = Field(default=0, ge=0, le=1)
    max_p95: float = Field(default=30, gt=0, le=3600)

    @model_validator(mode="after")
    def explicit_target(self):
        url = urlsplit(self.target)
        if (
            not self.allow_live
            or not url.hostname
            or url.username is not None
            or (url.password is not None or url.query or url.fragment or url.path not in ("", "/"))
        ):
            raise ValueError("Explicit origin and live opt-in required")
        if url.scheme != "https" and not (
            url.scheme == "http" and url.hostname in ("localhost", "127.0.0.1", "::1")
        ):
            raise ValueError("Use HTTPS or loopback HTTP")
        return self


async def measure(
    config: LoadConfig, token: str, question: str, client: httpx.AsyncClient
) -> dict[str, object]:
    adapter = HTTPAgentAdapter(config.target, token, allow_live=config.allow_live, client=client)
    case = Case(id="load", query=question)
    work = iter(range(config.requests))
    latencies: list[float] = []
    errors: Counter[str] = Counter()
    inflight = peak = 0
    start = perf_counter()
    deadline = start + config.duration

    async def worker() -> None:
        nonlocal inflight, peak
        for _ in work:
            began = perf_counter()
            if began >= deadline:
                break
            inflight += 1
            peak = max(peak, inflight)
            try:
                async with asyncio.timeout(deadline - began):
                    result = await adapter.execute(case)
                    if result.error:
                        errors[result.error] += 1
                    elif not result.answer:
                        errors["empty_answer"] += 1
            except (TimeoutError, httpx.TimeoutException):
                errors["timeout"] += 1
            except httpx.HTTPError:
                errors["http"] += 1
            except CleanupFailure:
                errors["cleanup"] += 1
            except (KeyError, ValueError, TypeError):
                errors["protocol"] += 1
            finally:
                inflight -= 1
                latencies.append(perf_counter() - began)

    async with asyncio.TaskGroup() as group:
        for _ in range(config.concurrency):
            group.create_task(worker())
    elapsed = perf_counter() - start
    count = len(latencies)
    p95 = quantile(latencies, 0.95)
    rate = errors.total() / count if count else None
    return {
        "target_fingerprint": fingerprint(config.target),
        "completed": count,
        "config": config.model_dump(exclude={"target"}),
        "elapsed_seconds": elapsed,
        "peak_client_inflight": peak,
        "errors": dict(errors),
        "error_rate": rate,
        "throughput_per_second": count / elapsed if elapsed > 0 else None,
        "p95_seconds": p95,
        "passed": rate is not None
        and rate <= config.max_error_rate
        and p95 is not None
        and p95 <= config.max_p95
        and not errors["cleanup"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--allow-live", action="store_true")
    for name, kind, default in (
        ("concurrency", int, 1),
        ("requests", int, 100),
        ("duration", float, 30),
        ("max-error-rate", float, 0),
        ("max-p95", float, 30),
    ):
        parser.add_argument("--" + name, type=kind, default=default)
    args = parser.parse_args()

    async def execute() -> dict[str, object]:
        config = LoadConfig.model_validate(vars(args))
        async with httpx.AsyncClient(timeout=30, trust_env=False) as client:
            return await measure(
                config,
                os.getenv("EVAL_AGENT_TOKEN", ""),
                os.getenv("EVAL_LOAD_QUERY", "Synthetic verification question"),
                client,
            )

    try:
        report = asyncio.run(execute())
    except Exception:
        print('{"error":"load_configuration_or_transport_failed"}')
        return 2
    print(json.dumps(report, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
