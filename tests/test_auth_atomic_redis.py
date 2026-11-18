"""Opt-in checks against a disposable, network-isolated Redis test container.

Set MEDEVIDENCE_TEST_REDIS_CONTAINER to that container's name. No TCP connection,
business Redis configuration, volumes, or global key cleanup is used here.
"""

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest

from med_evidence_agentic_rag.models_and_database.redis import VERIFY_EMAIL_CODE_SCRIPT

container = os.environ.get("MEDEVIDENCE_TEST_REDIS_CONTAINER", "")
pytestmark = pytest.mark.skipif(not container, reason="requires disposable Redis test container")


def command(*args):
    return subprocess.check_output(
        ["docker", "exec", container, "redis-cli", "--raw", *map(str, args)],
        text=True,
        encoding="utf-8",
    ).strip()


def test_lua_limits_parallel_wrong_guesses_and_preserves_code_ttl():
    code_key = "synthetic:" + uuid4().hex
    failure_key = code_key + ":failures"
    try:
        command("SET", code_key, "123456", "PX", 120000)

        def verify(_):
            return command("EVAL", VERIFY_EMAIL_CODE_SCRIPT, 2, code_key, failure_key, "000000", 5)

        with ThreadPoolExecutor(max_workers=10) as pool:
            results = list(pool.map(verify, range(20)))
        assert results.count("invalid") == 4
        assert results.count("exhausted") == 16
        assert command("GET", failure_key) == "5"
        ttl_script = "return {redis.call('PTTL', KEYS[1]), redis.call('PTTL', KEYS[2])}"
        code_ttl, failure_ttl = map(
            int, command("EVAL", ttl_script, 2, code_key, failure_key).split()
        )
        assert 0 < failure_ttl <= code_ttl
        assert (
            command("EVAL", VERIFY_EMAIL_CODE_SCRIPT, 2, code_key, failure_key, "123456", 5)
            == "exhausted"
        )
    finally:
        command("DEL", code_key, failure_key)


def test_lua_consumes_parallel_success_once_and_clears_failures():
    code_key = "synthetic:" + uuid4().hex
    failure_key = code_key + ":failures"
    try:
        command("SET", code_key, "123456", "PX", 120000)
        command("SET", failure_key, 3, "PX", 120000)

        def verify(_):
            return command("EVAL", VERIFY_EMAIL_CODE_SCRIPT, 2, code_key, failure_key, "123456", 5)

        with ThreadPoolExecutor(max_workers=10) as pool:
            results = list(pool.map(verify, range(20)))
        assert results.count("success") == 1
        assert results.count("missing") == 19
        assert command("EXISTS", code_key, failure_key) == "0"
    finally:
        command("DEL", code_key, failure_key)


def test_getdel_consumes_parallel_human_token_once():
    key = "synthetic:" + uuid4().hex
    try:
        command("SET", key, "verified", "PX", 120000)
        with ThreadPoolExecutor(max_workers=10) as pool:
            results = list(pool.map(lambda _: command("GETDEL", key), range(20)))
        assert results.count("verified") == 1
        assert results.count("") == 19
    finally:
        command("DEL", key)
