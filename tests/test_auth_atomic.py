import asyncio
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from med_evidence_agentic_rag.api import auth
from med_evidence_agentic_rag.models_and_database import redis as store


def test_human_credentials_use_getdel(monkeypatch):
    client = AsyncMock()
    client.getdel.side_effect = ["ABCDE", "verified", None]
    monkeypatch.setattr(store, "redis_client", client)

    async def run():
        assert await store.verify_human_challenge("challenge", "ABCDE")
        assert await store.consume_human_verification("token")
        assert not await store.consume_human_verification("token")

    asyncio.run(run())
    assert client.getdel.await_count == 3
    client.get.assert_not_awaited()
    client.delete.assert_not_awaited()


@pytest.mark.parametrize("result,status", [("exhausted", 429), ("invalid", 400), ("missing", 400)])
def test_login_rejects_atomic_verification_failure(monkeypatch, result, status):
    verify = AsyncMock(return_value=result)
    monkeypatch.setattr(auth, "verify_email_code", verify, raising=False)
    user = AsyncMock()
    monkeypatch.setattr(auth, "get_or_create_user", user)
    with pytest.raises(HTTPException) as error:
        asyncio.run(
            auth.login_with_email(auth.EmailLoginRequest(email="a@example.org", code="000000"))
        )
    assert error.value.status_code == status
    user.assert_not_awaited()


def test_email_verification_uses_one_atomic_bounded_script(monkeypatch):
    client = AsyncMock()
    client.eval.return_value = "success"
    monkeypatch.setattr(store, "redis_client", client)
    assert asyncio.run(store.verify_email_code("a@example.org", "123456")) == "success"
    script, count, code_key, attempts_key, code, limit = client.eval.await_args.args
    assert count == 2 and code_key != attempts_key
    assert code == "123456" and limit == 5
    assert all(command in script for command in ["INCR", "PTTL", "PEXPIRE", "DEL"])
    assert "exhausted" in script
