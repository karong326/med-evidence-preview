"""Real auth and conversation routes, isolated persistence and no live network calls."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import jwt
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from med_evidence_agentic_rag.api import auth
from med_evidence_agentic_rag.api import conversations as routes


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setattr(auth, "jwt_secret_key", "isolated-test-secret-" * 3)
    monkeypatch.setattr(auth, "access_token_code", "test-user-code")
    monkeypatch.setattr(auth, "admin_access_code", "test-admin-code")
    app = FastAPI()
    app.include_router(auth.auth_router)
    app.include_router(routes.conversation_router)
    app.state.checkpointer = SimpleNamespace(adelete_thread=AsyncMock())
    user_id = uuid4()
    headers = {"Authorization": "Bearer " + auth.create_access_token(user_id, "test@example.org")}
    with TestClient(app) as client:
        yield client, headers, user_id, app


def test_protected_conversations_require_valid_identity(api):
    client, headers, _, _ = api
    assert client.get("/api/conversations").status_code == 401
    assert (
        client.get("/api/conversations", headers={"Authorization": "Bearer invalid"}).status_code
        == 401
    )
    expired = jwt.encode(
        {
            "sub": str(uuid4()),
            "email": "test@example.org",
            "exp": datetime.now(UTC) - timedelta(seconds=1),
        },
        auth.jwt_secret_key,
        algorithm=auth.jwt_algorithm,
    )
    assert (
        client.get("/api/conversations", headers={"Authorization": "Bearer " + expired}).status_code
        == 401
    )


def test_access_login_consumes_captcha_before_creating_user(api, monkeypatch):
    client, _, user_id, _ = api
    consume = AsyncMock(return_value=False)
    create = AsyncMock(return_value=SimpleNamespace(id=user_id, email="test@example.org"))
    monkeypatch.setattr(auth, "consume_human_verification", consume)
    monkeypatch.setattr(auth, "get_or_create_access_user", create)
    body = {"human_token": "test-human", "access_code": "test-user-code", "visitor_id": "v" * 20}
    assert client.post("/api/auth/login/access-token", json=body).status_code == 400
    create.assert_not_awaited()
    consume.return_value = True
    response = client.post("/api/auth/login/access-token", json=body)
    assert response.status_code == 200
    token = response.json()["access_token"]
    claims = jwt.decode(token, auth.jwt_secret_key, algorithms=[auth.jwt_algorithm])
    assert claims["sub"] == str(user_id)
    assert claims["role"] == "user"


def test_wrong_access_code_and_normal_user_admin_boundary(api, monkeypatch):
    client, _, user_id, _ = api
    monkeypatch.setattr(auth, "consume_human_verification", AsyncMock(return_value=True))
    body = {"human_token": "test-human", "access_code": "wrong", "visitor_id": "v" * 20}
    assert client.post("/api/auth/login/access-token", json=body).status_code == 401
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as denied:
        auth.require_admin(auth.AuthenticatedUser(user_id, "test@example.org"))
    assert denied.value.status_code == 403
    body["access_code"] = "test-admin-code"
    response = client.post("/api/auth/login/access-token", json=body)
    assert response.status_code == 200
    assert response.json()["role"] == "admin"


def test_history_and_delete_keep_owner_and_checkpoint_boundary(api, monkeypatch, isolated_journal):
    client, headers, user_id, app = api
    conversation_id = uuid4()
    history = AsyncMock(return_value=None)
    owned = AsyncMock(return_value=None)
    deleted = AsyncMock(return_value=True)
    monkeypatch.setattr(routes, "list_chat_turns", history)
    monkeypatch.setattr(routes, "get_owned_conversation", owned)
    monkeypatch.setattr(routes, "delete_owned_conversation", deleted)
    url = f"/api/conversations/{conversation_id}"
    assert client.get(url + "/turns", headers=headers).status_code == 404
    history.assert_awaited_once_with(user_id, conversation_id)
    assert client.delete(url, headers=headers).status_code == 404
    deleted.assert_not_awaited()
    app.state.checkpointer.adelete_thread.assert_not_awaited()
    owned.return_value = SimpleNamespace(id=conversation_id)
    assert client.delete(url, headers=headers).status_code == 204
    deleted.assert_awaited_once_with(user_id, conversation_id)
    app.state.checkpointer.adelete_thread.assert_awaited_once_with(str(conversation_id))


def test_delete_keeps_record_when_checkpoint_service_missing(api, monkeypatch):
    client, headers, _, app = api
    monkeypatch.setattr(routes, "get_owned_conversation", AsyncMock(return_value=object()))
    deleted = AsyncMock()
    monkeypatch.setattr(routes, "delete_owned_conversation", deleted)
    del app.state.checkpointer
    assert client.delete(f"/api/conversations/{uuid4()}", headers=headers).status_code == 503
    deleted.assert_not_awaited()
