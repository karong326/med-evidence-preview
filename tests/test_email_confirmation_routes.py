import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from med_evidence_agentic_rag.api import email_actions
from med_evidence_agentic_rag.api.auth import AuthenticatedUser, require_user
from med_evidence_agentic_rag.services.email_approval import EmailApprovalRegistry
from med_evidence_agentic_rag.tools import send_email as mail


def test_agent_tool_only_proposes_and_page_confirmation_sends_owner_payload(
    monkeypatch, email_sessions
):
    registry = EmailApprovalRegistry(email_sessions)
    monkeypatch.setattr(mail, "email_approvals", registry)
    monkeypatch.setattr(email_actions, "email_approvals", registry)
    monkeypatch.setattr(mail, "resolve_email_recipient", AsyncMock(return_value="test@example.org"))
    owner, conversation = uuid4(), uuid4()
    runtime = SimpleNamespace(context={"user_id": str(owner), "conversation_id": str(conversation)})
    sends = []
    monkeypatch.setattr(mail, "send_email", lambda *args: sends.append(args) or True)
    asyncio.run(mail.send_email_tool.coroutine("contact", "subject", "body", runtime))
    assert sends == []
    app = FastAPI()
    app.include_router(email_actions.email_actions_router)
    app.dependency_overrides[require_user] = lambda: AuthenticatedUser(owner, "test@example.org")
    with TestClient(app) as client:
        pending = client.get(
            "/api/email-actions", params={"conversation_id": str(conversation)}
        ).json()
        assert len(pending) == 1
        assert pending[0]["body"] == "body"
        url = f"/api/email-actions/{pending[0]['id']}/confirm"
        assert client.post(url).json()["status"] == "sent"
        assert client.post(url).json()["status"] == "sent"
        assert len(sends) == 1


def test_email_confirmation_does_not_accept_foreign_owner_or_payload(monkeypatch, email_sessions):
    registry = EmailApprovalRegistry(email_sessions)
    monkeypatch.setattr(email_actions, "email_approvals", registry)
    owner, other = uuid4(), uuid4()
    action = asyncio.run(registry.propose(owner, uuid4(), "test@example.org", "subject", "body"))
    app = FastAPI()
    app.include_router(email_actions.email_actions_router)
    app.dependency_overrides[require_user] = lambda: AuthenticatedUser(other, "other@example.org")
    with TestClient(app) as client:
        assert client.post(f"/api/email-actions/{action.id}/confirm").status_code == 404
        assert client.post(f"/api/email-actions/{action.id}/reject").status_code == 404
    assert action.status == "pending"
