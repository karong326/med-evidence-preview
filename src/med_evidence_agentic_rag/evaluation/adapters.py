"""Offline replay and lazy opt-in product adapters."""

import asyncio
import json
import re
from time import perf_counter
from typing import TYPE_CHECKING, Literal, Protocol
from uuid import UUID

import httpx

from .schemas import Case, Observation, fingerprint

if TYPE_CHECKING:
    from med_evidence_agentic_rag.rag_retrieval.models import RetrievalResult

_CLEANUP_TIMEOUT_SECONDS = 10.0


class CleanupFailure(RuntimeError):
    """A temporary conversation could not be removed during cancellation."""


def evidence_key(source: str, content: str) -> str:
    """Stable chunk key independent of per-query citation numbering."""
    return fingerprint({'source': source, 'content': content})


class Adapter(Protocol):
    metadata: dict[str, object]

    async def execute(self, case: Case) -> Observation: ...


class ReplayAdapter:
    def __init__(self, results: list[Observation]):
        self.results = {r.id: r for r in results}
        if len(self.results) != len(results):
            raise ValueError('duplicate result ID')
        self.metadata: dict[str, object] = {'adapter': 'replay', 'version': 1}

    async def execute(self, case: Case) -> Observation:
        if case.id not in self.results:
            raise ValueError('missing replay case')
        return self.results[case.id].model_copy(deep=True)


class RetrievalAdapter:
    """Runs production hybrid/rerank retrieval, or an explicit dense ablation.

    Explicit user UUID is required; omission must never widen access. Imports are lazy.
    This synchronous product retriever is serialized: cancellation cannot stop a worker
    thread already inside the embedding/vector backend.
    """

    def __init__(self, *, allow_live: bool = False, user_id: str | None = None,
                 top_k: int = 5, mode: Literal['production', 'dense'] = 'production'):
        if not allow_live:
            raise ValueError('allow_live must be explicit')
        if user_id is None or top_k < 1 or mode not in ('production', 'dense'):
            raise ValueError('explicit user_id and positive top_k required')
        self.user_id = UUID(user_id)
        self.top_k = top_k
        self.mode = mode
        self._lock = asyncio.Lock()
        self.metadata: dict[str, object] = {'adapter': 'retrieval', 'version': 2,
                         'retrieval_mode': mode,
                         'scope_fingerprint': fingerprint(str(self.user_id))}

    def _retrieve(self, query: str) -> tuple['RetrievalResult', int]:
        if self.mode == 'dense':
            from med_evidence_agentic_rag.rag_retrieval.retrieval import retrieve

            return retrieve(query, top_k=self.top_k, user_id=self.user_id), self.top_k
        from med_evidence_agentic_rag.rag_retrieval.reranker import (
            reranker_top_k,
            retrieve_with_rerank,
        )

        return retrieve_with_rerank(query, user_id=self.user_id), reranker_top_k

    async def execute(self, case: Case) -> Observation:
        async with self._lock:
            # Keep lock held until a timed-out backend thread exits.
            task = asyncio.create_task(asyncio.to_thread(
                self._retrieve, case.query))
            try:
                result, depth = await asyncio.shield(task)
            except asyncio.CancelledError:
                await task
                raise
        self.metadata['configured_depth'] = depth
        return Observation(id=case.id, retrieval_depth=depth,
                           evidence_keys=[evidence_key(e.source, e.content)
                                          for e in result.evidences],
                           evidence_contexts=[e.content for e in result.evidences],
                           citation_ids=[e.citation_id for e in result.evidences])


class HTTPAgentAdapter:
    """Authenticated fresh-conversation NDJSON evaluation, with cleanup by default."""

    def __init__(self, base_url: str, token: str, *, allow_live: bool = False,
                 cleanup: bool = True, client: httpx.AsyncClient | None = None):
        if not allow_live:
            raise ValueError('allow_live must be explicit')
        if not token:
            raise ValueError('authenticated bearer token required')
        self.url = base_url.rstrip('/')
        self.headers = {'Authorization': f'Bearer {token}'}
        self.cleanup = cleanup
        self.client = client
        self.metadata: dict[str, object] = {
            'adapter': 'http-agent', 'version': 1, 'cleanup': cleanup,
            # Conservative identity: rotating a credential requires a fresh baseline.
            # High-entropy bearer credentials are hashed; the credential never enters reports.
            'scope_fingerprint': fingerprint({'bearer_credential': token}),
            'target_fingerprint': fingerprint(self.url)}

    async def execute(self, case: Case) -> Observation:
        if self.client is not None:
            return await self._execute(case, self.client)
        async with httpx.AsyncClient(timeout=60, follow_redirects=False) as client:
            return await self._execute(case, client)

    async def _cleanup_conversation(self, client: httpx.AsyncClient,
                                    conversation_id: str) -> bool:
        async def delete() -> None:
            async with asyncio.timeout(_CLEANUP_TIMEOUT_SECONDS):
                response = await client.delete(
                    self.url + '/api/conversations/' + conversation_id, headers=self.headers)
                response.raise_for_status()

        # Own the task until it has finished or acknowledged its own deadline cancellation.
        # Shield alone is insufficient: caller cancellation must not outlive the client.
        task = asyncio.create_task(delete())
        cancellation: asyncio.CancelledError | None = None
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as error:
                cancellation = error
            except Exception:
                break
        failed = False
        try:
            task.result()
        except (Exception, asyncio.CancelledError):
            failed = True
        if cancellation is not None:
            if failed:
                raise CleanupFailure('temporary conversation cleanup failed') from None
            raise cancellation
        return not failed

    async def _execute(self, case: Case, client: httpx.AsyncClient) -> Observation:
        created = await client.post(self.url + '/api/conversations', headers=self.headers)
        created.raise_for_status()
        conversation_id = str(UUID(created.json()['id']))
        result = Observation(id=case.id)
        parts: list[str] = []
        done = False
        start = perf_counter()
        try:
            async with client.stream('POST', self.url + '/api/ask', headers=self.headers,
                                     json={'question': case.query,
                                           'conversation_id': conversation_id}) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    event = json.loads(line)
                    kind = event['type']
                    if done:
                        result.error = 'protocol'
                        break
                    if kind == 'answer':
                        content = event['content']
                        if not isinstance(content, str):
                            raise ValueError('invalid answer chunk')
                        parts.append(content)
                        if content and result.ttft_seconds is None:
                            result.ttft_seconds = perf_counter() - start
                    elif kind == 'evidence':
                        for evidence in event['evidences']:
                            key = evidence_key(evidence['source'], evidence['content'])
                            citation = evidence['citation_id']
                            if citation in result.citation_ids:
                                continue
                            result.evidence_keys.append(key)
                            result.evidence_contexts.append(evidence['content'])
                            result.citation_ids.append(citation)
                    elif kind == 'trace':
                        trace = event['trace']
                        if trace.get('kind') == 'tool_call':
                            name = trace.get('tool_name')
                            if name is None:
                                match = re.search(r'原生名称：[ ]*([a-zA-Z_][\w]*)',
                                                  trace.get('detail', ''))
                                name = match.group(1) if match else None
                            if isinstance(name, str):
                                result.tools.append(name)
                    elif kind == 'done':
                        done = True
                    elif kind == 'error':
                        result.error = 'agent'
                        break
            if not done and result.error is None:
                result.error = 'incomplete'
        except httpx.TimeoutException:
            result.error = 'timeout'
        except httpx.HTTPStatusError:
            result.error = 'http'
        except httpx.RequestError:
            result.error = 'network'
        except (ValueError, KeyError, TypeError):
            result.error = 'protocol'
        finally:
            if self.cleanup:
                if not await self._cleanup_conversation(client, conversation_id):
                    result.error = 'cleanup'
        result.answer = ''.join(parts) if parts else None
        # Streaming agent has no fixed k; only observed ranks can be scored.
        result.retrieval_depth = len(result.evidence_keys)
        result.cited_ids = re.findall(r'\[([A-Za-z0-9][\w.-]*)\]', result.answer or '')
        return Observation.model_validate(result.model_dump())
