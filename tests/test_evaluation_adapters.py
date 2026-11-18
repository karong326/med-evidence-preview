import asyncio
import json

import httpx
import pytest

from med_evidence_agentic_rag.evaluation.adapters import HTTPAgentAdapter, RetrievalAdapter
from med_evidence_agentic_rag.evaluation.judges import ReferenceJudge, RemoteJudge
from med_evidence_agentic_rag.evaluation.schemas import Case, Observation


def test_live_and_remote_require_opt_in():
    with pytest.raises(ValueError, match='allow_live'):
        RetrievalAdapter()
    with pytest.raises(ValueError, match='allow_live'):
        HTTPAgentAdapter('https://example.invalid', 'secret')
    with pytest.raises(ValueError, match='allow_remote'):
        RemoteJudge('https://example.invalid', 'model', 'secret')


def test_http_scope_fingerprint_distinguishes_credentials_without_storing_them():
    first = HTTPAgentAdapter('https://example.invalid', 'synthetic_token_one', allow_live=True)
    second = HTTPAgentAdapter('https://example.invalid', 'synthetic_token_two', allow_live=True)
    assert first.metadata['scope_fingerprint'] != second.metadata['scope_fingerprint']
    assert 'synthetic_token_one' not in json.dumps(first.metadata)
    assert 'synthetic_token_two' not in json.dumps(second.metadata)


def test_reference_judge_is_labeled_lexical_not_clinical():
    scores = asyncio.run(ReferenceJudge().score(
        Case(id='a', query='q', reference_answer='alpha beta'),
        Observation(id='a', answer='alpha')))
    assert scores.reference_token_f1 == pytest.approx(2 / 3)
    assert scores.answer_correctness is None
    assert scores.faithfulness is None


@pytest.mark.parametrize('terminal,error', [('done', None), ('error', 'agent'),
                                          ('', 'incomplete')])
def test_http_ndjson_partial_evidence_and_cleanup(terminal, error):
    requests = []

    def respond(request):
        requests.append((request.method, request.url.path))
        assert request.headers['Authorization'] == 'Bearer secret'
        if request.method == 'DELETE':
            return httpx.Response(204)
        if request.url.path == '/api/conversations':
            return httpx.Response(200, json={'id': '01234567-0123-0123-0123-012345678901'})
        events = [{'type': 'answer', 'content': 'partial [E1]'},
                  {'type': 'evidence', 'evidences': [
                      {'source': 's', 'content': 'c', 'citation_id': 'E1'}]},
                  {'type': 'trace', 'trace': {'kind': 'tool_call',
                                            'detail': '原生名称：search_knowledge'}}]
        if terminal:
            events.append({'type': terminal})
        return httpx.Response(200, text='\n'.join(json.dumps(e) for e in events))

    async def invoke():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            adapter = HTTPAgentAdapter('https://example.invalid', 'secret',
                                       allow_live=True, client=client)
            return await adapter.execute(Case(id='a', query='q'))

    result = asyncio.run(invoke())
    assert result.error == error
    assert result.answer == 'partial [E1]'
    assert result.cited_ids == ['E1']
    assert result.tools == ['search_knowledge']
    assert result.evidence_contexts == ['c']
    assert result.ttft_seconds is not None
    assert requests[-1][0] == 'DELETE'


def test_remote_judge_structured_scores_and_usage():
    def respond(request):
        body = json.loads(request.content)
        assert body['response_format']['type'] == 'json_schema'
        return httpx.Response(200, json={
            'choices': [{'message': {'content': json.dumps({
                'reference_exact_match': None, 'reference_token_f1': None,
                'answer_correctness': 0.75, 'faithfulness': None})}}],
            'usage': {'prompt_tokens': 20, 'completion_tokens': 5}})

    async def invoke():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            judge = RemoteJudge('https://example.invalid/v1', 'synthetic', 'secret',
                                allow_remote=True, client=client)
            return await judge.score(Case(id='a', query='q', reference_answer='r'),
                                     Observation(id='a', answer='a'))

    scores = asyncio.run(invoke())
    assert scores.answer_correctness == 0.75
    assert scores.provider_usage == {'input_tokens': 20, 'output_tokens': 5, 'cost_usd': None}


def test_retrieval_preserves_user_scope(monkeypatch):
    import sys
    import types
    from uuid import UUID

    calls = []

    def retrieve(query, *, top_k, user_id):
        calls.append((query, top_k, user_id))
        return types.SimpleNamespace(evidences=[types.SimpleNamespace(
            source='s', content='c', citation_id='E1')])

    module = types.ModuleType('med_evidence_agentic_rag.rag_retrieval.retrieval')
    module.retrieve = retrieve
    monkeypatch.setitem(sys.modules, module.__name__, module)
    user_id = '01234567-0123-0123-0123-012345678901'
    result = asyncio.run(RetrievalAdapter(allow_live=True, user_id=user_id,
                                         top_k=3, mode='dense').execute(Case(id='a', query='q')))
    assert calls == [('q', 3, UUID(user_id))]
    assert result.retrieval_depth == 3
    assert len(result.evidence_keys[0]) == 64
    assert result.evidence_contexts == ['c']


def test_default_retrieval_uses_product_reranking_and_actual_cutoff(monkeypatch):
    import sys
    import types
    from uuid import UUID

    from med_evidence_agentic_rag.evaluation.metrics import score_retrieval

    calls = []

    def retrieve_with_rerank(query, *, user_id):
        calls.append((query, user_id))
        return types.SimpleNamespace(evidences=[types.SimpleNamespace(
            source='s', content='actual synthetic passage', citation_id='E1')])

    module = types.ModuleType('med_evidence_agentic_rag.rag_retrieval.reranker')
    module.retrieve_with_rerank = retrieve_with_rerank
    module.reranker_top_k = 3
    monkeypatch.setitem(sys.modules, module.__name__, module)
    dense = types.ModuleType('med_evidence_agentic_rag.rag_retrieval.retrieval')

    def unexpected_dense(*args, **kwargs):
        raise AssertionError('default retrieval must use the production reranker')

    dense.retrieve = unexpected_dense
    monkeypatch.setitem(sys.modules, dense.__name__, dense)
    user_id = '01234567-0123-0123-0123-012345678901'
    adapter = RetrievalAdapter(allow_live=True, user_id=user_id, top_k=5)
    result = asyncio.run(adapter.execute(Case(id='a', query='q')))
    assert calls == [('q', UUID(user_id))]
    assert result.retrieval_depth == 3
    assert result.evidence_contexts == ['actual synthetic passage']
    assert adapter.metadata['retrieval_mode'] == 'production'
    assert score_retrieval(result.evidence_keys, result, [1, 3, 5])['hit@5'] is None


@pytest.mark.parametrize('reference', [None, 'synthetic reference'])
def test_remote_grounding_receives_actual_context(reference):
    context = 'Synthetic evidence: alpha is supported by this passage.'

    def respond(request):
        payload = json.loads(request.content)
        data = json.loads(payload['messages'][1]['content'])
        assert data['evidence_contexts'] == [context]
        assert data['reference'] == reference
        return httpx.Response(200, json={'choices': [{'message': {'content': json.dumps({
            'reference_exact_match': None, 'reference_token_f1': None,
            'answer_correctness': 0.9 if reference else None, 'faithfulness': 0.8})}}]})

    async def invoke():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            return await RemoteJudge('https://example.invalid/v1', 'synthetic', 'secret',
                                     allow_remote=True, client=client).score(
                Case(id='a', query='q', reference_answer=reference),
                Observation(id='a', answer='alpha', evidence_contexts=[context]))

    scores = asyncio.run(invoke())
    assert scores.faithfulness == 0.8
    assert scores.answer_correctness == (0.9 if reference else None)


def test_remote_grounding_rejects_score_without_actual_context():
    def respond(request):
        data = json.loads(json.loads(request.content)['messages'][1]['content'])
        assert data['evidence_contexts'] == []
        return httpx.Response(200, json={'choices': [{'message': {'content': json.dumps({
            'reference_exact_match': None, 'reference_token_f1': None,
            'answer_correctness': 0.9, 'faithfulness': 0.8})}}]})

    async def invoke():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            return await RemoteJudge('https://example.invalid/v1', 'synthetic', 'secret',
                                     allow_remote=True, client=client).score(
                Case(id='a', query='q', reference_answer='r'), Observation(id='a', answer='a'))

    with pytest.raises(ValueError, match='unavailable'):
        asyncio.run(invoke())


def test_http_cleanup_finishes_before_run_timeout_returns():
    from med_evidence_agentic_rag.evaluation.runner import run
    from med_evidence_agentic_rag.evaluation.schemas import RunConfig

    async def invoke():
        deleted = asyncio.Event()

        async def respond(request):
            if request.method == 'DELETE':
                await asyncio.sleep(0.05)
                deleted.set()
                return httpx.Response(204)
            if request.url.path == '/api/conversations':
                return httpx.Response(200, json={'id': '01234567-0123-0123-0123-012345678901'})
            return httpx.Response(200, text='{"type":"done"}\n')

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            report = await run([Case(id='a', query='q')], HTTPAgentAdapter(
                'https://example.invalid', 'secret', allow_live=True, client=client),
                RunConfig(timeout_seconds=0.01))
            assert report['cases'][0]['error'] == 'timeout'
            assert deleted.is_set()
            assert len(asyncio.all_tasks()) == 1

    asyncio.run(invoke())


def test_http_cleanup_deadline_cancels_and_joins_delete(monkeypatch):
    from med_evidence_agentic_rag.evaluation import adapters

    monkeypatch.setattr(adapters, '_CLEANUP_TIMEOUT_SECONDS', 0.01, raising=False)

    async def invoke():
        acknowledged = asyncio.Event()

        async def respond(request):
            if request.method == 'DELETE':
                try:
                    await asyncio.Event().wait()
                finally:
                    acknowledged.set()
            if request.url.path == '/api/conversations':
                return httpx.Response(200, json={'id': '01234567-0123-0123-0123-012345678901'})
            return httpx.Response(200, text='{"type":"done"}\n')

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            result = await HTTPAgentAdapter('https://example.invalid', 'secret',
                                             allow_live=True, client=client).execute(
                Case(id='a', query='q'))
            assert result.error == 'cleanup'
            assert acknowledged.is_set()
            assert len(asyncio.all_tasks()) == 1

    asyncio.run(invoke())
