import asyncio
import copy
import json

import pytest
from pydantic import ValidationError

from med_evidence_agentic_rag.evaluation.adapters import ReplayAdapter
from med_evidence_agentic_rag.evaluation.metrics import score_retrieval
from med_evidence_agentic_rag.evaluation.runner import compare, gate, run
from med_evidence_agentic_rag.evaluation.schemas import Case, Observation, RunConfig, load_jsonl


def test_schema_rejects_unknown_nonfinite_and_duplicates(tmp_path):
    with pytest.raises(ValidationError):
        Case(id='a', query='x', surprise=True)
    with pytest.raises(ValidationError):
        Observation(id='a', cost_usd=float('nan'))
    path = tmp_path / 'data.jsonl'
    path.write_text('\n'.join(json.dumps({'id': 'a', 'query': 'x'}) for _ in range(2)))
    with pytest.raises(ValueError, match='duplicate'):
        load_jsonl(path, Case)


def test_metrics_respect_observed_depth_and_deduplicate():
    result = Observation(id='a', evidence_keys=['x', 'x', 'z'], retrieval_depth=3)
    scores = score_retrieval(['z'], result, [1, 3, 10])
    assert scores['hit@1'] == 0
    assert scores['recall@3'] == 1
    assert scores['mrr@3'] == pytest.approx(1 / 3)
    assert scores['ndcg@3'] == pytest.approx(0.5)
    assert scores['hit@10'] is None


def test_runner_privacy_repeats_comparison_and_gate():
    cases = [Case(id='a', query='private query', reference_answer='private answer',
                  relevant_evidence_keys=['x'], tags=['synthetic'])]
    adapter = ReplayAdapter([Observation(id='a', answer='private answer',
                                        evidence_keys=['x'], retrieval_depth=1)])
    report = asyncio.run(run(cases, adapter, RunConfig(ks=[1], repeats=2)))
    assert len(report['cases']) == 2
    assert report['aggregate']['hit@1'] == 1
    assert report['aggregate']['cost_usd'] is None
    assert 'private query' not in json.dumps(report)
    assert 'private answer' not in json.dumps(report)
    assert report['tags']['synthetic']['hit@1'] == 1
    assert compare(report, report)['delta']['hit@1'] == 0
    assert gate(report, {'hit@1': 1}) == []
    assert gate(report, {'missing': 0})
    changed = dict(report, dataset_fingerprint='0' * 64)
    with pytest.raises(ValueError, match='comparable'):
        compare(report, changed)


def test_runner_timeout_and_safe_errors():
    class Slow:
        metadata = {'adapter': 'slow'}

        async def execute(self, case):
            await asyncio.sleep(1)

    report = asyncio.run(run([Case(id='a', query='x')], Slow(),
                             RunConfig(timeout_seconds=0.01)))
    assert report['cases'][0]['error'] == 'timeout'
    assert report['aggregate']['error_rate'] == 1


def test_unscored_answer_dimensions_are_null_and_nonfinite_report_rejected():
    report = asyncio.run(run([Case(id='a', query='x')],
                             ReplayAdapter([Observation(id='a')]), RunConfig()))
    assert report['cases'][0]['metrics']['answer_correctness'] is None
    assert report['cases'][0]['metrics']['faithfulness'] is None
    report['cases'][0]['metrics']['hit@1'] = float('nan')
    with pytest.raises(ValueError):
        gate(report, {'error_rate': 0})


def test_invalid_duplicate_report_rows_and_boolean_scores_rejected():
    report = asyncio.run(run([Case(id='a', query='x')],
                             ReplayAdapter([Observation(id='a')]), RunConfig(repeats=2)))
    report['cases'][1]['repeat'] = 0
    with pytest.raises(ValueError):
        gate(report, {'error_rate': 0})
    report['cases'][1]['repeat'] = 1
    report['cases'][0]['metrics']['answer_correctness'] = True
    with pytest.raises(ValueError):
        gate(report, {'error_rate': 0})


def test_baselines_different_user_scope_are_incomparable():
    adapter = ReplayAdapter([Observation(id='a')])
    adapter.metadata['scope_fingerprint'] = 'scope_one'
    first = asyncio.run(run([Case(id='a', query='q')], adapter, RunConfig()))
    adapter.metadata['scope_fingerprint'] = 'scope_two'
    second = asyncio.run(run([Case(id='a', query='q')], adapter, RunConfig()))
    with pytest.raises(ValueError, match='comparable'):
        compare(first, second)


def test_gate_fails_incomplete_runs_unless_explicitly_allowed():
    cases = [Case(id='a', query='q', relevant_evidence_keys=['x']),
             Case(id='b', query='q', relevant_evidence_keys=['x'])]
    observations = [Observation(id='a', evidence_keys=['x'], retrieval_depth=1),
                    Observation(id='b', error='agent')]
    report = asyncio.run(run(cases, ReplayAdapter(observations), RunConfig(ks=[1])))
    assert 'error_rate' in gate(report, {'hit@1': 1})
    assert gate(report, {'hit@1': 1}, {'error_rate': 0.5}) == []
    report['aggregate']['hit@1'] = 2.0
    with pytest.raises(ValueError):
        gate(report, {'hit@1': 1})


def test_context_replay_is_not_written_to_default_report(tmp_path):
    path = tmp_path / 'observations.jsonl'
    path.write_text(json.dumps({'id': 'a', 'answer': 'synthetic answer',
                                'evidence_contexts': ['private synthetic context']}),
                    encoding='utf-8')
    adapter = ReplayAdapter(load_jsonl(path, Observation))
    assert adapter.results['a'].evidence_contexts == ['private synthetic context']
    report = asyncio.run(run([Case(id='a', query='q')], adapter, RunConfig()))
    assert 'private synthetic context' not in json.dumps(report)


@pytest.mark.parametrize('tamper', ['error_rate', 'score', 'count', 'tag', 'coverage', 'config'])
def test_gate_rejects_reports_whose_summaries_or_metadata_contradict_rows(tamper):
    cases = [Case(id='a', query='q', relevant_evidence_keys=['x'], tags=['synthetic']),
             Case(id='b', query='q', relevant_evidence_keys=['x'], tags=['synthetic'])]
    adapter = ReplayAdapter([Observation(id='a', evidence_keys=['wrong'], retrieval_depth=1),
                             Observation(id='b', error='agent')])
    genuine = asyncio.run(run(cases, adapter, RunConfig(ks=[1], repeats=2)))
    report = copy.deepcopy(genuine)
    if tamper == 'error_rate':
        report['aggregate']['error_rate'] = 0.0
    elif tamper == 'score':
        report['aggregate']['hit@1'] = 1.0
    elif tamper == 'count':
        report['aggregate']['hit@1_scored_count'] = 4.0
    elif tamper == 'tag':
        report['tags']['synthetic']['hit@1'] = 1.0
    elif tamper == 'coverage':
        report['cases'][2]['repeat'] = 2
    else:
        report['metadata']['config']['ks'] = [10]
    with pytest.raises(ValueError):
        gate(report, {'hit@1': 0})
    with pytest.raises(ValueError):
        compare(genuine, report)
