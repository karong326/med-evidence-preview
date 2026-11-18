import json

from med_evidence_agentic_rag.evaluation.__main__ import main


def test_cli_replay_validate_compare_and_fail_closed_gate(tmp_path):
    data = tmp_path / 'input.jsonl'
    results = tmp_path / 'results.jsonl'
    output = tmp_path / 'report.json'
    comparison = tmp_path / 'comparison.json'
    data.write_text(json.dumps({'id': 's1', 'query': 'synthetic private text',
                                'relevant_evidence_keys': ['key']}), encoding='utf-8')
    results.write_text(json.dumps({'id': 's1', 'evidence_keys': ['key'],
                                   'retrieval_depth': 1}), encoding='utf-8')
    assert main(['validate', '--dataset', str(data), '--results', str(results)]) == 0
    assert main(['replay', '--dataset', str(data), '--results', str(results),
                 '--output', str(output), '--k', '1', '10']) == 0
    report = json.loads(output.read_text(encoding='utf-8'))
    assert report['aggregate']['hit@1'] == 1
    assert report['aggregate']['hit@10'] is None
    assert 'synthetic private text' not in output.read_text(encoding='utf-8')
    assert main(['compare', '--baseline', str(output), '--current', str(output),
                 '--output', str(comparison)]) == 0
    assert main(['gate', '--report', str(output), '--minimum', 'hit@1=1']) == 0
    assert main(['gate', '--report', str(output), '--minimum', 'hit@10=1']) == 1
    assert main(['gate', '--report', str(output)]) == 2
    assert main(['gate', '--report', str(output), '--minimum', 'hit@1=nan']) == 2
    # Existing outputs are never silently replaced.
    assert main(['replay', '--dataset', str(data), '--results', str(results),
                 '--output', str(output)]) == 2


def test_cli_never_runs_live_without_explicit_flag(tmp_path):
    data = tmp_path / 'input.jsonl'
    data.write_text('{"id":"s","query":"synthetic"}', encoding='utf-8')
    assert main(['run', '--dataset', str(data), '--output', str(tmp_path / 'out.json'),
                 '--adapter', 'http', '--base-url', 'https://example.invalid']) == 2
