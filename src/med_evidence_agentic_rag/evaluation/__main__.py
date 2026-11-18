"""Explicit-file CLI: validate, replay, run, compare, gate. No automatic live calls."""

import argparse
import asyncio
import json
import os
from pathlib import Path

from .adapters import HTTPAgentAdapter, ReplayAdapter, RetrievalAdapter
from .judges import ReferenceJudge, RemoteJudge
from .runner import compare, gate, run
from .schemas import Case, Observation, RunConfig, load_jsonl


def _write(path: Path, value: object) -> None:
    # Refuse overwriting source inputs or previous results by accident.
    with path.open('x', encoding='utf-8') as output:
        json.dump(value, output, ensure_ascii=False, allow_nan=False, indent=2)
        output.write('\n')


def _bounds(values: list[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for value in values:
        key, number = value.split('=', 1)
        if key in result or not key:
            raise ValueError('duplicate or empty threshold')
        result[key] = float(number)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    validate = commands.add_parser('validate')
    validate.add_argument('--dataset', type=Path, required=True)
    validate.add_argument('--results', type=Path)
    for name in ('replay', 'run'):
        command = commands.add_parser(name)
        command.add_argument('--dataset', type=Path, required=True)
        command.add_argument('--output', type=Path, required=True)
        command.add_argument('--k', type=int, nargs='+', default=[1, 3, 5])
        command.add_argument('--repeats', type=int, default=1)
        command.add_argument('--concurrency', type=int, default=1)
        command.add_argument('--timeout', type=float, default=60)
        command.add_argument('--corpus-version', default='unspecified')
        command.add_argument('--include-private-text', action='store_true')
        command.add_argument('--judge', choices=['none', 'reference', 'remote'], default='none')
        command.add_argument('--allow-remote-judge', action='store_true')
        command.add_argument('--judge-url')
        command.add_argument('--judge-model')
        if name == 'replay':
            command.add_argument('--results', type=Path, required=True)
        else:
            command.add_argument('--adapter', choices=['retrieval', 'http'], required=True)
            command.add_argument('--allow-live', action='store_true')
            command.add_argument('--user-id')
            command.add_argument('--retrieval-mode', choices=['production', 'dense'],
                                 default='production')
            command.add_argument('--base-url')
            command.add_argument('--keep-conversation', action='store_true')
    comparison = commands.add_parser('compare')
    comparison.add_argument('--baseline', type=Path, required=True)
    comparison.add_argument('--current', type=Path, required=True)
    comparison.add_argument('--output', type=Path, required=True)
    quality = commands.add_parser('gate')
    quality.add_argument('--report', type=Path, required=True)
    quality.add_argument('--minimum', action='append', default=[])
    quality.add_argument('--maximum', action='append', default=[])
    args = parser.parse_args(argv)
    try:
        if args.command == 'validate':
            cases = load_jsonl(args.dataset, Case)
            if args.results:
                results = load_jsonl(args.results, Observation)
                if {c.id for c in cases} != {r.id for r in results}:
                    raise ValueError('dataset/result ID mismatch')
            print(json.dumps({'valid': True, 'cases': len(cases)}))
            return 0
        if args.command == 'compare':
            _write(args.output, compare(json.loads(args.baseline.read_text(encoding='utf-8')),
                                        json.loads(args.current.read_text(encoding='utf-8'))))
            return 0
        if args.command == 'gate':
            if not args.minimum and not args.maximum:
                raise ValueError('at least one explicit threshold required')
            failures = gate(json.loads(args.report.read_text(encoding='utf-8')),
                            _bounds(args.minimum), _bounds(args.maximum))
            print(json.dumps({'passed': not failures, 'failed_metrics': failures}))
            return 1 if failures else 0
        cases = load_jsonl(args.dataset, Case)
        config = RunConfig(ks=args.k, repeats=args.repeats, concurrency=args.concurrency,
                           timeout_seconds=args.timeout, corpus_version=args.corpus_version,
                           include_private_text=args.include_private_text)
        if args.command == 'replay':
            adapter = ReplayAdapter(load_jsonl(args.results, Observation))
        elif args.adapter == 'retrieval':
            adapter = RetrievalAdapter(allow_live=args.allow_live, user_id=args.user_id,
                                       top_k=max(args.k), mode=args.retrieval_mode)
        else:
            if not args.base_url:
                raise ValueError('base-url required')
            adapter = HTTPAgentAdapter(args.base_url, os.environ.get('EVAL_AGENT_TOKEN', ''),
                                       allow_live=args.allow_live,
                                       cleanup=not args.keep_conversation)
        judge = None
        if args.judge == 'reference':
            judge = ReferenceJudge()
        elif args.judge == 'remote':
            if not args.judge_url or not args.judge_model:
                raise ValueError('judge-url and judge-model required')
            judge = RemoteJudge(args.judge_url, args.judge_model,
                                os.environ.get('EVAL_JUDGE_API_KEY', ''),
                                allow_remote=args.allow_remote_judge)
        if args.output.exists():
            raise ValueError('output already exists')
        report = asyncio.run(run(cases, adapter, config, judge))
        _write(args.output, report)
        print(json.dumps({'written': True, 'error_rate': report['aggregate']['error_rate']}))
        return 1 if report['aggregate']['error_rate'] else 0
    except Exception:
        # Dataset validation/transport exceptions may contain private text or credentials.
        print(json.dumps({'error': 'evaluation_failed',
                          'hint': 'check schemas, explicit flags, credentials and file paths'}))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
