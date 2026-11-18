"""Concurrent runs, privacy-preserving reports, comparable baselines and gates."""

import asyncio
import copy
import math
import platform
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from statistics import mean
from time import perf_counter
from typing import Any, cast

import httpx

from .adapters import Adapter, CleanupFailure, ReplayAdapter
from .judges import Judge
from .metrics import score_case
from .schemas import Case, JudgeScores, Observation, ReportRow, RunConfig, fingerprint


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _aggregate(rows: list[dict[str, Any]], elapsed: float) -> dict[str, float | None]:
    names = {name for row in rows for name in row['metrics']}
    summary: dict[str, float | None] = {}
    for name in sorted(names):
        scored = [row['metrics'][name] for row in rows
                  if row['error'] is None and row['metrics'].get(name) is not None]
        summary[name] = mean(scored) if scored else None
        summary[name + '_scored_count'] = float(len(scored))
    times = [row['latency_seconds'] for row in rows]
    ttfts = [row['ttft_seconds'] for row in rows if row['ttft_seconds'] is not None]
    summary.update({'count': float(len(rows)),
                    'error_rate': sum(row['error'] is not None for row in rows) / len(rows),
                    'p50_seconds': _quantile(times, 0.5),
                    'p95_seconds': _quantile(times, 0.95),
                    'ttft_p50_seconds': _quantile(ttfts, 0.5),
                    'ttft_p95_seconds': _quantile(ttfts, 0.95),
                    'throughput_per_second': len(rows) / elapsed if elapsed else None})
    for field in ('input_tokens', 'output_tokens', 'cost_usd',
                  'judge_input_tokens', 'judge_output_tokens', 'judge_cost_usd'):
        values = [row[field] for row in rows]
        # A partial total must not look like the complete run cost.
        summary[field] = float(sum(values)) if all(v is not None for v in values) else None
    return summary


def _config_fingerprint(config: RunConfig, adapter: dict[str, object],
                        judge: object) -> str:
    return fingerprint({'config': config.model_dump(),
                        'scope': {k: v for k, v in adapter.items()
                                  if k in ('scope_fingerprint', 'target_fingerprint',
                                           'retrieval_mode', 'configured_depth')},
                        'judge': judge})


async def run(cases: list[Case], adapter: Adapter, config: RunConfig,
              judge: Judge | None = None) -> dict[str, Any]:
    if not cases or len({c.id for c in cases}) != len(cases):
        raise ValueError('empty dataset or duplicate IDs')
    if isinstance(adapter, ReplayAdapter) and set(adapter.results) != {c.id for c in cases}:
        raise ValueError('replay IDs must exactly match dataset IDs')
    semaphore = asyncio.Semaphore(config.concurrency)

    async def one(case: Case, repeat: int) -> dict[str, Any]:
        async with semaphore:
            start = perf_counter()
            observation = Observation(id=case.id)
            scores: dict[str, float | None] = JudgeScores().model_dump()
            judge_usage = JudgeScores().provider_usage
            phase = 'execution'
            try:
                async with asyncio.timeout(config.timeout_seconds):
                    observation = await adapter.execute(case)
                    observation = Observation.model_validate(observation.model_dump())
                    if observation.id != case.id:
                        raise ValueError('result case mismatch')
                    scores.update(score_case(case, observation, config.ks))
                    if judge is not None and observation.error is None:
                        phase = 'judge'
                        judged = await judge.score(case, observation)
                        scores.update(judged.model_dump())
                        judge_usage = judged.provider_usage
            except (TimeoutError, httpx.TimeoutException):
                observation.error = 'timeout'
            except httpx.HTTPStatusError:
                observation.error = 'http'
            except httpx.RequestError:
                observation.error = 'network'
            except CleanupFailure:
                observation.error = 'cleanup'
            except Exception:
                observation.error = 'judge' if phase == 'judge' else 'execution'
            row = {'id': case.id, 'repeat': repeat, 'tags': case.tags,
                   'error': observation.error, 'metrics': scores,
                   'latency_seconds': perf_counter() - start,
                   'ttft_seconds': observation.ttft_seconds,
                   'input_tokens': observation.input_tokens,
                   'output_tokens': observation.output_tokens, 'cost_usd': observation.cost_usd}
            row.update({'judge_' + key: value for key, value in judge_usage.items()})
            if config.include_private_text:
                row.update({'query': case.query, 'answer': observation.answer,
                            'reference_answer': case.reference_answer})
            return row

    start = perf_counter()
    rows = await asyncio.gather(*(one(case, repeat) for repeat in range(config.repeats)
                                  for case in cases))
    elapsed = perf_counter() - start
    try:
        package_version = version('med-evidence-agentic-rag')
    except PackageNotFoundError:
        package_version = 'source-checkout'
    return {'schema_version': 1, 'created_at': datetime.now(UTC).isoformat(),
            'dataset_fingerprint': fingerprint([c.model_dump() for c in cases]),
            'config_fingerprint': _config_fingerprint(
                config, adapter.metadata, judge.metadata if judge else None),
            'metadata': {'python': platform.python_version(), 'package': package_version,
                         'adapter': copy.deepcopy(adapter.metadata),
                         'judge': copy.deepcopy(judge.metadata) if judge else None,
                         'config': config.model_dump(),
                         'dataset_case_ids': [case.id for case in cases],
                         'latency_scope': 'adapter_and_judge',
                         'percentile_method': 'linear_interpolation',
                         'elapsed_seconds': elapsed},
            'cases': rows, 'aggregate': _aggregate(rows, elapsed),
            'tags': {tag: _aggregate([row for row in rows if tag in row['tags']], elapsed)
                     for tag in sorted({tag for c in cases for tag in c.tags})}}


def validate_report(report: dict[str, Any]) -> None:
    """Verify row coverage, metadata and every aggregate before comparison or gating.

    This verifies internal consistency, not authenticity of an unsigned report.
    """
    def finite_tree(value: Any) -> None:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError('nonfinite report value')
        if isinstance(value, dict):
            for child in cast(dict[str, Any], value).values():
                finite_tree(child)
        if isinstance(value, list):
            for child in cast(list[Any], value):
                finite_tree(child)

    finite_tree(report)
    for key in ('dataset_fingerprint', 'config_fingerprint'):
        if not isinstance(report.get(key), str) or len(report[key]) != 64:
            raise ValueError('invalid fingerprint')
    if report.get('schema_version') != 1 or not isinstance(report.get('cases'), list):
        raise ValueError('invalid report schema')
    raw_summary = report.get('aggregate')
    if not isinstance(raw_summary, dict) or not raw_summary or not report['cases']:
        raise ValueError('empty report')
    summary = cast(dict[str, float | None], raw_summary)
    for value in summary.values():
        if value is not None and (type(value) not in (float, int) or not math.isfinite(value)):
            raise ValueError('invalid nonfinite summary')
    if summary.get('count') != len(report['cases']):
        raise ValueError('report count mismatch')
    metadata: dict[str, Any] = report.get('metadata', {})
    config = RunConfig.model_validate(metadata.get('config'))
    if metadata.get('percentile_method') != 'linear_interpolation':
        raise ValueError('unsupported percentile method')
    elapsed = metadata.get('elapsed_seconds')
    if not isinstance(elapsed, (float, int)) or isinstance(elapsed, bool):
        raise ValueError('invalid elapsed time')
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError('invalid elapsed time')
    adapter_metadata: dict[str, object] = metadata.get('adapter', {})
    if _config_fingerprint(config, adapter_metadata, metadata.get('judge')) != (
            report['config_fingerprint']):
        raise ValueError('configuration fingerprint mismatch')
    dataset_ids: list[str] = metadata.get('dataset_case_ids', [])
    if not dataset_ids or any(type(identifier) is not str for identifier in dataset_ids):
        raise ValueError('invalid dataset case IDs')
    if len(set(dataset_ids)) != len(dataset_ids):
        raise ValueError('duplicate dataset case IDs')
    seen: set[tuple[str, int]] = set()
    metric_names: set[str] = set()
    validated_rows: list[dict[str, Any]] = []
    case_tags: dict[str, list[str]] = {}
    for raw in report['cases']:
        row = ReportRow.model_validate(raw)
        validated_rows.append(row.model_dump())
        metric_names.update(row.metrics)
        key = (row.id, row.repeat)
        if key in seen:
            raise ValueError('duplicate report case/repeat')
        seen.add(key)
        if len(row.tags) != len(set(row.tags)) or (
                row.id in case_tags and case_tags[row.id] != row.tags):
            raise ValueError('inconsistent case tags')
        case_tags[row.id] = row.tags
    expected = {(identifier, repeat) for identifier in dataset_ids
                for repeat in range(config.repeats)}
    if seen != expected:
        raise ValueError('incomplete or invalid repeat coverage')
    for name, value in summary.items():
        if value is not None and (value < 0 or (
                (name in metric_names or name == 'error_rate') and value > 1)):
            raise ValueError('out of range aggregate value')

    def matching_summary(actual: dict[str, Any], expected: dict[str, float | None]) -> None:
        if actual.keys() != expected.keys():
            raise ValueError('aggregate metric keys mismatch')
        for name, value in expected.items():
            given = actual[name]
            if value is None:
                if given is not None:
                    raise ValueError('aggregate nullability mismatch')
            elif type(given) not in (float, int) or not math.isclose(
                    given, value, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError('aggregate contradicts report rows')

    matching_summary(summary, _aggregate(validated_rows, elapsed))
    tag_summaries: dict[str, dict[str, Any]] = report.get('tags', {})
    expected_tags = {tag for row in validated_rows for tag in row['tags']}
    if tag_summaries.keys() != expected_tags:
        raise ValueError('tag summary keys mismatch')
    for tag in expected_tags:
        matching_summary(tag_summaries[tag], _aggregate(
            [row for row in validated_rows if tag in row['tags']], elapsed))


def compare(baseline: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    validate_report(baseline)
    validate_report(current)
    if any(baseline[key] != current[key]
           for key in ('dataset_fingerprint', 'config_fingerprint')):
        raise ValueError('reports are not comparable')
    keys = baseline['aggregate'].keys() | current['aggregate'].keys()
    return {'delta': {key: current['aggregate'][key] - baseline['aggregate'][key]
                      if baseline['aggregate'].get(key) is not None
                      and current['aggregate'].get(key) is not None else None
                      for key in sorted(keys)}}


def gate(report: dict[str, Any], minimum: dict[str, float],
         maximum: dict[str, float] | None = None) -> list[str]:
    validate_report(report)
    failures: list[str] = []
    upper_bounds = {'error_rate': 0.0, **(maximum or {})}
    for bounds, is_min in ((minimum, True), (upper_bounds, False)):
        for name, threshold in bounds.items():
            if not math.isfinite(threshold):
                raise ValueError('nonfinite threshold')
            value = report['aggregate'].get(name)
            if value is None or (value < threshold if is_min else value > threshold):
                failures.append(name)
    return sorted(set(failures))
