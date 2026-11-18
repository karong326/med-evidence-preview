"""Strict external dataset and result contracts. No bundled datasets."""

import hashlib
import json
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

Text = Annotated[str, Field(min_length=1, pattern=r'\S')]
Score = Annotated[float, Field(ge=0, le=1)]
Nonnegative = Annotated[float, Field(ge=0)]
ErrorCode = Literal['timeout', 'network', 'http', 'protocol', 'incomplete',
                    'agent', 'cleanup', 'execution', 'judge']


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, allow_inf_nan=False)


class Case(StrictModel):
    id: Text
    query: Text
    reference_answer: str | None = None
    relevant_evidence_keys: list[Text] = Field(default_factory=list)
    answerable: bool | None = None
    expected_tools: list[Text] = Field(default_factory=list)
    tags: list[Text] = Field(default_factory=list)

    @model_validator(mode='after')
    def unique_lists(self):
        for values in (self.relevant_evidence_keys, self.expected_tools, self.tags):
            if len(values) != len(set(values)):
                raise ValueError('duplicate labels')
        return self


class Observation(StrictModel):
    id: Text
    answer: str | None = None
    evidence_keys: list[Text] = Field(default_factory=list)
    # Actual retrieved passage text for optional grounding judgment; never emitted by default.
    evidence_contexts: list[Text] = Field(default_factory=list)
    # Largest rank cutoff actually requested, even if an exhausted corpus returned fewer hits.
    retrieval_depth: int = Field(default=0, ge=0)
    citation_ids: list[Text] = Field(default_factory=list)
    cited_ids: list[Text] = Field(default_factory=list)
    tools: list[Text] = Field(default_factory=list)
    abstained: bool | None = None
    ttft_seconds: Nonnegative | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cost_usd: Nonnegative | None = None
    error: ErrorCode | None = None

    @model_validator(mode='after')
    def check_depth(self):
        if len(self.evidence_keys) > self.retrieval_depth:
            raise ValueError('evidence exceeds declared retrieval depth')
        if len(self.citation_ids) != len(set(self.citation_ids)):
            raise ValueError('duplicate citation IDs')
        return self


class JudgeScores(StrictModel):
    reference_exact_match: Score | None = None
    reference_token_f1: Score | None = None
    answer_correctness: Score | None = None
    faithfulness: Score | None = None
    _provider_usage: dict[str, int | float | None] = PrivateAttr(default_factory=lambda: {
        'input_tokens': None, 'output_tokens': None, 'cost_usd': None})

    @property
    def provider_usage(self) -> dict[str, int | float | None]:
        return self._provider_usage.copy()

    def set_provider_usage(self, usage: dict[str, object]) -> None:
        parsed = Observation.model_validate({
            'id': 'judge', 'input_tokens': usage.get('prompt_tokens'),
            'output_tokens': usage.get('completion_tokens'), 'cost_usd': usage.get('cost_usd')})
        self._provider_usage = {name: getattr(parsed, name)
                                for name in ('input_tokens', 'output_tokens', 'cost_usd')}


class RunConfig(StrictModel):
    ks: list[int] = Field(default_factory=lambda: [1, 3, 5])
    repeats: int = Field(default=1, ge=1)
    concurrency: int = Field(default=1, ge=1)
    timeout_seconds: float = Field(default=60.0, gt=0)
    include_private_text: bool = False
    corpus_version: str = 'unspecified'

    @model_validator(mode='after')
    def check_ks(self):
        if not self.ks or any(k <= 0 for k in self.ks) or len(set(self.ks)) != len(self.ks):
            raise ValueError('ks must be unique positive cutoffs')
        return self


class ReportRow(StrictModel):
    id: Text
    repeat: int = Field(ge=0)
    tags: list[Text]
    error: ErrorCode | None
    metrics: dict[str, Score | None]
    latency_seconds: Nonnegative
    ttft_seconds: Nonnegative | None
    input_tokens: int | None = Field(ge=0)
    output_tokens: int | None = Field(ge=0)
    cost_usd: Nonnegative | None
    judge_input_tokens: int | None = Field(ge=0)
    judge_output_tokens: int | None = Field(ge=0)
    judge_cost_usd: Nonnegative | None
    query: str | None = None
    answer: str | None = None
    reference_answer: str | None = None

    @model_validator(mode='before')
    @classmethod
    def numeric_scores(cls, values: dict[str, Any]) -> dict[str, Any]:
        metrics: dict[str, Any] = values.get('metrics', {})
        if any(v is not None and type(v) not in (int, float) for v in metrics.values()):
            raise ValueError('invalid numeric metric')
        return values


def load_jsonl[T: StrictModel](path: Path, model: type[T]) -> list[T]:
    values: list[T] = []
    seen: set[str] = set()
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        item = model.model_validate_json(line)
        identifier = getattr(item, 'id')
        if identifier in seen:
            raise ValueError('duplicate ID')
        seen.add(identifier)
        values.append(item)
    if not values:
        raise ValueError('empty JSONL input')
    return values


def fingerprint(value: object) -> str:
    canonical = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(canonical.encode()).hexdigest()
