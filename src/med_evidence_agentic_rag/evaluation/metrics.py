"""Binary relevance metrics at explicitly observable rank cutoffs."""

import math

from .schemas import Case, Observation


def score_retrieval(relevant: list[str], result: Observation,
                    ks: list[int]) -> dict[str, float | None]:
    scores: dict[str, float | None] = {}
    gold = set(relevant)
    for k in ks:
        names = [f'{name}@{k}' for name in ('hit', 'recall', 'mrr', 'ndcg')]
        if not gold or k > result.retrieval_depth:
            scores.update(dict.fromkeys(names, None))
            continue
        seen: set[str] = set()
        ranks: list[int] = []
        for rank, key in enumerate(result.evidence_keys[:k], 1):
            if key in gold and key not in seen:
                ranks.append(rank)
            seen.add(key)
        ideal = sum(1 / math.log2(i + 1) for i in range(1, min(k, len(gold)) + 1))
        values = [float(bool(ranks)), len(ranks) / len(gold),
                  1 / ranks[0] if ranks else 0.0,
                  sum(1 / math.log2(r + 1) for r in ranks) / ideal]
        scores.update(zip(names, values, strict=True))
    return scores


def score_case(case: Case, result: Observation, ks: list[int]) -> dict[str, float | None]:
    scores = score_retrieval(case.relevant_evidence_keys, result, ks)
    scores['citation_validity'] = (
        sum(c in result.citation_ids for c in result.cited_ids) / len(result.cited_ids)
        if result.cited_ids else None
    )
    scores['expected_tool_recall'] = (
        len(set(case.expected_tools) & set(result.tools)) / len(case.expected_tools)
        if case.expected_tools and result.tools is not None else None
    )
    scores['forbidden_tool_compliance'] = (
        float(set(case.forbidden_tools).isdisjoint(result.tools))
        if case.forbidden_tools and result.tools is not None else None
    )
    scores['side_effect_accuracy'] = (
        sum(result.effect_counts[k] == v for k, v in case.expected_effects.items())
        / len(case.expected_effects)
        if case.expected_effects and all(result.effect_counts.get(k) is not None
                                         for k in case.expected_effects) else None
    )
    scores['answerability_accuracy'] = (
        float(result.abstained != case.answerable)
        if case.answerable is not None and result.abstained is not None else None
    )
    return scores
