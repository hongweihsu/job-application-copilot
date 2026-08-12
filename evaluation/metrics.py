import re
from collections import defaultdict
from dataclasses import asdict, dataclass

STATUSES = ("supported", "partial", "missing")


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    split: str
    expected_status: str
    predicted_status: str
    expected_evidence_ids: list[str]
    retrieved_evidence_ids: list[str]
    forbidden_claims: list[str]
    generated_text: str


def _safe_divide(numerator: int | float, denominator: int | float) -> float:
    return numerator / denominator if denominator else 0.0


def classification_metrics(results: list[CaseResult]) -> dict:
    correct = sum(result.expected_status == result.predicted_status for result in results)
    confusion = {expected: {predicted: 0 for predicted in STATUSES} for expected in STATUSES}
    per_class = {}

    for result in results:
        confusion[result.expected_status][result.predicted_status] += 1

    for status in STATUSES:
        true_positive = confusion[status][status]
        predicted_positive = sum(confusion[expected][status] for expected in STATUSES)
        actual_positive = sum(confusion[status].values())
        precision = _safe_divide(true_positive, predicted_positive)
        recall = _safe_divide(true_positive, actual_positive)
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        per_class[status] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": actual_positive,
        }

    return {
        "accuracy": _safe_divide(correct, len(results)),
        "macro_f1": sum(metrics["f1"] for metrics in per_class.values()) / len(STATUSES),
        "false_supported_rate": _safe_divide(
            sum(
                result.predicted_status == "supported" and result.expected_status != "supported"
                for result in results
            ),
            sum(result.expected_status != "supported" for result in results),
        ),
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def retrieval_metrics(results: list[CaseResult], cutoffs: tuple[int, ...] = (1, 3, 5)) -> dict:
    answerable = [result for result in results if result.expected_evidence_ids]
    metrics: dict[str, float | int] = {"answerable_cases": len(answerable)}

    for cutoff in cutoffs:
        recalls = []
        precisions = []
        hit_rates = []
        for result in answerable:
            expected = set(result.expected_evidence_ids)
            retrieved = result.retrieved_evidence_ids[:cutoff]
            relevant = sum(evidence_id in expected for evidence_id in retrieved)
            recalls.append(_safe_divide(relevant, len(expected)))
            precisions.append(_safe_divide(relevant, len(retrieved)))
            hit_rates.append(float(relevant > 0))
        metrics[f"recall_at_{cutoff}"] = _safe_divide(sum(recalls), len(recalls))
        metrics[f"precision_at_{cutoff}"] = _safe_divide(sum(precisions), len(precisions))
        metrics[f"hit_rate_at_{cutoff}"] = _safe_divide(sum(hit_rates), len(hit_rates))

    reciprocal_ranks = []
    for result in answerable:
        expected = set(result.expected_evidence_ids)
        rank = next(
            (
                index
                for index, evidence_id in enumerate(result.retrieved_evidence_ids, start=1)
                if evidence_id in expected
            ),
            None,
        )
        reciprocal_ranks.append(1 / rank if rank else 0.0)
    metrics["mrr"] = _safe_divide(sum(reciprocal_ranks), len(reciprocal_ranks))
    return metrics


def citation_metrics(results: list[CaseResult]) -> dict:
    citations = [
        (evidence_id, set(result.expected_evidence_ids))
        for result in results
        for evidence_id in result.retrieved_evidence_ids
    ]
    correct_citations = sum(evidence_id in expected for evidence_id, expected in citations)
    answerable = [result for result in results if result.expected_evidence_ids]
    covered = sum(
        bool(set(result.retrieved_evidence_ids) & set(result.expected_evidence_ids))
        for result in answerable
    )
    return {
        "citation_precision": _safe_divide(correct_citations, len(citations)),
        "citation_coverage": _safe_divide(covered, len(answerable)),
        "total_citations": len(citations),
    }


def safety_metrics(results: list[CaseResult]) -> dict:
    non_assertion_cues = (
        "if ",
        "unless ",
        "no ",
        "not ",
        "none ",
        "lack ",
        "lacks ",
        "without ",
        "does not ",
        "do not ",
        "did not ",
    )
    violations = []
    for result in results:
        sentences = [
            sentence.strip().lower()
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", result.generated_text)
            if sentence.strip()
        ]
        matched_claims = [
            claim
            for claim in result.forbidden_claims
            if any(
                claim.lower() in sentence and not any(cue in sentence for cue in non_assertion_cues)
                for sentence in sentences
            )
        ]
        if matched_claims:
            violations.append({"case_id": result.case_id, "claims": matched_claims})
    return {
        "unsupported_claim_rate": _safe_divide(len(violations), len(results)),
        "violations": violations,
    }


def metrics_by_split(results: list[CaseResult]) -> dict:
    grouped: dict[str, list[CaseResult]] = defaultdict(list)
    grouped["all"] = results
    for result in results:
        grouped[result.split].append(result)

    return {
        split: {
            "cases": len(split_results),
            "classification": classification_metrics(split_results),
            "retrieval": retrieval_metrics(split_results),
            "citations": citation_metrics(split_results),
            "safety": safety_metrics(split_results),
            "case_results": [asdict(result) for result in split_results],
        }
        for split, split_results in grouped.items()
    }
