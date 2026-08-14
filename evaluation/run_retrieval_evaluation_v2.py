import argparse
import json
from pathlib import Path

from app.analyzer import analyze
from app.retrieval import BM25Retriever
from evaluation.metrics import CaseResult, retrieval_metrics

EVALUATION_DIR = Path(__file__).parent
DATASET = EVALUATION_DIR / "golden_dataset_v2.json"


def _safe_divide(numerator: int | float, denominator: int | float) -> float:
    return numerator / denominator if denominator else 0.0


def _load_cases(split: str) -> list[dict]:
    dataset = json.loads(DATASET.read_text())
    profiles = {profile["id"]: profile for profile in dataset["profiles"]}
    cases = []
    for case in dataset["cases"]:
        profile = profiles[case["profile_id"]]
        if profile["split"] == split:
            cases.append(
                {
                    **case,
                    "split": split,
                    "resume_evidence": profile["resume_evidence"],
                }
            )
    return cases


def _metric_result(case: dict, retrieved_ids: list[str]) -> CaseResult:
    return CaseResult(
        case_id=case["id"],
        split=case["split"],
        expected_status=case["expected_status"],
        predicted_status=case["expected_status"],
        expected_evidence_ids=case["expected_evidence_ids"],
        retrieved_evidence_ids=retrieved_ids,
        forbidden_claims=[],
        generated_text="",
    )


def _summarize(results: list[CaseResult], cases_by_id: dict[str, dict]) -> dict:
    metrics = retrieval_metrics(results)
    no_gold = [result for result in results if not result.expected_evidence_ids]
    retrieved_no_gold = sum(bool(result.retrieved_evidence_ids) for result in no_gold)
    metrics.update(
        {
            "no_gold_cases": len(no_gold),
            "no_gold_candidate_rate": _safe_divide(retrieved_no_gold, len(no_gold)),
            "average_candidates_retrieved": _safe_divide(
                sum(len(result.retrieved_evidence_ids) for result in results), len(results)
            ),
        }
    )
    return {
        "metrics": metrics,
        "case_results": [
            {
                "case_id": result.case_id,
                "profile_id": cases_by_id[result.case_id]["profile_id"],
                "tags": cases_by_id[result.case_id]["tags"],
                "expected_evidence_ids": result.expected_evidence_ids,
                "retrieved_evidence_ids": result.retrieved_evidence_ids,
            }
            for result in results
        ],
    }


def evaluate_retrieval_v2(split: str = "dev", top_k: int = 5) -> dict:
    if split not in {"dev", "validation", "test"}:
        raise ValueError("split must be dev, validation, or test")
    if top_k < 5:
        raise ValueError("top_k must be at least 5 to calculate Recall@5")

    cases = _load_cases(split)
    cases_by_id = {case["id"]: case for case in cases}
    bm25 = BM25Retriever()
    baseline_results = []
    bm25_results = []

    for case in cases:
        evidence_units = [
            (evidence["id"], evidence["text"]) for evidence in case["resume_evidence"]
        ]
        resume_text = "\n".join(text for _, text in evidence_units)

        baseline = analyze(resume_text, case["requirement"])
        if len(baseline.matches) != 1:
            raise ValueError(f"{case['id']} must produce exactly one requirement")
        baseline_ids = [evidence.evidence_id for evidence in baseline.matches[0].evidence]
        baseline_results.append(_metric_result(case, baseline_ids[:top_k]))

        retrieved = bm25.retrieve(case["requirement"], evidence_units, top_k=top_k)
        bm25_results.append(_metric_result(case, [evidence.evidence_id for evidence in retrieved]))

    return {
        "dataset": DATASET.name,
        "evaluated_split": split,
        "cases": len(cases),
        "profiles": len({case["profile_id"] for case in cases}),
        "top_k": top_k,
        "methods": {
            "deterministic": _summarize(baseline_results, cases_by_id),
            "bm25": {
                "configuration": {"k1": bm25.k1, "b": bm25.b},
                **_summarize(bm25_results, cases_by_id),
            },
        },
    }


def _percent(value: float) -> str:
    return f"{value:.1%}"


def markdown_report(report: dict) -> str:
    lines = [
        "# Retrieval Evaluation V2",
        "",
        f"Dataset: `{report['dataset']}`",
        f"Evaluated split: `{report['evaluated_split']}` "
        f"({report['profiles']} profiles, {report['cases']} cases)",
        "",
        "## Summary",
        "",
        "| Method | Answerable | Recall@1 | Recall@3 | Recall@5 | Precision@5 | "
        "Hit@5 | MRR | No-gold candidate rate | Avg candidates |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in ("deterministic", "bm25"):
        metrics = report["methods"][method]["metrics"]
        lines.append(
            f"| {method} | {metrics['answerable_cases']} | "
            f"{_percent(metrics['recall_at_1'])} | "
            f"{_percent(metrics['recall_at_3'])} | "
            f"{_percent(metrics['recall_at_5'])} | "
            f"{_percent(metrics['precision_at_5'])} | "
            f"{_percent(metrics['hit_rate_at_5'])} | "
            f"{_percent(metrics['mrr'])} | "
            f"{_percent(metrics['no_gold_candidate_rate'])} | "
            f"{metrics['average_candidates_retrieved']:.2f} |"
        )

    baseline = report["methods"]["deterministic"]["metrics"]
    bm25 = report["methods"]["bm25"]["metrics"]
    lines.extend(
        [
            "",
            "## Baseline comparison",
            "",
            f"BM25 changes Recall@5 by "
            f"{bm25['recall_at_5'] - baseline['recall_at_5']:+.1%}, "
            f"Precision@5 by {bm25['precision_at_5'] - baseline['precision_at_5']:+.1%}, "
            f"and MRR by {bm25['mrr'] - baseline['mrr']:+.1%}.",
            "",
            "No-gold candidate rate is the percentage of cases with no supporting gold evidence "
            "where the retriever still returns candidates. Such candidates may be irrelevant, "
            "adjacent, or contradictory; the metric does not by itself distinguish those types.",
            "",
            "Only the dev split is intended for retriever error analysis and implementation "
            "changes.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("dev", "validation", "test"), default="dev")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--write-report", action="store_true")
    parser.add_argument("--allow-test", action="store_true")
    args = parser.parse_args()

    if args.split == "test" and not args.allow_test:
        raise ValueError("Refusing to evaluate test without --allow-test")

    report = evaluate_retrieval_v2(split=args.split, top_k=args.top_k)
    print(markdown_report(report))
    if args.write_report:
        report_json = EVALUATION_DIR / f"retrieval_v2_{args.split}_report.json"
        report_markdown = EVALUATION_DIR / f"retrieval_v2_{args.split}_report.md"
        report_json.write_text(json.dumps(report, indent=2) + "\n")
        report_markdown.write_text(markdown_report(report))
        print(f"Wrote {report_json} and {report_markdown}")


if __name__ == "__main__":
    main()
