import argparse
import json
from collections import defaultdict
from pathlib import Path

from app.analyzer import analyze
from app.retrieval import BM25Retriever
from evaluation.metrics import CaseResult, retrieval_metrics
from evaluation.run_evaluation import _validate_case

EVALUATION_DIR = Path(__file__).parent
DATASET = EVALUATION_DIR / "golden_dataset.json"
REPORT_JSON = EVALUATION_DIR / "retrieval_report.json"
REPORT_MARKDOWN = EVALUATION_DIR / "retrieval_report.md"
SPLIT_ORDER = ("dev", "validation", "test", "all")


def _metric_result(case: dict, retrieved_evidence_ids: list[str]) -> CaseResult:
    return CaseResult(
        case_id=case["id"],
        split=case["split"],
        expected_status=case["expected_status"],
        predicted_status=case["expected_status"],
        expected_evidence_ids=case["expected_evidence_ids"],
        retrieved_evidence_ids=retrieved_evidence_ids,
        forbidden_claims=[],
        generated_text="",
    )


def _summarize(results: list[CaseResult]) -> dict:
    grouped: dict[str, list[CaseResult]] = defaultdict(list)
    grouped["all"] = results
    for result in results:
        grouped[result.split].append(result)

    return {
        split: {
            "cases": len(grouped[split]),
            "metrics": retrieval_metrics(grouped[split]),
            "case_results": [
                {
                    "case_id": result.case_id,
                    "expected_evidence_ids": result.expected_evidence_ids,
                    "retrieved_evidence_ids": result.retrieved_evidence_ids,
                }
                for result in grouped[split]
            ],
        }
        for split in SPLIT_ORDER
    }


def evaluate_retrieval(top_k: int = 5) -> dict:
    cases = json.loads(DATASET.read_text())
    bm25 = BM25Retriever()
    baseline_results = []
    bm25_results = []

    for case in cases:
        _validate_case(case)
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
        "dataset_cases": len(cases),
        "top_k": top_k,
        "methods": {
            "deterministic": {"splits": _summarize(baseline_results)},
            "bm25": {
                "configuration": {"k1": bm25.k1, "b": bm25.b},
                "splits": _summarize(bm25_results),
            },
        },
    }


def _percent(value: float) -> str:
    return f"{value:.1%}"


def markdown_report(report: dict) -> str:
    lines = [
        "# Retrieval Evaluation",
        "",
        f"Dataset: `{report['dataset']}` ({report['dataset_cases']} cases)",
        "",
        "Only cases with labelled evidence are included in retrieval metrics. Missing cases "
        "remain in the case-level output but do not have a relevant document to retrieve.",
        "",
        "## Summary",
        "",
        "| Method | Split | Answerable | Recall@1 | Recall@3 | Recall@5 | "
        "Precision@5 | Hit@5 | MRR |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in ("deterministic", "bm25"):
        for split in SPLIT_ORDER:
            metrics = report["methods"][method]["splits"][split]["metrics"]
            lines.append(
                f"| {method} | {split} | {metrics['answerable_cases']} | "
                f"{_percent(metrics['recall_at_1'])} | "
                f"{_percent(metrics['recall_at_3'])} | "
                f"{_percent(metrics['recall_at_5'])} | "
                f"{_percent(metrics['precision_at_5'])} | "
                f"{_percent(metrics['hit_rate_at_5'])} | {_percent(metrics['mrr'])} |"
            )

    baseline = report["methods"]["deterministic"]["splits"]["all"]["metrics"]
    bm25 = report["methods"]["bm25"]["splits"]["all"]["metrics"]
    recall_delta = bm25["recall_at_5"] - baseline["recall_at_5"]
    mrr_delta = bm25["mrr"] - baseline["mrr"]
    lines.extend(
        [
            "",
            "## Comparison",
            "",
            f"Across all answerable cases, BM25 changes Recall@5 by "
            f"{recall_delta:+.1%} and MRR by {mrr_delta:+.1%} versus the deterministic baseline.",
            "",
            "These metrics evaluate retrieval only. They do not measure requirement "
            "classification, recommendation quality, or LLM generation quality.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()

    if args.top_k < 5:
        raise ValueError("top_k must be at least 5 to calculate Recall@5")

    report = evaluate_retrieval(top_k=args.top_k)
    for method in ("deterministic", "bm25"):
        metrics = report["methods"][method]["splits"]["all"]["metrics"]
        print(
            f"{method}: Recall@5={_percent(metrics['recall_at_5'])}, "
            f"Precision@5={_percent(metrics['precision_at_5'])}, "
            f"MRR={_percent(metrics['mrr'])}"
        )

    if args.write_report:
        REPORT_JSON.write_text(json.dumps(report, indent=2) + "\n")
        REPORT_MARKDOWN.write_text(markdown_report(report))
        print(f"Wrote {REPORT_JSON} and {REPORT_MARKDOWN}")


if __name__ == "__main__":
    main()
