import argparse
import json
from pathlib import Path

from app.retrieval import BM25Retriever, BM25StopwordRetriever
from evaluation.run_retrieval_evaluation_v2 import (
    DATASET,
    _load_cases,
    _metric_result,
    _summarize,
)

REPORT_JSON = Path(__file__).with_name("bm25_stopword_experiment_dev.json")
REPORT_MARKDOWN = Path(__file__).with_name("bm25_stopword_experiment_dev.md")


def _evaluate_method(cases: list[dict], retriever, top_k: int) -> dict:
    results = []
    cases_by_id = {case["id"]: case for case in cases}
    for case in cases:
        evidence_units = [
            (evidence["id"], evidence["text"]) for evidence in case["resume_evidence"]
        ]
        retrieved = retriever.retrieve(case["requirement"], evidence_units, top_k=top_k)
        results.append(_metric_result(case, [item.evidence_id for item in retrieved]))
    return _summarize(results, cases_by_id)


def evaluate_stopword_experiment(top_k: int = 5) -> dict:
    if top_k < 5:
        raise ValueError("top_k must be at least 5 to calculate Recall@5")

    cases = _load_cases("dev")
    raw = BM25Retriever()
    stopword = BM25StopwordRetriever()
    report = {
        "dataset": DATASET.name,
        "evaluated_split": "dev",
        "cases": len(cases),
        "profiles": len({case["profile_id"] for case in cases}),
        "top_k": top_k,
        "methods": {
            "bm25_raw": {
                "configuration": {"k1": raw.k1, "b": raw.b, "query_stopwords": []},
                **_evaluate_method(cases, raw, top_k),
            },
            "bm25_stopwords": {
                "configuration": {
                    "k1": stopword.k1,
                    "b": stopword.b,
                    "query_stopwords": sorted(stopword.query_stopwords),
                },
                **_evaluate_method(cases, stopword, top_k),
            },
        },
    }
    raw_recall = report["methods"]["bm25_raw"]["metrics"]["recall_at_5"]
    stopword_metrics = report["methods"]["bm25_stopwords"]["metrics"]
    report["decision"] = {
        "status": "rejected_as_standalone",
        "minimum_recall_at_5": raw_recall,
        "recall_gate_passed": stopword_metrics["recall_at_5"] >= raw_recall,
        "precision_improved": (
            stopword_metrics["precision_at_5"]
            > report["methods"]["bm25_raw"]["metrics"]["precision_at_5"]
        ),
        "reason": "Precision improved, but Recall@5 fell below the raw BM25 baseline.",
    }
    return report


def _percent(value: float) -> str:
    return f"{value:.1%}"


def markdown_report(report: dict) -> str:
    lines = [
        "# BM25 Stopword Experiment",
        "",
        f"Dataset: `{report['dataset']}`",
        f"Evaluated split: `dev` ({report['profiles']} profiles, {report['cases']} cases)",
        "",
        "This experiment removes conservative boilerplate terms from the query only. Resume "
        "document tokens and BM25 parameters are unchanged.",
        "",
        "## Results",
        "",
        "| Method | Recall@1 | Recall@3 | Recall@5 | Precision@5 | Hit@5 | MRR | "
        "No-gold candidate rate | Avg candidates |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in ("bm25_raw", "bm25_stopwords"):
        metrics = report["methods"][method]["metrics"]
        lines.append(
            f"| {method} | {_percent(metrics['recall_at_1'])} | "
            f"{_percent(metrics['recall_at_3'])} | "
            f"{_percent(metrics['recall_at_5'])} | "
            f"{_percent(metrics['precision_at_5'])} | "
            f"{_percent(metrics['hit_rate_at_5'])} | "
            f"{_percent(metrics['mrr'])} | "
            f"{_percent(metrics['no_gold_candidate_rate'])} | "
            f"{metrics['average_candidates_retrieved']:.2f} |"
        )

    raw = report["methods"]["bm25_raw"]["metrics"]
    improved = report["methods"]["bm25_stopwords"]["metrics"]
    candidate_delta = improved["average_candidates_retrieved"] - raw["average_candidates_retrieved"]
    lines.extend(
        [
            "",
            "## Delta versus raw BM25",
            "",
            f"- Recall@5: {improved['recall_at_5'] - raw['recall_at_5']:+.1%}",
            f"- Precision@5: {improved['precision_at_5'] - raw['precision_at_5']:+.1%}",
            f"- MRR: {improved['mrr'] - raw['mrr']:+.1%}",
            f"- Average candidates: {candidate_delta:+.2f}",
            "",
            "## Decision",
            "",
            "**Rejected as a standalone replacement for raw BM25.** Precision improves and "
            "candidate volume falls, but Recall@5 drops below the pre-registered 84.9% floor.",
            "",
            "Four cases lose gold evidence: AWS observability, stakeholder communication, "
            "business partnering, and mobile performance. In several of them, raw BM25 found "
            "semantic gold evidence only through accidental matches on words such as `and` or "
            "`with`. Filtering exposes the semantic gap but does not solve it.",
            "",
            "The stopword variant remains useful as an experimental component and may be combined "
            "with word normalization or embedding retrieval later. It is not wired into the LLM "
            "pipeline.",
            "",
            "Scope-bearing terms including `production`, `ownership`, `formal`, and "
            "`administration` are intentionally retained.",
            "",
            "Validation and test were not evaluated.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()

    report = evaluate_stopword_experiment(top_k=args.top_k)
    print(markdown_report(report))
    if args.write_report:
        REPORT_JSON.write_text(json.dumps(report, indent=2) + "\n")
        REPORT_MARKDOWN.write_text(markdown_report(report))
        print(f"Wrote {REPORT_JSON} and {REPORT_MARKDOWN}")


if __name__ == "__main__":
    main()
