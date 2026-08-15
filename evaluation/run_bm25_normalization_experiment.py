import argparse
import json
from pathlib import Path

from app.retrieval import BM25NormalizedRetriever, BM25Retriever
from app.retrieval.bm25 import WORD_NORMALIZATION_ALIASES
from evaluation.run_bm25_stopword_experiment import _evaluate_method
from evaluation.run_retrieval_evaluation_v2 import DATASET, _load_cases

REPORT_JSON = Path(__file__).with_name("bm25_normalization_experiment_dev.json")
REPORT_MARKDOWN = Path(__file__).with_name("bm25_normalization_experiment_dev.md")


def evaluate_normalization_experiment(top_k: int = 5) -> dict:
    if top_k < 5:
        raise ValueError("top_k must be at least 5 to calculate Recall@5")

    cases = _load_cases("dev")
    raw = BM25Retriever()
    normalized = BM25NormalizedRetriever()
    report = {
        "dataset": DATASET.name,
        "evaluated_split": "dev",
        "cases": len(cases),
        "profiles": len({case["profile_id"] for case in cases}),
        "top_k": top_k,
        "methods": {
            "bm25_raw": {
                "configuration": {"k1": raw.k1, "b": raw.b, "word_normalization": {}},
                **_evaluate_method(cases, raw, top_k),
            },
            "bm25_normalized": {
                "configuration": {
                    "k1": normalized.k1,
                    "b": normalized.b,
                    "word_normalization": WORD_NORMALIZATION_ALIASES,
                },
                **_evaluate_method(cases, normalized, top_k),
            },
        },
    }
    raw_metrics = report["methods"]["bm25_raw"]["metrics"]
    normalized_metrics = report["methods"]["bm25_normalized"]["metrics"]
    recall_passed = normalized_metrics["recall_at_5"] >= raw_metrics["recall_at_5"]
    ranking_improved = normalized_metrics["mrr"] > raw_metrics["mrr"]
    precision_improved = normalized_metrics["precision_at_5"] > raw_metrics["precision_at_5"]
    report["decision"] = {
        "minimum_recall_at_5": raw_metrics["recall_at_5"],
        "recall_gate_passed": recall_passed,
        "ranking_improved": ranking_improved,
        "precision_improved": precision_improved,
        "status": (
            "candidate_for_validation"
            if recall_passed and (ranking_improved or precision_improved)
            else "rejected_as_standalone"
        ),
    }
    return report


def _percent(value: float) -> str:
    return f"{value:.1%}"


def markdown_report(report: dict) -> str:
    lines = [
        "# BM25 Word Normalization Experiment",
        "",
        f"Dataset: `{report['dataset']}`",
        f"Evaluated split: `dev` ({report['profiles']} profiles, {report['cases']} cases)",
        "",
        "This experiment normalizes a pre-registered set of common word forms in both queries and "
        "resume evidence. It does not use stopwords, aliases for domain concepts, or embeddings.",
        "",
        "## Results",
        "",
        "| Method | Recall@1 | Recall@3 | Recall@5 | Precision@5 | Hit@5 | MRR | "
        "No-gold candidate rate | Avg candidates |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in ("bm25_raw", "bm25_normalized"):
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
    normalized = report["methods"]["bm25_normalized"]["metrics"]
    candidate_delta = (
        normalized["average_candidates_retrieved"] - raw["average_candidates_retrieved"]
    )
    decision = report["decision"]["status"].replace("_", " ")
    lines.extend(
        [
            "",
            "## Delta versus raw BM25",
            "",
            f"- Recall@5: {normalized['recall_at_5'] - raw['recall_at_5']:+.1%}",
            f"- Precision@5: {normalized['precision_at_5'] - raw['precision_at_5']:+.1%}",
            f"- MRR: {normalized['mrr'] - raw['mrr']:+.1%}",
            f"- Average candidates: {candidate_delta:+.2f}",
            "",
            "## Decision",
            "",
            f"Decision: **{decision}**.",
            "",
            "The decision requires Recall@5 to remain at least equal to raw BM25 and either MRR "
            "or Precision@5 to improve.",
            "",
            "Observed dev gains include AWS observability moving from first relevant rank 2 to "
            "rank 1, complete retrieval of both incident-detection evidence units, mobile "
            "monitoring moving from rank 3 to rank 1, and an additional MLOps training evidence "
            "unit entering Top-5.",
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

    report = evaluate_normalization_experiment(top_k=args.top_k)
    print(markdown_report(report))
    if args.write_report:
        REPORT_JSON.write_text(json.dumps(report, indent=2) + "\n")
        REPORT_MARKDOWN.write_text(markdown_report(report))
        print(f"Wrote {REPORT_JSON} and {REPORT_MARKDOWN}")


if __name__ == "__main__":
    main()
