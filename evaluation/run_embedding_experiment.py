import argparse
import json

from app.retrieval import (
    BM25Retriever,
    EmbeddingConfig,
    EmbeddingRetriever,
    OpenAIEmbeddingProvider,
)
from evaluation.run_bm25_stopword_experiment import _evaluate_method
from evaluation.run_retrieval_evaluation_v2 import DATASET, _load_cases


def evaluate_embedding_experiment(provider, top_k: int = 5, split: str = "dev") -> dict:
    if top_k < 5:
        raise ValueError("top_k must be at least 5 to calculate Recall@5")
    if split not in {"dev", "validation"}:
        raise ValueError("embedding experiments are limited to dev and validation")

    cases = _load_cases(split)
    raw = BM25Retriever()
    embedding = EmbeddingRetriever(provider)
    report = {
        "dataset": DATASET.name,
        "evaluated_split": split,
        "cases": len(cases),
        "profiles": len({case["profile_id"] for case in cases}),
        "top_k": top_k,
        "methods": {
            "bm25_raw": {
                "configuration": {"k1": raw.k1, "b": raw.b},
                **_evaluate_method(cases, raw, top_k),
            },
            "embedding": {
                "configuration": {"model": provider.model_name, "similarity": "cosine"},
                **_evaluate_method(cases, embedding, top_k),
            },
        },
        "embedding_usage": {
            "api_requests": getattr(provider, "request_count", None),
            "input_tokens": getattr(provider, "input_tokens", None),
            "cached_documents": embedding.cached_documents,
        },
    }
    raw_metrics = report["methods"]["bm25_raw"]["metrics"]
    embedding_metrics = report["methods"]["embedding"]["metrics"]
    recall_passed = embedding_metrics["recall_at_5"] >= raw_metrics["recall_at_5"]
    ranking_improved = embedding_metrics["mrr"] > raw_metrics["mrr"]
    precision_improved = embedding_metrics["precision_at_5"] > raw_metrics["precision_at_5"]
    report["decision"] = {
        "minimum_recall_at_5": raw_metrics["recall_at_5"],
        "recall_gate_passed": recall_passed,
        "ranking_improved": ranking_improved,
        "precision_improved": precision_improved,
        "status": "pending",
    }
    passed = recall_passed and (ranking_improved or precision_improved)
    if split == "dev":
        report["decision"]["status"] = (
            "candidate_for_validation" if passed else "not_a_standalone_replacement"
        )
    else:
        report["decision"]["status"] = (
            "approved_for_promotion" if passed else "rejected_after_validation"
        )
    return report


def _percent(value: float) -> str:
    return f"{value:.1%}"


def markdown_report(report: dict) -> str:
    lines = [
        "# Embedding Retrieval Experiment",
        "",
        f"Dataset: `{report['dataset']}`",
        f"Evaluated split: `{report['evaluated_split']}` "
        f"({report['profiles']} profiles, {report['cases']} cases)",
        f"Embedding model: `{report['methods']['embedding']['configuration']['model']}`",
        "",
        "## Results",
        "",
        "| Method | Recall@1 | Recall@3 | Recall@5 | Precision@5 | Hit@5 | MRR | "
        "No-gold candidate rate | Avg candidates |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in ("bm25_raw", "embedding"):
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
    embedding = report["methods"]["embedding"]["metrics"]
    usage = report["embedding_usage"]
    decision = report["decision"]["status"].replace("_", " ")
    lines.extend(
        [
            "",
            "## Delta versus raw BM25",
            "",
            f"- Recall@5: {embedding['recall_at_5'] - raw['recall_at_5']:+.1%}",
            f"- Precision@5: {embedding['precision_at_5'] - raw['precision_at_5']:+.1%}",
            f"- MRR: {embedding['mrr'] - raw['mrr']:+.1%}",
            "",
            "## Embedding usage",
            "",
            f"- API requests: {usage['api_requests']}",
            f"- Input tokens: {usage['input_tokens']}",
            f"- Cached resume evidence units: {usage['cached_documents']}",
            "",
            "Document embeddings are cached by exact text for the duration of the run. Each "
            "resume is embedded once even though it is evaluated against four requirements.",
            "",
            "## Decision",
            "",
            f"Decision: **{decision}**.",
            "",
            "A standalone candidate must preserve raw BM25 Recall@5 and improve MRR or "
            "Precision@5. Even if it fails this gate, case-level complementary wins may still "
            "justify a later hybrid experiment.",
            "",
        ]
    )
    if report["evaluated_split"] == "dev":
        lines.extend(
            [
                "Dev improvements include semantic stakeholder communication, Azure scope, mobile "
                "performance, expanded SIEM evidence, and the RAG prototype case. Remaining "
                "partial misses include one MLOps training unit and one cross-functional "
                "leadership unit.",
                "",
                "Validation and test were not evaluated.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "The embedding model, cosine similarity, and Top-k were frozen after dev; no "
                "configuration was changed using validation results.",
                "",
                "Test was not evaluated.",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paid embedding evaluation")
    parser.add_argument("--split", choices=("dev", "validation"), default="dev")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()

    provider = OpenAIEmbeddingProvider(EmbeddingConfig.from_environment())
    report = evaluate_embedding_experiment(provider, top_k=args.top_k, split=args.split)
    print(markdown_report(report))
    if args.write_report:
        report_json = DATASET.with_name(f"embedding_experiment_{args.split}.json")
        report_markdown = DATASET.with_name(f"embedding_experiment_{args.split}.md")
        report_json.write_text(json.dumps(report, indent=2) + "\n")
        report_markdown.write_text(markdown_report(report))
        print(f"Wrote {report_json} and {report_markdown}")


if __name__ == "__main__":
    main()
