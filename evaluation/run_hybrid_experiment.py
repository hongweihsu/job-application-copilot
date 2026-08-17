import argparse
import json

from app.retrieval import (
    BM25Retriever,
    EmbeddingConfig,
    EmbeddingRetriever,
    OpenAIEmbeddingProvider,
    ReciprocalRankFusionRetriever,
)
from evaluation.run_bm25_stopword_experiment import _evaluate_method
from evaluation.run_retrieval_evaluation_v2 import DATASET, _load_cases


def evaluate_hybrid_experiment(
    provider,
    top_k: int = 5,
    split: str = "dev",
    rrf_k: int = 60,
    retrieval_depth: int = 20,
) -> dict:
    if top_k < 5:
        raise ValueError("top_k must be at least 5 to calculate Recall@5")
    if split != "dev":
        raise ValueError("hybrid configuration must be selected on dev before validation")

    cases = _load_cases(split)
    bm25 = BM25Retriever()
    embedding = EmbeddingRetriever(provider)
    hybrid = ReciprocalRankFusionRetriever(
        [bm25, embedding], rrf_k=rrf_k, retrieval_depth=retrieval_depth
    )
    report = {
        "dataset": DATASET.name,
        "evaluated_split": split,
        "cases": len(cases),
        "profiles": len({case["profile_id"] for case in cases}),
        "top_k": top_k,
        "methods": {
            "bm25_raw": {
                "configuration": {"k1": bm25.k1, "b": bm25.b},
                **_evaluate_method(cases, bm25, top_k),
            },
            "embedding": {
                "configuration": {"model": provider.model_name, "similarity": "cosine"},
                **_evaluate_method(cases, embedding, top_k),
            },
            "hybrid_rrf": {
                "configuration": {
                    "children": [bm25.method_name, embedding.method_name],
                    "rrf_k": rrf_k,
                    "retrieval_depth": retrieval_depth,
                },
                **_evaluate_method(cases, hybrid, top_k),
            },
        },
        "embedding_usage": {
            "api_requests": getattr(provider, "request_count", None),
            "input_tokens": getattr(provider, "input_tokens", None),
            "cached_documents": embedding.cached_documents,
        },
    }
    baselines = [report["methods"][name]["metrics"] for name in ("bm25_raw", "embedding")]
    hybrid_metrics = report["methods"]["hybrid_rrf"]["metrics"]
    recall_floor = max(metrics["recall_at_5"] for metrics in baselines)
    best_mrr = max(metrics["mrr"] for metrics in baselines)
    best_precision = max(metrics["precision_at_5"] for metrics in baselines)
    recall_passed = hybrid_metrics["recall_at_5"] >= recall_floor
    quality_improved = (
        hybrid_metrics["mrr"] > best_mrr or hybrid_metrics["precision_at_5"] > best_precision
    )
    report["decision"] = {
        "minimum_recall_at_5": recall_floor,
        "recall_gate_passed": recall_passed,
        "ranking_or_precision_improved": quality_improved,
        "status": (
            "candidate_for_validation"
            if recall_passed and quality_improved
            else "not_better_than_best_component"
        ),
    }
    return report


def _percent(value: float) -> str:
    return f"{value:.1%}"


def markdown_report(report: dict) -> str:
    lines = [
        "# Hybrid Retrieval Experiment",
        "",
        f"Dataset: `{report['dataset']}`",
        f"Evaluated split: `dev` ({report['profiles']} profiles, {report['cases']} cases)",
        "",
        "BM25 and embedding rankings are fused with Reciprocal Rank Fusion (RRF). Raw BM25 "
        "scores and cosine similarities are never added directly.",
        "",
        "## Results",
        "",
        "| Method | Recall@1 | Recall@3 | Recall@5 | Precision@5 | MRR | Avg candidates |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in ("bm25_raw", "embedding", "hybrid_rrf"):
        metrics = report["methods"][method]["metrics"]
        lines.append(
            f"| {method} | {_percent(metrics['recall_at_1'])} | "
            f"{_percent(metrics['recall_at_3'])} | "
            f"{_percent(metrics['recall_at_5'])} | "
            f"{_percent(metrics['precision_at_5'])} | "
            f"{_percent(metrics['mrr'])} | "
            f"{metrics['average_candidates_retrieved']:.2f} |"
        )

    config = report["methods"]["hybrid_rrf"]["configuration"]
    usage = report["embedding_usage"]
    decision = report["decision"]["status"].replace("_", " ")
    lines.extend(
        [
            "",
            "## Configuration",
            "",
            f"- RRF k: {config['rrf_k']}",
            f"- Child retrieval depth: {config['retrieval_depth']}",
            f"- Final Top-k: {report['top_k']}",
            "",
            "## Embedding usage",
            "",
            f"- API requests: {usage['api_requests']}",
            f"- Input tokens: {usage['input_tokens']}",
            f"- Cached resume evidence units: {usage['cached_documents']}",
            "",
            "## Decision",
            "",
            f"Decision: **{decision}**.",
            "",
            "The hybrid must preserve the best component's Recall@5 and improve either MRR or "
            "Precision@5. Validation and test were not evaluated.",
            "",
            "Equal-weight RRF weakened the stronger embedding ranking. It dropped one MLOps "
            "evidence unit, completely lost the generative-AI gold evidence from Top-5, and "
            "pushed several otherwise rank-1 results lower. This configuration will not advance "
            "to validation.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paid hybrid retrieval evaluation on dev")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--retrieval-depth", type=int, default=20)
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()

    provider = OpenAIEmbeddingProvider(EmbeddingConfig.from_environment())
    report = evaluate_hybrid_experiment(
        provider,
        top_k=args.top_k,
        rrf_k=args.rrf_k,
        retrieval_depth=args.retrieval_depth,
    )
    output = markdown_report(report)
    print(output)
    if args.write_report:
        report_json = DATASET.with_name("hybrid_experiment_dev.json")
        report_markdown = DATASET.with_name("hybrid_experiment_dev.md")
        report_json.write_text(json.dumps(report, indent=2) + "\n")
        report_markdown.write_text(output)
        print(f"Wrote {report_json} and {report_markdown}")


if __name__ == "__main__":
    main()
