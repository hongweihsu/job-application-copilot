import argparse
import json
from time import perf_counter

from app.llm.analyzer import analyze_with_llm
from app.llm.config import LLMConfig
from app.llm.provider import OpenAIRequirementDecisionProvider
from app.retrieval import (
    BM25Retriever,
    EmbeddingConfig,
    EmbeddingRetriever,
    OpenAIEmbeddingProvider,
)
from evaluation.metrics import CaseResult, citation_metrics, classification_metrics, safety_metrics
from evaluation.run_retrieval_evaluation_v2 import DATASET, _load_cases


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile)
    return ordered[index]


def evaluate_llm_pipeline(cases: list[dict], provider, retriever, top_k: int = 5) -> dict:
    results = []
    latencies = []
    for index, case in enumerate(cases, start=1):
        print(f"[{retriever.method_name} {index}/{len(cases)}] {case['id']}", flush=True)
        resume_text = "\n".join(item["text"] for item in case["resume_evidence"])
        started = perf_counter()
        analysis = analyze_with_llm(
            resume_text,
            case["requirement"],
            provider,
            retriever=retriever,
            top_k=top_k,
        )
        latencies.append(perf_counter() - started)
        if len(analysis.matches) != 1:
            raise ValueError(f"{case['id']} must produce exactly one requirement")
        match = analysis.matches[0]
        results.append(
            CaseResult(
                case_id=case["id"],
                split="dev",
                expected_status=case["expected_status"],
                predicted_status=match.status,
                expected_evidence_ids=case["expected_evidence_ids"],
                retrieved_evidence_ids=[item.evidence_id for item in match.evidence],
                forbidden_claims=case["forbidden_claims"],
                generated_text=match.recommendation,
            )
        )

    return {
        "metrics": {
            "classification": classification_metrics(results),
            "citations": citation_metrics(results),
            "safety": safety_metrics(results),
            "latency_seconds": {
                "mean": sum(latencies) / len(latencies) if latencies else 0.0,
                "p50": _percentile(latencies, 0.50),
                "p95": _percentile(latencies, 0.95),
            },
        },
        "llm_usage": {
            "requests": getattr(provider, "request_count", None),
            "input_tokens": getattr(provider, "input_tokens", None),
            "output_tokens": getattr(provider, "output_tokens", None),
            "provider_elapsed_seconds": getattr(provider, "elapsed_seconds", None),
        },
        "case_results": [
            {
                "case_id": result.case_id,
                "expected_status": result.expected_status,
                "predicted_status": result.predicted_status,
                "expected_evidence_ids": result.expected_evidence_ids,
                "cited_evidence_ids": result.retrieved_evidence_ids,
                "generated_text": result.generated_text,
            }
            for result in results
        ],
    }


def evaluate_comparison(
    llm_config: LLMConfig,
    embedding_config: EmbeddingConfig,
    top_k: int = 5,
) -> dict:
    if top_k < 5:
        raise ValueError("top_k must be at least 5")
    cases = _load_cases("dev")

    bm25_provider = OpenAIRequirementDecisionProvider(llm_config)
    embedding_llm_provider = OpenAIRequirementDecisionProvider(llm_config)
    embedding_provider = OpenAIEmbeddingProvider(embedding_config)
    embedding_retriever = EmbeddingRetriever(embedding_provider)

    report = {
        "dataset": DATASET.name,
        "evaluated_split": "dev",
        "cases": len(cases),
        "profiles": len({case["profile_id"] for case in cases}),
        "top_k": top_k,
        "model": llm_config.model,
        "pipelines": {
            "bm25_llm": evaluate_llm_pipeline(cases, bm25_provider, BM25Retriever(), top_k=top_k),
            "embedding_llm": evaluate_llm_pipeline(
                cases, embedding_llm_provider, embedding_retriever, top_k=top_k
            ),
        },
        "embedding_usage": {
            "model": embedding_config.model,
            "requests": embedding_provider.request_count,
            "input_tokens": embedding_provider.input_tokens,
            "cached_documents": embedding_retriever.cached_documents,
        },
    }
    bm25_metrics = report["pipelines"]["bm25_llm"]["metrics"]
    embedding_metrics = report["pipelines"]["embedding_llm"]["metrics"]
    quality_improved = (
        embedding_metrics["classification"]["accuracy"] > bm25_metrics["classification"]["accuracy"]
        and embedding_metrics["classification"]["macro_f1"]
        > bm25_metrics["classification"]["macro_f1"]
    )
    citation_precision_preserved = (
        embedding_metrics["citations"]["citation_precision"]
        >= bm25_metrics["citations"]["citation_precision"]
    )
    safety_preserved = (
        embedding_metrics["classification"]["false_supported_rate"]
        <= bm25_metrics["classification"]["false_supported_rate"]
        and embedding_metrics["safety"]["unsupported_claim_rate"]
        <= bm25_metrics["safety"]["unsupported_claim_rate"]
    )
    report["decision"] = {
        "classification_improved": quality_improved,
        "citation_precision_preserved": citation_precision_preserved,
        "safety_preserved": safety_preserved,
        "status": (
            "candidate_for_validation"
            if quality_improved and citation_precision_preserved and safety_preserved
            else "needs_error_analysis"
        ),
    }
    return report


def _percent(value: float) -> str:
    return f"{value:.1%}"


def markdown_report(report: dict) -> str:
    lines = [
        "# LLM Retrieval Pipeline Comparison",
        "",
        f"Dataset: `{report['dataset']}`",
        f"Evaluated split: `dev` ({report['profiles']} profiles, {report['cases']} cases)",
        f"LLM: `{report['model']}`; final Top-k: {report['top_k']}",
        "",
        "## End-to-end results",
        "",
        "| Pipeline | Accuracy | Macro F1 | Citation precision | Citation coverage | "
        "False supported | Unsupported claims | p50 latency | p95 latency |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("bm25_llm", "embedding_llm"):
        metrics = report["pipelines"][name]["metrics"]
        classification = metrics["classification"]
        citations = metrics["citations"]
        safety = metrics["safety"]
        latency = metrics["latency_seconds"]
        lines.append(
            f"| {name} | {_percent(classification['accuracy'])} | "
            f"{_percent(classification['macro_f1'])} | "
            f"{_percent(citations['citation_precision'])} | "
            f"{_percent(citations['citation_coverage'])} | "
            f"{_percent(classification['false_supported_rate'])} | "
            f"{_percent(safety['unsupported_claim_rate'])} | "
            f"{latency['p50']:.2f}s | {latency['p95']:.2f}s |"
        )

    lines.extend(["", "## Actual API usage", ""])
    for name in ("bm25_llm", "embedding_llm"):
        usage = report["pipelines"][name]["llm_usage"]
        lines.append(
            f"- {name}: {usage['requests']} LLM requests, {usage['input_tokens']} input tokens, "
            f"{usage['output_tokens']} output tokens"
        )
    embedding = report["embedding_usage"]
    decision = report["decision"]
    lines.extend(
        [
            f"- embedding: {embedding['requests']} requests, {embedding['input_tokens']} input "
            f"tokens, {embedding['cached_documents']} cached evidence units",
            "",
            "## Decision",
            "",
            f"Decision: **{decision['status'].replace('_', ' ')}**.",
            "",
            f"- Classification improved: {decision['classification_improved']}",
            f"- Citation precision preserved: {decision['citation_precision_preserved']}",
            f"- Safety preserved: {decision['safety_preserved']}",
            "",
            "Only dev was evaluated. Validation and test remain untouched until the end-to-end "
            "candidate and decision gate are reviewed.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare BM25 and embedding LLM pipelines on dev")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()

    report = evaluate_comparison(
        LLMConfig.from_environment(),
        EmbeddingConfig.from_environment(),
        top_k=args.top_k,
    )
    output = markdown_report(report)
    print(output)
    if args.write_report:
        report_json = DATASET.with_name("llm_retrieval_comparison_dev.json")
        report_markdown = DATASET.with_name("llm_retrieval_comparison_dev.md")
        report_json.write_text(json.dumps(report, indent=2) + "\n")
        report_markdown.write_text(output)
        print(f"Wrote {report_json} and {report_markdown}")


if __name__ == "__main__":
    main()
