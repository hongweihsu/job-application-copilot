import argparse
import json
from pathlib import Path

from app.llm.config import LLMConfig
from app.llm.prompt import PROMPT_VERSION
from app.llm.provider import OpenAIRequirementDecisionProvider
from app.retrieval import EmbeddingConfig, EmbeddingRetriever, OpenAIEmbeddingProvider
from evaluation.run_llm_retrieval_comparison import evaluate_llm_pipeline
from evaluation.run_retrieval_evaluation_v2 import DATASET, _load_cases

BASELINE_REPORT = Path(__file__).with_name("llm_retrieval_comparison_dev.json")


def evaluate_evidence_relationship_candidate(
    llm_provider,
    embedding_provider,
    baseline_report: dict,
    top_k: int = 5,
) -> dict:
    if top_k < 5:
        raise ValueError("top_k must be at least 5")
    if baseline_report.get("evaluated_split") != "dev":
        raise ValueError("baseline report must contain the dev split")

    cases = _load_cases("dev")
    retriever = EmbeddingRetriever(embedding_provider)
    candidate = evaluate_llm_pipeline(cases, llm_provider, retriever, top_k=top_k)
    baseline = baseline_report["pipelines"]["embedding_llm"]
    baseline_metrics = baseline["metrics"]
    candidate_metrics = candidate["metrics"]

    accuracy_preserved = (
        candidate_metrics["classification"]["accuracy"]
        >= baseline_metrics["classification"]["accuracy"]
    )
    macro_f1_preserved = (
        candidate_metrics["classification"]["macro_f1"]
        >= baseline_metrics["classification"]["macro_f1"]
    )
    citation_precision_improved = (
        candidate_metrics["citations"]["citation_precision"]
        > baseline_metrics["citations"]["citation_precision"]
    )
    citation_coverage_preserved = (
        candidate_metrics["citations"]["citation_coverage"]
        >= baseline_metrics["citations"]["citation_coverage"]
    )
    safety_preserved = (
        candidate_metrics["classification"]["false_supported_rate"]
        <= baseline_metrics["classification"]["false_supported_rate"]
        and candidate_metrics["safety"]["unsupported_claim_rate"]
        <= baseline_metrics["safety"]["unsupported_claim_rate"]
    )
    passed = all(
        (
            accuracy_preserved,
            macro_f1_preserved,
            citation_precision_improved,
            citation_coverage_preserved,
            safety_preserved,
        )
    )

    return {
        "dataset": DATASET.name,
        "evaluated_split": "dev",
        "cases": len(cases),
        "profiles": len({case["profile_id"] for case in cases}),
        "top_k": top_k,
        "prompt_version": PROMPT_VERSION,
        "baseline": {
            "prompt_version": "requirement-match-v1",
            "metrics": baseline_metrics,
        },
        "candidate": candidate,
        "embedding_usage": {
            "model": embedding_provider.model_name,
            "requests": getattr(embedding_provider, "request_count", None),
            "input_tokens": getattr(embedding_provider, "input_tokens", None),
            "cached_documents": retriever.cached_documents,
        },
        "decision": {
            "accuracy_preserved": accuracy_preserved,
            "macro_f1_preserved": macro_f1_preserved,
            "citation_precision_improved": citation_precision_improved,
            "citation_coverage_preserved": citation_coverage_preserved,
            "safety_preserved": safety_preserved,
            "status": "candidate_for_validation" if passed else "needs_error_analysis",
        },
    }


def _percent(value: float) -> str:
    return f"{value:.1%}"


def markdown_report(report: dict) -> str:
    baseline = report["baseline"]["metrics"]
    candidate = report["candidate"]["metrics"]
    lines = [
        "# Evidence Relationship Prompt Experiment",
        "",
        f"Dataset: `{report['dataset']}`",
        f"Evaluated split: `dev` ({report['profiles']} profiles, {report['cases']} cases)",
        f"Candidate prompt: `{report['prompt_version']}`",
        "",
        "## Results",
        "",
        "| Version | Accuracy | Macro F1 | Citation precision | Citation coverage | "
        "False supported | Unsupported claims | p50 latency | p95 latency |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in (("v1", baseline), ("v2", candidate)):
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

    llm_usage = report["candidate"]["llm_usage"]
    embedding_usage = report["embedding_usage"]
    decision = report["decision"]
    lines.extend(
        [
            "",
            "## Candidate API usage",
            "",
            f"- LLM: {llm_usage['requests']} requests, {llm_usage['input_tokens']} input tokens, "
            f"{llm_usage['output_tokens']} output tokens",
            f"- Embedding: {embedding_usage['requests']} requests, "
            f"{embedding_usage['input_tokens']} input tokens, "
            f"{embedding_usage['cached_documents']} cached evidence units",
            "",
            "## Decision",
            "",
            f"Decision: **{decision['status'].replace('_', ' ')}**.",
            "",
            f"- Accuracy preserved: {decision['accuracy_preserved']}",
            f"- Macro F1 preserved: {decision['macro_f1_preserved']}",
            f"- Citation precision improved: {decision['citation_precision_improved']}",
            f"- Citation coverage preserved: {decision['citation_coverage_preserved']}",
            f"- Safety preserved: {decision['safety_preserved']}",
            "",
            "The v1 baseline is read from the previously committed report; it is not rerun. Only "
            "dev was evaluated. Validation and test remain untouched.",
            "",
            "The candidate improved citation precision and safety but became too conservative. "
            "Evidence that both supports one material part and disproves another was forced into "
            "a single mutually exclusive relationship group. Several valid partial cases were "
            "therefore downgraded to missing. The next schema revision needs explicit partial "
            "evidence rather than more prompt tuning.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate evidence relationship prompt v2 on dev")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()

    llm_provider = OpenAIRequirementDecisionProvider(LLMConfig.from_environment())
    embedding_provider = OpenAIEmbeddingProvider(EmbeddingConfig.from_environment())
    report = evaluate_evidence_relationship_candidate(
        llm_provider,
        embedding_provider,
        json.loads(BASELINE_REPORT.read_text()),
        top_k=args.top_k,
    )
    output = markdown_report(report)
    print(output)
    if args.write_report:
        report_json = DATASET.with_name("evidence_relationship_experiment_dev.json")
        report_markdown = DATASET.with_name("evidence_relationship_experiment_dev.md")
        report_json.write_text(json.dumps(report, indent=2) + "\n")
        report_markdown.write_text(output)
        print(f"Wrote {report_json} and {report_markdown}")


if __name__ == "__main__":
    main()
