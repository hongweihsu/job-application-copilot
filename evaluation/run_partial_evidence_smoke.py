import argparse
import json

from app.llm.config import LLMConfig
from app.llm.prompt import PROMPT_VERSION
from app.llm.provider import OpenAIRequirementDecisionProvider
from app.retrieval import EmbeddingConfig, EmbeddingRetriever, OpenAIEmbeddingProvider
from evaluation.run_llm_retrieval_comparison import evaluate_llm_pipeline
from evaluation.run_retrieval_evaluation_v2 import DATASET, _load_cases

TARGET_CASE_IDS = (
    "v2-dev-security-terraform-ownership",
    "v2-dev-mobile-ios-ownership",
    "v2-dev-ml-kubernetes-operations",
    "v2-dev-delivery-cloud-migration",
    "v2-dev-mobile-react-native",
)


def load_target_cases() -> list[dict]:
    cases_by_id = {case["id"]: case for case in _load_cases("dev")}
    return [cases_by_id[case_id] for case_id in TARGET_CASE_IDS]


def evaluate_partial_evidence_smoke(llm_provider, embedding_provider, top_k: int = 5) -> dict:
    if top_k < 5:
        raise ValueError("top_k must be at least 5")
    cases = load_target_cases()
    candidate = evaluate_llm_pipeline(
        cases,
        llm_provider,
        EmbeddingRetriever(embedding_provider),
        top_k=top_k,
    )
    case_results = candidate["case_results"]
    correct = sum(item["predicted_status"] == item["expected_status"] for item in case_results)
    partial_relationship_correct = all(
        item["predicted_status"] == "partial" and bool(item["partial_evidence_ids"])
        for item in case_results
        if item["case_id"] != "v2-dev-mobile-react-native"
    )
    missing_control = next(
        item for item in case_results if item["case_id"] == "v2-dev-mobile-react-native"
    )
    missing_control_correct = (
        missing_control["predicted_status"] == "missing"
        and not missing_control["supporting_evidence_ids"]
        and not missing_control["partial_evidence_ids"]
    )
    return {
        "dataset": DATASET.name,
        "evaluated_split": "dev",
        "target_case_ids": list(TARGET_CASE_IDS),
        "prompt_version": PROMPT_VERSION,
        "top_k": top_k,
        "candidate": candidate,
        "embedding_usage": {
            "model": embedding_provider.model_name,
            "requests": getattr(embedding_provider, "request_count", None),
            "input_tokens": getattr(embedding_provider, "input_tokens", None),
        },
        "decision": {
            "correct_cases": correct,
            "total_cases": len(case_results),
            "partial_relationship_correct": partial_relationship_correct,
            "missing_control_correct": missing_control_correct,
            "status": (
                "eligible_for_full_dev"
                if correct == len(case_results)
                and partial_relationship_correct
                and missing_control_correct
                else "needs_error_analysis"
            ),
        },
    }


def markdown_report(report: dict) -> str:
    lines = [
        "# Partial Evidence Targeted Smoke Test",
        "",
        f"Dataset: `{report['dataset']}`",
        f"Evaluated split: `dev` ({len(report['target_case_ids'])} targeted cases)",
        f"Prompt: `{report['prompt_version']}`",
        "",
        "## Case results",
        "",
        "| Case | Expected | Predicted | Supporting | Partial | Related | Contradictory |",
        "|---|---|---|---|---|---|---|",
    ]
    for result in report["candidate"]["case_results"]:
        lines.append(
            f"| {result['case_id']} | {result['expected_status']} | "
            f"{result['predicted_status']} | "
            f"{', '.join(result['supporting_evidence_ids']) or '—'} | "
            f"{', '.join(result['partial_evidence_ids']) or '—'} | "
            f"{', '.join(result['related_evidence_ids']) or '—'} | "
            f"{', '.join(result['contradictory_evidence_ids']) or '—'} |"
        )
    decision = report["decision"]
    usage = report["candidate"]["llm_usage"]
    embedding = report["embedding_usage"]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"Decision: **{decision['status'].replace('_', ' ')}**.",
            "",
            f"- Correct cases: {decision['correct_cases']}/{decision['total_cases']}",
            f"- Partial relationships correct: {decision['partial_relationship_correct']}",
            f"- Missing safety control correct: {decision['missing_control_correct']}",
            "",
            "## API usage",
            "",
            f"- LLM: {usage['requests']} requests, {usage['input_tokens']} input tokens, "
            f"{usage['output_tokens']} output tokens",
            f"- Embedding: {embedding['requests']} requests, "
            f"{embedding['input_tokens']} input tokens",
            "",
            "This smoke test is only a precondition for a full dev evaluation. Validation and test "
            "were not evaluated.",
            "",
            "A partial relationship also needs an activity-strength boundary. Meaningful delivery "
            "participation with limited ownership may be partial, while merely evaluating, "
            "observing, or studying a technology without doing the requested work is not.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test explicit partial evidence on dev")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()

    llm_provider = OpenAIRequirementDecisionProvider(LLMConfig.from_environment())
    embedding_provider = OpenAIEmbeddingProvider(EmbeddingConfig.from_environment())
    report = evaluate_partial_evidence_smoke(
        llm_provider,
        embedding_provider,
        top_k=args.top_k,
    )
    output = markdown_report(report)
    print(output)
    if args.write_report:
        report_json = DATASET.with_name("partial_evidence_smoke_dev.json")
        report_markdown = DATASET.with_name("partial_evidence_smoke_dev.md")
        report_json.write_text(json.dumps(report, indent=2) + "\n")
        report_markdown.write_text(output)
        print(f"Wrote {report_json} and {report_markdown}")


if __name__ == "__main__":
    main()
