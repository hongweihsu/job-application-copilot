import argparse
import json
from pathlib import Path

from app.llm.config import LLMConfig
from app.llm.provider import OpenAIRequirementDecisionProvider
from evaluation.metrics import CaseResult, metrics_by_split
from evaluation.run_evaluation import DATASET

REPORT = Path(__file__).with_name("llm_report.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paid LLM evaluation on one dataset split")
    parser.add_argument("--split", choices=("dev", "validation", "test"), default="validation")
    parser.add_argument("--write-report", action="store_true")
    parser.add_argument(
        "--reuse-report",
        action="store_true",
        help="Recalculate metrics from saved model outputs without making API requests",
    )
    args = parser.parse_args()

    cases = [case for case in json.loads(DATASET.read_text()) if case["split"] == args.split]
    results = []
    cached_report = json.loads(REPORT.read_text()) if args.reuse_report else None
    cached_results = (
        {item["case_id"]: item for item in cached_report["splits"][args.split]["case_results"]}
        if cached_report
        else {}
    )
    if args.reuse_report and set(cached_results) != {case["id"] for case in cases}:
        raise ValueError("Saved report cases do not match the requested dataset split")

    provider = None
    model_name = cached_report["model"] if cached_report else None
    if not args.reuse_report:
        config = LLMConfig.from_environment()
        provider = OpenAIRequirementDecisionProvider(config)
        model_name = provider.model_name

    for case in cases:
        if provider:
            evidence_units = [(item["id"], item["text"]) for item in case["resume_evidence"]]
            decision = provider.decide(case["requirement"], evidence_units)
            predicted_status = decision.status
            retrieved_evidence_ids = decision.evidence_ids
            generated_text = f"{decision.explanation}\n{decision.recommendation}"
        else:
            cached = cached_results[case["id"]]
            predicted_status = cached["predicted_status"]
            retrieved_evidence_ids = cached["retrieved_evidence_ids"]
            generated_text = cached["generated_text"]
        results.append(
            CaseResult(
                case_id=case["id"],
                split=case["split"],
                expected_status=case["expected_status"],
                predicted_status=predicted_status,
                expected_evidence_ids=case["expected_evidence_ids"],
                retrieved_evidence_ids=retrieved_evidence_ids,
                forbidden_claims=case["forbidden_claims"],
                generated_text=generated_text,
            )
        )
        print(f"{case['id']}: expected={case['expected_status']} predicted={predicted_status}")

    report = {
        "pipeline": "llm-structured-v1",
        "model": model_name,
        "dataset": DATASET.name,
        "evaluated_split": args.split,
        "splits": metrics_by_split(results),
    }
    split_metrics = report["splits"][args.split]
    print(f"Accuracy: {split_metrics['classification']['accuracy']:.1%}")
    print(f"Macro F1: {split_metrics['classification']['macro_f1']:.1%}")
    print(f"Evidence Recall@5: {split_metrics['retrieval']['recall_at_5']:.1%}")
    print(f"Citation precision: {split_metrics['citations']['citation_precision']:.1%}")

    if args.write_report:
        REPORT.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
