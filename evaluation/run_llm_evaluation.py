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
    args = parser.parse_args()

    config = LLMConfig.from_environment()
    provider = OpenAIRequirementDecisionProvider(config)
    cases = [case for case in json.loads(DATASET.read_text()) if case["split"] == args.split]
    results = []

    for case in cases:
        evidence_units = [(item["id"], item["text"]) for item in case["resume_evidence"]]
        decision = provider.decide(case["requirement"], evidence_units)
        generated_text = f"{decision.explanation}\n{decision.recommendation}"
        results.append(
            CaseResult(
                case_id=case["id"],
                split=case["split"],
                expected_status=case["expected_status"],
                predicted_status=decision.status,
                expected_evidence_ids=case["expected_evidence_ids"],
                retrieved_evidence_ids=decision.evidence_ids,
                forbidden_claims=case["forbidden_claims"],
                generated_text=generated_text,
            )
        )
        print(f"{case['id']}: expected={case['expected_status']} predicted={decision.status}")

    report = {
        "pipeline": "llm-structured-v1",
        "model": provider.model_name,
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
