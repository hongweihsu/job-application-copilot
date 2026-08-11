import argparse
import json
from pathlib import Path

from app.analyzer import analyze
from evaluation.metrics import STATUSES, CaseResult, metrics_by_split

EVALUATION_DIR = Path(__file__).parent
DATASET = EVALUATION_DIR / "golden_dataset.json"
THRESHOLDS = EVALUATION_DIR / "baseline_thresholds.json"
REPORT_JSON = EVALUATION_DIR / "baseline_report.json"
REPORT_MARKDOWN = EVALUATION_DIR / "baseline_report.md"


def _validate_case(case: dict) -> None:
    required = {
        "id",
        "split",
        "resume_evidence",
        "requirement",
        "expected_status",
        "expected_evidence_ids",
        "forbidden_claims",
    }
    missing = required - case.keys()
    if missing:
        raise ValueError(f"{case.get('id', '<unknown>')} is missing fields: {sorted(missing)}")
    if case["split"] not in {"dev", "validation", "test"}:
        raise ValueError(f"{case['id']} has an invalid split")
    if case["expected_status"] not in STATUSES:
        raise ValueError(f"{case['id']} has an invalid expected status")

    evidence_ids = [evidence["id"] for evidence in case["resume_evidence"]]
    expected_ids = [f"resume-s{index}" for index in range(1, len(evidence_ids) + 1)]
    if evidence_ids != expected_ids:
        raise ValueError(f"{case['id']} evidence IDs must be sequential: {expected_ids}")
    unknown_gold_ids = set(case["expected_evidence_ids"]) - set(evidence_ids)
    if unknown_gold_ids:
        raise ValueError(f"{case['id']} references unknown evidence IDs: {unknown_gold_ids}")
    if case["expected_status"] == "missing" and case["expected_evidence_ids"]:
        raise ValueError(f"{case['id']} cannot be missing and have gold evidence")


def evaluate_cases() -> dict:
    cases = json.loads(DATASET.read_text())
    results = []
    for case in cases:
        _validate_case(case)
        resume_text = "\n".join(evidence["text"] for evidence in case["resume_evidence"])
        analysis = analyze(resume_text, case["requirement"])
        if len(analysis.matches) != 1:
            raise ValueError(f"{case['id']} must produce exactly one requirement")
        match = analysis.matches[0]
        results.append(
            CaseResult(
                case_id=case["id"],
                split=case["split"],
                expected_status=case["expected_status"],
                predicted_status=match.status,
                expected_evidence_ids=case["expected_evidence_ids"],
                retrieved_evidence_ids=[evidence.evidence_id for evidence in match.evidence],
                forbidden_claims=case["forbidden_claims"],
                generated_text=match.recommendation,
            )
        )
    return {
        "dataset": DATASET.name,
        "dataset_cases": len(cases),
        "splits": metrics_by_split(results),
    }


def _percent(value: float) -> str:
    return f"{value:.1%}"


def markdown_report(report: dict) -> str:
    lines = [
        "# Deterministic Baseline Evaluation",
        "",
        f"Dataset: `{report['dataset']}` ({report['dataset_cases']} cases)",
        "",
        "## Summary",
        "",
        "| Split | Cases | Accuracy | Macro F1 | Recall@5 | Citation precision | False supported |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("dev", "validation", "test", "all"):
        metrics = report["splits"][split]
        classification = metrics["classification"]
        retrieval = metrics["retrieval"]
        citations = metrics["citations"]
        lines.append(
            f"| {split} | {metrics['cases']} | {_percent(classification['accuracy'])} | "
            f"{_percent(classification['macro_f1'])} | "
            f"{_percent(retrieval['recall_at_5'])} | "
            f"{_percent(citations['citation_precision'])} | "
            f"{_percent(classification['false_supported_rate'])} |"
        )

    test = report["splits"]["test"]
    lines.extend(
        [
            "",
            "## Held-out test classification",
            "",
            "| Class | Precision | Recall | F1 | Support |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for status in STATUSES:
        metrics = test["classification"]["per_class"][status]
        lines.append(
            f"| {status} | {_percent(metrics['precision'])} | {_percent(metrics['recall'])} | "
            f"{_percent(metrics['f1'])} | {metrics['support']} |"
        )

    lines.extend(["", "## Held-out test confusion matrix", ""])
    lines.append("| Expected \\ Predicted | supported | partial | missing |")
    lines.append("|---|---:|---:|---:|")
    confusion = test["classification"]["confusion_matrix"]
    for expected in STATUSES:
        lines.append(
            f"| {expected} | {confusion[expected]['supported']} | "
            f"{confusion[expected]['partial']} | {confusion[expected]['missing']} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "These numbers describe a transparent keyword-overlap baseline, not production model "
            "quality. The held-out test split is reserved for comparisons with BM25, vector, and "
            "hybrid retrieval. Dataset examples are synthetic and manually labelled.",
            "",
        ]
    )
    return "\n".join(lines)


def check_thresholds(report: dict) -> list[str]:
    thresholds = json.loads(THRESHOLDS.read_text())
    failures = []
    for check in thresholds["checks"]:
        value = report["splits"][check["split"]]
        for key in check["path"]:
            value = value[key]
        if "minimum" in check and value < check["minimum"]:
            failures.append(f"{check['name']}: {value:.4f} is below minimum {check['minimum']:.4f}")
        elif "maximum" in check and value > check["maximum"]:
            failures.append(f"{check['name']}: {value:.4f} is above maximum {check['maximum']:.4f}")
        else:
            boundary = (
                f">= {check['minimum']:.4f}" if "minimum" in check else f"<= {check['maximum']:.4f}"
            )
            print(f"PASS {check['name']}: {value:.4f} {boundary}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-report", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    report = evaluate_cases()
    test = report["splits"]["test"]
    print(f"Dataset cases: {report['dataset_cases']}")
    print(f"Test accuracy: {_percent(test['classification']['accuracy'])}")
    print(f"Test macro F1: {_percent(test['classification']['macro_f1'])}")
    print(f"Test Evidence Recall@5: {_percent(test['retrieval']['recall_at_5'])}")
    print(f"Test citation precision: {_percent(test['citations']['citation_precision'])}")

    if args.write_report:
        REPORT_JSON.write_text(json.dumps(report, indent=2) + "\n")
        REPORT_MARKDOWN.write_text(markdown_report(report))
        print(f"Wrote {REPORT_JSON} and {REPORT_MARKDOWN}")

    if args.check:
        failures = check_thresholds(report)
        if failures:
            for failure in failures:
                print(f"FAIL {failure}")
            raise SystemExit(1)


if __name__ == "__main__":
    main()
