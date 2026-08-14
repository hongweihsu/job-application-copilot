import argparse
import json
from pathlib import Path

import tiktoken

from app.llm.prompt import SYSTEM_PROMPT, build_user_prompt
from app.retrieval import BM25Retriever
from evaluation.run_evaluation import _validate_case
from evaluation.run_retrieval_evaluation import DATASET, evaluate_retrieval

REPORT_JSON = Path(__file__).with_name("retrieval_impact_report.json")
REPORT_MARKDOWN = Path(__file__).with_name("retrieval_impact_report.md")


def _encoding_for_model(model_name: str):
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        return tiktoken.get_encoding("o200k_base")


def _prompt_tokens(encoding, requirement: str, evidence_units: list[tuple[str, str]]) -> int:
    prompt_text = f"{SYSTEM_PROMPT}\n{build_user_prompt(requirement, evidence_units)}"
    return len(encoding.encode(prompt_text))


def evaluate_retrieval_impact(model_name: str = "gpt-5-mini", top_k: int = 5) -> dict:
    if top_k < 5:
        raise ValueError("top_k must be at least 5 to compare Recall@5")

    cases = json.loads(DATASET.read_text())
    encoding = _encoding_for_model(model_name)
    retriever = BM25Retriever()
    case_results = []

    for case in cases:
        _validate_case(case)
        evidence_units = [
            (evidence["id"], evidence["text"]) for evidence in case["resume_evidence"]
        ]
        retrieved = retriever.retrieve(case["requirement"], evidence_units, top_k=top_k)
        retrieved_units = [(item.evidence_id, item.text) for item in retrieved]
        case_results.append(
            {
                "case_id": case["id"],
                "split": case["split"],
                "full_evidence_units": len(evidence_units),
                "retrieved_evidence_units": len(retrieved_units),
                "full_prompt_tokens": _prompt_tokens(encoding, case["requirement"], evidence_units),
                "retrieved_prompt_tokens": _prompt_tokens(
                    encoding, case["requirement"], retrieved_units
                ),
            }
        )

    full_units = sum(case["full_evidence_units"] for case in case_results)
    retrieved_units = sum(case["retrieved_evidence_units"] for case in case_results)
    full_tokens = sum(case["full_prompt_tokens"] for case in case_results)
    retrieved_tokens = sum(case["retrieved_prompt_tokens"] for case in case_results)
    retrieval_report = evaluate_retrieval(top_k=top_k)
    baseline_recall = retrieval_report["methods"]["deterministic"]["splits"]["all"]["metrics"][
        "recall_at_5"
    ]
    bm25_recall = retrieval_report["methods"]["bm25"]["splits"]["all"]["metrics"]["recall_at_5"]

    return {
        "dataset": DATASET.name,
        "dataset_cases": len(cases),
        "model": model_name,
        "encoding": encoding.name,
        "top_k": top_k,
        "token_estimate_scope": (
            "System and user prompt text only; API message framing and output tokens excluded."
        ),
        "retrieval": {
            "deterministic_recall_at_5": baseline_recall,
            "bm25_recall_at_5": bm25_recall,
            "recall_at_5_delta": bm25_recall - baseline_recall,
        },
        "tokens": {
            "full_context": {
                "evidence_units": full_units,
                "prompt_tokens": full_tokens,
            },
            "bm25_top_k": {
                "evidence_units": retrieved_units,
                "prompt_tokens": retrieved_tokens,
            },
            "evidence_unit_reduction": 1 - retrieved_units / full_units,
            "prompt_token_reduction": 1 - retrieved_tokens / full_tokens,
            "prompt_tokens_saved": full_tokens - retrieved_tokens,
        },
        "case_results": case_results,
    }


def _percent(value: float) -> str:
    return f"{value:.1%}"


def markdown_report(report: dict) -> str:
    retrieval = report["retrieval"]
    tokens = report["tokens"]
    return "\n".join(
        [
            "# Retrieval Impact Report",
            "",
            f"Dataset: `{report['dataset']}` ({report['dataset_cases']} cases)",
            f"Tokenizer: `{report['encoding']}` for `{report['model']}`",
            "",
            "## Retrieval quality",
            "",
            "| Comparison | Recall@5 | Delta |",
            "|---|---:|---:|",
            f"| Deterministic baseline | {_percent(retrieval['deterministic_recall_at_5'])} | — |",
            f"| BM25 Top-{report['top_k']} | {_percent(retrieval['bm25_recall_at_5'])} | "
            f"{retrieval['recall_at_5_delta']:+.1%} |",
            "",
            "## Prompt size",
            "",
            "| Pipeline | Evidence units sent | Estimated prompt tokens |",
            "|---|---:|---:|",
            f"| Full context | {tokens['full_context']['evidence_units']} | "
            f"{tokens['full_context']['prompt_tokens']} |",
            f"| BM25 Top-{report['top_k']} | {tokens['bm25_top_k']['evidence_units']} | "
            f"{tokens['bm25_top_k']['prompt_tokens']} |",
            "",
            f"BM25 reduced evidence units by {_percent(tokens['evidence_unit_reduction'])} and "
            f"estimated prompt tokens by {_percent(tokens['prompt_token_reduction'])} "
            f"({tokens['prompt_tokens_saved']} tokens) while changing Recall@5 by "
            f"{retrieval['recall_at_5_delta']:+.1%}.",
            "",
            "Token counts include system and user prompt text. They exclude API message framing "
            "and model output tokens, so they are reproducible estimates rather than billing data.",
            "",
            "The golden cases contain short synthetic resumes, so token savings on longer real "
            "resumes may differ.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()

    report = evaluate_retrieval_impact(model_name=args.model, top_k=args.top_k)
    print(markdown_report(report))
    if args.write_report:
        REPORT_JSON.write_text(json.dumps(report, indent=2) + "\n")
        REPORT_MARKDOWN.write_text(markdown_report(report))
        print(f"Wrote {REPORT_JSON} and {REPORT_MARKDOWN}")


if __name__ == "__main__":
    main()
