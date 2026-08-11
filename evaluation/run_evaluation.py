import json
from pathlib import Path

from app.analyzer import analyze

DATASET = Path(__file__).with_name("golden_dataset.json")


def main() -> None:
    cases = json.loads(DATASET.read_text())
    passed = 0
    for case in cases:
        result = analyze(case["resume"], case["job_description"])
        match = result.matches[0]
        valid = match.status == case["expected_status"]
        passed += valid
        print(f"{'PASS' if valid else 'FAIL'} {case['id']}: {match.status}")
    accuracy = passed / len(cases)
    print(f"\nBaseline status accuracy: {accuracy:.0%} ({passed}/{len(cases)})")
    raise SystemExit(0 if accuracy >= 0.9 else 1)


if __name__ == "__main__":
    main()
