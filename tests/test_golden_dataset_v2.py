import json
from collections import Counter, defaultdict
from pathlib import Path

DATASET = Path(__file__).parents[1] / "evaluation" / "golden_dataset_v2.json"


def load_dataset() -> dict:
    return json.loads(DATASET.read_text())


def test_v2_has_expected_profile_and_case_structure():
    dataset = load_dataset()

    assert dataset["version"] == "2.0"
    assert len(dataset["profiles"]) == 13
    assert len(dataset["cases"]) == 52
    assert Counter(profile["split"] for profile in dataset["profiles"]) == {
        "dev": 7,
        "validation": 3,
        "test": 3,
    }


def test_v2_profiles_have_realistic_stable_evidence_units():
    for profile in load_dataset()["profiles"]:
        evidence = profile["resume_evidence"]
        assert 10 <= len(evidence) <= 25
        assert [item["id"] for item in evidence] == [
            f"resume-s{index}" for index in range(1, len(evidence) + 1)
        ]
        assert all(len(item["text"]) >= 30 for item in evidence)


def test_v2_cases_reference_known_profiles_and_evidence():
    dataset = load_dataset()
    profiles = {profile["id"]: profile for profile in dataset["profiles"]}
    cases_by_profile = defaultdict(list)
    case_ids = []

    for case in dataset["cases"]:
        case_ids.append(case["id"])
        assert case["profile_id"] in profiles
        assert case["expected_status"] in {"supported", "partial", "missing"}
        assert case["tags"]
        valid_ids = {item["id"] for item in profiles[case["profile_id"]]["resume_evidence"]}
        assert set(case["expected_evidence_ids"]) <= valid_ids
        if case["expected_status"] == "missing":
            assert case["expected_evidence_ids"] == []
        else:
            assert case["expected_evidence_ids"]
        cases_by_profile[case["profile_id"]].append(case)

    assert all(len(cases) == 4 for cases in cases_by_profile.values())
    assert len(case_ids) == len(set(case_ids))


def test_v2_cases_keep_resume_profiles_isolated_by_split():
    dataset = load_dataset()
    profile_splits = {profile["id"]: profile["split"] for profile in dataset["profiles"]}

    assert Counter(profile_splits[case["profile_id"]] for case in dataset["cases"]) == {
        "dev": 28,
        "validation": 12,
        "test": 12,
    }


def test_v2_status_and_challenge_coverage():
    cases = load_dataset()["cases"]
    statuses = Counter(case["expected_status"] for case in cases)
    tags = Counter(tag for case in cases for tag in case["tags"])

    assert statuses == {"supported": 26, "partial": 13, "missing": 13}
    assert tags["hard-negative"] >= 10
    assert tags["semantic-paraphrase"] >= 10
    assert tags["multi-evidence"] >= 5
    assert tags["no-evidence"] >= 5
    assert tags["negation"] >= 5
