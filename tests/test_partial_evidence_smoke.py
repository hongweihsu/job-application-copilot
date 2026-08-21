from evaluation.run_partial_evidence_smoke import TARGET_CASE_IDS, load_target_cases


def test_partial_evidence_smoke_uses_only_targeted_dev_cases():
    cases = load_target_cases()

    assert tuple(case["id"] for case in cases) == TARGET_CASE_IDS
    assert len(cases) == 5
    assert all(case["split"] == "dev" for case in cases)
    assert "v2-dev-mobile-react-native" in TARGET_CASE_IDS
