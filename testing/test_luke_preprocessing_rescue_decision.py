import json

import pandas as pd

from testing.luke_preprocessing_rescue_decision import run


def _window(path, duration, gain):
    path.mkdir()
    decision = {
        "decision": "advance_single_pass_to_broader_validation_with_strict_candidate_tracking",
        "duration_s": duration,
        "ks_good_effective_families": {
            "current": 100,
            "single_pass": 100 + gain,
            "difference": gain,
            "relative_change": gain / 100,
        },
        "single_pass_ks_good_related_to_current_fraction": 0.5,
        "ks_good_family_gain_by_agreement_threshold": {"0.1": gain, "0.2": gain - 1},
        "conservative_independent_candidate_count": 3,
        "moderate_independent_candidate_count": 10,
    }
    (path / "decision.json").write_text(json.dumps(decision))
    pd.DataFrame(
        [
            {
                "agreement_threshold": 0.2,
                "population": "KS-good",
                "condition": "current",
                "effective_cross_condition_families": 100,
            }
        ]
    ).to_csv(path / "family_threshold_sensitivity.csv", index=False)


def test_combined_decision_requires_replication_and_stays_preproduction(tmp_path):
    shared, pathological = tmp_path / "shared", tmp_path / "pathological"
    _window(shared, 240, 20)
    _window(pathological, 120, 10)
    conditioning = tmp_path / "conditioning.csv"
    pd.DataFrame(
        [
            {
                "condition": "Current no motion",
                "window": "120 s pathological",
                "recovered": "23/27",
            },
            {
                "condition": "Single KS preprocessing",
                "window": "120 s pathological",
                "recovered": "22/27",
            },
            {
                "condition": "Current conditioning",
                "window": "240 s shared",
                "recovered": "119/126",
            },
            {
                "condition": "Single KS preprocessing",
                "window": "240 s shared",
                "recovered": "120/126",
            },
        ]
    ).to_csv(conditioning, index=False)
    result = run(shared, pathological, conditioning, tmp_path / "out")
    assert result["decision"] == "advance_single_pass_to_full_session_validation_not_production"
    assert len(result["windows"]) == 2
    assert result["event_recovery_guardrail"]["shared"]["single_pass"] == "120/126"
