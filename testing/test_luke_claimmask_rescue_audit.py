import pandas as pd

from testing.luke_claimmask_rescue_audit import (
    EXPECTED_SETTINGS,
    EXPECTED_WINDOWS,
    build_decision,
    summarize_short_window,
)


def _scores():
    rows = []
    for window, n_events in EXPECTED_WINDOWS.items():
        for index, setting in enumerate(sorted(EXPECTED_SETTINGS)):
            is_off = setting == "claim_off"
            recovery = 0.9 if is_off else 0.8 - 0.02 * index
            duplicate = 0.6 if is_off else 0.4 - 0.02 * index
            for population in (
                "visual_neural_unmatched",
                "automatic_neural_like_unmatched",
                "all_reviewed",
            ):
                rows.append(
                    {
                        "window": window,
                        "setting": setting,
                        "claim_ms": 0.0 if is_off else 0.1,
                        "claim_um": 0.0 if is_off else 25.0,
                        "population": population,
                        "n_events": n_events,
                        "observed_recovery": recovery,
                        "n_spikes": 1000 if is_off else 800,
                        "n_units": 10,
                        "n_ks_good": 5,
                        "median_contamination_pct": 30.0,
                        "cross_unit_near_coincident_fraction": duplicate,
                        "median_unit_refractory_violation_fraction": 0.01,
                    }
                )
    return pd.DataFrame(rows)


def test_summary_uses_within_window_claim_off_baseline():
    summary = summarize_short_window(_scores())
    baseline = summary[summary.setting == "claim_off"]
    assert len(summary) == 12
    assert (baseline.recovery_change_pp == 0).all()
    assert (baseline.spike_retention == 1).all()


def test_decision_rejects_masks_when_every_nonzero_setting_loses_recovery():
    short = summarize_short_window(_scores())
    full = pd.DataFrame(
        [
            {
                "variant": "patched",
                "population": "visual_neural_unmatched",
                "n_events": 62,
                "observed_recovery": 0.2,
                "recovery_above_null": 0.1,
            },
            {
                "variant": "dredge",
                "population": "visual_neural_unmatched",
                "n_events": 62,
                "observed_recovery": 0.8,
                "recovery_above_null": 0.7,
            },
        ]
    )
    decision = build_decision(short, full)
    assert decision["decision"] == "reject_current_claim_mask_for_luke_rescue"
    assert decision["causal_short_window_result"][
        "all_nonzero_masks_reduce_visual_neural_recovery_in_both_windows"
    ]
    assert decision["full_session_replication"][
        "patched_to_claim_off_recovery_ratio"
    ] == 0.25
