import pandas as pd

from testing import luke_conditioning_final_decision as decision


def test_decision_retains_recovery_supported_pipeline_and_defers_motion():
    factorial = pd.DataFrame(
        {
            "cell": ["blank_interpolate_include", "blank_exclude191"],
            "n_ks_good": [20, 10],
            "median_contamination_pct": [5.0, 8.0],
        }
    )
    harder = pd.DataFrame(
        [
            {"window": window, "condition": condition, "neural_unmatched_recovery": value}
            for window in ("neutral_template", "pathological")
            for condition, value in (
                ("legacy_blank_interpolate", 0.9),
                ("unchanged_interpolate", 0.5),
            )
        ]
    )
    artifact = pd.DataFrame(
        {
            "window": ["neutral_template", "pathological"],
            "artifact_threshold_counts": [300, 300],
            "rejected_batch_fraction": [0.2, 0.1],
            "reviewed_neural_erased_fraction": [0.2, 0.1],
        }
    )
    result = decision.build_decision(factorial, harder, artifact)
    assert result["decision"].startswith("retain_legacy_conditioning")
    assert result["sorter_settings"]["external_motion_correction"] is False
    assert result["sorter_settings"]["kilosort_artifact_threshold"] == "infinite/disabled"
    assert result["positive_polarity_excess"]["status"].startswith("deferred")
