import numpy as np
import pandas as pd

from testing.luke_multichannel_event_validation import (
    GateThresholds,
    evaluate_gate,
    half_amplitude_width_ms,
    stratified_sample,
)


def test_stratified_sample_is_balanced_unique_and_reproducible():
    rows = []
    sample = 0
    for status in ("missed", "other"):
        for window in ("a", "b"):
            for unit in (1, 2):
                for _ in range(8):
                    rows.append(
                        {
                            "unit_id": unit,
                            "window": window,
                            "sample_index": sample,
                            "classification": status,
                        }
                    )
                    sample += 1
    candidates = pd.DataFrame(rows)
    # A candidate can be locally matched for one representative unit and
    # unmatched for another; it must still appear only once in the review.
    candidates.loc[len(candidates)] = {
        "unit_id": 9,
        "window": "a",
        "sample_index": 0,
        "classification": "missed",
    }
    first = stratified_sample(candidates, n_per_class=12, seed=7)
    second = stratified_sample(candidates, n_per_class=12, seed=7)
    assert first.equals(second)
    assert first.groupby("status").size().to_dict() == {"matched": 12, "unmatched": 12}
    assert not first.duplicated(["status", "sample_index"]).any()
    assert not first["sample_index"].duplicated().any()
    assert first["review_id"].is_unique


def test_half_amplitude_width_uses_contiguous_trough_samples():
    waveform = np.array([0.0, -1.0, -3.0, -4.0, -3.0, -1.0, 0.0])
    assert half_amplitude_width_ms(waveform, trough=3, fs=1000.0) == 3.0


def test_gate_reports_specific_failure_reasons():
    metrics = {
        "peak_snr": 8.0,
        "local_energy_fraction": 0.8,
        "common_mode_ratio": 0.1,
        "active_channels": 5,
        "trough_width_ms": 0.3,
        "peak_offset_ms": 0.1,
        "near_saturation": False,
    }
    passed, failures = evaluate_gate(metrics, GateThresholds())
    assert passed
    assert failures == ""
    metrics["common_mode_ratio"] = 0.9
    passed, failures = evaluate_gate(metrics, GateThresholds())
    assert not passed
    assert failures == "common_mode"
