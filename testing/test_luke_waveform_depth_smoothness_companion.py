import numpy as np
import pandas as pd

from testing.luke_waveform_depth_smoothness_companion import (
    GATE,
    leave_one_depth_state_out,
    summarize_smoothness,
    waveform_depth_um,
)


def test_waveform_depth_is_energy_weighted():
    positions = np.column_stack((np.zeros(3), [0.0, 20.0, 40.0]))
    waveform = np.zeros((5, 3), dtype=float)
    waveform[2, 0] = 1.0
    waveform[2, 2] = 1.0
    assert waveform_depth_um(waveform, positions) == 20.0


def test_linear_depth_family_beats_nearest_state():
    base = np.zeros((5, 3), dtype=float)
    base[2, 0] = -1.0
    slope = np.zeros_like(base)
    slope[2, 1] = -0.5
    states = [
        ("left", 0.0, base),
        ("held", 10.0, base + slope),
        ("right", 20.0, base + 2 * slope),
    ]
    result = leave_one_depth_state_out(1, states)
    assert len(result) == 1
    assert result.iloc[0].predicted_residual_fraction < result.iloc[0].nearest_residual_fraction
    assert result.iloc[0].delta_cosine_vs_nearest > 0


def test_too_few_families_is_unvalidated_even_with_good_metrics():
    manifest = pd.DataFrame(
        {
            "family_id": [1, 2],
            "states": [3, 3],
            "depth_span_um": [40.0, 40.0],
            "polarity_stable": [True, True],
            "eligible": [True, True],
        }
    )
    metrics = pd.DataFrame(
        {
            "family_id": [1, 2],
            "held_template_id": ["a", "b"],
            "delta_residual_vs_nearest": [-0.1, -0.1],
            "delta_cosine_vs_nearest": [0.1, 0.1],
            "delta_absolute_amplitude_error_vs_nearest": [-0.1, -0.1],
        }
    )
    _, decision = summarize_smoothness(metrics, manifest)
    assert GATE.minimum_eligible_families == 3
    assert decision["status"] == "unvalidated"
    assert decision["smoothness_supported"] is False
