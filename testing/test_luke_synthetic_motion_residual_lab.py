import numpy as np
import pandas as pd

from testing.luke_synthetic_motion_residual_lab import (
    Kernel,
    interpolate_field_at,
    spatial_warp,
    summarize_candidates,
)


def test_field_interpolation_handles_time_depth_and_edges():
    displacement = np.array([[0.0, 10.0], [10.0, 20.0]])
    result = interpolate_field_at(
        displacement,
        np.array([0.0, 2.0]),
        np.array([100.0, 200.0]),
        1.0,
        np.array([50.0, 100.0, 150.0, 200.0, 250.0]),
    )
    np.testing.assert_allclose(result, [5.0, 5.0, 10.0, 15.0, 15.0])


def test_zero_spatial_warp_is_identity():
    locations = np.column_stack([np.zeros(4), np.arange(4) * 20.0])
    waveform = np.arange(20, dtype=np.float32).reshape(5, 4)
    result = spatial_warp(
        waveform,
        locations,
        np.zeros(4),
        Kernel("nearest", "nearest"),
    )
    np.testing.assert_array_equal(result, waveform)


def test_summary_gate_requires_generator_robust_gain_without_shape_tradeoff():
    rows = []
    for generator in ["g1", "g2"]:
        for case in range(2):
            common = {
                "snippet": f"s{case}",
                "template_id": "t1",
                "generator": generator,
                "peak_channel_error": 0,
            }
            rows.append(
                {
                    **common,
                    "candidate": "no_motion",
                    "residual_fraction": 0.10,
                    "template_cosine": 0.90,
                    "amplitude_retention": 0.90,
                }
            )
            rows.append(
                {
                    **common,
                    "candidate": "robust",
                    "residual_fraction": 0.09,
                    "template_cosine": 0.91,
                    "amplitude_retention": 0.905,
                }
            )
            rows.append(
                {
                    **common,
                    "candidate": "oversmoothed",
                    "residual_fraction": 0.08,
                    "template_cosine": 0.92,
                    "amplitude_retention": 0.70,
                }
            )
    summary = summarize_candidates(pd.DataFrame(rows)).set_index("candidate")
    assert bool(summary.loc["robust", "robust_screen_pass"])
    assert not bool(summary.loc["oversmoothed", "robust_screen_pass"])
    assert not bool(summary.loc["no_motion", "robust_screen_pass"])
