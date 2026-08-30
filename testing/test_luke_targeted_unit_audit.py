import numpy as np
import pandas as pd

from testing.luke_targeted_unit_audit import classify, template_metrics


def test_template_metrics_recognizes_edge_peak():
    locations = np.c_[np.zeros(5), np.arange(5) * 20.0]
    template = np.zeros((7, 5))
    template[3, 0] = -4
    template[4, 0] = 1
    result = template_metrics(template, locations)
    assert result["template_peak_channel"] == 0
    assert result["template_peak_distance_to_strip_edge_um"] == 0
    assert result["template_opposite_to_dominant_ratio"] == 0.25


def test_classification_keeps_boundary_and_motion_distinct():
    boundary = pd.Series({
        "edge_spike_fraction": 0.9, "template_peak_distance_to_strip_edge_um": 20,
        "first_last_pc_cosine": 0.9, "median_amplitude": 50,
        "mean_rate_hz": 1, "depth_excursion_p95_p5_um": 20,
    })
    motion = pd.Series({
        "edge_spike_fraction": 0, "template_peak_distance_to_strip_edge_um": 200,
        "first_last_pc_cosine": 0.5, "median_amplitude": 60,
        "mean_rate_hz": 0.2, "depth_excursion_p95_p5_um": 50,
    })
    assert classify(boundary) == "strip_boundary_truncation_candidate"
    assert classify(motion) == "motion_or_family_fragmentation_candidate"
