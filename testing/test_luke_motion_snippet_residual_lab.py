import numpy as np
import pandas as pd

from testing.luke_motion_snippet_residual_lab import (
    Snippet,
    Variant,
    best_scaled_residual,
    choose_units,
    decompose_motion,
    ks_center_car_lower_median,
    qualify_coherent_waveform_families,
    summarize,
    variant_displacement,
)


def test_motion_decomposition_and_shrinkage_are_exact():
    field = np.array([[1.0, 2.0, 3.0], [4.0, 6.0, 8.0]])
    rigid, residual = decompose_motion(field)
    np.testing.assert_allclose(rigid, [[2.0], [6.0]])
    np.testing.assert_allclose(residual, [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0]])
    variant = Variant("test", 0.25, 0.10, 10.0)
    np.testing.assert_allclose(variant_displacement(field, variant), 0.25 * rigid + 0.10 * residual)


def test_scaled_residual_recovers_shift_and_gain():
    template = np.zeros((9, 2))
    template[3, 0] = -2
    template[4, 1] = 1
    observed = np.zeros_like(template)
    observed[4, 0] = -4
    observed[5, 1] = 2
    result = best_scaled_residual(observed, template, maximum_shift=2)
    assert result["time_shift_samples"] == 1
    assert np.isclose(result["coefficient"], 2)
    assert result["residual_fraction"] < 1e-12
    assert np.isclose(result["template_cosine"], 1)


def test_car_uses_lower_median_like_torch_for_even_channel_count():
    values = np.array([[0.0, 1.0, 10.0, 20.0], [2.0, 3.0, 12.0, 22.0]])
    result = ks_center_car_lower_median(values)
    centered = values - values.mean(axis=0, keepdims=True)
    expected_lower = np.sort(centered, axis=1)[:, 1]
    np.testing.assert_allclose(result, centered - expected_lower[:, None])


def test_choose_units_requires_both_snippet_folds():
    continuity = pd.DataFrame(
        {
            "unit_id": [1, 2],
            "ks_good": [True, True],
            "presence_fraction_300s": [1.0, 1.0],
            "first_last_pc_cosine": [0.9, 0.9],
            "edge_spike_fraction": [0.0, 0.0],
        }
    )
    snippets = (
        Snippet("a", 0, 1, "quiet"),
        Snippet("b", 1, 1, "high_motion"),
    )
    times = np.array([10, 20, 1010, 1020, 500])
    clusters = np.array([1, 1, 1, 1, 2])
    units, events = choose_units(
        continuity,
        times,
        clusters,
        1000.0,
        snippets,
        maximum_units=2,
        minimum_events_per_fold=2,
    )
    assert units.unit_id.tolist() == [1]
    assert set(events.fold) == {0, 1}


def test_summary_pairs_each_event_to_no_motion():
    frame = pd.DataFrame(
        {
            "variant": ["no_motion", "candidate"],
            "unit_id": [1, 1],
            "sample_index": [10, 10],
            "snippet": ["a", "a"],
            "motion_class": ["high_motion", "high_motion"],
            "fold": [0, 0],
            "n_training_events": [3, 3],
            "n_local_channels": [4, 4],
            "residual_fraction": [0.4, 0.3],
            "template_cosine": [0.7, 0.8],
            "coefficient": [1.0, 1.0],
            "time_shift_samples": [0, 0],
            "peak_residual": [2.0, 1.0],
        }
    )
    paired, summary = summarize(frame)
    candidate = paired[paired.variant == "candidate"].iloc[0]
    assert np.isclose(candidate.delta_residual_vs_no_motion, -0.1)
    assert np.isclose(candidate.delta_cosine_vs_no_motion, 0.1)
    assert len(summary) == 2


def test_family_qualification_keeps_medoid_neighborhood_in_both_folds():
    events = pd.DataFrame(
        {
            "unit_id": [1, 1, 1, 1, 1],
            "sample_index": [10, 20, 30, 40, 50],
            "fold": [0, 0, 1, 1, 1],
            "snippet": ["a", "a", "b", "b", "b"],
            "motion_class": ["quiet"] * 5,
        }
    )
    base = np.zeros((7, 3))
    base[3, 1] = -8
    waves = {
        (1, 10): base,
        (1, 20): base * 0.9,
        (1, 30): base * 1.1,
        (1, 40): base * 1.2,
        (1, 50): np.roll(base, 1, axis=1),
    }
    templates = np.zeros((2, 7, 3))
    templates[1] = base
    locations = np.column_stack([np.zeros(3), np.arange(3) * 20.0])
    result = qualify_coherent_waveform_families(
        events,
        waves,
        templates,
        locations,
        np.ones(3),
        minimum_family_cosine=0.8,
        minimum_peak_snr=4,
        minimum_events_per_fold=2,
    )
    assert set(result.sample_index) == {10, 20, 30, 40}
