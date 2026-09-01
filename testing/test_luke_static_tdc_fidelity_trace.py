import numpy as np

from testing.luke_static_tdc_fidelity_trace import (
    classify_first_gate,
    deterministic_balanced_sample,
    fit_template,
    one_to_one_reference_mask,
)


def test_one_to_one_reference_mask_does_not_reuse_target_event():
    matched = one_to_one_reference_mask(
        np.array([10, 11, 30]),
        np.array([1, 1, 2]),
        np.array([10, 30]),
        np.array([1, 2]),
        tolerance=2,
    )
    assert matched.tolist() == [True, False, True]


def test_balanced_sample_contains_both_replay_strata():
    replayed = np.array([True] * 10 + [False] * 30)
    selected = deterministic_balanced_sample(replayed, 12, seed=4)
    assert replayed[selected].sum() == 6
    assert (~replayed[selected]).sum() == 6
    assert len(np.unique(selected)) == 12


def test_fit_template_recovers_shift_and_amplitude():
    sparse = np.array([[0.0], [-1.0], [-2.0], [-1.0], [0.0]])
    traces = np.zeros((21, 2), dtype=float)
    traces[9:14, 0] = sparse[:, 0] * 1.2
    result = fit_template(
        traces,
        sparse,
        np.array([True, False]),
        center=11,
        nbefore=2,
        shifts=np.array([-1, 0, 1]),
    )
    assert result["shift"] == 0
    assert np.isclose(result["amplitude"], 1.2)
    assert np.isclose(result["fit_improvement"], 1.0)


def test_gate_classification_reflects_asymmetric_tdc_amplitude_behavior():
    common = dict(
        fast_peak=True,
        correct_template_candidate=True,
        selected_correct_template=True,
        static_any_label=False,
        static_same_label=False,
    )
    assert (
        classify_first_gate(fitted_amplitude=0.5, **common)
        == "below_tdc_amplitude_floor"
    )
    assert (
        classify_first_gate(fitted_amplitude=2.0, **common)
        == "passes_isolated_gates_but_missing_in_full_peeler"
    )
    assert (
        classify_first_gate(
            fitted_amplitude=1.0,
            **{**common, "fast_peak": False},
        )
        == "no_fast_detector_peak"
    )
