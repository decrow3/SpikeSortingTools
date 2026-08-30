import copy

import numpy as np
import pytest

from testing.luke_injected_ground_truth_benchmark import (
    REQUIRED_STRATA,
    InjectionEvent,
    build_benchmark_plan,
    inject_float32_raw_domain,
    run_synthetic_validation,
    score_sample_detections,
    template_sha256,
    validate_benchmark_plan,
    validate_template,
)


def small_template() -> np.ndarray:
    template = np.zeros((7, 2), dtype=np.float32)
    template[2:5, 0] = [-1.0, -3.0, -1.0]
    return template


def test_plan_encodes_sealed_disjoint_paired_balanced_protocol_and_phase_2():
    plan = build_benchmark_plan()
    assert set(plan["strata"]) == REQUIRED_STRATA
    assert plan["safety_contract"]["array_dtype"] == "float32"
    assert plan["safety_contract"]["stored_int16_injection_allowed"] is False
    assert plan["sealed_inputs"]["seal_before_condition_selection"] is True
    assert set(plan["splits"]) == {
        "template_extraction",
        "template_qualification",
        "evaluation_background",
    }
    assert plan["paired_conditions"]["conditions"] == ["uninjected", "injected"]
    assert plan["phase_2"]["enabled_after_phase_1_freeze"] is True
    assert "residual_energy_fraction" in plan["ground_truth_metrics"]["failure_modes"]


def test_plan_validation_rejects_direct_int16_permission():
    plan = copy.deepcopy(build_benchmark_plan())
    plan["safety_contract"]["stored_int16_injection_allowed"] = True
    with pytest.raises(ValueError, match="int16"):
        validate_benchmark_plan(plan)


def test_injection_rejects_int16_background_and_template():
    with pytest.raises(TypeError, match="int16 injection is forbidden"):
        inject_float32_raw_domain(
            np.zeros((30, 4), dtype=np.int16), {"t": small_template()}, []
        )
    with pytest.raises(TypeError, match="templates must be float32"):
        inject_float32_raw_domain(
            np.zeros((30, 4), dtype=np.float32),
            {"t": small_template().astype(np.int16)},
            [],
        )


def test_template_validation_rejects_nonzero_temporal_edges():
    template = small_template()
    template[0, 1] = 0.01
    with pytest.raises(ValueError, match="nonzero temporal edges"):
        validate_template(template)


def test_paired_injection_preserves_background_and_adds_collisions():
    background = np.zeros((40, 5), dtype=np.float32)
    original = background.copy()
    template = small_template()
    events = [
        InjectionEvent("one", "t", 15, 1.0, 1, "pair"),
        InjectionEvent("two", "t", 15, 0.5, 1, "pair"),
    ]
    injected = inject_float32_raw_domain(background, {"t": template}, events)
    assert np.array_equal(background, original)
    assert injected[15, 1] == pytest.approx(-4.5)
    assert np.count_nonzero(injected) == 3


def test_injection_rejects_boundary_truncation():
    with pytest.raises(ValueError, match="time boundary"):
        inject_float32_raw_domain(
            np.zeros((20, 4), dtype=np.float32),
            {"t": small_template()},
            [InjectionEvent("edge", "t", 1)],
        )


def test_template_hash_includes_shape_and_content():
    first = small_template()
    second = first.copy()
    second[3, 0] -= 1
    assert template_sha256(first) == template_sha256(first.copy())
    assert template_sha256(first) != template_sha256(second)


def test_detection_metrics_separate_duplicates_and_false_positives():
    metrics = score_sample_detections([100, 200], [99, 100, 201, 500], tolerance_samples=2)
    assert metrics["true_positive_count"] == 2
    assert metrics["duplicate_count"] == 1
    assert metrics["false_positive_count"] == 2
    assert metrics["recall"] == 1.0
    assert metrics["precision"] == 0.5


def test_synthetic_validation_runs_without_spikeinterface_or_raw_data():
    result = run_synthetic_validation()
    assert result["passed"] is True
    assert result["dtype"] == "float32"
    assert result["paired_background_unchanged"] is True
    assert result["spikeinterface_used"] is False
