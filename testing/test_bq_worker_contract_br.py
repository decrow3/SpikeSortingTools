from pathlib import Path

import numpy as np
import pytest

from testing.experimental.bq_worker_contract import (
    WorkerContractError,
    compose_actual_lineage,
    flatten_with_lineage,
    keyword_only_parameters,
    linkage_accounting,
    partition_delta,
    qda_accounting,
    require_shared_state,
    stratified_isi_summary,
    validate_prefix_contract,
)


DARTSORT = Path("/home/huklab/Documents/DARTsort/src/dartsort")
H = "a" * 64


def prefix():
    return {
        "stages": ["pcmerge", "tmm", "agglomerate"],
        "recluster_after_matching": False,
        "waveform_config_sha256": H,
        "template_state_sha256": H,
        "rng_state_sha256": H,
        "merge_cutoff": 0.6,
        "force_cutoff": 0.3,
        "cross_merge_cutoff": 0.5,
        "cross_merge_cutoff_used": False,
    }


def test_executed_source_apis_are_keyword_only():
    motion = keyword_only_parameters(
        DARTSORT / "util/motion.py", "from_motion_est", class_name="MotionInfo"
    )
    agglomerate = keyword_only_parameters(
        DARTSORT / "clustering/agglomerate.py", "agglomerate"
    )
    assert motion == ("geom", "dredge_motion_est", "si_motion", "rgeom")
    assert agglomerate == (
        "sorting", "recording", "template_merge_cfg", "refinement_cfg", "motion",
        "template_data", "computation_cfg", "waveform_cfg", "show_progress",
    )


def test_checkout_and_installed_shift_semantics_are_not_interchangeable():
    checkout = (DARTSORT / "clustering/agglomerate.py").read_text()
    installed = Path(
        "/home/huklab/anaconda3/envs/spike-sort-challengers/lib/python3.12/"
        "site-packages/dartsort/clustering/agglomerate.py"
    ).read_text()
    assert 'TEMPLATE_SHIFT_CONVENTION = "reference_peak_minus_source_peak_v1"' in checkout
    assert "shifts=-best_lag.numpy(force=True)" in checkout
    assert "TEMPLATE_SHIFT_CONVENTION" not in installed
    assert "shifts=best_lag.numpy(force=True)" in installed


def test_prefix_cannot_omit_pcmerge_or_use_silent_defaults():
    validate_prefix_contract(prefix())
    bad = prefix()
    bad["stages"] = ["tmm", "agglomerate"]
    with pytest.raises(WorkerContractError, match="pcmerge"):
        validate_prefix_contract(bad)
    bad = prefix()
    bad.pop("waveform_config_sha256")
    with pytest.raises(WorkerContractError, match="fields"):
        validate_prefix_contract(bad)


def test_sparse_ids_flatten_once_and_compose_actual_reorder():
    flat, original = flatten_with_lineage([10, 90, -1, 30, 10])
    assert flat.tolist() == [0, 2, -1, 1, 0]
    assert original.tolist() == [10, 30, 90]
    rows = compose_actual_lineage(original, [5, 5, 9], [1, 0])
    assert [row["actual_final_component"] for row in rows] == [1, 1, 0]
    assert [row["original_unit_id"] for row in rows] == [10, 30, 90]
    with pytest.raises(WorkerContractError, match="actual recluster"):
        compose_actual_lineage(original, [5, 5, 9], [0, 1, 2])


def test_partition_accounting_is_permutation_invariant_and_linear_grain():
    before = np.array([10, 10, 30, 30, 90, -1])
    renamed = np.array([7, 7, 3, 3, 8, -1])
    delta = partition_delta(before, renamed)
    assert delta.changed_event_count == 0
    assert delta.changed_before_group_count == 0
    split = np.array([7, 6, 3, 3, 8, -1])
    delta = partition_delta(before, split)
    assert delta.changed_event_count == 2
    assert delta.changed_before_group_count == 1
    assert delta.changed_after_group_count == 2


def test_isi_pairs_do_not_cross_frozen_state_boundaries():
    summary = stratified_isi_summary(
        [95, 99, 101, 103, 201, 203, 250],
        [10, 10, 10, 10, 10, 10, -1],
        {"rest": [(0, 100), (200, 300)], "episode": [(100, 200)]},
        refractory_samples=4,
    )
    assert summary["rest"] == {
        "spike_count": 4, "adjacent_pair_denominator": 2,
        "violation_count": 2, "refractory_samples_inclusive": 4,
    }
    assert summary["episode"]["adjacent_pair_denominator"] == 1
    assert summary["episode"]["violation_count"] == 1


def test_gate_h_requires_present_shared_rng_template_and_arrays():
    required = ["rng_state_sha256", "template_state_sha256", "labels", "times_samples"]
    state = {
        "rng_state_sha256": H, "template_state_sha256": H,
        "labels": np.array([0, 1]), "times_samples": np.array([10, 20]),
    }
    require_shared_state(state, {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in state.items()}, required)
    missing = dict(state)
    missing.pop("labels")
    with pytest.raises(WorkerContractError, match="missing"):
        require_shared_state(missing, missing, required)


def test_zero_qda_sentinels_do_not_decide_completion():
    scheduled = [[False, True, True], [False, False, False], [False, False, False]]
    status = [
        ["diagonal", "scheduled_unknown", "completed"],
        ["lower", "diagonal", "lower"],
        ["lower", "lower", "diagonal"],
    ]
    accepted = [[False, False, False], [False, False, False], [False, False, False]]
    assert qda_accounting(scheduled, status, accepted) == {
        "scheduled_pairs": 2, "completed_pairs": 1,
        "unknown_pairs": 1, "completed_rejections": 1,
    }


def test_direct_force_edges_are_not_expanded_comember_pairs():
    result = linkage_accounting(
        [[0.0, 0.2, 1.0], [9.0, 0.0, 0.2], [8.0, 7.0, 0.0]],
        threshold=0.3,
    )
    assert result["direct_pair_count"] == 2
    assert result["expanded_pair_count"] == 3
    assert result["component_count"] == 1
