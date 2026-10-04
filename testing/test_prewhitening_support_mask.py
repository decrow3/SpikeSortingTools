import hashlib
import json
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import tempfile
import unittest

from kilosort.io import BinaryFiltered, BinaryRWFile
from kilosort.preprocessing import get_highpass_filter
from spikeinterface.preprocessing import get_spatial_interpolation_kernel

from npx_preprocessing.motion.prewhitening_support_mask import (
    filter_then_mask_before_whitening,
    si_stepwise_states,
    support_for_shift,
    support_mask_from_geometry_and_field,
)
from npx_preprocessing.motion.lattice_remap_si import exact_coordinate_mapping
from testing.en_b384_q40_support_mask_replay import (
    batch_state_classifications,
    endpoint_neighbor_report,
    manifest_owned_binary_path,
    validate_endpoint_coverage,
)


GEOMETRY = np.array([[0.0, 0.0], [0.0, 40.0], [0.0, 80.0], [0.0, 120.0]])


def as_numpy(value):
    return np.asarray(value.cpu() if hasattr(value, "cpu") else value)


class FakeFilteredFile:
    def __init__(self, prewhite, whitening):
        self.prewhite = prewhite
        self.whiten_mat = whitening
        self.dshift = None
        self.calls = []

    def filter(self, raw, ops=None, ibatch=None):
        self.calls.append({
            "whitening_disabled": self.whiten_mat is None,
            "ibatch": ibatch,
            "raw": raw.clone(),
        })
        return self.prewhite.clone()


def run_candidate(prewhite, whitening, centers, shifts, *, fs=8.0, start=0):
    fake = FakeFilteredFile(prewhite, whitening)
    result = filter_then_mask_before_whitening(
        fake,
        torch.full_like(prewhite, 123.0),
        ops={"fs": fs},
        ibatch=7,
        global_start_frame=start,
        channel_locations_um=GEOMETRY,
        temporal_centers_s=np.asarray(centers),
        shifts_um=np.asarray(shifts),
    )
    assert len(fake.calls) == 1
    assert fake.calls[0]["whitening_disabled"] is True
    assert fake.calls[0]["ibatch"] == 7
    assert torch.equal(fake.calls[0]["raw"], torch.full_like(prewhite, 123.0))
    assert fake.whiten_mat is whitening
    return result


def test_all_valid_state_is_bitwise_equal_to_baseline():
    prewhite = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    whitening = torch.tensor([
        [1.0, 0.2, 0.0, -0.1], [0.1, 1.0, 0.3, 0.0],
        [0.0, -0.2, 1.0, 0.4], [0.3, 0.0, 0.2, 1.0],
    ])
    result = run_candidate(prewhite, whitening, [0.25, 0.75], [0.0, 0.0], fs=8.0)
    assert torch.all(result.support)
    assert torch.equal(result.candidate_prewhite, prewhite)
    assert torch.equal(result.candidate_whitened, result.baseline_whitened)


def test_stable_positive_state_zeros_only_unsupported_row_before_whitening():
    prewhite = torch.tensor([
        [1.0, 2.0, 3.0], [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0], [100.0, 200.0, 300.0],
    ])
    whitening = torch.eye(4)
    result = run_candidate(prewhite, whitening, [0.25, 0.75], [40.0, 40.0], fs=4.0)
    assert torch.all(result.support[:3])
    assert not torch.any(result.support[3])
    assert torch.equal(result.candidate_prewhite[:3], prewhite[:3])
    assert torch.count_nonzero(result.candidate_prewhite[3]) == 0


def test_nonsymmetric_whitening_uses_columns_and_preserves_adjacent_signal():
    prewhite = torch.tensor([[2.0], [3.0], [5.0], [11.0]])
    whitening = torch.tensor([
        [1.0, 2.0, 3.0, 4.0], [0.0, 1.0, -1.0, 2.0],
        [3.0, 0.5, 1.0, -2.0], [1.0, -3.0, 2.0, 0.25],
    ])
    result = run_candidate(prewhite, whitening, [0.25, 0.75], [40.0, 40.0], fs=4.0)
    expected_prewhite = prewhite.clone()
    expected_prewhite[3] = 0
    assert torch.equal(result.candidate_prewhite, expected_prewhite)
    assert torch.equal(result.candidate_prewhite[2], prewhite[2])
    # Row 2 is the nearest supported neighbor to invalid row 3. Its known value
    # must survive unchanged before whitening; only Wrot may mix it afterward.
    assert float(result.candidate_prewhite[2, 0]) == 5.0
    assert torch.equal(result.candidate_whitened, whitening @ expected_prewhite)
    assert not torch.equal(result.candidate_whitened, result.baseline_whitened)
    postwhite_row_mask = (whitening @ prewhite).clone()
    postwhite_row_mask[3] = 0
    assert not torch.equal(result.candidate_whitened, postwhite_row_mask)


def test_positive_negative_states_and_half_open_transition_assignment():
    centers = np.array([0.125, 0.375, 0.625])
    shifts = np.array([0.0, 40.0, -40.0])
    frames = np.array([0, 1, 2, 3, 4])
    states = si_stepwise_states(centers, shifts, frames, 8.0)
    assert states.tolist() == [0.0, 0.0, 40.0, 40.0, -40.0]
    support = support_mask_from_geometry_and_field(GEOMETRY, centers, shifts, frames, 8.0)
    assert np.all(support[:, :2])
    assert np.all(support[:3, 2:4]) and not np.any(support[3, 2:4])
    assert not support[0, 4] and np.all(support[1:, 4])


def test_local_batch_frames_receive_global_origin_without_shifting_field_clock():
    centers = np.array([10.125, 10.375, 10.625])
    shifts = np.array([0.0, 40.0, -40.0])
    local_frames = np.array([0, 2, 4])
    states = si_stepwise_states(
        centers, shifts, local_frames, 8.0, sample_frame_origin=80
    )
    assert states.tolist() == [0.0, 40.0, -40.0]
    # Without converting local frames to the field's global clock, all three
    # samples clamp to the first field cell.  A common shift of both clocks
    # would therefore fail to exercise the actual crop-origin distinction.
    unconverted = si_stepwise_states(centers, shifts, local_frames, 8.0)
    assert unconverted.tolist() == [0.0, 0.0, 0.0]


def test_support_is_geometric_not_inferred_from_signal_amplitude():
    positive = support_for_shift(GEOMETRY, 40.0)
    negative = support_for_shift(GEOMETRY, -40.0)
    assert positive.tolist() == [True, True, True, False]
    assert negative.tolist() == [False, True, True, True]
    # Signal values never enter support_for_shift; a legitimate zero on a
    # supported row and a large created value on an unsupported row do not
    # alter membership.
    synthetic_signal = np.array([0.0, 5.0, 7.0, 1e6])
    assert positive[0] and synthetic_signal[0] == 0
    assert not positive[3] and synthetic_signal[3] != 0


def test_support_matches_installed_si_bounds_and_nearest_kernel():
    irregular = np.array([[0.0, 0.0], [0.0, 50.0], [0.0, 100.0]])
    for shift, expected in ((25.0, [True, True, False]), (-25.0, [False, True, True])):
        moved = irregular.copy()
        moved[:, 1] += shift
        kernel = get_spatial_interpolation_kernel(
            irregular, moved, method="nearest", force_extrapolate=False
        )
        installed_support = np.any(kernel != 0, axis=0)
        observed = support_for_shift(irregular, shift)
        assert observed.tolist() == expected
        assert np.array_equal(observed, installed_support)
    assert support_for_shift(irregular, 0.0).tolist() == [True, True, True]
    just_positive = np.spacing(100.0)
    just_negative = -np.spacing(100.0)
    assert support_for_shift(irregular, just_positive).tolist() == [True, True, False]
    assert support_for_shift(irregular, just_negative).tolist() == [False, True, True]


def test_actual_binaryfiltered_highpass_integration_q0_and_signed_support():
    rng = np.random.default_rng(42)
    file_object = rng.integers(-100, 101, size=(80, 4), dtype=np.int16)
    whitening = torch.tensor([
        [1.0, 0.2, 0.0, -0.1], [0.1, 1.0, 0.3, 0.0],
        [0.0, -0.2, 1.0, 0.4], [0.3, 0.0, 0.2, 1.0],
    ])
    hp = get_highpass_filter(fs=1000.0, cutoff=100.0, device=torch.device("cpu"))
    bfile = BinaryFiltered(
        "in-memory", 4, fs=1000, NT=64, nt=5, nt0min=2,
        chan_map=np.arange(4), hp_filter=hp, whiten_mat=whitening,
        dshift=None, device=torch.device("cpu"), do_CAR=True,
        dtype="int16", file_object=file_object,
    )
    raw, inds = BinaryRWFile.padded_batch_to_torch(bfile, 0, return_inds=True)
    native = bfile.filter(raw.clone(), ops={"fs": 1000.0}, ibatch=0)
    q0 = filter_then_mask_before_whitening(
        bfile, raw, ops={"fs": 1000.0}, ibatch=0, global_start_frame=int(inds[0]),
        channel_locations_um=GEOMETRY,
        temporal_centers_s=np.array([0.0, 0.05]), shifts_um=np.array([0.0, 0.0]),
    )
    assert torch.equal(q0.baseline_whitened, native)
    assert torch.equal(q0.candidate_whitened, native)

    signed = filter_then_mask_before_whitening(
        bfile, raw, ops={"fs": 1000.0}, ibatch=0, global_start_frame=int(inds[0]),
        channel_locations_um=GEOMETRY,
        temporal_centers_s=np.array([-0.025, 0.025, 0.075]),
        shifts_um=np.array([40.0, 40.0, -40.0]),
    )
    frames = np.arange(int(inds[0]), int(inds[0]) + raw.shape[1])
    positive_columns = frames < 50
    negative_columns = frames >= 50
    assert not torch.any(signed.support[3, positive_columns])
    assert torch.all(signed.support[:3, positive_columns])
    assert not torch.any(signed.support[0, negative_columns])
    assert torch.all(signed.support[1:, negative_columns])
    assert torch.equal(signed.candidate_prewhite[:3, positive_columns], signed.prewhite[:3, positive_columns])
    assert torch.equal(signed.candidate_prewhite[1:, negative_columns], signed.prewhite[1:, negative_columns])


def test_b384_executed_lineage_geometry_and_lattice_support():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/en_common_support_crop_screen.v1.json").read_text())
    arm = config["arms"]["B384"]
    manifest_path = Path(arm["recording_manifest_path"])
    manifest = json.loads(manifest_path.read_text())
    provenance = json.loads((manifest_path.parent / "provenance.json").read_text())
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == arm["recording_manifest_sha256"]
    assert manifest["recording_binary_files"][0]["sha256"] == arm["accepted_binary_sha256"]
    assert provenance["class"] == "npx_preprocessing.motion.lattice_remap_si.ExactLatticeRemapRecording"
    assert arm["bad_channels"] == []
    assert arm["retained_channel_map_half_open"] == [0, 384]

    ops = np.load(
        "/media/huklaban5/Data/en_common_support_crop_screen_20260930_v1/B384/sorter_output/ops.npy",
        allow_pickle=True,
    ).item()
    chan_map = np.asarray(as_numpy(ops["probe"]["chanMap"]), dtype=np.int64)
    geometry = np.column_stack((as_numpy(ops["xc"]), as_numpy(ops["yc"])))
    original_geometry = np.asarray(manifest["channel_locations_um"], dtype=np.float64)[:, :2]
    assert int(ops["Nchan"]) == 384 and int(ops["n_chan_bin"]) == 384
    assert np.array_equal(chan_map, np.arange(384))
    assert np.array_equal(geometry, original_geometry)

    b368 = np.load(
        "/media/huklaban5/Data/en_common_support_crop_screen_20260930_v1/B368/sorter_output/ops.npy",
        allow_pickle=True,
    ).item()
    b368_map = np.asarray(as_numpy(b368["probe"]["chanMap"]), dtype=np.int64)
    assert int(b368["Nchan"]) == 368
    assert np.array_equal(b368_map, np.arange(12, 380))

    field_path = Path(
        "/mnt/NPX/Luke/20250804/shared_analysis/"
        "luke_improved_motion_two_machine_20260909_v1/estimation_huklaban1_v1/candidate_fields.npz"
    )
    with np.load(field_path, allow_pickle=False) as field:
        displacement = np.asarray(field["rigid_displacement_um"], dtype=np.float64)
        centers = np.asarray(field["time_s"], dtype=np.float64)
    reference = float(np.median(displacement))
    rounded = 40.0 * np.copysign(
        np.floor(np.abs((displacement - reference) / 40.0) + 0.5),
        displacement - reference,
    )
    expected_supported = {-120.0: 372, -80.0: 376, -40.0: 380, 0.0: 384, 40.0: 380}
    expected_invalid = {
        -120.0: list(range(12)), -80.0: list(range(8)), -40.0: list(range(4)),
        0.0: [], 40.0: list(range(380, 384)),
    }
    for state in np.unique(rounded):
        exact = exact_coordinate_mapping(original_geometry, float(state), direction="y", atol_um=1e-6) >= 0
        bounds = support_for_shift(original_geometry, float(state))
        assert np.array_equal(exact, bounds)
        assert int(bounds.sum()) == expected_supported[float(state)]
        assert np.flatnonzero(~bounds).tolist() == expected_invalid[float(state)]

    # Within a stable +40-um cell, the top four rows are unsupported for every
    # sample, not inferred from their voltage values.
    stable = support_mask_from_geometry_and_field(
        original_geometry, centers[:2], np.array([40.0, 40.0]),
        np.arange(8, dtype=np.int64), float(ops["fs"]),
    )
    assert np.all(stable[:380])
    assert not np.any(stable[380:])


def endpoint_fixture():
    config = {
        "expected_selected_endpoints": 2,
        "require_every_selected_batch_has_endpoints": True,
    }
    ops = {"tmin": 10.0, "tmax": 30.0, "fs": 10.0}
    batches = pd.DataFrame([
        {"batch_index": 4, "batch_core_start_global": 100,
         "batch_core_stop_global_exclusive": 150},
        {"batch_index": 5, "batch_core_start_global": 150,
         "batch_core_stop_global_exclusive": 200},
    ])
    endpoints = pd.DataFrame([
        {"batch_index": 4, "event_global": 120, "template_id": 0},
        {"batch_index": 5, "event_global": 180, "template_id": 6},
    ])
    return config, ops, batches, endpoints


def test_endpoint_coverage_accepts_only_selected_batches_and_valid_time_template():
    config, ops, batches, endpoints = endpoint_fixture()
    result = validate_endpoint_coverage(config, ops, batches, endpoints, bank_templates=7)
    assert result["endpoint_rows_validated"] == 2
    assert result["endpoint_batch_indices"] == [4, 5]


def test_endpoint_coverage_rejects_unselected_batch():
    config, ops, batches, endpoints = endpoint_fixture()
    endpoints.loc[1, "batch_index"] = 6
    with np.testing.assert_raises_regex(RuntimeError, "unselected batch"):
        validate_endpoint_coverage(config, ops, batches, endpoints, bank_templates=7)


def test_endpoint_coverage_rejects_invalid_global_time_and_template_id():
    config, ops, batches, endpoints = endpoint_fixture()
    bad_time = endpoints.copy()
    bad_time.loc[0, "event_global"] = 150
    with np.testing.assert_raises_regex(RuntimeError, "global time"):
        validate_endpoint_coverage(config, ops, batches, bad_time, bank_templates=7)
    bad_template = endpoints.copy()
    bad_template.loc[1, "template_id"] = 7
    with np.testing.assert_raises_regex(RuntimeError, "template ID"):
        validate_endpoint_coverage(config, ops, batches, bad_template, bank_templates=7)


def test_manifest_owned_binary_path_rejects_same_name_in_different_directory():
    with tempfile.TemporaryDirectory() as root_text:
        root = Path(root_text)
        accepted = root / "accepted"
        replacement = root / "replacement"
        accepted.mkdir(); replacement.mkdir()
        manifest_path = accepted / "rescue_recording_manifest.json"
        manifest_path.write_text("{}")
        (accepted / "traces.raw").write_bytes(b"")
        (replacement / "traces.raw").write_bytes(b"")
        manifest = {"recording_binary_files": [{"name": "traces.raw"}]}
        owned = manifest_owned_binary_path(manifest_path, manifest)
        assert owned == (accepted / "traces.raw").resolve()
        assert owned != (replacement / "traces.raw").resolve()


def test_manifest_owned_binary_path_rejects_path_in_filename_field():
    with tempfile.TemporaryDirectory() as root_text:
        root = Path(root_text)
        manifest_path = root / "manifest.json"
        manifest_path.write_text("{}")
        with np.testing.assert_raises_regex(RuntimeError, "basename"):
            manifest_owned_binary_path(
                manifest_path,
                {"recording_binary_files": [{"name": "../traces.raw"}]},
            )


def test_endpoint_neighbor_report_retains_all_frozen_local_cases():
    times = np.array([100, 101, 100, 101, 130, 131, 99], dtype=np.int64)
    templates = np.array([7, 7, 8, 9, 10, 11, 12], dtype=np.int64)
    amplitudes = np.arange(1, 8, dtype=np.float64)
    scores = np.arange(10, 17, dtype=np.float64)
    summary, neighbors = endpoint_neighbor_report(
        100, 7, times, templates, amplitudes, scores
    )
    assert summary == {
        "exact_same_time_template_count": 1,
        "exact_same_time_template_retained": True,
        "same_template_within_1_count": 2,
        "same_template_within_1_retained": True,
        "any_template_within_1_count": 5,
        "any_template_within_30_count": 6,
    }
    assert [(row["neighbor_template_id"], row["sample_offset"]) for row in neighbors] == [
        (7, 0), (7, 1), (8, 0), (9, 1), (10, 30), (12, -1)
    ]
    assert all(row["neighbor_template_id"] != 11 for row in neighbors)
    assert [row["amplitude"] for row in neighbors] == [1.0, 2.0, 3.0, 4.0, 5.0, 7.0]
    assert [row["threshold_score"] for row in neighbors] == [10.0, 11.0, 12.0, 13.0, 14.0, 16.0]


def test_batch_state_classification_preserves_stable_and_transition_labels():
    batches = pd.DataFrame([
        {"batch_index": 1, "padded_read_start_global": 0,
         "padded_read_stop_global_exclusive": 4},
        {"batch_index": 2, "padded_read_start_global": 2,
         "padded_read_stop_global_exclusive": 6},
    ])
    observed = batch_state_classifications(
        batches,
        np.array([0.125, 0.375, 0.625]),
        np.array([40.0, 40.0, 0.0]),
        8.0,
        {"1": "stable", "2": "transition"},
    )
    assert observed.batch_classification.tolist() == ["stable", "transition"]
    assert observed.state_changes_in_padded_batch.tolist() == [0, 1]


class PrewhiteningSupportMaskTests(unittest.TestCase):
    def test_all_valid(self):
        test_all_valid_state_is_bitwise_equal_to_baseline()

    def test_stable_unsupported(self):
        test_stable_positive_state_zeros_only_unsupported_row_before_whitening()

    def test_nonsymmetric_whitening_and_adjacent_signal(self):
        test_nonsymmetric_whitening_uses_columns_and_preserves_adjacent_signal()

    def test_positive_negative_transition(self):
        test_positive_negative_states_and_half_open_transition_assignment()

    def test_nonzero_origin(self):
        test_local_batch_frames_receive_global_origin_without_shifting_field_clock()

    def test_geometric_not_amplitude(self):
        test_support_is_geometric_not_inferred_from_signal_amplitude()

    def test_installed_si_support_kernel(self):
        test_support_matches_installed_si_bounds_and_nearest_kernel()

    def test_actual_binaryfiltered_integration(self):
        test_actual_binaryfiltered_highpass_integration_q0_and_signed_support()

    def test_b384_executed_lineage(self):
        test_b384_executed_lineage_geometry_and_lattice_support()

    def test_endpoint_coverage(self):
        test_endpoint_coverage_accepts_only_selected_batches_and_valid_time_template()

    def test_endpoint_unselected_batch_rejected(self):
        test_endpoint_coverage_rejects_unselected_batch()

    def test_endpoint_time_and_template_rejected(self):
        test_endpoint_coverage_rejects_invalid_global_time_and_template_id()

    def test_manifest_owned_path_rejects_replacement_directory(self):
        test_manifest_owned_binary_path_rejects_same_name_in_different_directory()

    def test_manifest_owned_path_rejects_embedded_path(self):
        test_manifest_owned_binary_path_rejects_path_in_filename_field()

    def test_endpoint_neighbor_report_all_cases(self):
        test_endpoint_neighbor_report_retains_all_frozen_local_cases()

    def test_batch_stable_transition_labels(self):
        test_batch_state_classification_preserves_stable_and_transition_labels()


if __name__ == "__main__":
    unittest.main()
