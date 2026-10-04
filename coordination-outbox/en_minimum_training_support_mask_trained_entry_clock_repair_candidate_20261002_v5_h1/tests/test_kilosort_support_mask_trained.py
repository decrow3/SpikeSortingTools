import json
from pathlib import Path
import tempfile

import numpy as np
import pytest
import torch

from kilosort import io, preprocessing, spikedetect, template_matching

from npx_preprocessing.motion import kilosort_support_mask_trained as trained
from testing.en_minimum_training_support_mask_trained_fixture import build_fixture, invoke


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_CONFIG = ROOT / "configs/en_minimum_training_support_mask.trained.review_pending.v1.json"
SMOKE_BASE = ROOT / "configs/en_minimum_training_support_mask.covariance_smoke.enabled.review_pending.v2.json"


def test_production_candidate_preserves_full_frozen_structure_and_is_disabled():
    config = json.loads(PRODUCTION_CONFIG.read_text())
    base = json.loads(SMOKE_BASE.read_text())
    trained.prevalidate_trained_structure(config)
    for key in (
        "recording", "crop", "field", "ops", "frozen_kilosort_settings",
        "covariance_gate", "native_invocation", "source_bindings", "prohibitions",
    ):
        assert key in config
        assert key in base
    assert config["execution_enabled"] is False
    assert config["trained"]["allow_saved_smoke_bank"] is False
    assert config["native_invocation"]["save_preprocessed_copy"] is False


def test_missing_required_key_fails_before_any_bound_source_or_manifest_access():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "fixture"
        path, digest = build_fixture(root, "run", missing_key=True)
        with pytest.raises(ValueError, match="missing required config key before data access"):
            trained.validate_trained_contract(
                path, expected_config_sha256=digest,
                allow_synthetic_orchestration=True, fixture_root=root,
            )
        assert not (root / "must_not_open").exists()
        assert not (root / "must_not_open_manifest").exists()


def test_actual_cli_success_reaches_completion_and_required_artifact_inventory():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "fixture"
        path, digest = build_fixture(root, "run")
        result = invoke(path, digest, root, "success")
        assert result.returncode == 0, result.stderr
        run = root / "run"
        complete = json.loads((run / "COMPLETE.json").read_text())
        inventory = json.loads((run / "launch_evidence/artifact_inventory.json").read_text())
        phases = json.loads((run / "launch_evidence/support_reads_by_phase.json").read_text())
        assert complete["required_artifacts_validated"] is True
        assert complete["smoke_bank_injected"] is False
        assert set(inventory["native"]) == set(trained.REQUIRED_NATIVE_FILES)
        assert set(inventory["pre_extraction_snapshot"]) == set(trained.REQUIRED_SNAPSHOT_FILES)
        assert set(phases["phase_counts"]) == {
            "covariance_and_whitening", "pca_and_universal_template_learning",
            "universal_detection_and_features", "learned_template_extraction",
        }
        assert inventory["semantic_validation"] == {
            "status": "semantic_validation_passed", "spikes": 3,
            "full_rows": 4, "channels": 4, "templates": 2,
            "wall3_templates": 2,
            "spike_times_frame": "global_acquisition_samples",
            "full_st_frame": "crop_local_samples",
            "crop_origin_samples": 0,
            "clock_relation": "spike_times=selected_full_st_local+crop_origin",
        }


def test_actual_cli_injected_failure_preserves_stage_and_never_completes():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "fixture"
        path, digest = build_fixture(root, "run")
        result = invoke(path, digest, root, "fail-after-whitening")
        assert result.returncode != 0
        run = root / "run"
        failure = json.loads((run / "launch_evidence/run_failure.json").read_text())
        assert failure["stage"] == "native_trained_pipeline"
        assert failure["exception_message"] == "injected synthetic downstream failure"
        assert failure["whitening_audit"]["covariance_validation"]["passed"] is True
        assert failure["completion_written"] is False
        assert not (run / "COMPLETE.json").exists()


def test_actual_cli_rejects_semantically_misaligned_saved_arrays():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "fixture"
        path, digest = build_fixture(root, "run")
        result = invoke(path, digest, root, "corrupt-alignment")
        assert result.returncode != 0
        assert "spike_clusters.npy is not aligned to spike_times.npy" in result.stderr
        failure = json.loads((root / "run/launch_evidence/run_failure.json").read_text())
        assert failure["stage"] == "validate_and_inventory_saved_artifacts"
        assert failure["completion_written"] is False
        assert not (root / "run/COMPLETE.json").exists()


@pytest.mark.parametrize("outcome, message", [
    ("permuted-ancestry", "spike_times global ancestry differs"),
    ("duplicate-kept", "integer kept_spikes indices must be unique"),
    ("out-of-bounds-id", "spike_templates is outside templates axis 0"),
    ("reordered-kept", "integer kept_spikes indices must be strictly increasing"),
    ("coordinated-permutation", "integer kept_spikes indices must be strictly increasing"),
])
def test_actual_cli_rejects_equal_length_semantic_corruption(outcome, message):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "fixture"
        path, digest = build_fixture(root, "run")
        result = invoke(path, digest, root, outcome)
        assert result.returncode != 0
        assert message in result.stderr
        failure = json.loads((root / "run/launch_evidence/run_failure.json").read_text())
        assert failure["stage"] == "validate_and_inventory_saved_artifacts"
        assert failure["completion_written"] is False
        assert not (root / "run/COMPLETE.json").exists()


def test_zero_origin_fixture_remains_a_valid_known_answer():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "fixture"
        path, digest = build_fixture(root, "run", crop_origin=0)
        result = invoke(path, digest, root, "success")
        assert result.returncode == 0, result.stderr
        inventory = json.loads((root / "run/launch_evidence/artifact_inventory.json").read_text())
        assert inventory["semantic_validation"]["crop_origin_samples"] == 0


def test_terminal_validator_nonzero_origin_wrong_offset_off_by_one_and_bounds():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "fixture"
        path, digest = build_fixture(root, "run", crop_origin=0)
        result = invoke(path, digest, root, "success")
        assert result.returncode == 0, result.stderr
        config = json.loads(path.read_text())
        origin = 208_498_882
        config["crop"] = {"imin": origin, "imax": origin + 80}
        config["output_clock"] = {
            "spike_times_frame": "global_acquisition_samples",
            "full_st_frame": "crop_local_samples",
            "crop_origin_samples": origin,
            "crop_local_bounds_half_open": [0, 80],
            "global_bounds_half_open": [origin, origin + 80],
            "exact_relation": "spike_times.npy == full_st.npy[kept_spikes,0] + crop.imin",
        }
        results = root / "run/native_results"
        snapshot = root / "run/pre_extraction_snapshot"
        local = np.array([1, 5, 9], dtype=np.int64)
        np.save(results / "spike_times.npy", local + origin, allow_pickle=False)
        receipt = trained.validate_saved_artifacts(results, snapshot, config)
        assert receipt["crop_origin_samples"] == origin
        np.save(results / "spike_times.npy", local, allow_pickle=False)
        with pytest.raises(ValueError, match="outside the half-open crop"):
            trained.validate_saved_artifacts(results, snapshot, config)
        np.save(results / "spike_times.npy", local + origin + 1, allow_pickle=False)
        with pytest.raises(ValueError, match="global ancestry differs"):
            trained.validate_saved_artifacts(results, snapshot, config)
        np.save(results / "spike_times.npy", local + origin, allow_pickle=False)
        full_st = np.load(results / "full_st.npy", allow_pickle=False)
        full_st[3, 0] = 80
        np.save(results / "full_st.npy", full_st, allow_pickle=False)
        with pytest.raises(ValueError, match="crop-local time is outside"):
            trained.validate_saved_artifacts(results, snapshot, config)


def test_clock_contract_wrong_origin_and_int64_overflow_fail_prevalidation():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "fixture"
        path, _ = build_fixture(root, "run")
        config = json.loads(path.read_text())
        config["output_clock"]["crop_origin_samples"] += 1
        with pytest.raises(ValueError, match="output_clock contract differs"):
            trained.prevalidate_trained_structure(config)
        config = json.loads(path.read_text())
        config["crop"]["imax"] = int(np.iinfo(np.int64).max) + 1
        with pytest.raises(ValueError, match="invalid for signed int64"):
            trained.prevalidate_trained_structure(config)


def test_actual_native_prepare_extract_hook_saves_before_first_batch_read():
    with tempfile.TemporaryDirectory() as td:
        snapshot = Path(td)
        phase_state = trained.PhaseState()
        data = np.random.default_rng(2).normal(0, .1, (50, 4)).astype("float32")
        bfile = io.BinaryFiltered(
            filename="synthetic", n_chan_bin=4, fs=1000, NT=20, nt=3,
            nt0min=1, chan_map=np.arange(4), whiten_mat=torch.eye(4),
            do_CAR=False, device=torch.device("cpu"), file_object=data,
        )
        class CheckingBfile:
            def __init__(self):
                self.n_batches = bfile.n_batches
            def padded_batch_to_torch(self, *args, **kwargs):
                assert (snapshot / "RECEIPT.json").is_file()
                return bfile.padded_batch_to_torch(*args, **kwargs)
        ops = {
            "settings": {"nearest_chans": 2, "position_limit": 100., "n_pcs": 1},
            "xc": np.zeros(4), "yc": np.arange(4) * 20., "nt": 3,
            "nt0min": 1, "batch_size": 20, "Th_learned": 100.,
            "max_peels": 2, "wPCA": torch.tensor([[0., 1., 0.]]),
            "wTEMP": torch.tensor([[0., 1., 0.]]), "chanMap": np.arange(4),
        }
        wall3 = torch.ones((1, 1, 4), dtype=torch.float32)
        with trained.installed_phase_and_snapshot_hooks(
            native_preprocessing=preprocessing, native_spikedetect=spikedetect,
            native_template_matching=template_matching,
            phase_state=phase_state, snapshot_dir=snapshot,
        ):
            st, features, returned_ops = template_matching.extract(
                ops, CheckingBfile(), wall3, device=torch.device("cpu")
            )
        assert st.shape[0] == features.shape[0] == 0
        assert returned_ops["iU"].shape == (1,)
        receipt = json.loads((snapshot / "RECEIPT.json").read_text())
        assert receipt["status"] == "saved_before_learned_batch_extraction"
        assert receipt["wall3_axis0_binding"] == "spike_detection_templates.npy"
        assert np.array_equal(np.load(snapshot / "iU_physical_binary_channel.npy"), np.array([0]))
