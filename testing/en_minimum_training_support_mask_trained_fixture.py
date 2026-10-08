#!/usr/bin/env python3
"""Full-structure synthetic vertical fixture for trained orchestration only."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
KS = ROOT / "environments/rescue-production/.venv/lib/python3.12/site-packages"
for import_root in (ROOT, KS):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))
BASE = ROOT / "configs/en_minimum_training_support_mask.trained.review_pending.v1.json"
LAUNCHER = ROOT / "testing/en_minimum_training_support_mask_trained_launch.py"
REVIEWED_PYTHON = ROOT / "environments/rescue-production/.venv/bin/python"

from kilosort import io as native_io, preprocessing, spikedetect, template_matching
from npx_preprocessing.motion import kilosort_support_mask_trained as trained_module


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_fixture(root: Path, run_name: str, *, missing_key: bool = False) -> tuple[Path, str]:
    root.mkdir(parents=True, exist_ok=False)
    config = copy.deepcopy(json.loads(BASE.read_text()))
    geometry = np.c_[np.zeros(4), np.arange(4) * 20.0]
    field = root / "field.npz"
    np.savez(field, time_s=np.arange(0.005, 0.080, 0.010),
             rigid_displacement_um=np.array([0., 20., 0., 20., 0., 20., 0., 20.]))
    ops_path = root / "ops.npy"
    frozen = {
        "fs": 1000.0, "batch_size": 20, "nt": 3, "nt0min": 1,
        "do_CAR": False, "artifact_threshold": float("inf"),
        "invert_sign": False, "nblocks": 0, "Th_universal": 12,
        "Th_learned": 9, "data_dtype": "float32",
        "settings": {
            "nskip": 25, "whitening_range": 32, "highpass_cutoff": 300,
            "templates_from_data": True, "n_templates": 6, "n_pcs": 6,
            "Th_single_ch": 6,
        },
        "chanMap": np.arange(4),
    }
    np.save(ops_path, frozen)
    binary = root / "synthetic.raw"
    binary.write_bytes(b"")
    manifest_path = root / "manifest.json"
    manifest = {
        "complete": True, "adapter": "synthetic ExactLattice orchestration adapter",
        "recording_content_sha256": "synthetic-content-not-opened",
        "num_samples": 80, "num_channels": 4,
        "sampling_frequency_hz": 1000.0, "dtype": "float32",
        "channel_locations_um": geometry.tolist(),
        "recording_binary_files": [{"name": binary.name, "sha256": "synthetic-binary-not-hashed"}],
    }
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))
    marker = root / "synthetic_source_binding.py"
    marker.write_text("# orchestration-only source marker\n")
    run_root = root / run_name
    config.update({
        "status": "synthetic_orchestration_only",
        "execution_enabled": True,
        "approval": {"h1_review_manifest_sha256": "0" * 64},
        "orchestration_only": {
            "synthetic": True,
            "label": "controlled_local_no_sort_training_or_detection",
            "geometry_um": geometry.tolist(),
            "substitutions": [
                "recording manifest and empty binary path", "field", "ops",
                "source bindings", "crop/clock/channel count", "run namespace",
                "device CPU", "native runner and saved artifacts",
            ],
        },
    })
    config["source_bindings"] = [{"path": str(marker), "sha256": sha(marker)}]
    config["recording"] = {
        "manifest_path": str(manifest_path), "manifest_sha256": sha(manifest_path),
        "required_adapter": manifest["adapter"],
        "recording_content_sha256": manifest["recording_content_sha256"],
        "num_samples": 80, "num_channels": 4, "sampling_frequency_hz": 1000.0,
        "dtype": "float32", "binary_path": str(binary), "binary_name": binary.name,
        "manifest_recorded_binary_sha256": "synthetic-binary-not-hashed",
        "binary_hash_policy": "orchestration-only empty file; never opened by runner",
    }
    config["crop"] = {"imin": 0, "imax": 80}
    config["field"] = {
        "path": str(field), "sha256": sha(field), "time_key": "time_s",
        "displacement_key": "rigid_displacement_um", "rounding_um": 20.0,
    }
    config["ops"] = {"path": str(ops_path), "sha256": sha(ops_path)}
    config["frozen_kilosort_settings"] = {
        "fs": 1000.0, "batch_size": 20, "nt": 3, "nt0min": 1,
        "do_CAR": False, "artifact_threshold": "Infinity", "invert_sign": False,
        "nblocks": 0, "Th_universal": 12, "Th_learned": 9,
        "data_dtype": "float32", "settings.nskip": 25,
        "settings.whitening_range": 32, "settings.highpass_cutoff": 300,
        "settings.templates_from_data": True, "settings.n_templates": 6,
        "settings.n_pcs": 6, "settings.Th_single_ch": 6,
    }
    config["native_invocation"] = {
        "filename": str(binary), "results_dir": str(run_root / "native_results"),
        "data_dtype": "float32", "do_CAR": False, "invert_sign": False,
        "device": "cpu", "save_extra_vars": True, "clear_cache": True,
        "save_preprocessed_copy": False,
        "settings": {
            "n_chan_bin": 4, "fs": 1000.0, "batch_size": 20,
            "tmin": 0.0, "tmax": 0.08, "nt": 3, "nt0min": 1,
            "nblocks": 0, "Th_universal": 12, "Th_learned": 9,
            "artifact_threshold": "Infinity", "nskip": 25,
            "whitening_range": 32, "highpass_cutoff": 300,
            "templates_from_data": True, "n_templates": 6, "n_pcs": 6,
            "Th_single_ch": 6,
        },
    }
    config["trained"]["run_root"] = str(run_root)
    config["trained"]["launch_evidence_dir"] = str(run_root / "launch_evidence")
    config["trained"]["pre_extraction_snapshot_dir"] = str(run_root / "pre_extraction_snapshot")
    if missing_key:
        del config["trained"]["required_snapshot_files"]
        config["source_bindings"] = [{"path": str(root / "must_not_open"), "sha256": "x" * 64}]
        config["recording"]["manifest_path"] = str(root / "must_not_open_manifest")
    config_path = root / "trained_fixture.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    return config_path, sha(config_path)


def invoke(config: Path, digest: str, root: Path, outcome: str) -> subprocess.CompletedProcess:
    return subprocess.run([
        str(REVIEWED_PYTHON), str(LAUNCHER), "--config", str(config),
        "--expected-config-sha256", digest, "--execute-trained",
        "--synthetic-fixture-root", str(root), "--synthetic-outcome", outcome,
    ], text=True, capture_output=True, check=False)


def actual_native_hook_fixture(root: Path) -> dict:
    root.mkdir()
    phase_state = trained_module.PhaseState()
    data = np.random.default_rng(2).normal(0, .1, (50, 4)).astype("float32")
    bfile = native_io.BinaryFiltered(
        filename="synthetic", n_chan_bin=4, fs=1000, NT=20, nt=3, nt0min=1,
        chan_map=np.arange(4), whiten_mat=torch.eye(4), do_CAR=False,
        device=torch.device("cpu"), file_object=data,
    )
    observed = {"first_batch_saw_snapshot": False}
    class CheckingBfile:
        def __init__(self): self.n_batches = bfile.n_batches
        def padded_batch_to_torch(self, *args, **kwargs):
            observed["first_batch_saw_snapshot"] = (root / "RECEIPT.json").is_file()
            if not observed["first_batch_saw_snapshot"]:
                raise RuntimeError("snapshot was not durable before first learned batch")
            return bfile.padded_batch_to_torch(*args, **kwargs)
    ops = {"settings": {"nearest_chans": 2, "position_limit": 100., "n_pcs": 1},
           "xc": np.zeros(4), "yc": np.arange(4) * 20., "nt": 3, "nt0min": 1,
           "batch_size": 20, "Th_learned": 100., "max_peels": 2,
           "wPCA": torch.tensor([[0., 1., 0.]]), "wTEMP": torch.tensor([[0., 1., 0.]]),
           "chanMap": np.arange(4)}
    wall3 = torch.ones((1, 1, 4), dtype=torch.float32)
    with trained_module.installed_phase_and_snapshot_hooks(
        native_preprocessing=preprocessing, native_spikedetect=spikedetect,
        native_template_matching=template_matching, phase_state=phase_state,
        snapshot_dir=root,
    ):
        st, features, returned = template_matching.extract(
            ops, CheckingBfile(), wall3, device=torch.device("cpu")
        )
    receipt = {"actual_native_prepare_extract_invoked": "iU" in returned,
               "actual_native_extract_returned": True,
               "first_batch_saw_snapshot": observed["first_batch_saw_snapshot"],
               "spikes": int(st.shape[0]), "feature_rows": int(features.shape[0]),
               "snapshot_receipt_sha256": sha(root / "RECEIPT.json")}
    (root / "HOOK_FIXTURE_RESULT.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cases = {}
    hook_receipt = actual_native_hook_fixture(output / "native_hook_fixture")
    success_root = output / "success_fixture"
    success_config, success_sha = build_fixture(success_root, "run")
    success = invoke(success_config, success_sha, success_root, "success")
    cases["success"] = {"returncode": success.returncode, "stdout": success.stdout,
                        "stderr": success.stderr, "config_sha256": success_sha}
    failure_root = output / "failure_fixture"
    failure_config, failure_sha = build_fixture(failure_root, "run")
    failure = invoke(failure_config, failure_sha, failure_root, "fail-after-whitening")
    cases["injected_downstream_failure"] = {
        "returncode": failure.returncode, "stdout": failure.stdout,
        "stderr": failure.stderr, "config_sha256": failure_sha,
    }
    corrupt_root = output / "corrupt_alignment_fixture"
    corrupt_config, corrupt_sha = build_fixture(corrupt_root, "run")
    corrupt = invoke(corrupt_config, corrupt_sha, corrupt_root, "corrupt-alignment")
    cases["corrupt_alignment"] = {
        "returncode": corrupt.returncode, "stdout": corrupt.stdout,
        "stderr": corrupt.stderr, "config_sha256": corrupt_sha,
        "failure_receipt_exists": (corrupt_root / "run/launch_evidence/run_failure.json").is_file(),
    }
    semantic_cases = {
        "permuted_ancestry": ("permuted-ancestry", "spike_times row ancestry differs"),
        "duplicate_kept": ("duplicate-kept", "integer kept_spikes indices must be unique"),
        "out_of_bounds_id": ("out-of-bounds-id", "spike_templates is outside templates axis 0"),
        "reordered_kept": ("reordered-kept", "integer kept_spikes indices must be strictly increasing"),
        "coordinated_permutation": ("coordinated-permutation", "integer kept_spikes indices must be strictly increasing"),
    }
    semantic_rejections = {}
    for case_name, (outcome, expected_message) in semantic_cases.items():
        case_root = output / f"{case_name}_fixture"
        case_config, case_sha = build_fixture(case_root, "run")
        result = invoke(case_config, case_sha, case_root, outcome)
        rejected = (
            result.returncode != 0
            and expected_message in result.stderr
            and (case_root / "run/launch_evidence/run_failure.json").is_file()
        )
        semantic_rejections[case_name] = rejected
        cases[case_name] = {
            "returncode": result.returncode, "stdout": result.stdout,
            "stderr": result.stderr, "config_sha256": case_sha,
            "failure_receipt_exists": (case_root / "run/launch_evidence/run_failure.json").is_file(),
        }
    missing_root = output / "missing_key_fixture"
    missing_config, missing_sha = build_fixture(missing_root, "run", missing_key=True)
    missing = invoke(missing_config, missing_sha, missing_root, "success")
    cases["missing_required_key"] = {
        "returncode": missing.returncode, "stdout": missing.stdout,
        "stderr": missing.stderr, "config_sha256": missing_sha,
        "sentinel_source_exists": (missing_root / "must_not_open").exists(),
        "sentinel_manifest_exists": (missing_root / "must_not_open_manifest").exists(),
    }
    success_complete = success_root / "run/COMPLETE.json"
    failure_receipt = failure_root / "run/launch_evidence/run_failure.json"
    receipt = {
        "schema": "trained-support-mask-vertical-fixture-v1",
        "orchestration_only": True,
        "no_recording_voltage_sort_training_detection": True,
        "success_reached_complete": success.returncode == 0 and success_complete.is_file(),
        "success_required_artifacts_validated": (
            json.loads(success_complete.read_text()).get("required_artifacts_validated") is True
            if success_complete.is_file() else False
        ),
        "injected_failure_preserved": failure.returncode != 0 and failure_receipt.is_file(),
        "corrupt_alignment_rejected": (
            corrupt.returncode != 0
            and "spike_clusters.npy is not aligned to spike_times.npy" in corrupt.stderr
            and (corrupt_root / "run/launch_evidence/run_failure.json").is_file()
        ),
        "equal_length_semantic_corruption_rejected": all(semantic_rejections.values()),
        "semantic_corruption_cases": semantic_rejections,
        "missing_key_failed_before_bound_paths": (
            missing.returncode != 0 and "missing required config key before data access" in missing.stderr
        ),
        "actual_native_hook_exercised": (
            hook_receipt["actual_native_prepare_extract_invoked"]
            and hook_receipt["actual_native_extract_returned"]
            and hook_receipt["first_batch_saw_snapshot"]
            and hook_receipt["spikes"] == hook_receipt["feature_rows"] == 0
        ),
        "actual_native_hook": hook_receipt,
        "cases": cases,
    }
    (output / "VERTICAL_FIXTURE_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return 0 if all(receipt[key] for key in (
        "success_reached_complete", "success_required_artifacts_validated",
        "injected_failure_preserved", "corrupt_alignment_rejected",
        "equal_length_semantic_corruption_rejected",
        "missing_key_failed_before_bound_paths", "actual_native_hook_exercised",
    )) else 1


if __name__ == "__main__":
    raise SystemExit(main())
