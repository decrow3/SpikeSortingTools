"""Saved-format integration fixture through the real composed CLI."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


PACKET = Path(__file__).resolve().parents[1]
SOURCE = PACKET / "source"
CONTRACT = PACKET / "configs" / "en_first_medium_saved_output_qualification.execution_disabled.v1.json"
sys.path.insert(0, str(SOURCE))

from pipeline.config import fingerprint
from testing.first_medium_evaluator import (
    KILOSORT_SOURCE_DIGEST,
    KILOSORT_SOURCE_FILES,
    spatial_event_lineage_sha256,
)
from testing.first_medium_saved_output_qualification import _cohort_digest
from testing.luke_amplitude_dropout_audit import read_curated_arrays


ORIGIN = 208_498_882
STOP = 17_999_901
FS = 29999.835983263598
SPATIAL_REGION = {
    "processing_depth_um": [1400.0, 2380.0],
    "scoring_depth_um": [1600.0, 2180.0],
    "minimum_edge_exclusion_um": 80.0,
    "source": "testing/configs/luke0804_hindsight_ladder_v1.json sha256 c0b564e0476f1af63a01c6c4efd3d0064471193329bf026e503f1bb25adc05d3",
}


def write_ref_saved_format(root: Path):
    curated = root / "REF384" / "sorter_output"
    qc = root / "REF384" / "qc" / "amp_truncation"
    curated.mkdir(parents=True)
    qc.mkdir(parents=True)
    local = np.array([100, 200, 300, 400, 1000, 1100, 1200, 1300], dtype=np.int64)
    clusters = np.array([1, 1, 1, 1, 2, 2, 2, 2], dtype=np.int64)
    positions = np.c_[np.zeros(8), np.full(8, 1800.0)].astype(np.float32)
    np.save(curated / "spike_times.npy", local + ORIGIN)
    np.save(curated / "spike_clusters.npy", clusters)
    np.save(curated / "full_st.npy", np.c_[local, np.zeros(8), np.full(8, 10.0)])
    np.save(curated / "kept_spikes.npy", np.arange(8, dtype=np.int64))
    np.save(curated / "spike_positions.npy", positions)
    np.save(curated / "ops.npy", {
        "nblocks": 0, "dshift": None, "chanMap": np.arange(384),
        "xc": np.tile([0.0, 32.0], 192), "yc": np.arange(384, dtype=float) * 10.0,
    }, allow_pickle=True)
    (curated / "cluster_KSLabel.tsv").write_text("cluster_id\tKSLabel\n1\tgood\n2\tgood\n")
    np.savez(
        qc / "truncation_qc.npz",
        cid=np.array([1, 1, 2, 2]),
        window_blocks=np.tile([[0, 1], [2, 3]], (2, 1)),
        popts=np.tile([[0.0, 1.0, 1.0]], (4, 1)),
        mpcts=np.full(4, 10.0),
    )
    raw, hashes = read_curated_arrays(curated)
    position_sha = hashlib.sha256((curated / "spike_positions.npy").read_bytes()).hexdigest()
    hashes["spike_positions.npy"] = position_sha
    labels = (curated / "cluster_KSLabel.tsv").read_bytes()
    identity = fingerprint({
        "curated": str(curated.resolve()), "files": hashes,
        "labels_sha256": hashlib.sha256(labels).hexdigest(),
    })
    ops = np.load(curated / "ops.npy", allow_pickle=True).item()
    geometry = np.stack([
        np.asarray(ops["xc"], dtype="<f8"), np.asarray(ops["yc"], dtype="<f8")
    ], axis=1)
    clock = {
        "saved_frame": "global_acquisition_samples",
        "crop_origin_global_frame": ORIGIN,
        "local_frames_half_open": [0, STOP],
        "global_frames_half_open": [ORIGIN, ORIGIN + STOP],
    }
    dependency_root = PACKET / "dependencies" / "kilosort" / KILOSORT_SOURCE_DIGEST
    spatial = {
        "schema": "evaluator-spatial-provenance-v1",
        "path": str((curated / "spike_positions.npy").resolve()),
        "sha256": position_sha,
        "ops_path": str((curated / "ops.npy").resolve()),
        "ops_sha256": hashlib.sha256((curated / "ops.npy").read_bytes()).hexdigest(),
        "row_count": 8,
        "event_lineage_sha256": spatial_event_lineage_sha256(
            local + ORIGIN, clusters, raw["full_st.npy"][raw["kept_spikes.npy"], 2], positions
        ),
        "reference_frame": "kilosort_probe_xy_feature_mass_unshifted_nblocks0",
        "units": "micrometers",
        "columns": [
            {"index": 0, "semantic": "x", "units": "micrometers"},
            {"index": 1, "semantic": "physical_probe_depth_y", "units": "micrometers"},
        ],
        "row_semantics": "exported_event_order_after_native_kept_spikes",
        "clock_binding": clock,
        "spatial_region": SPATIAL_REGION,
        "geometry_float64_sha256": hashlib.sha256(
            np.ascontiguousarray(geometry).tobytes()
        ).hexdigest(),
        "source_generation": {
            "function": "kilosort.postprocessing.compute_spike_positions",
            "dependency": {
                "schema": "content-addressed-python-source-root-v1",
                "root": str(dependency_root.resolve()),
                "digest_sha256": KILOSORT_SOURCE_DIGEST,
                "files": KILOSORT_SOURCE_FILES,
            },
        },
    }
    arm = {
        "arm_id": "REF384", "evidence_kind": "real_saved",
        "curated_output": str(curated), "qc_dir": str(qc.parent),
        "provenance": {
            "arm_contract_id": "REF384", "input_identity": "fixture",
            "clock_identity": "fixture", "geometry_identity": "fixture",
            "runtime_identity": "fixture", "config_identity": "fixture",
        },
        "expected_identity_digest": identity,
        "expected_qc_request_digest": hashlib.sha256(
            (qc / "truncation_qc.npz").read_bytes()
        ).hexdigest(),
        "clock_normalization": clock,
        "spatial_provenance": spatial,
    }
    return arm


def run_cli(request_path: Path, output: Path):
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": str(SOURCE), "PYTHONDONTWRITEBYTECODE": "1",
        "NUMBA_CACHE_DIR": str(request_path.parent / "numba-cache"),
        "MPLCONFIGDIR": str(request_path.parent / "mpl-cache"),
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "NUMBA_NUM_THREADS": "1",
    })
    return subprocess.run(
        [sys.executable, "-m", "testing.first_medium_saved_output_qualification",
         "--contract", str(CONTRACT), "--request", str(request_path),
         "--output", str(output)],
        cwd=request_path.parent, env=env, capture_output=True, text=True,
    )


def test_phase1_saved_format_fixture_through_real_cli(tmp_path):
    arm = write_ref_saved_format(tmp_path / "data")
    cohort = np.array([1, 2], dtype=np.int64)
    request = {
        "schema": "first-medium-saved-output-qualification-request-v1",
        "phase": "phase1_controls",
        "data_scope": "synthetic_saved_format_fixture",
        "fixture_token": "PACKAGED_SAVED_FORMAT_FIXTURE_ONLY_V1",
        "reference_cohort": cohort.tolist(),
        "reference_cohort_sha256": _cohort_digest(cohort),
        "arms": [arm],
    }
    request_path = tmp_path / "phase1-request.json"
    request_path.write_text(json.dumps(request))
    output = tmp_path / "phase1-output"
    result = run_cli(request_path, output)
    assert result.returncode == 0, result.stderr
    report = json.loads((output / "PHASE1_CONTROL_REPORT.json").read_text())
    assert report["label"] == "DIAGNOSTIC_CONTROLS_ONLY_NOT_EFFICACY"
    assert report["cross_arm_scalar_emitted"] is False
    assert report["candidate_ranked_output_emitted"] is False
    assert report["controls"]["ref_self_R_exact"] == 1.0
    assert report["controls"]["bijective_label_permutation_R_exact"] == 1.0
    assert report["controls"]["exclusive_duplicate_control_R_exact"] == 1.0
    assert report["controls"]["final_stop_rejected"] is True
    manifest_sha = hashlib.sha256((output / "MANIFEST.sha256").read_bytes()).hexdigest()
    complete = json.loads((output / "COMPLETE.json").read_text())
    assert complete["manifest_sha256"] == manifest_sha
    assert complete["scientific_advancement_authorized"] is False
    assert not (output / "PHASE2_EXISTING_B_REPORT.json").exists()

    retry = run_cli(request_path, output)
    assert retry.returncode != 0
    assert json.loads((output / "COMPLETE.json").read_text()) == complete


def test_phase2_is_disabled_even_for_fixture(tmp_path):
    request = {
        "schema": "first-medium-saved-output-qualification-request-v1",
        "phase": "phase2_existing_b",
        "data_scope": "synthetic_saved_format_fixture",
        "fixture_token": "PACKAGED_SAVED_FORMAT_FIXTURE_ONLY_V1",
        "reference_cohort": [1], "reference_cohort_sha256": "not-reached", "arms": [],
    }
    request_path = tmp_path / "phase2-request.json"
    request_path.write_text(json.dumps(request))
    output = tmp_path / "phase2-output"
    result = run_cli(request_path, output)
    assert result.returncode != 0
    assert "only phase1 is available to packaged fixtures" in result.stderr
    assert not output.exists()
