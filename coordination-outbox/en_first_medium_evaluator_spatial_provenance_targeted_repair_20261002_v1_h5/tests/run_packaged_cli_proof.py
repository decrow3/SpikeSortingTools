#!/usr/bin/env python3
"""Run the packet's CLI from an unrelated cwd against saved-format fixtures."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

PACKET = Path(__file__).resolve().parents[1]
SOURCE = PACKET / "source"
sys.path.insert(0, str(SOURCE))
from pipeline.config import fingerprint
from testing.first_medium_evaluator import (
    KILOSORT_SOURCE_DIGEST,
    KILOSORT_SOURCE_FILES,
    spatial_event_lineage_sha256,
)
from testing.luke_amplitude_dropout_audit import read_curated_arrays


def write_sort(root, name, offset):
    curated = root / name / "sorter_output"
    qc = root / name / "qc" / "amp_truncation"
    curated.mkdir(parents=True)
    qc.mkdir(parents=True)
    local = np.array([100, 200, 1100, 1200, 2100, 2200, 3100, 3200], dtype=np.int64)
    clusters = np.array([1, 2] * 4, dtype=np.int64) + offset
    np.save(curated / "spike_times.npy", local + 10_000)
    np.save(curated / "spike_clusters.npy", clusters)
    np.save(curated / "full_st.npy", np.c_[local, np.zeros(8), np.ones(8) * 10])
    np.save(curated / "kept_spikes.npy", np.arange(8, dtype=np.int64))
    np.save(curated / "spike_positions.npy", np.c_[np.zeros(8), np.ones(8) * 200])
    np.save(curated / "ops.npy", {
        "nblocks": 0, "dshift": None, "chanMap": np.arange(384),
        "xc": np.tile([0.0, 32.0], 192), "yc": np.arange(384, dtype=float) * 10.0,
    }, allow_pickle=True)
    (curated / "cluster_KSLabel.tsv").write_text(
        "cluster_id\tKSLabel\n" + "".join(f"{cid}\tgood\n" for cid in np.unique(clusters))
    )
    np.savez(qc / "truncation_qc.npz", cid=np.repeat(np.unique(clusters), 2),
             window_blocks=np.tile([[0, 1], [2, 3]], (2, 1)),
             popts=np.tile([[0., 1., 1.]], (4, 1)), mpcts=np.full(4, 10.))
    raw, hashes = read_curated_arrays(curated)
    hashes["spike_positions.npy"] = hashlib.sha256((curated / "spike_positions.npy").read_bytes()).hexdigest()
    labels = (curated / "cluster_KSLabel.tsv").read_bytes()
    identity = fingerprint({"curated": str(curated.resolve()), "files": hashes,
                            "labels_sha256": hashlib.sha256(labels).hexdigest()})
    qc_digest = hashlib.sha256((qc / "truncation_qc.npz").read_bytes()).hexdigest()
    positions = np.load(curated / "spike_positions.npy")
    ops = np.load(curated / "ops.npy", allow_pickle=True).item()
    geometry = np.stack([np.asarray(ops["xc"], dtype="<f8"),
                         np.asarray(ops["yc"], dtype="<f8")], axis=1)
    dependency_root = PACKET / "dependencies" / "kilosort" / KILOSORT_SOURCE_DIGEST
    clock = {"saved_frame": "global_acquisition_samples", "crop_origin_global_frame": 10000,
             "local_frames_half_open": [0, 4000], "global_frames_half_open": [10000, 14000]}
    spatial_region = {"processing_depth_um": [0.0, 400.0], "scoring_depth_um": [100.0, 300.0],
                      "minimum_edge_exclusion_um": 50.0, "source": "packaged known-answer fixture"}
    spatial = {
        "schema": "evaluator-spatial-provenance-v1",
        "path": str((curated / "spike_positions.npy").resolve()), "sha256": hashes["spike_positions.npy"],
        "ops_path": str((curated / "ops.npy").resolve()),
        "ops_sha256": hashlib.sha256((curated / "ops.npy").read_bytes()).hexdigest(),
        "row_count": 8,
        "event_lineage_sha256": spatial_event_lineage_sha256(
            local + 10_000, clusters, raw["full_st.npy"][raw["kept_spikes.npy"], 2], positions),
        "reference_frame": "kilosort_probe_xy_feature_mass_unshifted_nblocks0",
        "units": "micrometers",
        "columns": [{"index": 0, "semantic": "x", "units": "micrometers"},
                    {"index": 1, "semantic": "physical_probe_depth_y", "units": "micrometers"}],
        "row_semantics": "exported_event_order_after_native_kept_spikes",
        "clock_binding": clock, "spatial_region": spatial_region,
        "geometry_float64_sha256": hashlib.sha256(np.ascontiguousarray(geometry).tobytes()).hexdigest(),
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
    return curated, qc.parent, identity, qc_digest, spatial


with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    rows = []
    for name, offset in (("REF384", 0), ("existing_corrected_B384", 10)):
        curated, qc_dir, identity, qc_digest, spatial = write_sort(root / "data", name, offset)
        rows.append({
            "arm_id": name, "evidence_kind": "real_saved", "curated_output": str(curated),
            "qc_dir": str(qc_dir), "expected_identity_digest": identity,
            "expected_qc_request_digest": qc_digest,
            "clock_normalization": {"saved_frame": "global_acquisition_samples",
                "crop_origin_global_frame": 10000, "local_frames_half_open": [0, 4000],
                "global_frames_half_open": [10000, 14000]},
            "spatial_provenance": spatial,
            "provenance": {"arm_contract_id": name, "input_identity": "fixture",
                "clock_identity": "fixture", "geometry_identity": "fixture",
                "runtime_identity": "fixture", "config_identity": "fixture"},
        })
    rows.extend([
        {"arm_id": "repaired_B384", "evidence_kind": "unavailable", "provenance": {}},
        {"arm_id": "REF384_repeat", "evidence_kind": "unavailable", "provenance": {}},
    ])
    request = {"schema": "en-first-medium-four-arm-evaluator-request-v3-spatial-provenance",
        "execution_enabled": False, "evaluation_config": {
        "sampling_frequency_hz": 1000., "duration_s": 4., "correspondence_tolerance_ms": 1.,
        "minimum_correspondence_overlap": .1, "primary_retention": .5,
        "minimum_valid_amplitude_windows": 2, "minimum_common_time_fraction": .5,
        "minimum_measurable_unit_fraction": .5, "coincidence_tolerance_ms": 1.,
        "coincidence_depth_um": 75., "coincidence_seed": 4, "longitudinal_bin_s": 1.
    }, "spatial_region": {"processing_depth_um": [0.0, 400.0],
        "scoring_depth_um": [100.0, 300.0], "minimum_edge_exclusion_um": 50.0,
        "source": "packaged known-answer fixture"}, "arms": rows}
    request_path = root / "request.json"
    request_path.write_text(json.dumps(request))
    outside = root / "outside"
    outside.mkdir()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SOURCE)
    result = subprocess.run([sys.executable, "-m", "testing.first_medium_evaluator",
        "--request", str(request_path), "--output", str(root / "output")],
        cwd=outside, env=env, capture_output=True, text=True)
    if result.returncode:
        raise SystemExit(result.stderr)
    report = json.loads((root / "output" / "FOUR_ARM_REPORT.json").read_text())
    if not all(Path(row["path"]).is_relative_to(SOURCE)
               for row in report["implementation_provenance"].values()):
        raise SystemExit("workspace import leaked into packaged CLI")
    print(json.dumps({"status": "passed", "scientific_rows": list(report["scientific_scorecard"]),
                      "module_paths": report["implementation_provenance"]}, indent=2, sort_keys=True))
