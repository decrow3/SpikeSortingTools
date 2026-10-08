#!/usr/bin/env python3
"""Build the full execution-disabled trained config from the reviewed smoke base."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs/en_minimum_training_support_mask.covariance_smoke.enabled.review_pending.v2.json"
OUTPUT = ROOT / "configs/en_minimum_training_support_mask.trained.review_pending.v1.json"
DELTA_OUTPUT = ROOT / "configs/en_minimum_training_support_mask.trained.review_pending.v1.delta.json"
RUN_ROOT = ROOT / "en_minimum_training_support_mask_trained_medium_20261002_v1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def typed(value):
    if isinstance(value, dict):
        return {"type": "dict", "length": len(value)}
    if isinstance(value, list):
        return {"type": "list", "length": len(value)}
    return {"type": type(value).__name__, "value": value}


def nodes(value, pointer=""):
    result = {pointer: typed(value)}
    if isinstance(value, dict):
        for key, child in value.items():
            token = key.replace("~", "~0").replace("/", "~1")
            result.update(nodes(child, pointer + "/" + token))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            result.update(nodes(child, pointer + "/" + str(index)))
    return result


def main() -> None:
    base = json.loads(BASE.read_text())
    config = copy.deepcopy(base)
    config.update({
        "schema": "en-minimum-training-support-mask-v5-trained",
        "status": "execution_disabled_pending_independent_h1_delta_review",
        "execution_enabled": False,
        "failure_reason": "No trained real-data execution before separate H1 delta review and GO",
        "approval": {"h1_review_manifest_sha256": None},
        "orchestration_only": {
            "synthetic": False,
            "label": "none_real_future_execution_requires_separate_h1_go",
        },
        "comparators": {
            "REF384": {
                "root": "/media/huklaban5/Data/en_common_support_crop_screen_20260930_v1/REF384",
                "receipt_sha256": "e3798289f79e51545a2da5fefaee99684c7a667cf863d9ec7cc687f6d1164ac9",
                "status_sha256": "c0373d7d6caa22f6944d4bfa9071dbbb9b5291b838c7d07ca0cb95466d8022e7",
                "ops_sha256": "5b3b365a83babe1dc5842fb92ff5e56f35a93aefe2557a1cd31ca42b46d72f58",
                "pre_extraction_receipt_sha256": "06f82d7a057ad09c05a7895b0f1b4d0bc3c8dbd6cdc21f5ac1b268fd41f0d9b9",
                "recording_manifest_sha256": "2d15cf9da243a55b06358544f2436f118da2967ec410a7bb7792755351cc9a67",
                "arrays_copied": False,
                "raw_recording_rehashed": False,
            },
            "B384": {
                "root": "/media/huklaban5/Data/en_common_support_crop_screen_20260930_v1/B384",
                "receipt_sha256": "8a89ea4330871dce820329ca753890e12877b2695158ce9beddc77e9f190de3b",
                "status_sha256": "13d08afe1219dddeaa81f388ab65c1e8077f394460126a89bfad65878c0c603e",
                "ops_sha256": "7a8f5d219de1b98776134abb00f637777f85ba94113758ae47a1e4b26b2a44f0",
                "pre_extraction_receipt_sha256": "72e9b2742374afb91a798d246cf6b4a9fbe7696efe8c9195958cd9b154beb4f5",
                "recording_manifest_sha256": "46a3845008cb19d9c35a85e67841b308843e071f1a27605e87195a8d4d0a42cf",
                "arrays_copied": False,
                "raw_recording_rehashed": False,
            },
            "screen_complete_sha256": "30207b59113925b42081dfd367d2f580af72b5b7c1ed508141aa72cae2019bee",
            "screen_plan_receipt_sha256": "08bffeb44254ad1054ebe9e389b8d32877c778d5396addfa7483647a807c5d3a",
            "frozen_config_sha256": "bb6dd8f21539e5ee57f4216702b7c9d4c36d1a0feeefe3859ba25449af52ad17",
        },
        "trained": {
            "mode": "full_native_trained",
            "stop_after_whitening": False,
            "run_root": str(RUN_ROOT),
            "launch_evidence_dir": str(RUN_ROOT / "launch_evidence"),
            "pre_extraction_snapshot_dir": str(RUN_ROOT / "pre_extraction_snapshot"),
            "allow_saved_smoke_bank": False,
            "recompute_adaptive_state": [
                "covariance_C", "whitening_W", "wPCA", "wTEMP",
                "universal_detections_and_features", "Wall3", "iU", "iCC",
                "iCC_mask", "learned_detections_and_features", "final_clusters",
                "final_templates",
            ],
            "required_native_files": [
                "ops.npy", "channel_map.npy", "channel_positions.npy",
                "channel_shanks.npy", "spike_times.npy", "spike_clusters.npy",
                "spike_templates.npy", "spike_detection_templates.npy",
                "spike_positions.npy", "amplitudes.npy", "kept_spikes.npy",
                "templates.npy", "templates_ind.npy", "similar_templates.npy",
                "whitening_mat.npy", "whitening_mat_dat.npy",
                "whitening_mat_inv.npy", "pc_features.npy", "pc_feature_ind.npy",
                "cluster_Amplitude.tsv", "cluster_ContamPct.tsv",
                "cluster_KSLabel.tsv", "cluster_group.tsv", "params.py",
                "kilosort4.log", "tF.npy", "Wall.npy", "full_st.npy",
                "full_clu.npy", "full_amp.npy",
            ],
            "required_snapshot_files": [
                "Wall3.npy", "wPCA.npy", "wTEMP.npy", "chanMap.npy", "iU.npy",
                "iCC.npy", "iCC_mask.npy", "iU_physical_binary_channel.npy",
                "RECEIPT.json",
            ],
            "completion_written_last": True,
            "failure_stage_preserved": True,
        },
        "resources": {
            "wall_seconds_max": 1800,
            "ram_bytes_max": 68719476736,
            "gpu_count_max": 1,
            "cpu_threads_max": 16,
            "persistent_output_bytes_max": 8589934592,
            "estimate_seconds_per_600s_arm": [600, 720],
            "estimate_fresh_repaired_plus_ref_repeat_seconds": [1200, 1440],
            "estimate_basis": "observed 120-s smoke 100-103 s and full 10473.6-s 12156-12348 s",
        },
    })
    config.pop("smoke", None)
    config["base_binding"] = {"path": str(BASE), "sha256": sha(BASE)}
    config["native_invocation"]["results_dir"] = str(RUN_ROOT / "native_results")
    bindings = [
        ROOT / "npx_preprocessing/motion/kilosort_support_mask_candidate.py",
        ROOT / "npx_preprocessing/motion/prewhitening_support_mask.py",
        ROOT / "npx_preprocessing/motion/kilosort_support_mask_trained.py",
        ROOT / "testing/en_minimum_training_support_mask_trained_launch.py",
        ROOT / "environments/rescue-production/.venv/lib/python3.12/site-packages/kilosort/io.py",
        ROOT / "environments/rescue-production/.venv/lib/python3.12/site-packages/kilosort/preprocessing.py",
        ROOT / "environments/rescue-production/.venv/lib/python3.12/site-packages/kilosort/spikedetect.py",
        ROOT / "environments/rescue-production/.venv/lib/python3.12/site-packages/kilosort/template_matching.py",
        ROOT / "environments/rescue-production/.venv/lib/python3.12/site-packages/kilosort/run_kilosort.py",
    ]
    config["source_bindings"] = [{"path": str(path), "sha256": sha(path)} for path in bindings]
    OUTPUT.write_text(json.dumps(config, indent=2, sort_keys=False) + "\n")
    before = nodes(base)
    after = nodes(config)
    missing = {"type": "missing"}
    differences = [
        {"pointer": pointer, "base": before.get(pointer, missing),
         "trained": after.get(pointer, missing)}
        for pointer in sorted(set(before) | set(after))
        if before.get(pointer, missing) != after.get(pointer, missing)
    ]
    DELTA_OUTPUT.write_text(json.dumps({
        "schema": "exact-recursive-trained-delta-v1",
        "base_path": str(BASE), "base_sha256": sha(BASE),
        "trained_path": str(OUTPUT), "trained_sha256": sha(OUTPUT),
        "difference_count": len(differences), "differences": differences,
    }, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
