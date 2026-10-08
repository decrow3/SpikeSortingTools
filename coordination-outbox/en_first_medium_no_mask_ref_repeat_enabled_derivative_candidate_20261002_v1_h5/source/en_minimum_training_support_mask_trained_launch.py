#!/usr/bin/env python3
"""CLI boundary for the execution-disabled trained support-mask candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
KS = ROOT / "environments/rescue-production/.venv/lib/python3.12/site-packages"
for path in (ROOT, KS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from kilosort import io, preprocessing
from npx_preprocessing.motion import kilosort_support_mask_trained as trained


def _write_synthetic_native_files(results: Path, crop_origin: int) -> None:
    channel_map = np.arange(4, dtype=np.int32)
    values = {
        "ops.npy": {"chanMap": channel_map},
        "channel_map.npy": channel_map, "channel_positions.npy": np.c_[np.zeros(4), np.arange(4) * 20.],
        "channel_shanks.npy": np.zeros(4, dtype=np.int32),
        "spike_times.npy": np.array([1, 5, 9], dtype=np.int64) + crop_origin,
        "spike_clusters.npy": np.array([0, 1, 0], dtype=np.int32),
        "spike_templates.npy": np.array([0, 1, 0], dtype=np.int32),
        "spike_detection_templates.npy": np.array([0, 1, 0], dtype=np.int32),
        "spike_positions.npy": np.array([[0., 0.], [0., 20.], [0., 0.]]),
        "amplitudes.npy": np.ones(3), "kept_spikes.npy": np.array([True, True, False, True]),
        "templates.npy": np.ones((2, 3, 4)), "templates_ind.npy": np.tile(channel_map, (2, 1)),
        "similar_templates.npy": np.eye(2), "whitening_mat.npy": np.eye(4),
        "whitening_mat_dat.npy": np.eye(4), "whitening_mat_inv.npy": np.eye(4),
        "pc_features.npy": np.ones((3, 2, 2)), "pc_feature_ind.npy": np.array([[0, 1], [1, 2]]),
        "tF.npy": np.ones((3, 2, 2)), "Wall.npy": np.ones((2, 2, 4)),
        "full_st.npy": np.array([[1., 0., 1.], [5., 1., 1.], [7., 1., 1.], [9., 0., 1.]]),
        "full_clu.npy": np.array([0, 1, 1, 0], dtype=np.int32), "full_amp.npy": np.ones(4),
    }
    for name, value in values.items():
        np.save(results / name, value, allow_pickle=name == "ops.npy")
    for name in ("cluster_Amplitude.tsv", "cluster_ContamPct.tsv", "cluster_KSLabel.tsv", "cluster_group.tsv"):
        (results / name).write_text("cluster_id\tvalue\n0\tgood\n1\tgood\n")
    (results / "params.py").write_text("sample_rate = 1000.0\n")
    (results / "kilosort4.log").write_text("orchestration-only synthetic log\n")


def _write_synthetic_snapshot(snapshot: Path) -> None:
    values = {
        "Wall3.npy": np.ones((2, 2, 4)), "wPCA.npy": np.ones((2, 3)),
        "wTEMP.npy": np.ones((2, 3)), "chanMap.npy": np.arange(4, dtype=np.int32),
        "iU.npy": np.array([0, 2], dtype=np.int32),
        "iCC.npy": np.array([[0, 1, 2, 3], [1, 0, 1, 2]], dtype=np.int32),
        "iCC_mask.npy": np.ones((2, 4), dtype=bool),
        "iU_physical_binary_channel.npy": np.array([0, 2], dtype=np.int32),
    }
    for name, value in values.items():
        np.save(snapshot / name, value, allow_pickle=False)
    trained.atomic_json(snapshot / "RECEIPT.json", {
        "status": "orchestration_only_substitution",
        "saved_before_learned_batch_extraction": True,
        "wall3_axis0_binding": "spike_detection_templates.npy",
        "not_native_scientific_output": True,
        "files": {name: trained.sha256_file(snapshot / name) for name in values},
    })


def synthetic_orchestration_runner(config: dict, outcome: str):
    """Exercise native I/O hook and saves without sorting/training/detection."""
    def runner(**kwargs):
        settings = kwargs["settings"]
        n_channels = int(settings["n_chan_bin"])
        batch = int(settings["batch_size"])
        nt = int(settings["nt"])
        frames = max(batch * 4, batch + 2 * nt + 1)
        data = np.random.default_rng(714).normal(
            size=(frames, n_channels)
        ).astype(np.float32)
        bfile = io.BinaryFiltered(
            filename=kwargs["filename"], n_chan_bin=n_channels,
            fs=float(settings["fs"]), NT=batch, nt=nt,
            nt0min=int(settings["nt0min"]), chan_map=np.arange(n_channels),
            whiten_mat=None, do_CAR=bool(kwargs["do_CAR"]),
            artifact_threshold=float("inf"), device=torch.device("cpu"),
            file_object=data,
        )
        geometry = np.asarray(config["orchestration_only"]["geometry_um"], dtype=float)
        whitening = preprocessing.get_whitening_matrix(
            bfile, geometry[:, 0], geometry[:, 1], nskip=25, nrange=32
        )
        bfile.whiten_mat = whitening
        stamped = bfile.support_mask_policy.audit.batches
        for phase_name in (
            "pca_and_universal_template_learning",
            "universal_detection_and_features",
            "learned_template_extraction",
        ):
            stamped.phase.name = phase_name
            bfile.padded_batch_to_torch(0)
        if outcome == "fail-after-whitening":
            raise RuntimeError("injected synthetic downstream failure")
        snapshot = Path(config["trained"]["pre_extraction_snapshot_dir"])
        results = Path(kwargs["results_dir"])
        _write_synthetic_snapshot(snapshot)
        crop_origin = int(config["crop"]["imin"])
        _write_synthetic_native_files(results, crop_origin)
        if outcome == "corrupt-alignment":
            np.save(results / "spike_clusters.npy", np.array([0, 1], dtype=np.int32), allow_pickle=False)
        elif outcome == "permuted-ancestry":
            np.save(results / "spike_times.npy", np.array([5, 1, 9], dtype=np.int64) + crop_origin, allow_pickle=False)
        elif outcome == "duplicate-kept":
            np.save(results / "kept_spikes.npy", np.array([0, 0, 3], dtype=np.int64), allow_pickle=False)
        elif outcome == "out-of-bounds-id":
            np.save(results / "spike_templates.npy", np.array([0, 2, 0], dtype=np.int32), allow_pickle=False)
        elif outcome == "reordered-kept":
            np.save(results / "kept_spikes.npy", np.array([1, 0, 3], dtype=np.int64), allow_pickle=False)
        elif outcome == "coordinated-permutation":
            np.save(results / "kept_spikes.npy", np.array([1, 0, 3], dtype=np.int64), allow_pickle=False)
            for name in (
                "spike_times.npy", "spike_clusters.npy", "spike_templates.npy",
                "spike_detection_templates.npy", "spike_positions.npy",
                "amplitudes.npy", "pc_features.npy", "tF.npy",
            ):
                values = np.load(results / name, allow_pickle=False)
                np.save(results / name, values[[1, 0, 2]], allow_pickle=False)
        return {"status": "orchestration_only_success"}
    return runner


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--expected-config-sha256", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--execute-trained", action="store_true")
    parser.add_argument("--synthetic-fixture-root", type=Path)
    parser.add_argument(
        "--synthetic-outcome", choices=(
            "success", "fail-after-whitening", "corrupt-alignment",
            "permuted-ancestry", "duplicate-kept", "out-of-bounds-id",
            "reordered-kept", "coordinated-permutation",
        ),
        default="success",
    )
    args = parser.parse_args()
    synthetic = args.synthetic_fixture_root is not None
    config, contract = trained.validate_trained_contract(
        args.config.resolve(), expected_config_sha256=args.expected_config_sha256,
        allow_synthetic_orchestration=synthetic,
        fixture_root=args.synthetic_fixture_root,
    )
    report = {
        "status": "validated",
        "config_sha256": args.expected_config_sha256,
        "contract_digest": contract.contract_digest,
        "execution_enabled": contract.execution_enabled,
        "orchestration_only_synthetic": config["orchestration_only"]["synthetic"],
        "recording_binary_opened": False,
        "trained_mode": config["trained"]["mode"],
        "saved_smoke_bank_allowed": config["trained"]["allow_saved_smoke_bank"],
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.validate_only:
        return 0
    runner = None
    if synthetic:
        runner = synthetic_orchestration_runner(config, args.synthetic_outcome)
    result, complete = trained.run_native_trained(
        config_path=args.config.resolve(),
        expected_config_sha256=args.expected_config_sha256,
        allow_synthetic_orchestration=synthetic,
        fixture_root=args.synthetic_fixture_root,
        native_runner=runner,
    )
    print(json.dumps({"result": result, "complete": complete}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
