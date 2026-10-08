#!/usr/bin/env python3
"""Frozen miniature Candidate-3 input adapter and DARTsort stage exercise."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/candidate3-mini-e2e-numba")

import h5py
import numpy as np


FS = 29999.835983263598
N_FRAMES = 297_000
N_CHANNELS = 24
SEED = 38172
DURATION_S = N_FRAMES / FS


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prepare(output: Path) -> None:
    import torch
    from kilosort.io import BinaryFiltered
    from kilosort.preprocessing import get_highpass_filter
    from npx_preprocessing.motion import ExactLatticeRemapRecording
    from spikeinterface.core import NumpyRecording

    output.mkdir(parents=True, exist_ok=False)
    rg = np.random.default_rng(SEED)
    traces = np.rint(rg.normal(0.0, 20.0, size=(N_FRAMES, N_CHANNELS))).astype(np.int16)
    truth = []
    t = np.arange(61) - 20
    temporal = -np.exp(-0.5 * (t / 3.2) ** 2) + 0.28 * np.exp(-0.5 * ((t - 9) / 6.0) ** 2)
    for unit in range(12):
        channel = 1 + 2 * (unit % 11)
        amp = 180.0 + 10.0 * unit
        start = 1500 + 83 * unit
        interval = 1500 + 7 * unit
        for sample in range(start, N_FRAMES - 1500, interval):
            spatial = np.exp(-0.5 * ((np.arange(N_CHANNELS) - channel) / 1.15) ** 2)
            wf = amp * temporal[:, None] * spatial[None, :]
            sl = slice(sample - 20, sample + 41)
            traces[sl] = np.clip(traces[sl].astype(np.float64) + wf, -32768, 32767).astype(np.int16)
            truth.append((sample, unit, channel))

    geom = np.c_[np.zeros(N_CHANNELS), 20.0 * np.arange(N_CHANNELS)]
    base = NumpyRecording(traces, sampling_frequency=FS)
    base.set_channel_locations(geom)
    truth_array = np.asarray(sorted(truth), dtype=np.int64)
    np.save(output / "truth_events.npy", truth_array, allow_pickle=False)

    arms = {"ordinary": [0.0, 0.0], "transition": [0.0, 20.0]}
    receipts = {}
    for arm, shifts in arms.items():
        remapped = ExactLatticeRemapRecording(
            base,
            temporal_bin_centers_s=np.array([DURATION_S / 4.0, 3.0 * DURATION_S / 4.0]),
            shifts_um=np.asarray(shifts),
            cell_width_s=DURATION_S / 2.0,
            direction="y",
            time_origins_s=[0.0],
        )
        remapped_traces = remapped.get_traces(start_frame=0, end_frame=N_FRAMES)
        producer = object.__new__(BinaryFiltered)
        producer.chan_map = np.arange(N_CHANNELS)
        producer.invert_sign = False
        producer.do_CAR = True
        producer.hp_filter = get_highpass_filter(fs=FS, cutoff=300, device=torch.device("cpu"))
        producer.artifact_threshold = np.inf
        producer.whiten_mat = None
        producer.dshift = None
        producer.device = torch.device("cpu")
        boundary = BinaryFiltered.filter(
            producer, torch.from_numpy(remapped_traces.T.copy()).float()
        ).numpy(force=True).T
        path = output / f"{arm}_prewhitening.npy"
        np.save(path, boundary.astype(np.float32, copy=False), allow_pickle=False)
        receipts[arm] = {
            "path": str(path),
            "sha256": sha256(path),
            "shape": list(boundary.shape),
            "dtype": str(boundary.dtype),
            "finite": int(np.isfinite(boundary).sum()),
            "minimum": float(boundary.min()),
            "maximum": float(boundary.max()),
            "rms": float(np.sqrt(np.mean(boundary.astype(np.float64) ** 2))),
            "zero_count": int(np.count_nonzero(boundary == 0)),
            "shifts_um": shifts,
        }
    receipt = {
        "schema": "candidate3-mini-input-v1",
        "seed": SEED,
        "sampling_frequency": FS,
        "frames": N_FRAMES,
        "duration_s": DURATION_S,
        "channels": N_CHANNELS,
        "truth_events": int(len(truth_array)),
        "truth_sha256": sha256(output / "truth_events.npy"),
        "arms": receipts,
        "adapter_source": inspect.getsourcefile(ExactLatticeRemapRecording),
        "adapter_source_sha256": sha256(Path(inspect.getsourcefile(ExactLatticeRemapRecording))),
        "kilosort_source": inspect.getsourcefile(BinaryFiltered),
        "kilosort_source_sha256": sha256(Path(inspect.getsourcefile(BinaryFiltered))),
    }
    (output / "INPUT_RECEIPT.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


def run_arm(input_dir: Path, output: Path, arm: str) -> None:
    import dartsort
    from spikeinterface.core import NumpyRecording

    boundary = np.load(input_dir / f"{arm}_prewhitening.npy", mmap_mode="r")
    rec = NumpyRecording(boundary, sampling_frequency=FS)
    rec.set_channel_locations(np.c_[np.zeros(N_CHANNELS), 20.0 * np.arange(N_CHANNELS)])
    cfg = dartsort.DARTsortInternalConfig(
        detection_type="threshold",
        initial_detection_cfg=dartsort.ThresholdingConfig(
            chunk_length_samples=30_000,
            detection_threshold=40.0,
            peak_sign="neg",
            spatial_dedup_radius_um=60.0,
        ),
        peeler_sampling_cfg=dartsort.FitSamplingConfig(
            max_waveforms_fit=3000,
            n_waveforms_fit=2000,
            n_residual_snips=512,
            seed=0,
            n_seconds_fit=10,
        ),
        featurization_cfg=dartsort.FeaturizationConfig(
            do_nn_denoise=False,
            do_tpca_denoise=True,
            nn_localization=False,
            tpca_rank=5,
            tpca_max_waveforms=3000,
        ),
        clustering_features_cfg=dartsort.ClusteringFeaturesConfig(n_main_channel_pcs=2),
        motion_estimation_cfg=dartsort.MotionEstimationConfig(do_motion_estimation=False),
        computation_cfg=dartsort.ComputationConfig(
            n_jobs_cpu=0,
            n_jobs_gpu=0,
            n_jobs_small=0,
            device="cpu",
        ),
        matching_iterations=1,
        save_intermediate_features=True,
        save_intermediate_labels=True,
        preprocessing="none",
        work_in_tmpdir=False,
        save_everything_on_error=False,
    )
    output.mkdir(parents=True, exist_ok=False)
    result = dartsort.dartsort(rec, output, cfg=cfg, overwrite=False)
    sorting = result["sorting"]
    receipt = {
        "arm": arm,
        "final_events": int(len(sorting)),
        "final_units_nonnegative": int(len(np.unique(sorting.labels[sorting.labels >= 0]))),
        "final_noise_events": int(np.count_nonzero(sorting.labels < 0)),
        "parent_h5": str(sorting.parent_h5_path),
    }
    (output / "RUN_RECEIPT.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


def summarize(input_dir: Path, runs_dir: Path, output: Path) -> None:
    import dartsort

    summary = {"input": json.loads((input_dir / "INPUT_RECEIPT.json").read_text()), "arms": {}}
    for arm in ("ordinary", "transition"):
        arm_dir = runs_dir / arm
        arm_summary = {"files": [], "hdf5": {}, "npz": {}}
        for path in sorted(arm_dir.rglob("*")):
            if not path.is_file():
                continue
            rel = str(path.relative_to(arm_dir))
            arm_summary["files"].append({"path": rel, "size": path.stat().st_size})
            if path.suffix == ".h5":
                with h5py.File(path, "r") as h5:
                    datasets = {
                        name: {"shape": list(ds.shape), "dtype": str(ds.dtype)}
                        for name, ds in h5.items()
                        if isinstance(ds, h5py.Dataset)
                    }
                    arm_summary["hdf5"][rel] = datasets
            elif path.suffix == ".npz":
                with np.load(path, allow_pickle=False) as z:
                    arm_summary["npz"][rel] = {
                        name: {"shape": list(z[name].shape), "dtype": str(z[name].dtype)}
                        for name in z.files
                    }
        final_path = arm_dir / "dartsort_sorting.npz"
        reloaded = dartsort.DARTsortSorting.from_npz(final_path)
        arm_summary["reload"] = {
            "events": int(len(reloaded)),
            "units_nonnegative": int(len(np.unique(reloaded.labels[reloaded.labels >= 0]))),
            "noise_events": int(np.count_nonzero(reloaded.labels < 0)),
            "times_sorted": bool(np.all(np.diff(reloaded.times_samples) >= 0)),
        }
        summary["arms"][arm] = arm_summary
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("output", type=Path)
    run = sub.add_parser("run")
    run.add_argument("input_dir", type=Path)
    run.add_argument("output", type=Path)
    run.add_argument("arm", choices=("ordinary", "transition"))
    summ = sub.add_parser("summarize")
    summ.add_argument("input_dir", type=Path)
    summ.add_argument("runs_dir", type=Path)
    summ.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.output)
    elif args.command == "run":
        run_arm(args.input_dir, args.output, args.arm)
    else:
        summarize(args.input_dir, args.runs_dir, args.output)
