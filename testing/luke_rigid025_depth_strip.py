"""Prepare, verify, sort, and score Luke's rigid-0.25 depth-strip pilot.

The saved 96-channel source strip uses a zero-based clock, whereas DREDGE's
motion bins use acquisition-absolute seconds.  This script explicitly rebases
those bins before applying the holdout-qualified p=2/force-extrapolate warp.
It refuses to reuse outputs that lack their completion receipt.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from testing.luke_claimmask_window_sweep import ClaimSetting, build_sorter_params
from testing.luke_two_axis_pilot import (
    DEFAULT_REVIEW,
    PILOTS,
    assert_gpu_and_patch,
    score_pilot,
)

ROOT = Path(
    "/media/huklab/Data/NPX/Ryansorting/Luke/"
    "Luke0804_two_axis_pilot_imec1"
)
SOURCE = ROOT / "recordings/core_depth_strip"
TARGET = ROOT / "recordings/core_depth_strip_rigid025_p2"
MOTION_DIR = Path(
    "/mnt/NPX/Luke/20250804/"
    "dredge_pipeline_results_Luke0804_V2V1_g0_imec1/motion/dredge-motion"
)
SORT = ROOT / "sorts/core_depth_strip/rigid025_p2_single_ks_preprocessing_claim_off"
LOG = ROOT / "logs/core_depth_strip_rigid025_p2.log"
OUTPUT = Path("testing/outputs/luke_rigid025_depth_strip")
BASELINE_SCORE = ROOT / "scores/core_depth_strip/summary.json"
GAIN = 0.25
CLAIM_OFF = ClaimSetting("claim_off", 0.0, 0.0)


@dataclass(frozen=True)
class ConditionPaths:
    target: Path
    sort: Path
    log: Path
    output: Path
    score_name: str


def condition_paths(halo_channels: int) -> ConditionPaths:
    if halo_channels < 0:
        raise ValueError("halo_channels cannot be negative")
    if halo_channels == 0:
        return ConditionPaths(TARGET, SORT, LOG, OUTPUT, "core_depth_strip_rigid025_p2")
    label = f"rigid025_p2_halo{halo_channels}"
    return ConditionPaths(
        ROOT / f"recordings/core_depth_strip_{label}",
        ROOT / f"sorts/core_depth_strip/{label}_single_ks_preprocessing_claim_off",
        ROOT / f"logs/core_depth_strip_{label}.log",
        Path(f"testing/outputs/luke_{label}_depth_strip"),
        f"core_depth_strip_{label}",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_motion_bins(temporal_bins_s: np.ndarray) -> tuple[np.ndarray, float]:
    """Convert center-based absolute DREDGE bins to the strip's zero clock."""
    bins = np.asarray(temporal_bins_s, dtype=float)
    if bins.ndim != 1 or len(bins) < 2 or not np.all(np.diff(bins) > 0):
        raise ValueError("Motion time bins must be a strictly increasing vector")
    step = float(np.median(np.diff(bins)))
    acquisition_start_s = float(bins[0] - step / 2.0)
    return bins - acquisition_start_s, acquisition_start_s


def rigid025_motion_arrays(
    displacement: np.ndarray,
    temporal_bins_s: np.ndarray,
    spatial_bins_um: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    displacement = np.asarray(displacement, dtype=float)
    spatial_bins_um = np.asarray(spatial_bins_um, dtype=float)
    if displacement.shape != (len(temporal_bins_s), len(spatial_bins_um)):
        raise ValueError("Motion array dimensions do not match its bins")
    relative_bins, acquisition_start_s = relative_motion_bins(temporal_bins_s)
    rigid = GAIN * np.nanmedian(displacement, axis=1, keepdims=True)
    rigid_depth = np.asarray([np.nanmedian(spatial_bins_um)], dtype=float)
    return rigid, relative_bins, rigid_depth, acquisition_start_s


def required_motion_files() -> list[Path]:
    return [MOTION_DIR / name for name in ("motion.npy", "time_bins.npy", "depth_bins.npy")]


def prepare(n_jobs: int, halo_channels: int = 0, chunk_duration_s: float = 10.0) -> None:
    import spikeinterface.core as sc
    from spikeinterface.core.motion import Motion
    from spikeinterface.sortingcomponents.motion import interpolate_motion

    paths = condition_paths(halo_channels)
    receipt = paths.target / "motion_manifest.json"
    if paths.target.exists():
        if not receipt.exists():
            raise RuntimeError(f"Partial or ambiguous target recording: {paths.target}")
        print(f"Reusing completed recording: {paths.target}")
        return
    missing = [str(path) for path in [SOURCE / "pilot_manifest.json", *required_motion_files()] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing inputs: {missing}")
    if halo_channels:
        from testing.luke_two_axis_pilot import load_source_recording

        core_ids = sc.load(SOURCE).get_channel_ids().astype(int)
        left_ids = np.arange(int(core_ids[0]) - halo_channels, int(core_ids[0]))
        right_ids = np.arange(int(core_ids[-1]) + 1, int(core_ids[-1]) + halo_channels + 1)
        expanded_ids = np.r_[left_ids, core_ids, right_ids]
        if expanded_ids[0] < 0 or expanded_ids[-1] >= 384:
            raise ValueError("Requested halo extends beyond the physical probe")
        # Reuse the verified conditioned core.  The halo excludes known bad
        # channel 191, so its matching legacy path is phase shift + saturation
        # blanking without the unnecessary all-channel interpolation wrapper.
        halo = load_source_recording(
            np.r_[left_ids, right_ids],
            conditioning_policy="legacy_no_bad_interpolation",
        )
        left = halo.channel_slice(channel_ids=left_ids)
        right = halo.channel_slice(channel_ids=right_ids)
        recording = sc.aggregate_channels([left, sc.load(SOURCE), right])
        if not np.array_equal(recording.get_channel_ids(), expanded_ids):
            raise RuntimeError("Aggregated halo/core channel order is incorrect")
    else:
        core_ids = sc.load(SOURCE).get_channel_ids().astype(int)
        expanded_ids = core_ids
        recording = sc.load(SOURCE)
    displacement = np.load(MOTION_DIR / "motion.npy")
    absolute_bins = np.load(MOTION_DIR / "time_bins.npy")
    depth_bins = np.load(MOTION_DIR / "depth_bins.npy")
    rigid, relative_bins, rigid_depth, acquisition_start_s = rigid025_motion_arrays(
        displacement, absolute_bins, depth_bins
    )
    duration_s = recording.get_num_samples() / recording.get_sampling_frequency()
    if relative_bins[0] > 1.0 or abs(relative_bins[-1] - (duration_s - 0.5)) > 1.1:
        raise RuntimeError(
            f"Rebased motion does not span recording: {relative_bins[[0, -1]]} vs {duration_s}"
        )
    motion = Motion(rigid, relative_bins, rigid_depth)
    corrected = interpolate_motion(
        recording.astype("float32"),
        motion,
        border_mode="force_extrapolate",
        spatial_interpolation_method="kriging",
        sigma_um=20.0,
        p=2,
    ).astype("int16")
    if halo_channels:
        corrected = corrected.channel_slice(channel_ids=core_ids)
    paths.target.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if chunk_duration_s <= 0:
        raise ValueError("chunk_duration_s must be positive")
    corrected.save(
        folder=paths.target,
        n_jobs=n_jobs,
        chunk_duration=f"{chunk_duration_s}s",
        progress_bar=True,
    )
    elapsed_s = time.perf_counter() - started
    manifest = {
        "condition": "rigid_gain025_p2_force_extrapolate",
        "source_recording": str(SOURCE),
        "source_manifest_sha256": sha256_file(SOURCE / "pilot_manifest.json"),
        "motion_dir": str(MOTION_DIR),
        "motion_input_sha256": {path.name: sha256_file(path) for path in required_motion_files()},
        "field_transform": "0.25 times depthwise nanmedian at each time bin",
        "halo_channels_each_side": halo_channels,
        "interpolation_input_channel_ids": expanded_ids.tolist(),
        "saved_core_channel_ids": core_ids.tolist(),
        "boundary_support": (
            "motion interpolation on expanded real-voltage strip, then crop to core"
            if halo_channels
            else "motion interpolation directly on core; retained only as boundary diagnostic"
        ),
        "source_clock": "zero-based saved strip",
        "motion_original_clock": "acquisition-absolute seconds",
        "inferred_acquisition_start_s": acquisition_start_s,
        "relative_motion_time_range_s": [float(relative_bins[0]), float(relative_bins[-1])],
        "rigid_displacement_range_um": [float(np.nanmin(rigid)), float(np.nanmax(rigid))],
        "interpolation": {"method": "kriging", "sigma_um": 20.0, "p": 2, "border_mode": "force_extrapolate"},
        "input_dtype": "float32",
        "saved_dtype": "int16",
        "n_jobs": n_jobs,
        "chunk_duration_s": chunk_duration_s,
        "materialization_runtime_s": elapsed_s,
        "expected_binary_bytes": int(recording.get_num_samples() * len(core_ids) * 2),
    }
    receipt.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Completed recording in {elapsed_s / 60:.1f} min: {paths.target}")


def verify(chunk_s: float = 1.0, halo_channels: int = 0) -> dict:
    import spikeinterface.core as sc

    paths = condition_paths(halo_channels)
    manifest_path = paths.target / "motion_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    source, target = sc.load(SOURCE), sc.load(paths.target)
    structural = {
        "same_num_samples": source.get_num_samples() == target.get_num_samples(),
        "same_num_channels": source.get_num_channels() == target.get_num_channels(),
        "same_channel_ids": np.array_equal(source.get_channel_ids(), target.get_channel_ids()),
        "same_channel_locations": np.allclose(source.get_channel_locations(), target.get_channel_locations()),
        "same_sampling_frequency": source.get_sampling_frequency() == target.get_sampling_frequency(),
        "target_dtype_int16": np.dtype(target.get_dtype()) == np.dtype("int16"),
    }
    raw_path = paths.target / "traces_cached_seg0.raw"
    structural["exact_binary_size"] = raw_path.stat().st_size == manifest["expected_binary_bytes"]
    if not all(structural.values()):
        raise RuntimeError(f"Structural integrity failure: {structural}")
    fs = float(source.get_sampling_frequency())
    width = max(1, int(round(chunk_s * fs)))
    last = source.get_num_samples() - width
    fractions = np.asarray([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    rows = []
    for fraction in fractions:
        start = int(round(fraction * last))
        a = source.get_traces(start_frame=start, end_frame=start + width).astype(np.float32)
        b = target.get_traces(start_frame=start, end_frame=start + width).astype(np.float32)
        delta = b - a
        rows.append({
            "fraction": float(fraction),
            "start_frame": start,
            "fraction_changed": float(np.mean(delta != 0)),
            "median_abs_change_counts": float(np.median(np.abs(delta))),
            "p99_abs_change_counts": float(np.quantile(np.abs(delta), 0.99)),
            "max_abs_change_counts": float(np.max(np.abs(delta))),
            "target_to_source_std_ratio": float(np.std(b) / np.std(a)) if np.std(a) else None,
            "source_zero_fraction": float(np.mean(a == 0)),
            "target_zero_fraction": float(np.mean(b == 0)),
        })
    if max(row["fraction_changed"] for row in rows) == 0:
        raise RuntimeError("Corrected recording is identical to its source in all verification chunks")
    receipt = {"structural_checks": structural, "sampled_chunk_duration_s": chunk_s, "sampled_chunks": rows}
    paths.output.mkdir(parents=True, exist_ok=True)
    (paths.output / "integrity_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return receipt


def run_sort(halo_channels: int = 0) -> None:
    import spikeinterface.core as sc
    from spikeinterface.sorters import run_sorter

    assert_gpu_and_patch()
    paths = condition_paths(halo_channels)
    result = paths.sort / "sorter_output/spike_times.npy"
    if result.exists():
        print(f"Reusing completed sort: {paths.sort}")
        return
    if paths.sort.exists():
        raise RuntimeError(f"Partial or ambiguous sort: {paths.sort}")
    if not (paths.output / "integrity_receipt.json").exists():
        raise FileNotFoundError("Run --verify before sorting")
    params = build_sorter_params(CLAIM_OFF)
    paths.sort.parent.mkdir(parents=True, exist_ok=True)
    paths.log.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(paths.log)
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    try:
        with paths.log.open("a") as log_file, contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            run_sorter("kilosort4", sc.load(paths.target), folder=str(paths.sort), verbose=True, remove_existing_folder=False, **params)
    finally:
        root_logger.removeHandler(handler)
        handler.close()
    print(f"Completed sort: {paths.sort}; log: {paths.log}")


def compare_summaries(baseline: dict, candidate: dict) -> dict:
    keys = (
        "n_final_spikes", "n_units", "n_ks_good", "median_contamination_pct",
        "cross_unit_coincidence_excess", "median_unit_refractory_violation_fraction",
        "spike_rate_cv_across_time_bins", "edge_spike_fraction_within_40um",
        "neural_unmatched_recovery", "neural_unmatched_recovery_excess",
    )
    deltas = {key: candidate[key] - baseline[key] for key in keys}
    gates = {
        "recovery_within_one_legacy_event": candidate["neural_unmatched_recovery"] >= baseline["neural_unmatched_recovery"] - 1 / 44,
        "ks_good_not_lower": candidate["n_ks_good"] >= baseline["n_ks_good"],
        "contamination_not_higher": candidate["median_contamination_pct"] <= baseline["median_contamination_pct"],
        "coincidence_excess_not_higher": candidate["cross_unit_coincidence_excess"] <= baseline["cross_unit_coincidence_excess"],
        "refractory_violations_not_higher": candidate["median_unit_refractory_violation_fraction"] <= baseline["median_unit_refractory_violation_fraction"],
        "edge_fraction_not_higher": candidate["edge_spike_fraction_within_40um"] <= baseline["edge_spike_fraction_within_40um"],
    }
    return {
        "baseline": baseline,
        "candidate": candidate,
        "candidate_minus_baseline": deltas,
        "prespecified_descriptive_gates": gates,
        "n_gates_passed": int(sum(gates.values())),
        "n_gates": len(gates),
        "strict_nondominance_pass": bool(all(gates.values())),
    }


def score(halo_channels: int = 0) -> dict:
    paths = condition_paths(halo_channels)
    candidate = score_pilot(
        PILOTS["core_depth_strip"], ROOT, DEFAULT_REVIEW, 300.0,
        conditioning_policy="legacy", result_override=paths.sort / "sorter_output",
        score_name=paths.score_name, log_override=paths.log,
    )
    baseline = json.loads(BASELINE_SCORE.read_text())
    comparison = compare_summaries(baseline, candidate)
    paths.output.mkdir(parents=True, exist_ok=True)
    (paths.output / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    print(json.dumps(comparison, indent=2))
    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument(
        "--chunk-duration-s",
        type=float,
        default=10.0,
        help="Materialization chunk size; use longer chunks to amortize lazy filter setup.",
    )
    parser.add_argument(
        "--halo-channels",
        type=int,
        default=0,
        help="Real-voltage channels added on each side during interpolation, then cropped.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/luke-rigid025-depth-strip-numba")
    if not (args.prepare or args.verify or args.run or args.score):
        raise SystemExit("Choose --prepare, --verify, --run, and/or --score")
    if args.n_jobs < 1:
        raise ValueError("--n-jobs must be positive")
    if args.prepare:
        prepare(args.n_jobs, args.halo_channels, args.chunk_duration_s)
    if args.verify:
        verify(halo_channels=args.halo_channels)
    if args.run:
        run_sort(args.halo_channels)
    if args.score:
        score(args.halo_channels)


if __name__ == "__main__":
    main()
