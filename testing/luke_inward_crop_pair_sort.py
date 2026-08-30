"""Run the paired full-duration inward-crop test for Luke rigid-0.25.

Both existing 96-channel recordings are sliced to physical channel ids
184--263 before Kilosort.  For the corrected condition, the removed eight
channels on each side therefore act as real-voltage interpolation support;
the no-motion input is cropped identically.  No voltage is resampled here.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from testing.luke_claimmask_window_sweep import ClaimSetting, build_sorter_params
from testing.luke_rigid025_depth_strip import compare_summaries
from testing.luke_two_axis_pilot import (
    DEFAULT_REVIEW,
    Pilot,
    assert_gpu_and_patch,
    pilot_channel_ids,
    score_pilot,
)

ROOT = Path(
    "/media/huklab/Data/NPX/Ryansorting/Luke/"
    "Luke0804_two_axis_pilot_imec1"
)
RECORDINGS = {
    "no_motion": ROOT / "recordings/core_depth_strip",
    "rigid025_p2": ROOT / "recordings/core_depth_strip_rigid025_p2",
}
SORT_ROOT = ROOT / "sorts/core_depth_strip_interior80"
LOG_ROOT = ROOT / "logs/core_depth_strip_interior80"
CROPPED_RECORDING_ROOT = ROOT / "recordings/core_depth_strip_interior80"
OUTPUT = Path("testing/outputs/luke_inward_crop_pair")
MARGIN_CHANNELS = 8
CORE_FIRST = 176
CORE_COUNT = 96
RETAINED_IDS = np.arange(CORE_FIRST + MARGIN_CHANNELS, CORE_FIRST + CORE_COUNT - MARGIN_CHANNELS)
CLAIM_OFF = ClaimSetting("claim_off", 0.0, 0.0)
PILOT = Pilot(
    name="core_depth_strip_interior80",
    axis="depth",
    role="full-duration 80-channel interior; 80-um real-voltage support removed from each source edge",
    first_channel=int(RETAINED_IDS[0]),
    n_channels=len(RETAINED_IDS),
)


def sort_path(condition: str) -> Path:
    return SORT_ROOT / f"{condition}_single_ks_preprocessing_claim_off"


def log_path(condition: str) -> Path:
    return LOG_ROOT / f"{condition}.log"


def cropped_recording_path(condition: str) -> Path:
    return CROPPED_RECORDING_ROOT / condition


def verify_inputs(chunk_s: float = 1.0) -> dict:
    import spikeinterface.core as sc

    if chunk_s <= 0:
        raise ValueError("chunk_s must be positive")
    recordings = {name: sc.load(path) for name, path in RECORDINGS.items()}
    reference = recordings["no_motion"]
    checks = {}
    for name, recording in recordings.items():
        checks[name] = {
            "frames_match": bool(recording.get_num_samples() == reference.get_num_samples()),
            "channels_match": bool(np.array_equal(recording.get_channel_ids(), reference.get_channel_ids())),
            "locations_match": bool(np.allclose(recording.get_channel_locations(), reference.get_channel_locations())),
            "sampling_frequency_match": bool(recording.get_sampling_frequency() == reference.get_sampling_frequency()),
            "dtype_int16": bool(np.dtype(recording.get_dtype()) == np.dtype("int16")),
            "retained_ids_present": bool(np.isin(RETAINED_IDS, recording.get_channel_ids()).all()),
        }
    if not all(all(values.values()) for values in checks.values()):
        raise RuntimeError(f"Input structural checks failed: {checks}")
    locations = reference.get_channel_locations()
    source_ids = reference.get_channel_ids()
    retained_locations = reference.channel_slice(RETAINED_IDS).get_channel_locations()
    source_depth_min = float(locations[:, 1].min())
    source_depth_max = float(locations[:, 1].max())
    retained_depth_min = float(retained_locations[:, 1].min())
    retained_depth_max = float(retained_locations[:, 1].max())
    support = {
        "source_channel_ids": [int(source_ids[0]), int(source_ids[-1])],
        "retained_channel_ids": [int(RETAINED_IDS[0]), int(RETAINED_IDS[-1])],
        "n_source_channels": int(len(source_ids)),
        "n_retained_channels": int(len(RETAINED_IDS)),
        "lower_support_um": retained_depth_min - source_depth_min,
        "upper_support_um": source_depth_max - retained_depth_max,
        "kriging_sigma_um": 20.0,
        "support_in_sigma": min(
            retained_depth_min - source_depth_min,
            source_depth_max - retained_depth_max,
        ) / 20.0,
    }
    if support["support_in_sigma"] < 4.0:
        raise RuntimeError(f"Insufficient inward support: {support}")
    fs = float(reference.get_sampling_frequency())
    width = int(round(chunk_s * fs))
    last = reference.get_num_samples() - width
    chunks = []
    no_motion = reference.channel_slice(RETAINED_IDS)
    corrected = recordings["rigid025_p2"].channel_slice(RETAINED_IDS)
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        start = int(round(fraction * last))
        a = no_motion.get_traces(start_frame=start, end_frame=start + width).astype(np.float32)
        b = corrected.get_traces(start_frame=start, end_frame=start + width).astype(np.float32)
        delta = b - a
        chunks.append({
            "fraction": fraction,
            "start_frame": start,
            "fraction_changed": float(np.mean(delta != 0)),
            "p99_abs_change_counts": float(np.quantile(np.abs(delta), 0.99)),
            "target_to_source_std_ratio": float(np.std(b) / np.std(a)),
        })
    if max(row["fraction_changed"] for row in chunks) == 0:
        raise RuntimeError("Corrected interior is identical to no motion")
    receipt = {
        "design": "crop both cached 96-channel voltages before sorting; no resampling in this script",
        "structural_checks": checks,
        "spatial_support": support,
        "sampled_chunks": chunks,
        "passed": True,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "input_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return receipt


def run_condition(condition: str) -> None:
    import spikeinterface.core as sc
    from spikeinterface.sorters import run_sorter

    if condition not in RECORDINGS:
        raise ValueError(f"Unknown condition: {condition}")
    if not (OUTPUT / "materialized_receipt.json").exists():
        raise FileNotFoundError("Run --verify-materialized before sorting")
    assert_gpu_and_patch()
    target = sort_path(condition)
    result = target / "sorter_output/spike_times.npy"
    if result.exists():
        print(f"Reusing completed sort: {target}")
        return
    if target.exists():
        raise RuntimeError(f"Partial or ambiguous sort: {target}")
    cropped_path = cropped_recording_path(condition)
    if not (cropped_path / "crop_manifest.json").exists():
        raise FileNotFoundError(f"Materialize the cropped recording first: {cropped_path}")
    recording = sc.load(cropped_path)
    if not np.array_equal(recording.get_channel_ids(), RETAINED_IDS):
        raise RuntimeError("Kilosort input channel order is incorrect")
    params = build_sorter_params(CLAIM_OFF)
    target.parent.mkdir(parents=True, exist_ok=True)
    log = log_path(condition)
    log.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log)
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    try:
        with log.open("a") as log_file, contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            run_sorter(
                "kilosort4", recording, folder=str(target), verbose=True,
                remove_existing_folder=False, **params,
            )
    finally:
        root_logger.removeHandler(handler)
        handler.close()
    print(f"Completed {condition}: {target}; log: {log}")


def materialize_condition(condition: str, n_jobs: int = 8) -> None:
    import spikeinterface.core as sc

    if condition not in RECORDINGS:
        raise ValueError(f"Unknown condition: {condition}")
    if n_jobs < 1:
        raise ValueError("n_jobs must be positive")
    if not (OUTPUT / "input_receipt.json").exists():
        raise FileNotFoundError("Run --verify before materializing")
    target = cropped_recording_path(condition)
    manifest = target / "crop_manifest.json"
    if target.exists():
        if not manifest.exists():
            raise RuntimeError(f"Partial or ambiguous cropped recording: {target}")
        print(f"Reusing cropped recording: {target}")
        return
    source = sc.load(RECORDINGS[condition])
    cropped = source.channel_slice(channel_ids=RETAINED_IDS)
    target.parent.mkdir(parents=True, exist_ok=True)
    cropped.save(folder=target, n_jobs=n_jobs, chunk_duration="10s", progress_bar=True)
    manifest.write_text(json.dumps({
        "condition": condition,
        "source_recording": str(RECORDINGS[condition]),
        "channel_ids": RETAINED_IDS.tolist(),
        "n_frames": int(cropped.get_num_samples()),
        "sampling_frequency_hz": float(cropped.get_sampling_frequency()),
        "dtype": str(cropped.get_dtype()),
        "expected_binary_bytes": int(cropped.get_num_samples() * len(RETAINED_IDS) * 2),
        "crop_before_sorting": True,
        "motion_resampling_in_this_step": False,
    }, indent=2) + "\n")
    print(f"Completed cropped recording: {target}")


def verify_materialized(chunk_s: float = 1.0) -> dict:
    import spikeinterface.core as sc

    rows = {}
    for condition, source_path in RECORDINGS.items():
        target_path = cropped_recording_path(condition)
        manifest_path = target_path / "crop_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        source = sc.load(source_path).channel_slice(channel_ids=RETAINED_IDS)
        target = sc.load(target_path)
        raw_path = target_path / "traces_cached_seg0.raw"
        checks = {
            "exact_binary_size": bool(raw_path.stat().st_size == manifest["expected_binary_bytes"]),
            "same_frames": bool(target.get_num_samples() == source.get_num_samples()),
            "same_channel_ids": bool(np.array_equal(target.get_channel_ids(), source.get_channel_ids())),
            "same_locations": bool(np.allclose(target.get_channel_locations(), source.get_channel_locations())),
            "same_dtype": bool(np.dtype(target.get_dtype()) == np.dtype(source.get_dtype())),
        }
        fs = float(source.get_sampling_frequency())
        width = int(round(chunk_s * fs))
        last = source.get_num_samples() - width
        sampled = []
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            start = int(round(fraction * last))
            a = source.get_traces(start_frame=start, end_frame=start + width)
            b = target.get_traces(start_frame=start, end_frame=start + width)
            sampled.append({
                "fraction": fraction,
                "start_frame": start,
                "exactly_equal": bool(np.array_equal(a, b)),
            })
        checks["sampled_chunks_exactly_equal"] = bool(all(item["exactly_equal"] for item in sampled))
        if not all(checks.values()):
            raise RuntimeError(f"Materialized integrity failure for {condition}: {checks}")
        rows[condition] = {"checks": checks, "sampled_chunks": sampled}
    receipt = {"conditions": rows, "passed": True}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "materialized_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return receipt


def score_pair() -> dict:
    summaries = {}
    for condition in RECORDINGS:
        summaries[condition] = score_pilot(
            PILOT,
            ROOT,
            DEFAULT_REVIEW,
            300.0,
            conditioning_policy="legacy",
            result_override=sort_path(condition) / "sorter_output",
            score_name=f"core_depth_strip_interior80_{condition}",
            log_override=log_path(condition),
        )
    comparison = compare_summaries(summaries["no_motion"], summaries["rigid025_p2"])
    comparison.update({
        "design": "paired 80-channel voltages cropped before sorting from matched 96-channel caches",
        "interpretation_scope": "KS4-specific sorting comparison; corrected retained channels had 80-um real-voltage support during resampling",
    })
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    print(json.dumps(comparison, indent=2))
    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--materialize-condition", choices=tuple(RECORDINGS))
    parser.add_argument("--verify-materialized", action="store_true")
    parser.add_argument("--run-condition", choices=tuple(RECORDINGS))
    parser.add_argument("--score", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/luke-inward-crop-pair-numba")
    if not (args.verify or args.materialize_condition or args.verify_materialized or args.run_condition or args.score):
        raise SystemExit("Choose --verify, --materialize-condition, --verify-materialized, --run-condition, and/or --score")
    if args.verify:
        verify_inputs()
    if args.materialize_condition:
        materialize_condition(args.materialize_condition)
    if args.verify_materialized:
        verify_materialized()
    if args.run_condition:
        run_condition(args.run_condition)
    if args.score:
        score_pair()


if __name__ == "__main__":
    main()
