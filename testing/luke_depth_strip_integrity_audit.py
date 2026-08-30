"""Write a reproducible integrity receipt for the materialized Luke depth strip."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


STRIP_DIR = Path(
    "/media/huklab/Data/NPX/Ryansorting/Luke/"
    "Luke0804_two_axis_pilot_imec1/recordings/core_depth_strip"
)
OUTPUT = Path("testing/outputs/luke_depth_strip_integrity_audit/receipt.json")
FRACTIONS = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)
REQUIRED_CLASSES = {
    "spikeinterface.preprocessing.phase_shift.PhaseShiftRecording",
    "spikeinterface.preprocessing.clip.BlankSaturationRecording",
    "spikeinterface.preprocessing.interpolate_bad_channels.InterpolateBadChannelsRecording",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strip-dir", type=Path, default=STRIP_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--chunk-duration-s", type=float, default=1.0)
    return parser.parse_args()


def nested_values(value, key: str) -> list:
    found = []
    if isinstance(value, dict):
        if key in value:
            found.append(value[key])
        for child in value.values():
            found.extend(nested_values(child, key))
    elif isinstance(value, list):
        for child in value:
            found.extend(nested_values(child, key))
    return found


def flatten_strings(values) -> list[str]:
    flattened: list[str] = []
    for value in values:
        if isinstance(value, (list, tuple)):
            flattened.extend(flatten_strings(value))
        elif isinstance(value, str):
            flattened.append(value)
    return flattened


def run_audit(strip_dir: Path, output: Path, chunk_duration_s: float) -> dict:
    if chunk_duration_s <= 0:
        raise ValueError("chunk duration must be positive")
    manifest = json.loads((strip_dir / "pilot_manifest.json").read_text())
    binary = json.loads((strip_dir / "binary.json").read_text())
    provenance = json.loads((strip_dir / "provenance.json").read_text())
    raw_path = strip_dir / binary["kwargs"]["file_paths"][0]
    n_channels = int(binary["kwargs"]["num_channels"])
    dtype = np.dtype(binary["kwargs"]["dtype"])
    fs = float(binary["kwargs"]["sampling_frequency"])
    n_frames = int(manifest["stop_frame"] - manifest["start_frame"])
    expected_bytes = n_frames * n_channels * dtype.itemsize
    actual_bytes = raw_path.stat().st_size
    chunk_frames = int(round(chunk_duration_s * fs))
    chunk_rows = []
    channel_sums = np.zeros(n_channels, dtype=np.float64)
    channel_sumsq = np.zeros(n_channels, dtype=np.float64)
    channel_count = 0
    with raw_path.open("rb") as stream:
        for fraction in FRACTIONS:
            start = min(int(fraction * n_frames), n_frames - chunk_frames)
            stream.seek(start * n_channels * dtype.itemsize)
            traces = np.frombuffer(
                stream.read(chunk_frames * n_channels * dtype.itemsize), dtype=dtype
            ).reshape(-1, n_channels)
            channel_sums += traces.sum(axis=0, dtype=np.float64)
            channel_sumsq += np.square(traces.astype(np.float64)).sum(axis=0)
            channel_count += traces.shape[0]
            chunk_rows.append(
                {
                    "fraction": fraction,
                    "start_frame": start,
                    "n_frames": int(traces.shape[0]),
                    "nonzero_fraction": float(np.mean(traces != 0)),
                    "minimum_counts": int(traces.min()),
                    "maximum_counts": int(traces.max()),
                    "mean_counts": float(traces.mean()),
                    "std_counts": float(traces.std()),
                }
            )
    channel_means = channel_sums / channel_count
    channel_stds = np.sqrt(channel_sumsq / channel_count - channel_means**2)
    classes = set(nested_values(provenance, "class"))
    file_paths = flatten_strings(nested_values(provenance, "file_paths"))
    source_present = any(path.endswith("imec1.ap.bin") for path in file_paths)
    checks = {
        "exact_binary_size": actual_bytes == expected_bytes == int(manifest["expected_binary_bytes"]),
        "channel_count": n_channels == 96 == len(manifest["channel_ids"]),
        "frame_count": n_frames == 314_204_094,
        "source_imec1_ap_present": source_present,
        "required_preprocessing_classes_present": REQUIRED_CLASSES.issubset(classes),
        "all_chunks_populated": all(row["nonzero_fraction"] > 0.5 for row in chunk_rows),
        "no_stuck_channels": bool(np.all(channel_stds > 1.0)),
    }
    receipt = {
        "strip_dir": str(strip_dir),
        "binary_path": str(raw_path),
        "expected_bytes": expected_bytes,
        "actual_bytes": actual_bytes,
        "n_frames": n_frames,
        "n_channels": n_channels,
        "sampling_frequency_hz": fs,
        "dtype": dtype.str,
        "source_file_paths": file_paths,
        "preprocessing_classes": sorted(classes & REQUIRED_CLASSES),
        "sampled_chunks": chunk_rows,
        "per_channel_std_min_counts": float(channel_stds.min()),
        "per_channel_std_max_counts": float(channel_stds.max()),
        "per_channel_std_min_channel": int(manifest["channel_ids"][int(channel_stds.argmin())]),
        "per_channel_std_max_channel": int(manifest["channel_ids"][int(channel_stds.argmax())]),
        "checks": checks,
        "passed": all(checks.values()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    args = parse_args()
    receipt = run_audit(args.strip_dir, args.output, args.chunk_duration_s)
    print(json.dumps(receipt, indent=2))
    if not receipt["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
