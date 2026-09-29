#!/usr/bin/env python3
"""Slice a completed full-session DARTsort output onto frozen validation windows."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def slice_arrays(
    times: np.ndarray,
    labels: np.ndarray,
    channels: np.ndarray,
    start: int,
    end: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = np.asarray(times, dtype=np.int64)
    labels = np.asarray(labels)
    channels = np.asarray(channels)
    if not (times.shape == labels.shape == channels.shape):
        raise ValueError("event arrays differ in shape")
    if np.any(np.diff(times) < 0):
        raise ValueError("full-session event times are not sorted")
    left = int(np.searchsorted(times, start, side="left"))
    right = int(np.searchsorted(times, end, side="left"))
    local = times[left:right] - start
    if np.any((local < 0) | (local >= end - start)):
        raise RuntimeError("sliced event lies outside half-open window")
    return local, labels[left:right], channels[left:right]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    contract = json.loads(args.contract.read_text())
    run = Path(contract["source_run"]["output"])
    receipt_path = run / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("status") != "complete" or receipt.get("exit_status") != 0:
        raise RuntimeError("full-session run is not complete and successful")
    config_path = run / "config.json"
    expected_config = contract["source_run"]["resolved_config_sha256"]
    if sha256(config_path) != expected_config or receipt.get("config_sha256") != expected_config:
        raise RuntimeError("full-session resolved configuration differs")
    sorting_path = Path(contract["source_run"]["sorting"])
    if sorting_path != run / "sort/dartsort_sorting.npz":
        raise RuntimeError("sorting path is outside the frozen run")

    source_sha = sha256(sorting_path)
    with np.load(sorting_path, allow_pickle=False) as saved:
        times = np.asarray(saved["times_samples"], dtype=np.int64)
        labels = np.asarray(saved["labels"])
        channels = np.asarray(saved["channels"])
        sampling_frequency = float(saved["sampling_frequency"])
        geom = np.asarray(saved["geom"], dtype=np.float64)
    expected_fs = float(contract["scope"]["sampling_frequency_hz"])
    if not np.isclose(sampling_frequency, expected_fs, rtol=0, atol=1e-9):
        raise RuntimeError("full-session sorting sampling frequency differs")
    expected_frames = int(contract["source_run"]["expected_frames"])
    if times.size and (times.min() < 0 or times.max() >= expected_frames):
        raise RuntimeError("full-session event lies outside recording support")

    args.output.mkdir(parents=True)
    windows = {}
    for name, specification in contract["windows"].items():
        reference = specification["exact_reference"]
        if sha256(Path(reference["path"])) != reference["sha256"]:
            raise RuntimeError(f"{name} exact reference hash differs")
        start, end = map(int, specification["frames_half_open"])
        local, window_labels, window_channels = slice_arrays(
            times, labels, channels, start, end
        )
        path = args.output / f"full_session_rounded_kriging_{name}.npz"
        np.savez_compressed(
            path,
            times_samples=local,
            labels=window_labels,
            channels=window_channels,
            sampling_frequency=np.asarray(sampling_frequency),
            geom=geom,
        )
        windows[name] = {
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "frames_half_open": [start, end],
            "events": int(local.size),
            "assigned_events": int(np.count_nonzero(window_labels >= 0)),
            "negative_events": int(np.count_nonzero(window_labels < 0)),
            "assigned_units": int(np.unique(window_labels[window_labels >= 0]).size),
            "clock": "window-local samples",
            "exact_reference": reference,
        }
    result = {
        "schema": "em2g-full-sort-window-slices-v1",
        "status": "complete",
        "voltage_read": False,
        "rf_evaluated": False,
        "contract_sha256": sha256(args.contract),
        "source_sorting": {
            "path": str(sorting_path),
            "sha256": source_sha,
            "events": int(times.size),
            "assigned_events": int(np.count_nonzero(labels >= 0)),
            "negative_events": int(np.count_nonzero(labels < 0)),
            "assigned_units": int(np.unique(labels[labels >= 0]).size),
        },
        "windows": windows,
    }
    result_path = args.output / "RESULT.json"
    atomic_json(result_path, result)
    products = [
        {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(args.output.iterdir())
        if path.name not in {"MANIFEST.json", "COMPLETE.json"}
    ]
    manifest_path = args.output / "MANIFEST.json"
    atomic_json(manifest_path, {"products": products})
    atomic_json(
        args.output / "COMPLETE.json",
        {"status": "complete", "manifest_sha256": sha256(manifest_path)},
    )


if __name__ == "__main__":
    main()
