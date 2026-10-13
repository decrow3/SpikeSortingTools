#!/usr/bin/env python3
"""Materialize only DARTsort matching1 times/labels for fast-loop metrics.

The adapter deliberately does not fabricate Kilosort amplitudes, positions, or
KS labels.  It preserves matching1 row order and session-clock sample values.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


KILOSORT_FIELDS_NOT_APPLICABLE = (
    "amplitudes.npy",
    "spike_positions.npy",
    "cluster_KSLabel.tsv",
)


def _exact_int64(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or np.issubdtype(array.dtype, np.bool_):
        raise ValueError(f"{name} must be a one-dimensional integer array")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must have integer dtype")
    if array.size and (
        np.any(array < np.iinfo(np.int64).min)
        or np.any(array > np.iinfo(np.int64).max)
    ):
        raise ValueError(f"{name} exceeds the int64 domain")
    return array.astype(np.int64, copy=False)


def load_matching1_times_labels(matching_h5: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load matching-stage arrays without sorting, filtering, or relabelling."""
    with h5py.File(matching_h5, "r") as h5:
        missing = [name for name in ("times_samples", "labels") if name not in h5]
        if missing:
            raise ValueError(f"matching1 datasets missing: {missing}")
        times = _exact_int64("times_samples", h5["times_samples"][:])
        labels = _exact_int64("labels", h5["labels"][:])
    if times.size != labels.size:
        raise ValueError("matching1 times_samples and labels lengths differ")
    return times, labels


def materialize_times_labels(matching_h5: Path, output_dir: Path) -> tuple[Path, Path]:
    """Create a two-array evaluator input without Kilosort-only substitutes."""
    times, labels = load_matching1_times_labels(Path(matching_h5))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    times_path = output_dir / "spike_times.npy"
    labels_path = output_dir / "spike_clusters.npy"
    np.save(times_path, times, allow_pickle=False)
    np.save(labels_path, labels, allow_pickle=False)
    return times_path, labels_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matching_h5", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    materialize_times_labels(args.matching_h5, args.output_dir)


if __name__ == "__main__":
    main()
