#!/usr/bin/env python3
"""Metadata-only feasibility audit for full KS batches within Arm-A states."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np


FS = 29999.835983263598
N_SAMPLES = 314204894
BATCH_SAMPLES = 60000
CELL_WIDTH_S = 0.25
FIELD = Path(
    "/mnt/NPX/Luke/DARTsort_motion_experiments/"
    "en_am3_imec0_field_request_20260929_v1/payload/"
    "luke0804_imec0_two_layer_motion.npz"
)
PUBLISHED = Path(
    "/mnt/NPX/Luke/DARTsort_motion_experiments/"
    "en_tier1_rounded_field_result_20260930_v1/STATE_METRICS_A_A.csv"
)
OUT = (
    Path(__file__).resolve().parent
    / "outputs/en_executed_provenance_mechanism_audit_20260930_v1/"
    "a_state_duration_audit.json"
)


def round_half_away(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.floor(np.abs(values) + 0.5)


def main() -> None:
    with np.load(FIELD, allow_pickle=False) as z:
        centers = np.asarray(z["time_s"], dtype=np.float64)
        displacement = np.asarray(z["displacement_um"][:, 0], dtype=np.float64)
    q = (40 * round_half_away(displacement / 40)).astype(np.int64)
    duration = N_SAMPLES / FS
    starts = np.maximum(0.0, centers - CELL_WIDTH_S / 2)
    ends = np.minimum(duration, centers + CELL_WIDTH_S / 2)

    runs: list[tuple[int, float, float]] = []
    run_start = starts[0]
    run_q = int(q[0])
    for i in range(1, q.size):
        if int(q[i]) != run_q:
            runs.append((run_q, float(run_start), float(ends[i - 1])))
            run_start = starts[i]
            run_q = int(q[i])
    runs.append((run_q, float(run_start), float(ends[-1])))

    published = {}
    with PUBLISHED.open(newline="") as stream:
        for row in csv.DictReader(stream):
            published[int(float(row["state_um"]))] = float(row["exposure_s"])

    batch_seconds = BATCH_SAMPLES / FS
    states = []
    for state in sorted(np.unique(q)):
        state_runs = [(a, b) for state_q, a, b in runs if state_q == int(state)]
        exposure = sum(b - a for a, b in state_runs)
        arbitrary_windows = sum(math.floor((b - a) / batch_seconds + 1e-12) for a, b in state_runs)
        aligned_batches = 0
        for a, b in state_runs:
            first = math.ceil((a * FS) / BATCH_SAMPLES - 1e-12)
            last_exclusive = math.floor((b * FS) / BATCH_SAMPLES + 1e-12)
            aligned_batches += max(0, last_exclusive - first)
        pub = published.get(int(state))
        states.append(
            {
                "state_um": int(state),
                "exposure_s": exposure,
                "published_exposure_s": pub,
                "exposure_minus_published_s": None if pub is None else exposure - pub,
                "run_count": len(state_runs),
                "max_contiguous_run_s": max((b - a for a, b in state_runs), default=0),
                "nonoverlap_60000_windows_with_arbitrary_alignment": arbitrary_windows,
                "globally_aligned_60000_sample_batches_wholly_inside": aligned_batches,
            }
        )

    max_exposure_error = max(
        abs(row["exposure_minus_published_s"])
        for row in states
        if row["exposure_minus_published_s"] is not None
    )
    if max_exposure_error > 1e-6:
        raise AssertionError(f"state exposure mismatch {max_exposure_error}")

    result = {
        "schema": "en-a-state-duration-feasibility-v1",
        "scope": "metadata only; no voltage read",
        "field": str(FIELD),
        "published_state_metrics": str(PUBLISHED),
        "fs_hz": FS,
        "num_samples": N_SAMPLES,
        "session_duration_s": duration,
        "field_cell_width_s": CELL_WIDTH_S,
        "batch_samples": BATCH_SAMPLES,
        "batch_duration_s": batch_seconds,
        "max_abs_exposure_reproduction_error_s": max_exposure_error,
        "states": states,
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
