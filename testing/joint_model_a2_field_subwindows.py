#!/usr/bin/env python3
"""Select A2 P4 subwindows using only the frozen rounded motion field."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np


SCHEMA = "joint-model-a2-field-subwindows-result-v1"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def round_half_away_lattice(values: np.ndarray, pitch_um: float = 40.0) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("field values must be a finite one-dimensional array")
    return np.sign(values) * np.floor(np.abs(values) / pitch_um + 0.5) * pitch_um


def nearest_boundaries_samples(time_s: np.ndarray, sampling_frequency_hz: float) -> np.ndarray:
    time_s = np.asarray(time_s, dtype=np.float64)
    if (
        time_s.ndim != 1
        or time_s.size < 2
        or not np.isfinite(time_s).all()
        or np.any(np.diff(time_s) <= 0)
    ):
        raise ValueError("field time knots must be finite, one-dimensional and strictly increasing")
    spacing = np.diff(time_s)
    if not np.allclose(spacing, spacing[0], rtol=0.0, atol=1e-12):
        raise ValueError("field time knots must be uniformly spaced")
    if not np.isfinite(sampling_frequency_hz) or sampling_frequency_hz <= 0:
        raise ValueError("sampling frequency must be positive and finite")
    midpoints = (time_s[:-1] + time_s[1:]) / 2.0
    return np.ceil(midpoints * sampling_frequency_hz).astype(np.int64)


def score_candidate(
    start: int,
    stop: int,
    rounded_q: np.ndarray,
    boundaries: np.ndarray,
) -> tuple[int, int]:
    if stop <= start or rounded_q.size != boundaries.size + 1:
        raise ValueError("invalid candidate or field boundary arrays")
    changed = rounded_q[:-1] != rounded_q[1:]
    switches = int(np.count_nonzero(changed & (boundaries >= start) & (boundaries < stop)))

    left = np.concatenate((np.array([np.iinfo(np.int64).min]), boundaries))
    right = np.concatenate((boundaries, np.array([np.iinfo(np.int64).max])))
    overlaps = np.maximum(
        0,
        np.minimum(right, np.int64(stop)) - np.maximum(left, np.int64(start)),
    )
    high_displacement_samples = int(overlaps[np.abs(rounded_q) >= 80.0].sum())
    return switches, high_displacement_samples


def select_subwindow(
    parent: tuple[int, int],
    rounded_q: np.ndarray,
    boundaries: np.ndarray,
    *,
    chunk_length_samples: int,
    chunks_per_subwindow: int,
) -> dict[str, Any]:
    parent_start, parent_stop = map(int, parent)
    length = int(chunk_length_samples * chunks_per_subwindow)
    first = ((parent_start + chunk_length_samples - 1) // chunk_length_samples) * chunk_length_samples
    last = ((parent_stop - length) // chunk_length_samples) * chunk_length_samples
    if first > last:
        raise ValueError("parent has no complete candidate subwindow")
    candidates = np.arange(first, last + chunk_length_samples, chunk_length_samples, dtype=np.int64)
    scored = []
    for start_value in candidates:
        start = int(start_value)
        switches, high_samples = score_candidate(start, start + length, rounded_q, boundaries)
        scored.append((switches, high_samples, -start, start))
    winner = max(scored)
    winner_start = winner[3]
    tied = [row for row in scored if row[:3] == winner[:3]]
    if len(tied) != 1:
        raise ValueError("selection did not produce one winner")
    return {
        "parent_half_open_samples": [parent_start, parent_stop],
        "selected_half_open_samples": [winner_start, winner_start + length],
        "chunk_starts_samples": list(
            range(winner_start, winner_start + length, chunk_length_samples)
        ),
        "state_switch_count": winner[0],
        "abs_q_at_least_80um_samples": winner[1],
        "candidate_count": int(candidates.size),
    }


def evaluate(contract: dict[str, Any]) -> dict[str, Any]:
    inputs = contract["inputs"]
    rule = contract["selection_rule"]
    field_path = Path(inputs["field_path"])
    observed_field_sha = sha256_path(field_path)
    if observed_field_sha != inputs["field_sha256"]:
        raise ValueError("field file hash mismatch")
    with np.load(field_path, allow_pickle=False) as archive:
        time_s = np.asarray(archive["time_s"], dtype=np.float64)
        displacement = np.asarray(archive["displacement_um"], dtype=np.float64)
    if displacement.shape != (time_s.size, 1):
        raise ValueError("expected the frozen one-depth AM.3 field")
    rounded_q = round_half_away_lattice(displacement[:, 0])
    rounded_hash = hashlib.sha256(rounded_q.tobytes()).hexdigest()
    if rounded_hash != inputs["expected_rounded_q_float64_bytes_sha256"]:
        raise ValueError("rounded field hash mismatch")
    fs = float(inputs["sampling_frequency_hz"])
    boundaries = nearest_boundaries_samples(time_s, fs)
    chunk = int(rule["chunk_length_samples"])
    count = int(rule["chunks_per_subwindow"])
    windows = {
        "high_displacement": select_subwindow(
            tuple(inputs["high_parent_half_open_samples"]), rounded_q, boundaries,
            chunk_length_samples=chunk, chunks_per_subwindow=count,
        ),
        "medium_displacement": select_subwindow(
            tuple(inputs["medium_parent_half_open_samples"]), rounded_q, boundaries,
            chunk_length_samples=chunk, chunks_per_subwindow=count,
        ),
    }
    return {
        "schema": SCHEMA,
        "status": "FIELD_ONLY_SUBWINDOWS_SELECTED",
        "inputs": {
            "field_path": str(field_path),
            "field_sha256": observed_field_sha,
            "rounded_q_float64_bytes_sha256": rounded_hash,
            "window_freeze_sha256": inputs["window_freeze_sha256"],
            "sampling_frequency_hz": fs,
        },
        "grid": {
            "origin_session_sample": 0,
            "chunk_length_samples": chunk,
            "chunks_per_subwindow": count,
            "subwindow_length_samples": chunk * count,
        },
        "windows": windows,
        "outcomes_inspected": False,
        "scope": contract["scope"],
    }


def write_json_create_only(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    fd, temporary_name = tempfile.mkstemp(prefix=".a2-field-window.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        temporary.unlink()
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    result = evaluate(contract)
    result["provenance"] = {
        "contract": str(args.contract.resolve()),
        "contract_sha256": sha256_path(args.contract),
        "source_sha256": sha256_path(Path(__file__).resolve()),
    }
    write_json_create_only(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
