"""CPU-only preparation primitives for the AU injected-truth worker.

This module deliberately cannot launch a sort, open raw voltage, or use CUDA.
It freezes donor selection, exact lattice remapping, matching-chunk trajectory
sampling, and exclusive event scoring before the AV handoff arrives.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


SCHEMA = "luke-au-cpu-preparation-v1"
FS_HZ = 30_000.0
WINDOW_START_S = 900.0
WINDOW_END_S = 1240.0
TRAIN_RATE_HZ = 5.0
TRAIN_GUARD_S = 1.0
MATCH_TOLERANCE_MS = 0.4
PITCH_UM = 40.0
N_DONORS = 30
N_DEPTH_STRATA = 6


@dataclass(frozen=True)
class DonorRule:
    """Frozen, outcome-independent donor selection rule."""

    quality_label: str = "good"
    max_refractory_violation_fraction: float = 0.005
    min_rest_spike_fraction: float = 0.80
    min_isolation_score: float = 0.80
    n_depth_strata: int = N_DEPTH_STRATA
    quota_per_stratum: int = N_DONORS // N_DEPTH_STRATA
    target_count: int = N_DONORS


DONOR_RULE = DonorRule()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    values = np.ascontiguousarray(values)
    header = f"{values.dtype.str}|{values.shape}".encode("ascii")
    return hashlib.sha256(header + values.tobytes()).hexdigest()


def immutable_regular_train(
    *,
    fs_hz: float = FS_HZ,
    start_s: float = WINDOW_START_S,
    end_s: float = WINDOW_END_S,
    rate_hz: float = TRAIN_RATE_HZ,
    guard_s: float = TRAIN_GUARD_S,
) -> np.ndarray:
    """Return the frozen 5 Hz train on the W2-local sample clock."""
    duration_s = float(end_s - start_s)
    first = int(round(guard_s * fs_hz))
    stop = int(round((duration_s - guard_s) * fs_hz))
    step = int(round(fs_hz / rate_hz))
    train = np.arange(first, stop, step, dtype=np.int64)
    train.flags.writeable = False
    return train


def _require_metadata_fields(rows: Sequence[Mapping]) -> None:
    required = {
        "unit_id",
        "quality",
        "depth_um",
        "ptp_uv",
        "refractory_violation_fraction",
        "rest_spike_fraction",
        "isolation_score",
    }
    for i, row in enumerate(rows):
        missing = required.difference(row)
        if missing:
            raise ValueError(f"donor metadata row {i} missing {sorted(missing)}")


def select_donors(rows: Sequence[Mapping], rule: DonorRule = DONOR_RULE) -> list[dict]:
    """Select up to 30 donors with fixed gates and balanced depth coverage.

    Eligibility is decided without any injected-arm outcome. Within each of six
    equal depth strata, candidates rank by larger static PTP, then smaller
    refractory fraction, then integer unit id. Five are taken per stratum.
    Empty quota slots are filled from the remaining eligible population by the
    same ranking. Fewer than 30 eligible donors is a hard downstream gate, not
    grounds to relax a threshold.
    """
    rows = [dict(row) for row in rows]
    _require_metadata_fields(rows)
    eligible = [
        row
        for row in rows
        if str(row["quality"]).lower() == rule.quality_label
        and float(row["refractory_violation_fraction"])
        <= rule.max_refractory_violation_fraction
        and float(row["rest_spike_fraction"]) >= rule.min_rest_spike_fraction
        and float(row["isolation_score"]) >= rule.min_isolation_score
        and np.isfinite(float(row["depth_um"]))
        and np.isfinite(float(row["ptp_uv"]))
        and float(row["ptp_uv"]) > 0
    ]
    if not eligible:
        return []
    depths = np.asarray([float(row["depth_um"]) for row in eligible])
    lo, hi = float(depths.min()), float(depths.max())
    edges = np.linspace(lo, np.nextafter(hi, np.inf), rule.n_depth_strata + 1)

    def rank(row: Mapping) -> tuple[float, float, int]:
        return (
            -float(row["ptp_uv"]),
            float(row["refractory_violation_fraction"]),
            int(row["unit_id"]),
        )

    chosen: list[dict] = []
    chosen_ids: set[int] = set()
    for stratum in range(rule.n_depth_strata):
        candidates = [
            row
            for row in eligible
            if edges[stratum] <= float(row["depth_um"]) < edges[stratum + 1]
        ]
        for row in sorted(candidates, key=rank)[: rule.quota_per_stratum]:
            item = dict(row, depth_stratum=stratum)
            chosen.append(item)
            chosen_ids.add(int(row["unit_id"]))
    if len(chosen) < rule.target_count:
        remainder = [row for row in eligible if int(row["unit_id"]) not in chosen_ids]
        for row in sorted(remainder, key=rank)[: rule.target_count - len(chosen)]:
            stratum = int(np.searchsorted(edges, float(row["depth_um"]), side="right") - 1)
            chosen.append(dict(row, depth_stratum=min(stratum, rule.n_depth_strata - 1)))
    return sorted(chosen, key=lambda row: (int(row["depth_stratum"]), *rank(row)))


def exact_same_column_map(
    geometry_um: np.ndarray, shift_um: float, *, atol_um: float = 1e-6
) -> np.ndarray:
    """Map each source channel to the exact same-x site at y + shift.

    Missing target sites are represented by -1. Ambiguous targets fail closed.
    """
    geom = np.asarray(geometry_um, dtype=np.float64)
    if geom.ndim != 2 or geom.shape[1] != 2 or not np.all(np.isfinite(geom)):
        raise ValueError("geometry must be a finite (channels, 2) x/y array")
    result = np.full(geom.shape[0], -1, dtype=np.int64)
    for source, (x_um, y_um) in enumerate(geom):
        hits = np.flatnonzero(
            np.isclose(geom[:, 0], x_um, atol=atol_um, rtol=0.0)
            & np.isclose(geom[:, 1], y_um + shift_um, atol=atol_um, rtol=0.0)
        )
        if hits.size > 1:
            raise ValueError(f"ambiguous exact target for channel {source}")
        if hits.size == 1:
            result[source] = int(hits[0])
    return result


def exact_remap(template: np.ndarray, source_to_target: np.ndarray) -> np.ndarray:
    """Move a time-by-channel template with no interpolation or clipping."""
    template = np.asarray(template)
    mapping = np.asarray(source_to_target, dtype=np.int64)
    if template.ndim != 2 or template.shape[1] != mapping.size:
        raise ValueError("template channels and mapping length differ")
    valid = mapping >= 0
    targets = mapping[valid]
    if np.unique(targets).size != targets.size:
        raise ValueError("mapping is not one-to-one")
    out = np.zeros_like(template)
    out[:, targets] = template[:, valid]
    return out


def centered_cosine(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64).ravel()
    second = np.asarray(second, dtype=np.float64).ravel()
    first = first - first.mean()
    second = second - second.mean()
    denom = np.linalg.norm(first) * np.linalg.norm(second)
    return float(np.dot(first, second) / denom) if denom else float("nan")


def qualify_exact_state(
    template: np.ndarray,
    geometry_um: np.ndarray,
    state_um: float,
) -> dict:
    """Measure the frozen no-interpolation qualification for one state."""
    template = np.asarray(template, dtype=np.float32)
    forward_map = exact_same_column_map(geometry_um, state_um)
    moved = exact_remap(template, forward_map)
    source_energy = float(np.sum(np.square(template, dtype=np.float64)))
    moved_energy = float(np.sum(np.square(moved, dtype=np.float64)))
    retained = moved_energy / source_energy if source_energy else float("nan")

    reverse_map = exact_same_column_map(geometry_um, -state_um)
    restored = exact_remap(moved, reverse_map)
    source_ptp = float(np.ptp(template, axis=0).max())
    restored_ptp = float(np.ptp(restored, axis=0).max())
    ptp_ratio = restored_ptp / source_ptp if source_ptp else float("nan")
    cosine = centered_cosine(template, restored)
    passed = bool(
        retained > 0.99
        and abs(ptp_ratio - 1.0) <= 0.02
        and cosine >= 0.99
    )
    return {
        "state_um": float(state_um),
        "energy_retention": retained,
        "roundtrip_ptp_ratio": ptp_ratio,
        "roundtrip_cosine": cosine,
        "passed": passed,
    }


def exclusive_pairs(truth_samples: Iterable[int], output_samples: Iterable[int], tol: int):
    """Maximum-cardinality interval-order 1:1 matching, inclusive at +/-tol."""
    truth = np.sort(np.asarray(list(truth_samples), dtype=np.int64))
    output = np.sort(np.asarray(list(output_samples), dtype=np.int64))
    ti: list[int] = []
    oi: list[int] = []
    i = j = 0
    while i < truth.size and j < output.size:
        delta = int(output[j]) - int(truth[i])
        if delta < -tol:
            j += 1
        elif delta > tol:
            i += 1
        else:
            ti.append(i)
            oi.append(j)
            i += 1
            j += 1
    return np.asarray(ti, dtype=np.int64), np.asarray(oi, dtype=np.int64)


def score_one_cluster(
    truth_samples: Iterable[int],
    output_samples: Iterable[int],
    *,
    fs_hz: float = FS_HZ,
    tolerance_ms: float = MATCH_TOLERANCE_MS,
) -> dict:
    """Corrected per-cluster exclusive score used by AU fixtures."""
    truth = np.sort(np.asarray(list(truth_samples), dtype=np.int64))
    output = np.sort(np.asarray(list(output_samples), dtype=np.int64))
    tol = int(round(tolerance_ms * fs_hz / 1000.0))
    ti, oi = exclusive_pairs(truth, output, tol)
    tp = int(ti.size)
    fp = int(output.size - oi.size)
    fn = int(truth.size - ti.size)
    denom = tp + fp + fn
    return {
        "tolerance_samples": tol,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "accuracy": tp / denom if denom else float("nan"),
    }


def sample_and_quantize_trajectory(
    field_time_s: np.ndarray,
    displacement_um: np.ndarray,
    chunk_start_samples: np.ndarray,
    chunk_length_samples: int,
    *,
    fs_hz: float = FS_HZ,
    pitch_um: float = PITCH_UM,
) -> dict:
    """Sample a rigid field at DARTsort chunk centres and round to 40 um states."""
    times = np.asarray(field_time_s, dtype=np.float64)
    disp = np.asarray(displacement_um, dtype=np.float64).reshape(-1)
    starts = np.asarray(chunk_start_samples, dtype=np.int64)
    if times.ndim != 1 or times.size != disp.size or times.size < 2:
        raise ValueError("rigid field must have matching 1D time/displacement arrays")
    if np.any(np.diff(times) <= 0):
        raise ValueError("field time must be strictly increasing")
    center_samples = starts + int(chunk_length_samples) // 2
    center_s = center_samples.astype(np.float64) / float(fs_hz)
    sampled = np.interp(center_s, times, disp)
    scaled = sampled / float(pitch_um)
    states = np.sign(scaled) * np.floor(np.abs(scaled) + 0.5)
    quantized = states * float(pitch_um)
    return {
        "chunk_center_samples": center_samples,
        "chunk_center_s": center_s,
        "sampled_displacement_um": sampled,
        "state_index": states.astype(np.int64),
        "quantized_displacement_um": quantized,
        "quantization_error_um": quantized - sampled,
    }

