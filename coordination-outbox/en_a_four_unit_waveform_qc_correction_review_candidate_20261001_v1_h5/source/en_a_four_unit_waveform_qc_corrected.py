#!/usr/bin/env python3
"""Execution-gated saved-array QC with the executed per-sample A remap."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def round_half_away(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.copysign(np.floor(np.abs(values) + 0.5), values)


def rigid_projection(time_s: np.ndarray, displacement_um: np.ndarray) -> np.ndarray:
    time_s = np.asarray(time_s, dtype=np.float64)
    displacement = np.asarray(displacement_um, dtype=np.float64)
    if time_s.ndim != 1 or time_s.size < 2 or not np.isfinite(time_s).all():
        raise ValueError("time_s must be a finite vector")
    if displacement.ndim == 1 and displacement.shape == time_s.shape:
        rigid = displacement
    elif displacement.ndim == 2 and displacement.shape[0] == time_s.size:
        rigid = np.median(displacement, axis=1)
    else:
        raise ValueError("displacement shape differs from executed rigid projection")
    if not np.isfinite(rigid).all():
        raise ValueError("rigid projection is nonfinite")
    return rigid


def rounded_field(time_s: np.ndarray, displacement_um: np.ndarray, period_um: float) -> tuple[np.ndarray, float, float]:
    rigid = rigid_projection(time_s, displacement_um)
    reference = float(np.median(rigid))
    states = period_um * round_half_away((rigid - reference) / period_um)
    steps = np.diff(np.asarray(time_s, dtype=np.float64))
    width = float(np.median(steps))
    tolerance = max(1e-12, abs(width) * 1e-9)
    if not np.allclose(steps, width, rtol=0, atol=tolerance):
        raise ValueError("field centers are not contiguous and uniform")
    return states.astype(np.float64), reference, width


def sample_stepwise_states(centers_s: np.ndarray, states_um: np.ndarray,
                           sample_times_s: np.ndarray, cell_width_s: float) -> np.ndarray:
    """Executed DD rule: [center-width/2, center+width/2), edge -> later."""
    centers = np.asarray(centers_s, dtype=np.float64)
    states = np.asarray(states_um, dtype=np.float64)
    times = np.asarray(sample_times_s, dtype=np.float64)
    if centers.ndim != 1 or states.shape != centers.shape or centers.size == 0:
        raise ValueError("field centers/states must be equal nonempty vectors")
    if np.any(np.diff(centers) <= 0) or not np.isfinite(times).all():
        raise ValueError("invalid field centers or sample times")
    tolerance = max(1e-12, abs(float(cell_width_s)) * 1e-9)
    if centers.size > 1 and not np.allclose(np.diff(centers), cell_width_s, rtol=0, atol=tolerance):
        raise ValueError("field centers are not contiguous at cell_width_s")
    positions = np.searchsorted(centers + cell_width_s / 2.0, times, side="right")
    if np.any(positions >= centers.size):
        raise ValueError("motion field does not cover requested sample")
    return states[positions]


def exact_mapping(geometry: np.ndarray, shift_um: float) -> np.ndarray:
    geometry = np.asarray(geometry, dtype=np.float64)
    if geometry.shape != (384, 2) or not np.isfinite(geometry).all():
        raise ValueError("geometry must be finite (384,2)")
    mapping = np.full(384, -1, dtype=np.int64)
    for target, (x, y) in enumerate(geometry):
        hits = np.flatnonzero(np.all(np.abs(geometry - [x, y + shift_um]) <= 1e-6, axis=1))
        if hits.size > 1:
            raise ValueError("ambiguous exact-coordinate source")
        if hits.size == 1:
            mapping[target] = int(hits[0])
    if np.unique(mapping[mapping >= 0]).size != np.count_nonzero(mapping >= 0):
        raise ValueError("mapping is not injective")
    return mapping


def remap_per_sample(original: np.ndarray, states_um: np.ndarray,
                     mappings: dict[float, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    original = np.asarray(original)
    states = np.asarray(states_um, dtype=np.float64)
    if original.ndim != 2 or original.shape[0] != 384 or original.shape[1] != states.size:
        raise ValueError("array/state shape mismatch")
    output = np.zeros_like(original)
    support = np.zeros(original.shape, dtype=bool)
    for state in np.unique(states):
        mapping = mappings[float(state)]
        columns = np.flatnonzero(states == state)
        targets = np.flatnonzero(mapping >= 0)
        output[np.ix_(targets, columns)] = original[np.ix_(mapping[targets], columns)]
        support[np.ix_(targets, columns)] = True
    return output, support


def stage_arrays(raw: np.ndarray) -> dict[str, np.ndarray]:
    raw64 = np.asarray(raw, dtype=np.float64)
    centered = raw64 - raw64.mean(axis=1, keepdims=True)
    car = centered - np.median(centered, axis=0, keepdims=True)
    return {"raw_counts": raw64, "temporal_centered": centered, "car_centered": car}


def frozen_windows(role: str, lag: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    effective_lag = lag if role == "short_pair" else 0
    if width != 121 + effective_lag:
        raise ValueError("member width differs from frozen rule")
    signal = np.arange(40, 101 + effective_lag)
    baseline = np.r_[np.arange(0, 30), np.arange(111 + effective_lag, 121 + effective_lag)]
    return signal, baseline


def channel_metrics(values: np.ndarray, signal: np.ndarray, baseline: np.ndarray) -> dict[str, np.ndarray]:
    base = values[:, baseline]
    sig = values[:, signal]
    median = np.median(base, axis=1)
    sigma = 1.4826 * np.median(np.abs(base - median[:, None]), axis=1)
    denominator = np.maximum(sigma, 1.0)
    ptp = np.ptp(sig, axis=1)
    return {
        "baseline_median": median,
        "robust_noise_sigma": sigma,
        "signal_ptp": ptp,
        "ptp_snr": ptp / denominator,
        "signal_rms_z": np.sqrt(np.mean((sig - median[:, None]) ** 2, axis=1)) / denominator,
    }


def local_domain(geometry: np.ndarray, anchor: int, radius_um: float) -> np.ndarray:
    return np.flatnonzero(np.linalg.norm(geometry - geometry[anchor], axis=1) <= radius_um + 1e-9)


def local_detectability(metrics: dict[str, np.ndarray], local: np.ndarray, threshold: float) -> dict[str, Any]:
    """Frozen local-domain maximum; global peak is descriptive and not a gate."""
    snr = metrics["ptp_snr"]
    local_peak = int(local[np.flatnonzero(snr[local] == snr[local].max())[0]])
    global_peak = int(np.flatnonzero(snr == snr.max())[0])
    value = float(snr[local_peak])
    return {"local_peak_channel": local_peak, "local_max_ptp_snr": value,
            "locally_detectable": bool(value >= threshold), "global_peak_channel_descriptive": global_peak,
            "global_peak_ptp_snr_descriptive": float(snr[global_peak])}


def negative_control(expected: np.ndarray, corrected: np.ndarray, alternative: np.ndarray,
                     states_um: np.ndarray, name: str) -> dict[str, Any]:
    nonzero = bool(np.any(np.asarray(states_um) != 0))
    distinct = bool(np.any(alternative != expected))
    nonvacuous = nonzero and distinct
    fraction = float(np.mean(alternative == corrected))
    return {"comparison": name, "enabled": nonvacuous, "nonvacuous": nonvacuous,
            "equality_fraction": fraction, "passed": None if not nonvacuous else fraction < 0.99}


def shape_correlation_supported(a: np.ndarray, b: np.ndarray, support: np.ndarray) -> float | None:
    left = np.asarray(a, dtype=np.float64)[support]
    right = np.asarray(b, dtype=np.float64)[support]
    if left.size == 0:
        return None
    left = left - left.mean(); right = right - right.mean()
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return None if denominator == 0 else float(np.dot(left, right) / denominator)


def atomic_json(path: Path, value: Any) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(partial, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 20 or len({row["slot_id"] for row in rows}) != 20:
        raise ValueError("expected 20 unique frozen slots")
    return rows


def metadata_preflight(config: dict[str, Any]) -> dict[str, Any]:
    """May read field/selection/geometry metadata, never retained waveforms."""
    if config["status"] != "review_pending_execution_disabled" or any(config["gates"].values()):
        raise PermissionError("candidate must remain execution-disabled")
    for name in ("selection_receipt", "channel_positions", "motion_field", "historical_fail_summary"):
        item = config["inputs"][name]
        if sha256(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"metadata hash mismatch: {name}")
    rows = load_rows(Path(config["inputs"]["selection_receipt"]["path"]))
    geometry = np.load(config["inputs"]["channel_positions"]["path"], allow_pickle=False)
    if geometry.shape != (384, 2):
        raise ValueError("geometry shape mismatch")
    with np.load(config["inputs"]["motion_field"]["path"], allow_pickle=False) as field:
        if set(field.files) != set(config["field_binding"]["keys"]):
            raise ValueError("motion-field keys differ")
        times = np.asarray(field[config["field_binding"]["time_key"]], dtype=np.float64)
        displacement = np.asarray(field[config["field_binding"]["displacement_key"]], dtype=np.float64)
        if str(field["sign"].tolist()) != config["field_binding"]["sign"]:
            raise ValueError("motion sign differs")
    states, reference, width = rounded_field(times, displacement, config["field_binding"]["period_um"])
    observed = {"centers": int(times.size), "first_center_s": float(times[0]), "last_center_s": float(times[-1]),
                "cell_width_s": width, "reference_um": reference,
                "rounded_states_um": np.unique(states).tolist()}
    if observed != config["field_binding"]["expected_metadata"]:
        raise ValueError(f"field metadata differs: {observed}")
    return {"selection_slots": len(rows), "geometry_shape": list(geometry.shape), **observed,
            "retained_archives_opened": 0, "historical_fail_preserved": True}


def execute(config: dict[str, Any], output: Path) -> None:
    """Reviewed real-data path; unreachable with the published disabled gates."""
    if config["gates"] != {"execution_enabled": True, "independent_h1_review_bound": True,
                            "outcome_archive_open_authorized": True}:
        raise PermissionError("execution remains disabled pending independent review")
    if output.exists():
        raise FileExistsError("fresh output root required")
    for name, item in config["inputs"].items():
        if sha256(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"input hash mismatch: {name}")
    rows = load_rows(Path(config["inputs"]["selection_receipt"]["path"]))
    geometry = np.load(config["inputs"]["channel_positions"]["path"], allow_pickle=False)
    with np.load(config["inputs"]["motion_field"]["path"], allow_pickle=False) as field:
        centers = np.asarray(field[config["field_binding"]["time_key"]], dtype=np.float64)
        displacement = np.asarray(field[config["field_binding"]["displacement_key"]], dtype=np.float64)
    states_table, _, cell_width = rounded_field(centers, displacement, config["field_binding"]["period_um"])
    mappings = {float(q): exact_mapping(geometry, float(q)) for q in np.unique(np.r_[states_table, -states_table])}
    archive_names = ("original_archive", "corrected_archive")
    with np.load(config["inputs"][archive_names[0]]["path"], allow_pickle=False) as left, \
         np.load(config["inputs"][archive_names[1]]["path"], allow_pickle=False) as right:
        keys = {row["slot_id"] for row in rows}
        if set(left.files) != keys or set(right.files) != keys:
            raise ValueError("archive keys differ from frozen selections")
        arrays = {"original": {key: left[key] for key in left.files},
                  "corrected_A": {key: right[key] for key in right.files}}
    numeric_bytes = sum(array.nbytes for representation in arrays.values() for array in representation.values())
    if numeric_bytes > config["resources"]["archive_numeric_bytes_max"]:
        raise RuntimeError("archive numeric-byte ceiling exceeded")
    output.mkdir(parents=False)
    sample_rows: list[dict[str, Any]] = []
    slot_rows: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    visible: dict[int, list[bool]] = {int(unit): [] for unit in config["unit_channel_identities"]}
    fs = float(config["clock_and_edges"]["sampling_frequency_hz"])
    origin = float(config["clock_and_edges"]["time_origin_s"])
    for row in rows:
        slot = row["slot_id"]; unit = int(row["unit_id"]); lag = int(row["lag_samples"])
        effective_lag = lag if row["role"] == "short_pair" else 0
        width = 121 + effective_lag
        original = arrays["original"][slot]; corrected = arrays["corrected_A"][slot]
        if original.shape != (384, width) or corrected.shape != original.shape:
            raise ValueError(f"member shape mismatch: {slot}")
        if original.dtype != np.int16 or corrected.dtype != np.int16:
            raise ValueError(f"member dtype mismatch: {slot}")
        start_frame = int(row["frame_1"]) - 60
        frames = start_frame + np.arange(width, dtype=np.int64)
        sample_times = origin + frames.astype(np.float64) / fs
        states = sample_stepwise_states(centers, states_table, sample_times, cell_width)
        expected, support = remap_per_sample(original, states, mappings)
        equality = float(np.mean(expected == corrected))
        comparisons.append({"slot_id": slot, "comparison": "per_sample_exact_remap", "enabled": True,
                            "nonvacuous": True, "passed": bool(np.array_equal(expected, corrected)),
                            "equality_fraction": equality})
        all_q0 = bool(np.all(states == 0))
        comparisons.append({"slot_id": slot, "comparison": "conditional_q0_identity", "enabled": all_q0,
                            "nonvacuous": all_q0, "passed": bool(np.array_equal(original, corrected)) if all_q0 else None,
                            "equality_fraction": float(np.mean(original == corrected)) if all_q0 else ""})
        no_shift = original
        wrong_sign, _ = remap_per_sample(original, -states, mappings)
        for control in (negative_control(expected, corrected, no_shift, states, "no_shift"),
                        negative_control(expected, corrected, wrong_sign, states, "wrong_sign")):
            comparisons.append({"slot_id": slot, **control})
        signal, baseline = frozen_windows(row["role"], lag, width)
        anchor = int(config["unit_channel_identities"][str(unit)]["corrected_anchor_channel"])
        local = local_domain(geometry, anchor, config["frozen_selections_and_thresholds"]["local_radius_um"])
        staged = {"aligned_original": stage_arrays(expected), "corrected_A": stage_arrays(corrected)}
        for representation, stages in staged.items():
            for stage, values in stages.items():
                metrics = channel_metrics(values, signal, baseline)
                detect = local_detectability(metrics, local,
                    config["frozen_selections_and_thresholds"]["local_ptp_snr_threshold"])
                slot_rows.append({"slot_id": slot, "unit_id": unit, "role": row["role"],
                                  "representation": representation, "stage": stage,
                                  "corrected_anchor_channel": anchor,
                                  "local_median_ptp_snr": float(np.median(metrics["ptp_snr"][local])), **detect})
                if representation == "corrected_A" and stage == "car_centered" and row["role"] != "short_pair":
                    visible[unit].append(bool(detect["locally_detectable"]))
        local_signal_mask = support[np.ix_(local, signal)]
        correlation = shape_correlation_supported(
            staged["aligned_original"]["car_centered"][np.ix_(local, signal)],
            staged["corrected_A"]["car_centered"][np.ix_(local, signal)], local_signal_mask)
        comparisons.append({"slot_id": slot, "comparison": "supported_local_car_correlation_descriptive",
                            "enabled": correlation is not None, "nonvacuous": correlation is not None,
                            "passed": None, "correlation": correlation if correlation is not None else ""})
        values, counts = np.unique(states, return_counts=True)
        sample_rows.append({"slot_id": slot, "global_start_frame": int(frames[0]),
                            "global_stop_frame_exclusive": int(frames[-1] + 1),
                            "unique_states_um": json.dumps(values.tolist()),
                            "state_sample_counts": json.dumps(counts.tolist()),
                            "receipt_event_center_state_um": int(row["state_um"]),
                            "first_event_sample_state_um": float(states[60]),
                            "second_event_sample_state_um": float(states[60 + lag]) if row["role"] == "short_pair" else "",
                            "supported_fraction": float(support.mean())})
    integrity_rows = [row for row in comparisons if row["comparison"] == "per_sample_exact_remap" or
                      (row["comparison"] == "conditional_q0_identity" and row["enabled"]) or
                      (row["comparison"] in {"no_shift", "wrong_sign"} and row["enabled"])]
    integrity = all(row["passed"] is True for row in integrity_rows)
    usable = {str(unit): sum(flags) >= config["frozen_selections_and_thresholds"]["unit_usable_minimum_detectable_singles"]
              for unit, flags in visible.items()}
    count = sum(usable.values())
    verdict = "PASS" if integrity and count >= 3 else ("CAUTION" if integrity and count == 2 else "FAIL")
    write_csv(output / "SAMPLE_STATE_SUMMARY.csv", sample_rows)
    write_csv(output / "SLOT_METRICS.csv", slot_rows)
    write_csv(output / "COMPARISONS.csv", comparisons)
    atomic_json(output / "VALIDATION.json", {"archives": 2, "members_per_archive": 20,
        "numeric_bytes": numeric_bytes, "recording_reads": 0, "historical_fail_preserved": True})
    atomic_json(output / "DECISION.json", {"verdict": verdict, "integrity_controls_passed": integrity,
        "unit_usable": usable, "usable_unit_count": count, "scope": config["decision_scope"],
        "prohibited_claims": config["prohibited_claims"]})
    if sum(path.stat().st_size for path in output.rglob("*") if path.is_file()) > config["resources"]["persistent_output_bytes_max"]:
        raise RuntimeError("persistent-output ceiling exceeded")
    atomic_json(output / "COMPLETE.json", {"status": "complete", "verdict": verdict,
        "recording_reads": 0, "waveform_payload_published": False})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--phase", choices=("preflight", "run"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if sha256(args.config) != args.expected_config_sha256:
        raise ValueError("config hash mismatch")
    config = json.loads(args.config.read_text())
    if args.phase == "preflight":
        print(json.dumps({"status": config["status"], "metadata": metadata_preflight(config)}, sort_keys=True))
    else:
        if args.output is None:
            raise ValueError("run phase requires --output")
        execute(config, args.output)


if __name__ == "__main__":
    main()
