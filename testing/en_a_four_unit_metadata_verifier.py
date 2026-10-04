#!/usr/bin/env python3
"""Independent, read-only verifier for the frozen EN arm-A four-unit selections.

This program reads only saved sorter/model metadata (``.npy/.npz``), an A-field
metadata file, and the frozen CSV selection contract.  It never accepts or
opens a recording binary.  Outputs contain event coordinates and scalar model
summaries only; numeric template arrays are never emitted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "en-a-four-unit-independent-metadata-verifier-v1"
OUTPUT_SCHEMA = "en-a-four-unit-independent-metadata-evidence-v1"
EXPECTED_UNITS = (278, 271, 588, 445)
PAIR_MAX_LAG = 29
CONTROL_MIN_NEIGHBOR_LAG = 300
FORBIDDEN_INPUT_SUFFIXES = {".raw", ".bin", ".dat", ".cbin"}
EXPECTED_RULES = {
    "short_pair": "earliest original adjacent same-unit both-supported pair with lag 1..29 in this state and one contiguous segment",
    "ordinary_event": "earliest same-unit event in this state with both same-unit neighboring lags at least 300 samples and not a selected pair endpoint",
    "q0_identity_control": "earliest q0 same-unit event with both same-unit neighboring lags at least 300 samples",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path: Path, expected: str, label: str) -> str:
    if path.suffix.lower() in FORBIDDEN_INPUT_SUFFIXES:
        raise ValueError(f"{label}: recording-like input is forbidden: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{label}: missing file: {path}")
    observed = sha256_file(path)
    if observed != expected.lower():
        raise ValueError(f"{label}: SHA-256 mismatch: {observed} != {expected}")
    return observed


def atomic_text(path: Path, text: str) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(text)
    os.replace(partial, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def half_away_round(values: np.ndarray, quantum: float) -> np.ndarray:
    scaled = np.asarray(values, dtype=np.float64) / float(quantum)
    return float(quantum) * np.copysign(np.floor(np.abs(scaled) + 0.5), scaled)


def load_contract(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"slot_id", "unit_id", "role", "state_um"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"selection contract must contain {sorted(required)}")
    normalized = []
    seen = set()
    for row in rows:
        slot = row["slot_id"]
        if slot in seen:
            raise ValueError(f"duplicate slot_id: {slot}")
        seen.add(slot)
        role = row["role"]
        if role not in {"short_pair", "ordinary_event", "q0_identity_control"}:
            raise ValueError(f"unsupported role in {slot}: {role}")
        if row.get("selection_rule") != EXPECTED_RULES[role]:
            raise ValueError(f"selection rule changed for {slot}")
        normalized.append({
            "slot_id": slot,
            "unit_id": int(row["unit_id"]),
            "role": role,
            "state_um": int(float(row["state_um"])),
            "source_row": row,
        })
    counts = {role: sum(r["role"] == role for r in normalized) for role in
              ("short_pair", "ordinary_event", "q0_identity_control")}
    if len(normalized) != 20 or counts != {
        "short_pair": 8, "ordinary_event": 8, "q0_identity_control": 4
    }:
        raise ValueError(f"expected frozen 20-slot 8/8/4 contract, got {counts}")
    if set(r["unit_id"] for r in normalized) != set(EXPECTED_UNITS):
        raise ValueError("selection contract unit set differs from frozen four units")
    return normalized


def field_cells(
    path: Path, *, time_key: str, displacement_key: str, rounding_um: float
) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        centers = np.asarray(saved[time_key], dtype=np.float64)
        displacement = np.asarray(saved[displacement_key], dtype=np.float64)
    if centers.ndim != 1 or centers.size < 2 or not np.isfinite(centers).all():
        raise ValueError("field time centers must be a finite 1-D array")
    steps = np.diff(centers)
    dt = float(np.median(steps))
    if not np.allclose(steps, dt, rtol=0, atol=1e-9):
        raise ValueError("field time centers are not uniform")
    if displacement.shape[0] != centers.size or not np.isfinite(displacement).all():
        raise ValueError("field displacement/time shape or finiteness mismatch")
    rigid = displacement if displacement.ndim == 1 else np.median(displacement, axis=tuple(range(1, displacement.ndim)))
    states = half_away_round(rigid, rounding_um).astype(np.int64)
    segments = np.cumsum(np.r_[True, states[1:] != states[:-1]]).astype(np.int64) - 1
    edges = np.r_[centers - dt / 2, centers[-1] + dt / 2]
    return {"centers": centers, "states": states, "segments": segments, "edges": edges}


def label_frames(frames: np.ndarray, fs: float, cells: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    seconds = np.asarray(frames, dtype=np.float64) / float(fs)
    knots = np.searchsorted(cells["edges"], seconds, side="right") - 1
    knots = np.clip(knots, 0, cells["states"].size - 1)
    return cells["states"][knots], cells["segments"][knots]


def support_from_saved_y(y: np.ndarray, states: np.ndarray, geometry: np.ndarray) -> np.ndarray:
    """Reviewed exact-lattice y-support: target y + displacement stays on probe bounds."""
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(y)
    source_y = y + np.asarray(states, dtype=np.float64)
    lo = float(np.min(geometry[:, 1]))
    hi = float(np.max(geometry[:, 1]))
    return finite & (source_y >= lo) & (source_y <= hi)


def check_global_order(times: np.ndarray, chunk: int = 2_000_000) -> None:
    previous = None
    total = int(times.size)
    for start in range(0, total, chunk):
        values = np.asarray(times[start:min(total, start + chunk)], dtype=np.int64).reshape(-1)
        if previous is not None and values.size and values[0] < previous:
            raise ValueError("spike_times are not globally nondecreasing")
        if values.size > 1 and np.any(values[1:] < values[:-1]):
            raise ValueError("spike_times are not globally nondecreasing")
        if values.size:
            previous = int(values[-1])


def unit_metadata(
    unit: int, times: np.ndarray, clusters: np.ndarray, templates: np.ndarray,
    positions: np.ndarray, fs: float, cells: dict[str, np.ndarray], geometry: np.ndarray,
) -> dict[str, np.ndarray]:
    indices = np.flatnonzero(np.asarray(clusters) == unit).astype(np.int64)
    if indices.size == 0:
        raise ValueError(f"unit {unit} has no events")
    frames = np.asarray(times[indices], dtype=np.int64).reshape(-1)
    labels = np.asarray(clusters[indices], dtype=np.int64).reshape(-1)
    tids = np.asarray(templates[indices], dtype=np.int64).reshape(-1)
    y = np.asarray(positions[indices, 1], dtype=np.float64).reshape(-1)
    if not np.all(labels == unit):
        raise AssertionError("unit extraction failed")
    states, segments = label_frames(frames, fs, cells)
    support = support_from_saved_y(y, states, geometry)
    return {
        "indices": indices, "frames": frames, "templates": tids, "y": y,
        "states": states, "segments": segments, "support": support,
    }


def event_row(slot: dict[str, Any], data: dict[str, np.ndarray], left: int, right: int | None = None) -> dict[str, Any]:
    i = int(left)
    j = None if right is None else int(right)
    row = {
        "slot_id": slot["slot_id"], "unit_id": slot["unit_id"], "role": slot["role"],
        "state_um": int(data["states"][i]), "event_index_1": int(data["indices"][i]),
        "event_index_2": "" if j is None else int(data["indices"][j]),
        "frame_1": int(data["frames"][i]), "frame_2": "" if j is None else int(data["frames"][j]),
        "seconds_1": float(data["frames"][i]), "seconds_2": "" if j is None else float(data["frames"][j]),
        "lag_samples": 0 if j is None else int(data["frames"][j] - data["frames"][i]),
        "previous_same_unit_lag_samples": "" if i == 0 else int(data["frames"][i] - data["frames"][i - 1]),
        "following_same_unit_lag_samples": "" if i + 1 == data["frames"].size else int(data["frames"][i + 1] - data["frames"][i]),
        "segment_id": int(data["segments"][i]),
        "support_1": "supported" if bool(data["support"][i]) else "unsupported",
        "support_2": "" if j is None else ("supported" if bool(data["support"][j]) else "unsupported"),
        "saved_y_1_um": float(data["y"][i]), "saved_y_2_um": "" if j is None else float(data["y"][j]),
        "template_id_1": int(data["templates"][i]), "template_id_2": "" if j is None else int(data["templates"][j]),
        "status": "resolved",
    }
    return row


def regenerate_selections(contract: list[dict[str, Any]], units: dict[int, dict[str, np.ndarray]], fs: float) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    used_pair_endpoints: dict[int, set[int]] = {unit: set() for unit in units}
    for slot in (s for s in contract if s["role"] == "short_pair"):
        data = units[slot["unit_id"]]
        lag = data["frames"][1:] - data["frames"][:-1]
        eligible = (
            (lag >= 1) & (lag <= PAIR_MAX_LAG)
            & (data["states"][:-1] == slot["state_um"])
            & (data["states"][1:] == slot["state_um"])
            & (data["segments"][:-1] == data["segments"][1:])
            & data["support"][:-1] & data["support"][1:]
        )
        choices = np.flatnonzero(eligible)
        if choices.size == 0:
            raise ValueError(f"no qualifying short pair for {slot['slot_id']}")
        left = int(choices[0]); right = left + 1
        used_pair_endpoints[slot["unit_id"]].update((left, right))
        selected[slot["slot_id"]] = event_row(slot, data, left, right)

    for slot in (s for s in contract if s["role"] != "short_pair"):
        data = units[slot["unit_id"]]
        previous = data["frames"][1:-1] - data["frames"][:-2]
        following = data["frames"][2:] - data["frames"][1:-1]
        target = 0 if slot["role"] == "q0_identity_control" else slot["state_um"]
        eligible = (
            (data["states"][1:-1] == target)
            & (previous >= CONTROL_MIN_NEIGHBOR_LAG)
            & (following >= CONTROL_MIN_NEIGHBOR_LAG)
        )
        choices = np.flatnonzero(eligible) + 1
        choices = np.asarray([i for i in choices if int(i) not in used_pair_endpoints[slot["unit_id"]]], dtype=np.int64)
        if choices.size == 0:
            raise ValueError(f"no qualifying control for {slot['slot_id']}")
        selected[slot["slot_id"]] = event_row(slot, data, int(choices[0]))

    output = []
    for slot in contract:
        row = selected[slot["slot_id"]]
        row["seconds_1"] = row["frame_1"] / float(fs)
        if row["frame_2"] != "":
            row["seconds_2"] = row["frame_2"] / float(fs)
        if row["state_um"] != slot["state_um"]:
            raise AssertionError(f"state mismatch for {slot['slot_id']}")
        output.append(row)
    return output


def template_summaries(
    template_bank: np.ndarray, geometry: np.ndarray,
    unit_template_ids: dict[int, list[int]],
) -> list[dict[str, Any]]:
    if template_bank.ndim != 3 or template_bank.shape[2] != geometry.shape[0]:
        raise ValueError("templates must have shape template x time x channel matching geometry")
    rows = []
    for unit in sorted(unit_template_ids):
        for template_id in unit_template_ids[unit]:
            if template_id < 0 or template_id >= template_bank.shape[0]:
                raise ValueError(f"template {template_id} lies outside exported bank")
            waveform = np.asarray(template_bank[template_id], dtype=np.float64)
            finite = np.isfinite(waveform)
            if not finite.all():
                raise ValueError(f"template {template_id} contains nonfinite values")
            p2p = np.ptp(waveform, axis=0)
            peak = int(np.argmax(p2p))
            maximum = float(p2p[peak])
            rows.append({
                "unit_id": unit, "exported_template_id": template_id,
                "template_time_samples": int(waveform.shape[0]), "channels": int(waveform.shape[1]),
                "all_values_finite": True, "finite_channels": int(finite.all(axis=0).sum()),
                "peak_channel_index": peak, "peak_x_um": float(geometry[peak, 0]),
                "peak_row_y_um": float(geometry[peak, 1]), "p2p_min": float(p2p.min()),
                "p2p_median": float(np.median(p2p)), "p2p_max": maximum,
                "active_channels_at_20pct_peak_p2p": int(np.count_nonzero(p2p >= 0.2 * maximum)),
                "coordinate_frame": "arm_A_materialized_sorter_channel_coordinates",
                "provenance": "post-sort exported templates.npy; not the unavailable pre-extraction detection bank",
                "interpretation": "model-derived scalar footprint summary; not measured identity or waveform export",
            })
    return rows


def scalar_ops_checks(ops: dict[str, Any], fs: float, template_bank: np.ndarray, geometry: np.ndarray) -> dict[str, Any]:
    observed_fs = float(ops["fs"])
    if not math.isclose(observed_fs, fs, rel_tol=0, abs_tol=1e-12):
        raise ValueError("ops sampling rate differs from requested clock")
    if int(ops["nt"]) != int(template_bank.shape[1]):
        raise ValueError("ops nt differs from exported template length")
    chan_map = np.asarray(ops["chanMap"], dtype=np.int64)
    if not np.array_equal(chan_map, np.arange(geometry.shape[0])):
        raise ValueError("ops chanMap is not full-probe identity")
    if "xc" in ops and "yc" in ops:
        ops_geometry = np.c_[np.asarray(ops["xc"]), np.asarray(ops["yc"])]
        if not np.array_equal(ops_geometry, geometry):
            raise ValueError("ops geometry differs from channel_positions")
    return {"fs": observed_fs, "nt": int(ops["nt"]), "identity_chanmap": True,
            "channels": int(geometry.shape[0])}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("spike-times", "spike-clusters", "spike-templates", "spike-positions",
                 "templates", "ops", "channel-positions", "field", "selection-contract"):
        parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument(f"--{name}-sha256", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--sampling-frequency-hz", type=float, required=True)
    parser.add_argument("--field-time-key", default="time_s")
    parser.add_argument("--field-displacement-key", default="displacement_um")
    parser.add_argument("--rounding-um", type=float, default=40.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = Path(__file__).resolve()
    require_hash(source_path, args.expected_source_sha256, "verifier source")
    bindings = {}
    for cli in ("spike_times", "spike_clusters", "spike_templates", "spike_positions",
                "templates", "ops", "channel_positions", "field", "selection_contract"):
        path = Path(getattr(args, cli))
        digest = getattr(args, f"{cli}_sha256")
        bindings[cli] = {"path": str(path.resolve()), "sha256": require_hash(path, digest, cli)}
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")

    times = np.load(args.spike_times, mmap_mode="r", allow_pickle=False).reshape(-1)
    clusters = np.load(args.spike_clusters, mmap_mode="r", allow_pickle=False).reshape(-1)
    assigned_templates = np.load(args.spike_templates, mmap_mode="r", allow_pickle=False).reshape(-1)
    positions = np.load(args.spike_positions, mmap_mode="r", allow_pickle=False)
    template_bank = np.load(args.templates, mmap_mode="r", allow_pickle=False)
    geometry = np.asarray(np.load(args.channel_positions, allow_pickle=False), dtype=np.float64)
    ops = np.load(args.ops, allow_pickle=True).item()
    n_events = int(times.size)
    if not (clusters.size == assigned_templates.size == positions.shape[0] == n_events):
        raise ValueError("saved event arrays are not aligned")
    if positions.ndim != 2 or positions.shape[1] < 2:
        raise ValueError("spike_positions must provide a y column")
    if geometry.ndim != 2 or geometry.shape[1] != 2:
        raise ValueError("channel_positions must be channels x 2")
    check_global_order(times)
    fs = float(args.sampling_frequency_hz)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("sampling frequency must be finite and positive")
    contract = load_contract(args.selection_contract)
    cells = field_cells(args.field, time_key=args.field_time_key,
                        displacement_key=args.field_displacement_key, rounding_um=args.rounding_um)
    per_unit = {
        unit: unit_metadata(unit, times, clusters, assigned_templates, positions, fs, cells, geometry)
        for unit in EXPECTED_UNITS
    }
    selections = regenerate_selections(contract, per_unit, fs)
    mappings = []
    unit_template_ids = {}
    for unit in sorted(EXPECTED_UNITS):
        tids, counts = np.unique(per_unit[unit]["templates"], return_counts=True)
        unit_template_ids[unit] = [int(t) for t in tids]
        mappings.append({"unit_id": unit, "event_count": int(counts.sum()),
                         "template_distribution": {str(int(t)): int(c) for t, c in zip(tids, counts)}})
    footprints = template_summaries(template_bank, geometry, unit_template_ids)
    ops_summary = scalar_ops_checks(ops, fs, template_bank, geometry)

    args.output.mkdir(parents=True)
    fields = list(selections[0])
    with (args.output / "SELECTION_EVIDENCE.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(selections)
    evidence = {
        "schema": OUTPUT_SCHEMA, "verifier_schema": SCHEMA,
        "verifier_source": {"path": str(source_path), "sha256": args.expected_source_sha256},
        "inputs": bindings, "recording_binary_opened": False, "voltage_read": False,
        "event_arrays_aligned": True, "global_event_count": n_events,
        "global_times_nondecreasing": True, "sampling_frequency_hz": fs,
        "field": {"rounding_um": float(args.rounding_um), "states": sorted(set(map(int, cells["states"])))},
        "ops_checks": ops_summary, "unit_template_mappings": mappings,
        "model_derived_template_footprints": footprints,
        "selection_rows": len(selections), "selection_status": "all_resolved",
        "interpretation_limits": [
            "saved y positions and exported templates are model-derived, not biological identity evidence",
            "short intervals do not establish contamination or purity",
            "exported templates are not the unavailable pre-extraction detection bank",
        ],
    }
    atomic_json(args.output / "EVIDENCE.json", evidence)
    products = []
    for path in sorted(args.output.iterdir()):
        products.append({"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    atomic_json(args.output / "MANIFEST.json", {"schema": OUTPUT_SCHEMA + "-manifest", "products": products})
    manifest_sha = sha256_file(args.output / "MANIFEST.json")
    atomic_json(args.output / "COMPLETE.json", {
        "schema": OUTPUT_SCHEMA + "-complete", "status": "complete_metadata_only",
        "manifest_sha256": manifest_sha, "recording_binary_opened": False, "voltage_read": False,
    })


if __name__ == "__main__":
    main()
