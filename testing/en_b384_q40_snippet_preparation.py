#!/usr/bin/env python3
"""Prepare, but never read voltage for, a bounded B384 q+40 snippet pilot."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from testing.en_common_support_crop_panel import crop_field_cells_from_global_midpoints
from testing.en_rounded_field import rounded_rigid_field


SCHEMA = "en-b384-q40-snippet-preparation-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def verify(config: dict[str, Any]) -> None:
    if config["voltage_reads_permitted"]:
        raise RuntimeError("preparation config must prohibit voltage reads")
    for item in config["frozen_inputs"]:
        if sha256(Path(item["path"])) != item["sha256"]:
            raise RuntimeError(f"frozen input differs: {item['path']}")


def load_cells(config: dict[str, Any]) -> Any:
    field = config["field"]
    with np.load(field["path"], allow_pickle=False) as saved:
        displacement = np.asarray(saved[field["displacement_key"]], dtype=float)
        if displacement.ndim == 2:
            displacement = np.median(displacement, axis=1)
        rounded, _ = rounded_rigid_field(displacement)
        return crop_field_cells_from_global_midpoints(
            np.asarray(saved[field["time_key"]], dtype=float), rounded,
            float(config["recording_duration_s"]),
            int(config["crop_start_frame"]) / float(config["sampling_frequency_hz"]),
            int(config["crop_stop_frame"]) / float(config["sampling_frequency_hz"]),
            float(config["accounting_block_seconds"]),
        )


def protocol() -> str:
    return """# Frozen snippet-preparation protocol

- Read only saved spike times, final clusters, detection-template IDs, frozen field, population
  membership, saved ops/bank metadata, and file metadata. Do not open the recording binary.
- Construct adjacency on every complete B384 final-unit event train before filtering. Keep positive
  lags 1..29 samples within the same original contiguous state segment and assign state from the
  second event. Restrict to q=+40 um only after adjacency exists.
- Use the original fixed median-depth population labels. Rank excluded units by q+40 short-pair
  count descending, then unit ID ascending; select two. Rank common-interior units the same way;
  select two controls. This is intentionally contribution-targeted, not representative sampling.
- For each unit, select the original 2 s Kilosort batch containing the most margin-safe q+40 short
  pairs, ties by lower batch index. Within it freeze four pairs at evenly spaced order-statistic
  positions (or all if fewer). Both events must be in the same batch core and at least 61 samples
  from its boundary. Also freeze the earliest event in that batch separated by more than 61 samples
  from both neighboring events in the same complete unit train; this is not global isolation.
  Do not re-pair after any filter.
- Future pilot may read at most the selected unique padded batches with saved `ops.npy` through
  Kilosort `bfile_from_ops`, CPU only. One padded batch is (60000+2*61)*384*2 logical raw bytes.
- Purpose: inspect whether actual close detections are compatible with one waveform counted twice,
  two distinguishable waveform components, or non-spike/artifact-like signal. Residual structure,
  by itself, is not biological identity or truth.
"""


def run(config: dict[str, Any], output: Path) -> None:
    verify(config)
    sorter = Path(config["sorter_output"])
    times_abs = np.asarray(np.load(sorter / "spike_times.npy", mmap_mode="r"), dtype=np.int64).reshape(-1)
    clusters = np.asarray(np.load(sorter / "spike_clusters.npy", mmap_mode="r"), dtype=np.int64).reshape(-1)
    templates = np.asarray(np.load(sorter / "spike_detection_templates.npy", mmap_mode="r"), dtype=np.int64).reshape(-1)
    if not (len(times_abs) == len(clusters) == len(templates)) or np.any(np.diff(times_abs) < 0):
        raise RuntimeError("saved event arrays are unaligned or unsorted")
    start, stop = int(config["crop_start_frame"]), int(config["crop_stop_frame"])
    if times_abs[0] < start or times_abs[-1] >= stop:
        raise RuntimeError("event outside crop")
    times = times_abs - start
    cells = load_cells(config)
    state_index, segment = cells.assign(times, float(config["sampling_frequency_hz"]))
    state_um = np.asarray(cells.states_um)[state_index]

    membership = pd.read_csv(config["population_membership_path"])
    population = dict(zip(membership.unit_id.astype(int), membership.population.astype(str)))
    median_depth = dict(zip(membership.unit_id.astype(int), membership.median_saved_y_um.astype(float)))
    order = np.lexsort((times, clusters))
    u, t, d = clusters[order], times[order], templates[order]
    s, g = state_um[order], segment[order]
    lag = np.diff(t)
    safe = (u[1:] == u[:-1]) & (g[1:] == g[:-1]) & (lag >= 1) & (lag <= 29)
    q40 = safe & (s[1:] == 40.0)
    pair_frame = pd.DataFrame({
        "unit_id": u[1:][q40], "first_time_local": t[:-1][q40], "second_time_local": t[1:][q40],
        "lag_samples": lag[q40], "first_template": d[:-1][q40], "second_template": d[1:][q40],
    })
    counts = pair_frame.groupby("unit_id").size().rename("q40_short_pairs").reset_index()
    counts["population"] = counts.unit_id.map(population)
    counts["original_median_depth_um"] = counts.unit_id.map(median_depth)
    counts = counts.sort_values(["q40_short_pairs", "unit_id"], ascending=[False, True])
    excluded = counts[counts.population.isin(["lower_excluded", "upper_excluded"])].head(2).copy()
    interior = counts[counts.population.eq("common")].head(2).copy()
    if len(excluded) != 2 or len(interior) != 2:
        raise RuntimeError("insufficient selected units")
    excluded["selection_role"] = "excluded_target"
    interior["selection_role"] = "interior_control"
    selected = pd.concat([excluded, interior], ignore_index=True)
    selected["selection_rank_within_role"] = selected.groupby("selection_role").cumcount() + 1

    nt, batch_size = int(config["nt"]), int(config["batch_size"])
    sample_rows, isolated_rows, batch_rows = [], [], []
    for row in selected.itertuples(index=False):
        pairs = pair_frame[pair_frame.unit_id == row.unit_id].copy()
        pairs["batch_index"] = (pairs.second_time_local // batch_size).astype(int)
        pairs["batch_core_start_local"] = pairs.batch_index * batch_size
        pairs["batch_core_stop_local"] = np.minimum((pairs.batch_index + 1) * batch_size, stop - start)
        margin = (
            (pairs.first_time_local >= pairs.batch_core_start_local + nt)
            & (pairs.second_time_local < pairs.batch_core_stop_local - nt)
            & ((pairs.first_time_local // batch_size) == pairs.batch_index)
        )
        safe_pairs = pairs[margin].copy()
        batch_counts = safe_pairs.groupby("batch_index").size().sort_index()
        best_count = int(batch_counts.max())
        batch_index = int(batch_counts[batch_counts == best_count].index.min())
        candidates = safe_pairs[safe_pairs.batch_index == batch_index].sort_values(
            ["second_time_local", "first_time_local"], kind="stable"
        )
        take = min(4, len(candidates))
        positions = np.unique(np.linspace(0, len(candidates) - 1, take, dtype=int))
        chosen = candidates.iloc[positions]
        batch_rows.append({
            "unit_id": int(row.unit_id), "selection_role": row.selection_role,
            "batch_index": batch_index, "safe_q40_short_pairs_in_batch": len(candidates),
            "batch_core_start_global": start + batch_index * batch_size,
            "batch_core_stop_global_exclusive": start + min((batch_index + 1) * batch_size, stop - start),
        })
        for sample_rank, pair in enumerate(chosen.itertuples(index=False), 1):
            sample_rows.append({
                "unit_id": int(row.unit_id), "selection_role": row.selection_role,
                "sample_rank": sample_rank, "batch_index": batch_index,
                "first_time_global": start + int(pair.first_time_local),
                "second_time_global": start + int(pair.second_time_local),
                "lag_samples": int(pair.lag_samples),
                "first_template": int(pair.first_template), "second_template": int(pair.second_template),
            })
        unit_events = np.flatnonzero((clusters == row.unit_id) & (state_um == 40.0))
        unit_local = times[unit_events]
        in_batch = (
            (unit_local >= batch_index * batch_size + nt)
            & (unit_local < min((batch_index + 1) * batch_size, stop - start) - nt)
        )
        candidate_unit_positions = np.flatnonzero(in_batch)
        candidate_indices = unit_events[candidate_unit_positions]
        candidates_local = times[candidate_indices]
        if len(candidates_local):
            left_pos = np.maximum(candidate_unit_positions - 1, 0)
            right_pos = np.minimum(candidate_unit_positions + 1, len(unit_local) - 1)
            left_gap = np.where(candidate_unit_positions > 0, candidates_local - unit_local[left_pos], np.iinfo(np.int64).max)
            right_gap = np.where(candidate_unit_positions + 1 < len(unit_local), unit_local[right_pos] - candidates_local, np.iinfo(np.int64).max)
            unit_isolated = (left_gap > nt) & (right_gap > nt)
            isolated_indices = candidate_indices[unit_isolated]
            if len(isolated_indices):
                event_index = int(isolated_indices[0])
                isolated_time = int(times[event_index])
                isolated_rows.append({
                    "unit_id": int(row.unit_id), "selection_role": row.selection_role,
                    "batch_index": batch_index, "time_global": start + isolated_time,
                    "detection_template": int(templates[event_index]),
                    "same_unit_event_exclusion_radius_samples": nt,
                })

    selected.to_csv(output / "SELECTED_UNITS.csv", index=False)
    pd.DataFrame(batch_rows).to_csv(output / "SELECTED_BATCHES.csv", index=False)
    pd.DataFrame(sample_rows).to_csv(output / "SELECTED_PAIRS.csv", index=False)
    pd.DataFrame(isolated_rows).to_csv(output / "SELECTED_ISOLATED_CONTROLS.csv", index=False)
    unique_batches = sorted({row["batch_index"] for row in batch_rows})
    bytes_per_padded_batch = (batch_size + 2 * nt) * int(config["n_chan_bin"]) * 2
    logical_bytes = len(unique_batches) * bytes_per_padded_batch
    state_metrics = pd.read_csv(config["state_metrics_path"])
    q = state_metrics[state_metrics.state_um.eq(40.0)].set_index(["arm", "scope"])
    all_excess = int(q.loc[("B384", "all_output"), "short_interval_count_1_29"] - q.loc[("REF384", "all_output"), "short_interval_count_1_29"])
    common_excess = int(q.loc[("B384", "common_unit_median_120_3780"), "short_interval_count_1_29"] - q.loc[("REF384", "common_unit_median_120_3780"), "short_interval_count_1_29"])
    excluded_excess = all_excess - common_excess
    result = {
        "schema": SCHEMA, "status": "prepared", "completed_at": datetime.now(timezone.utc).isoformat(),
        "raw_voltage_bytes_read": 0, "sorts_launched": 0, "rf_accesses": 0,
        "q40_all_output_excess_numerator": all_excess,
        "q40_common_interior_excess_numerator": common_excess,
        "q40_excluded_depth_excess_numerator": excluded_excess,
        "selected_units": selected.to_dict("records"),
        "selected_unique_batches": unique_batches,
        "future_padded_batch_count": len(unique_batches),
        "future_logical_recording_read_bytes": logical_bytes,
        "future_logical_recording_read_mib": logical_bytes / 2**20,
        "future_limits_pass": len(unique_batches) <= 4 and logical_bytes <= 512 * 2**20,
        "interpretation": "Selection/read-budget preparation only; no waveform outcome exists.",
    }
    atomic_json(output / "RESULT.json", result)
    products = ["PROTOCOL.md", "FREEZE.json", "SELECTED_UNITS.csv", "SELECTED_BATCHES.csv", "SELECTED_PAIRS.csv", "SELECTED_ISOLATED_CONTROLS.csv", "RESULT.json"]
    atomic_json(output / "MANIFEST.json", {"schema": SCHEMA, "products": {
        name: {"bytes": (output / name).stat().st_size, "sha256": sha256(output / name)} for name in products
    }})
    atomic_json(output / "COMPLETE.json", {"schema": SCHEMA, "status": "prepared",
        "manifest_sha256": sha256(output / "MANIFEST.json"), "result_sha256": sha256(output / "RESULT.json")})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=("freeze", "prepare"), required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    verify(config)
    if args.phase == "freeze":
        if any((output / name).exists() for name in ("RESULT.json", "MANIFEST.json", "COMPLETE.json")):
            raise FileExistsError("preparation outcome already exists")
        (output / "PROTOCOL.md").write_text(protocol())
        atomic_json(output / "FREEZE.json", {"schema": SCHEMA, "status": "frozen_preparation",
            "frozen_at": datetime.now(timezone.utc).isoformat(), "config_sha256": sha256(args.config),
            "source_sha256": sha256(Path(__file__)), "protocol_sha256": sha256(output / "PROTOCOL.md")})
        return
    freeze = json.loads((output / "FREEZE.json").read_text())
    if freeze["config_sha256"] != sha256(args.config) or freeze["source_sha256"] != sha256(Path(__file__)):
        raise RuntimeError("preparation source/config differs from freeze")
    run(config, output)


if __name__ == "__main__":
    main()
