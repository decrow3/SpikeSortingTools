#!/usr/bin/env python3
"""Frozen saved-event relevance check for the B384 template 398/399 replay pair.

The check constructs adjacency on each complete final-cluster event train before
selecting motion state or detection-template category. It reads no voltage and
does not call a sorter, matcher, fitter, or RF code.
"""

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


SCHEMA = "en-saved-bank-template-pair-relevance-v1"
TARGETS = (398, 399)
STATES_UM = (0.0, 40.0)


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


def verify_inputs(config: dict[str, Any]) -> None:
    for item in config["frozen_inputs"]:
        path = Path(item["path"])
        if sha256(path) != item["sha256"]:
            raise RuntimeError(f"frozen input differs: {path}")


def protocol() -> str:
    return """# Frozen counting protocol

- Scope: B384 saved spike times, final clusters, and preclustering detection-template IDs only.
- Templates: 398 and 399; states: q0 and q+40 um; sampling rate and crop/field are frozen inputs.
- State/segment assignment: build full-session midpoint cells, crop them, preserve original contiguous
  segment IDs, and assign each saved event after subtracting the crop start frame.
- Membership: for each target template and state, count its events by final cluster. No eligibility filter.
- Adjacency: within each complete final-cluster train, sort stably by saved time and form consecutive
  pairs before filtering. Retain only positive lags 1..29 samples whose two events share the same
  original contiguous state segment. Attribute a pair to the state of its second event.
- Containing units: the union of final clusters containing template 398 or 399 in q0 or q+40.
- Partition of each state's original short-pair numerator in those units: exact 398/399 cross-pair
  (both orientations), same detection template (any ID), and all other different-template pairs.
  Separately count target-same (398/398 or 399/399), target-other, and pairs involving neither target.
- Report the exact 1..29 lag histogram for 398->399 and 399->398. Never re-pair after filtering.
- Geometry: report captured bank iU selected channel, physical binary channel, and x/y geometry for
  B384 templates 398/399 and REF384 template 409, plus pairwise anchor-channel distances.
- Interpretation: outcome-guided descriptive relevance check. It cannot identify biological spikes,
  prove duplicate detections, estimate causal prevalence, or validate the synthetic replay on recording data.
- Prohibited: voltage reads, sorting, fitting/reclustering, matcher/model changes, RF, and holdout access.
"""


def load_cells(config: dict[str, Any]) -> Any:
    field = config["field"]
    with np.load(field["path"], allow_pickle=False) as saved:
        displacement = np.asarray(saved[field["displacement_key"]], dtype=float)
        if displacement.ndim == 2:
            displacement = np.median(displacement, axis=1)
        rounded, _ = rounded_rigid_field(displacement)
        return crop_field_cells_from_global_midpoints(
            np.asarray(saved[field["time_key"]], dtype=float),
            rounded,
            float(config["recording_duration_s"]),
            int(config["crop"]["start_frame"]) / float(config["sampling_frequency_hz"]),
            int(config["crop"]["stop_frame"]) / float(config["sampling_frequency_hz"]),
            float(config["accounting_block_seconds"]),
        )


def geometry_rows(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for arm, template_ids in (("B384", TARGETS), ("REF384", (409,))):
        bank = config["banks"][arm]
        iu = np.asarray(np.load(bank["iU_path"], mmap_mode="r"), dtype=np.int64).reshape(-1)
        physical = np.asarray(np.load(bank["iU_physical_path"], mmap_mode="r"), dtype=np.int64).reshape(-1)
        positions = np.asarray(np.load(bank["channel_positions_path"], mmap_mode="r"), dtype=float)
        for template_id in template_ids:
            selected = int(iu[template_id])
            rows.append({
                "arm": arm,
                "template_id": int(template_id),
                "iU_selected_channel": selected,
                "iU_physical_binary_channel": int(physical[template_id]),
                "x_um": float(positions[selected, 0]),
                "y_um": float(positions[selected, 1]),
            })
    frame = pd.DataFrame(rows)
    distances = []
    for left in range(len(frame)):
        for right in range(left + 1, len(frame)):
            a, b = frame.iloc[left], frame.iloc[right]
            distances.append({
                "left": f"{a.arm}:{int(a.template_id)}",
                "right": f"{b.arm}:{int(b.template_id)}",
                "dx_um": float(b.x_um - a.x_um),
                "dy_um": float(b.y_um - a.y_um),
                "distance_um": float(np.hypot(b.x_um - a.x_um, b.y_um - a.y_um)),
            })
    return frame, pd.DataFrame(distances)


def run(config: dict[str, Any], output: Path) -> None:
    verify_inputs(config)
    if (output / "RESULT.json").exists() or (output / "COMPLETE.json").exists():
        raise FileExistsError("refusing to overwrite completed outcome")

    root = Path(config["b384_sorter_output"])
    times_absolute = np.asarray(np.load(root / "spike_times.npy", mmap_mode="r"), dtype=np.int64).reshape(-1)
    clusters = np.asarray(np.load(root / "spike_clusters.npy", mmap_mode="r"), dtype=np.int64).reshape(-1)
    templates = np.asarray(np.load(root / "spike_detection_templates.npy", mmap_mode="r"), dtype=np.int64).reshape(-1)
    if not (len(times_absolute) == len(clusters) == len(templates)):
        raise RuntimeError("saved B384 event arrays are not aligned")
    if len(times_absolute) and np.any(np.diff(times_absolute) < 0):
        raise RuntimeError("saved B384 spike times are not sorted")
    start, stop = int(config["crop"]["start_frame"]), int(config["crop"]["stop_frame"])
    if len(times_absolute) and (times_absolute[0] < start or times_absolute[-1] >= stop):
        raise RuntimeError("saved B384 event lies outside frozen crop")

    times = times_absolute - np.int64(start)
    cells = load_cells(config)
    state_index, segment = cells.assign(times, float(config["sampling_frequency_hz"]))
    state_um = np.asarray(cells.states_um, dtype=float)[state_index]
    if not all(value in set(state_um.tolist()) for value in STATES_UM):
        raise RuntimeError("one requested state is absent")

    membership_rows = []
    containing_units: set[int] = set()
    for target in TARGETS:
        for requested_state in STATES_UM:
            keep = (templates == target) & (state_um == requested_state)
            units, counts = np.unique(clusters[keep], return_counts=True)
            total = int(counts.sum())
            for unit, count in zip(units, counts):
                containing_units.add(int(unit))
                membership_rows.append({
                    "template_id": target,
                    "state_um": requested_state,
                    "final_cluster": int(unit),
                    "events": int(count),
                    "fraction_of_template_state_events": float(count / total) if total else None,
                    "template_state_total_events": total,
                })
    membership = pd.DataFrame(membership_rows)

    # Construct original adjacency on the full train of every containing final unit.
    order = np.lexsort((times, clusters))
    u, t = clusters[order], times[order]
    d, s, g = templates[order], state_um[order], segment[order]
    adjacent = u[1:] == u[:-1]
    same_segment = g[1:] == g[:-1]
    lag = np.diff(t)
    short = adjacent & same_segment & (lag >= 1) & (lag <= 29) & np.isin(u[1:], list(containing_units))
    prev_d, next_d, unit, pair_state, pair_lag = d[:-1][short], d[1:][short], u[1:][short], s[1:][short], lag[short]

    summary_rows, lag_rows, unit_rows = [], [], []
    for requested_state in STATES_UM:
        q = pair_state == requested_state
        a, b, qunit, qlag = prev_d[q], next_d[q], unit[q], pair_lag[q]
        target_cross = ((a == 398) & (b == 399)) | ((a == 399) & (b == 398))
        same_any = a == b
        other_different = (~same_any) & (~target_cross)
        target_involved = np.isin(a, TARGETS) | np.isin(b, TARGETS)
        target_same = same_any & np.isin(a, TARGETS)
        target_other = target_involved & (~target_cross) & (~target_same)
        total = len(a)
        summary_rows.append({
            "state_um": requested_state,
            "containing_final_units": len(containing_units),
            "original_short_pairs_1_29": int(total),
            "target_cross_398_399_both_orientations": int(target_cross.sum()),
            "target_cross_fraction_of_original_short_pairs": float(target_cross.sum() / total) if total else None,
            "same_detection_template_any_id": int(same_any.sum()),
            "other_different_template": int(other_different.sum()),
            "partition_sum": int(target_cross.sum() + same_any.sum() + other_different.sum()),
            "target_same_398_or_399": int(target_same.sum()),
            "target_other_template": int(target_other.sum()),
            "neither_target": int((~target_involved).sum()),
        })
        for before, after in ((398, 399), (399, 398)):
            orientation = (a == before) & (b == after)
            for exact_lag in range(1, 30):
                lag_rows.append({
                    "state_um": requested_state,
                    "previous_template": before,
                    "next_template": after,
                    "lag_samples": exact_lag,
                    "pairs": int(np.sum(orientation & (qlag == exact_lag))),
                })
        for final_unit in sorted(containing_units):
            uq = qunit == final_unit
            unit_rows.append({
                "state_um": requested_state,
                "final_cluster": final_unit,
                "original_short_pairs_1_29": int(uq.sum()),
                "target_cross_398_399": int(np.sum(uq & target_cross)),
                "same_detection_template_any_id": int(np.sum(uq & same_any)),
                "other_different_template": int(np.sum(uq & other_different)),
            })

    geometry, distances = geometry_rows(config)
    products = {
        "TEMPLATE_FINAL_CLUSTER_MEMBERSHIP.csv": membership,
        "SHORT_PAIR_SUMMARY.csv": pd.DataFrame(summary_rows),
        "TARGET_PAIR_LAG_HISTOGRAM.csv": pd.DataFrame(lag_rows),
        "CONTAINING_UNIT_SHORT_PAIRS.csv": pd.DataFrame(unit_rows),
        "BANK_IU_GEOMETRY.csv": geometry,
        "BANK_IU_DISTANCES.csv": distances,
    }
    for name, frame in products.items():
        frame.to_csv(output / name, index=False)

    result = {
        "schema": SCHEMA,
        "status": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "outcome_guided_descriptive_followup": True,
        "raw_voltage_bytes_read": 0,
        "sorts_launched": 0,
        "rf_accesses": 0,
        "target_templates": list(TARGETS),
        "states_um": list(STATES_UM),
        "containing_final_units": sorted(containing_units),
        "event_rows": int(len(times)),
        "short_pair_summary": summary_rows,
        "interpretation": "Saved-output relevance only; not independent confirmation of duplicate biological spikes.",
    }
    atomic_json(output / "RESULT.json", result)

    manifest_products = {}
    for name in list(products) + ["RESULT.json", "PROTOCOL.md", "FREEZE.json"]:
        path = output / name
        manifest_products[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    atomic_json(output / "MANIFEST.json", {"schema": SCHEMA, "products": manifest_products})
    atomic_json(output / "COMPLETE.json", {
        "schema": SCHEMA,
        "status": "complete",
        "manifest_sha256": sha256(output / "MANIFEST.json"),
        "result_sha256": sha256(output / "RESULT.json"),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=("freeze", "run"), required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    verify_inputs(config)
    if args.phase == "freeze":
        if any((output / name).exists() for name in ("RESULT.json", "MANIFEST.json", "COMPLETE.json")):
            raise FileExistsError("outcome already exists")
        (output / "PROTOCOL.md").write_text(protocol())
        atomic_json(output / "FREEZE.json", {
            "schema": SCHEMA,
            "status": "frozen_preoutcome",
            "frozen_at": datetime.now(timezone.utc).isoformat(),
            "config_path": str(args.config.resolve()),
            "config_sha256": sha256(args.config),
            "source_path": str(Path(__file__).resolve()),
            "source_sha256": sha256(Path(__file__)),
            "protocol_sha256": sha256(output / "PROTOCOL.md"),
        })
        return
    freeze = json.loads((output / "FREEZE.json").read_text())
    if freeze["config_sha256"] != sha256(args.config) or freeze["source_sha256"] != sha256(Path(__file__)):
        raise RuntimeError("source or config differs from pre-outcome freeze")
    if freeze["protocol_sha256"] != sha256(output / "PROTOCOL.md"):
        raise RuntimeError("protocol differs from pre-outcome freeze")
    run(config, output)


if __name__ == "__main__":
    main()
