#!/usr/bin/env python3
"""Ready-item audit of W3 saved pre/post timing arrays; no replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("w3", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    pre_path = args.w3 / "POST_TMM_STATE.npz"
    saved_path = args.w3 / "gate/ALL_FORCE_REPLAY.npz"
    arrays_path = args.w3 / "ALL_FORCE_ARRAYS.npz"
    route_path = args.w3 / "gate/FORCE_ROUTE_ARRAYS.npz"
    with np.load(pre_path, allow_pickle=False) as z:
        row = z["event_row_ids"]
        tpre = z["times_samples"]
        source = z["labels"]
    with np.load(saved_path, allow_pickle=False) as z:
        tpost = z["times_samples"]
        final = z["labels"]
    with np.load(arrays_path, allow_pickle=False) as z:
        shifts = z["shifts"]
        mapping = z["merge_mapping"]
        saved_times = z["final_times_samples"]
        saved_labels = z["final_labels"]
    with np.load(route_path, allow_pickle=False) as z:
        route_rows = z["event_row_ids"]
        episode = z["fixed_state_episode"]

    if not np.array_equal(row, np.arange(row.size)):
        raise AssertionError("source row IDs are not dense identity")
    if not np.array_equal(row, route_rows):
        raise AssertionError("captured route state row IDs differ")
    if not np.array_equal(tpost, saved_times) or not np.array_equal(final, saved_labels):
        raise AssertionError("saved final arrays disagree")
    applied = tpre - tpost
    offsets = np.zeros(shifts.shape[0], dtype=np.int32)
    for unit in range(shifts.shape[0]):
        vals = np.unique(applied[source == unit])
        if vals.size != 1:
            raise AssertionError(f"source {unit} has nonconstant saved offsets")
        offsets[unit] = vals[0]

    records = []
    for unit in np.unique(final[final >= 0]):
        ix = np.flatnonzero(final == unit)
        ix = ix[np.lexsort((row[ix], tpost[ix]))]
        dt = np.diff(tpost[ix])
        for pos in np.flatnonzero((dt == 8) | ((dt >= 9) & (dt <= 29)) | (dt == 30)):
            ia, ib = int(ix[pos]), int(ix[pos + 1])
            a, b = int(source[ia]), int(source[ib])
            pre_lag = int(tpre[ib] - tpre[ia])
            post_lag = int(tpost[ib] - tpost[ia])
            delta = int(applied[ib] - applied[ia])
            records.append(
                {
                    "lag_group": "8" if post_lag == 8 else "30" if post_lag == 30 else "9_29",
                    "row_a": int(row[ia]),
                    "row_b": int(row[ib]),
                    "final_unit": int(unit),
                    "source_a": a,
                    "source_b": b,
                    "same_source": a == b,
                    "s_ab": int(shifts[a, b]),
                    "s_ba": int(shifts[b, a]),
                    "u_b_minus_u_a": delta,
                    "e_ab": int(shifts[a, b]) - delta,
                    "pre_lag_samples": pre_lag,
                    "predicted_post_lag_samples": pre_lag - delta,
                    "observed_post_lag_samples": post_lag,
                    "original_lag_le_7": abs(pre_lag) <= 7,
                    "captured_episode_state_a": bool(episode[ia]),
                    "captured_episode_state_b": bool(episode[ib]),
                }
            )
    table = pd.DataFrame.from_records(records)
    if not np.array_equal(table.predicted_post_lag_samples, table.observed_post_lag_samples):
        raise AssertionError("saved signed timing identity failed")
    table_path = args.output / "CU_W3_SAVED_TIMING_BOUNDARIES.csv"
    table.to_csv(table_path, index=False)
    lag_groups = {}
    for name, group in table.groupby("lag_group"):
        lag_groups[str(name)] = {
            "pairs": int(len(group)),
            "same_source": int(group.same_source.sum()),
            "cross_source": int((~group.same_source).sum()),
            "nonzero_applied_difference": int((group.u_b_minus_u_a != 0).sum()),
            "original_lag_le_7": int(group.original_lag_le_7.sum()),
            "state_boundary_crossing": int(
                (group.captured_episode_state_a != group.captured_episode_state_b).sum()
            ),
        }
    summary = {
        "status": "pass_saved_arrays_only_no_replay",
        "events": int(row.size),
        "final_assigned": int((final >= 0).sum()),
        "source_units": int(np.unique(source[source >= 0]).size),
        "final_units": int(np.unique(final[final >= 0]).size),
        "applied_offset_range_samples": [int(applied.min()), int(applied.max())],
        "signed_formula_exact_all_rows": bool(np.array_equal(tpost, tpre - applied)),
        "matrix_antisymmetry_checked_not_assumed": bool(np.array_equal(shifts, -shifts.T)),
        "lag_groups": lag_groups,
        "inputs_sha256": {
            pre_path.name: digest(pre_path),
            saved_path.name: digest(saved_path),
            arrays_path.name: digest(arrays_path),
            route_path.name: digest(route_path),
        },
        "table_sha256": digest(table_path),
        "state_note": "uses only captured fixed_state_episode; no state reconstructed or invented",
    }
    (args.output / "CU_W3_SAVED_TIMING_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
