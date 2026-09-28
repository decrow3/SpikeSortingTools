#!/usr/bin/env python3
"""Bounded independent checks for the corrected CW v2 saved-data packet."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def endpoints(labels: np.ndarray, times: np.ndarray) -> dict[str, int]:
    counts = {"8": 0, "9_29": 0, "30": 0}
    for unit in np.unique(labels[labels >= 0]):
        rows = np.flatnonzero(labels == unit)
        dt = np.diff(np.sort(times[rows], kind="stable"))
        counts["8"] += int(np.count_nonzero(dt == 8))
        counts["9_29"] += int(np.count_nonzero((dt >= 9) & (dt <= 29)))
        counts["30"] += int(np.count_nonzero(dt == 30))
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packet", type=Path, required=True)
    ap.add_argument("--experiment-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source = {
        "W2": (
            args.experiment_root / "luke0804-imec1-cp-direct-outcomes-v1/W2_saved_replay/ALL_FORCE_REPLAY.npz",
            args.experiment_root / "luke0804-imec1-cp-direct-outcomes-v1/W2_saved_replay/NO_FORCE_FINAL.npz",
        ),
        "W3": (
            args.experiment_root / "luke0804-imec1-cl-w3-force-gate-v1/gate/ALL_FORCE_REPLAY.npz",
            args.experiment_root / "luke0804-imec1-cl-w3-force-gate-v1/gate/NO_FORCE_FINAL.npz",
        ),
    }
    manifest_failures = []
    if (args.packet / "MANIFEST.json").exists():
        manifest = json.loads((args.packet / "MANIFEST.json").read_text())
        manifest_failures = [
            row["path"] for row in manifest["files"]
            if sha256(args.packet / row["path"]) != row["sha256"]
        ]
    result = {"status": "pass", "manifest_hash_failures": manifest_failures, "windows": {}}
    for window, (force_path, noforce_path) in source.items():
        with np.load(force_path, allow_pickle=False) as z:
            force = {k: np.asarray(z[k]) for k in z.files}
        with np.load(noforce_path, allow_pickle=False) as z:
            noforce = {k: np.asarray(z[k]) for k in z.files}
        parents = pd.read_csv(args.packet / f"ALL_PARENTS_{window}.csv")
        children = pd.read_csv(args.packet / f"ALL_PARENT_CHILD_RELATIONS_{window}.csv")
        pairs = pd.read_csv(args.packet / f"ALL_MULTI_CHILD_PAIRS_{window}.csv")
        expected_parents = set(map(int, np.unique(force["labels"][force["labels"] >= 0])))
        parent_set = set(map(int, parents.all_force_parent))
        expected_pair_rows = int(sum(n * (n - 1) // 2 for n in parents.actual_noforce_children))
        expected_primary_pairs = int(sum(max(0, n - 1) for n in parents.actual_noforce_children))

        # Deterministic one-family check: largest multi-child parent, ties by ID.
        family = parents[parents.actual_noforce_children >= 2].sort_values(
            ["parent_rows", "all_force_parent"], ascending=[False, True]
        ).iloc[0]
        p = int(family.all_force_parent)
        own = force["labels"] == p
        labs, count = np.unique(noforce["labels"][own & (noforce["labels"] >= 0)], return_counts=True)
        saved = children[children.all_force_parent == p].sort_values("noforce_final_child")
        global_counts = {int(c): int(np.count_nonzero(noforce["labels"] == c)) for c in labs}
        one_family_pass = (
            saved.noforce_final_child.astype(int).tolist() == labs.astype(int).tolist()
            and saved.within_parent_rows.astype(int).tolist() == count.astype(int).tolist()
            and saved.global_child_rows.astype(int).tolist() == [global_counts[int(c)] for c in labs]
        )
        independent_endpoints = {
            "force": endpoints(force["labels"], force["times_samples"]),
            "noforce": endpoints(noforce["labels"], noforce["times_samples"]),
        }
        ledger_check = None
        ledger_path = args.packet / "COMPLETE_FINAL_ADJACENCY_LEDGER.csv"
        if ledger_path.exists():
            ledger = pd.read_csv(ledger_path)
            ledger = ledger[ledger.window == window]
            ledger_check = {}
            for branch in ("force", "noforce"):
                got = ledger[ledger.branch == branch].endpoint.value_counts().to_dict()
                ledger_check[branch] = {
                    "saved": {"8": int(got.get("boundary_8", 0)), "9_29": int(got.get("endpoint_9_29", 0)), "30": int(got.get("boundary_30", 0))},
                    "passes": independent_endpoints[branch] == {"8": int(got.get("boundary_8", 0)), "9_29": int(got.get("endpoint_9_29", 0)), "30": int(got.get("boundary_30", 0))},
                }
        result["windows"][window] = {
            "all_parent_set_exact": parent_set == expected_parents,
            # The parent tables intentionally cover only rows assigned in the
            # all-force branch.  All-force noise is outside the parent
            # universe, so closure is against labels >= 0 rather than every
            # immutable event row.
            "all_force_assigned_rows": int(np.count_nonzero(force["labels"] >= 0)),
            "all_force_noise_rows_outside_parent_universe": int(np.count_nonzero(force["labels"] < 0)),
            "parent_rows_close": int(parents.parent_rows.sum()) == int(np.count_nonzero(force["labels"] >= 0)),
            "assigned_plus_noise_close": int((parents.noforce_assigned_rows + parents.noforce_noise_rows).sum()) == int(np.count_nonzero(force["labels"] >= 0)),
            "relation_rows_close": int(children.within_parent_rows.sum()) == int(parents.noforce_assigned_rows.sum()),
            "multi_child_families": int((parents.actual_noforce_children >= 2).sum()),
            "all_pair_rows": int(len(pairs)), "expected_all_pair_rows": expected_pair_rows,
            "primary_amplitude_max_pairs": int((pairs.high_amplitude_rank_desc == 1).sum()),
            "expected_primary_amplitude_max_pairs": expected_primary_pairs,
            "one_family_parent": p, "one_family_pass": bool(one_family_pass),
            "independent_endpoints": independent_endpoints,
            "saved_ledger_endpoint_check": ledger_check,
        }

    h5_path = args.packet / "COMMON_TARGET_NULL_SEGMENTS.h5"
    with h5py.File(h5_path, "r", locking=False) as h5:
        null = {}
        for window in ("W2", "W3"):
            segments, summary = h5[window], h5[f"{window}_offset_summary"]
            null[window] = {
                "segment_rows": int(len(segments)), "summary_rows": int(len(summary)),
                "summary_offsets": int(np.unique(summary["offset_index"]).size),
                "summary_has_explicit_zero_query_cells": bool(np.any((summary["observed_queries"] == 0) | (summary["null_queries"] == 0))),
                "segment_ledger_is_sparse": bool(not np.any((segments["observed_queries"] == 0) & (segments["null_queries"] == 0))),
                "support_positive_for_written_segment_rows": bool(np.all(segments["support_samples"] > 0)),
            }
    result["common_target_null_serialization"] = null
    checks = []
    for value in result["windows"].values():
        checks.extend([
            value["all_parent_set_exact"], value["parent_rows_close"], value["assigned_plus_noise_close"],
            value["relation_rows_close"], value["all_pair_rows"] == value["expected_all_pair_rows"],
            value["primary_amplitude_max_pairs"] == value["expected_primary_amplitude_max_pairs"],
            value["one_family_pass"],
        ])
        if value["saved_ledger_endpoint_check"] is not None:
            checks.extend(x["passes"] for x in value["saved_ledger_endpoint_check"].values())
    checks.extend([not manifest_failures, all(v["summary_offsets"] == 200 for v in null.values())])
    result["status"] = "pass" if all(checks) else "fail"
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
