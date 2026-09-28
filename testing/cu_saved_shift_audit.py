#!/usr/bin/env python3
"""Audit saved W2 shift orientation and exact 8/9--29/30 boundaries."""

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
    ap.add_argument("--experiment-root", type=Path, required=True)
    ap.add_argument("--episode-catalogue", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    ce = args.experiment_root / "luke0804-imec1-ce-paired-grouping-v1/3000"
    cs = args.experiment_root / "cs_merge_alignment_lineage_20260928"
    grouping_path = ce / "GROUPING_ARRAYS.npz"
    ledger_path = cs / "CE3000_EVENT_STAGE_LEDGER.npz"
    pair_path = cs / "CE3000_PREDEDUP_CANDIDATE_PAIRS.csv"
    manifest_path = (
        args.experiment_root
        / "luke0804-imec1-ao-d2l-v2/windows/W2/runs/D2L/input-manifest.json"
    )

    with np.load(grouping_path, allow_pickle=False) as z:
        shifts = z["shifts"]
        merge_mapping = z["merge_mapping"]
        masks = {
            k: z[k]
            for k in (
                "force_mask",
                "qda_requested_mask",
                "qda_accept_mask",
                "final_union_mask",
                "distance_mask",
            )
        }
    with np.load(ledger_path, allow_pickle=False) as z:
        source_row = z["source_row_id"]
        constituent = z["constituent_id"]
        applied = z["applied_offset_samples"]
        time_pre = z["time_pre_samples"]
        time_post = z["time_post_alignment_samples"]
        final_label = z["label_final_depth_reorder"]

    offsets = np.empty(shifts.shape[0], dtype=np.int32)
    for unit in range(shifts.shape[0]):
        vals = np.unique(applied[constituent == unit])
        if vals.size != 1:
            raise ValueError(f"constituent {unit} has offsets {vals.tolist()}")
        offsets[unit] = vals[0]

    saved_pairs = pd.read_csv(pair_path)
    saved_pairs = saved_pairs[saved_pairs.final_consecutive_9_29].copy()
    saved_lookup = {
        tuple(sorted((int(r.row_a), int(r.row_b)))): r
        for r in saved_pairs.itertuples(index=False)
    }

    input_manifest = json.loads(manifest_path.read_text())
    window = input_manifest["window"]
    start_frame = int(window["start_frame"])
    fs = float(window["sampling_frequency"])
    episodes = pd.read_csv(args.episode_catalogue)

    def state_for_sample(sample: int) -> str:
        absolute_s = (start_frame + sample) / fs
        hit = episodes[(episodes.start_s <= absolute_s) & (absolute_s < episodes.end_s)]
        return "rest" if hit.empty else str(hit.iloc[-1].status)

    records: list[dict[str, object]] = []
    reconstructed_9_29: set[tuple[int, int]] = set()
    keep = final_label >= 0
    for unit in np.unique(final_label[keep]):
        ix = np.flatnonzero(final_label == unit)
        ix = ix[np.lexsort((source_row[ix], time_post[ix]))]
        post_lags = np.diff(time_post[ix])
        selected = np.flatnonzero(
            (post_lags == 8)
            | ((post_lags >= 9) & (post_lags <= 29))
            | (post_lags == 30)
        )
        for pos in selected:
            ia, ib = int(ix[pos]), int(ix[pos + 1])
            row_a, row_b = int(source_row[ia]), int(source_row[ib])
            a, b = int(constituent[ia]), int(constituent[ib])
            post_lag = int(time_post[ib] - time_post[ia])
            pre_lag = int(time_pre[ib] - time_pre[ia])
            delta = int(offsets[b] - offsets[a])
            key = tuple(sorted((row_a, row_b)))
            saved = saved_lookup.get(key)
            if 9 <= post_lag <= 29:
                reconstructed_9_29.add(key)
            record: dict[str, object] = {
                "lag_group": "8" if post_lag == 8 else "30" if post_lag == 30 else "9_29",
                "row_a": row_a,
                "row_b": row_b,
                "final_unit": int(unit),
                "constituent_a": a,
                "constituent_b": b,
                "same_constituent": a == b,
                "s_ab": int(shifts[a, b]),
                "s_ba": int(shifts[b, a]),
                "u_a": int(offsets[a]),
                "u_b": int(offsets[b]),
                "u_b_minus_u_a": delta,
                "e_ab": int(shifts[a, b]) - delta,
                "pre_lag_samples": pre_lag,
                "predicted_post_lag_samples": pre_lag - delta,
                "observed_post_lag_samples": post_lag,
                "original_lag_le_7": abs(pre_lag) <= 7,
                "state_exact_clock": state_for_sample(int(time_post[ia])),
                "published_route": "" if saved is None else str(saved.route),
            }
            for name, matrix in masks.items():
                record[name] = bool(matrix[a, b])
            records.append(record)

    table = pd.DataFrame.from_records(records).sort_values(
        ["lag_group", "row_a", "row_b"], kind="stable"
    )
    if not np.array_equal(
        table.predicted_post_lag_samples.to_numpy(),
        table.observed_post_lag_samples.to_numpy(),
    ):
        raise AssertionError("signed lag formula failed")
    published = set(saved_lookup)
    if reconstructed_9_29 != published:
        raise AssertionError(
            f"candidate closure failed: missing={len(reconstructed_9_29-published)} "
            f"extra={len(published-reconstructed_9_29)}"
        )
    table_path = args.output / "CU_SAVED_SHIFT_ORIENTATION_TABLE.csv"
    table.to_csv(table_path, index=False)

    within_component_residuals: list[int] = []
    for component in np.unique(merge_mapping):
        ids = np.flatnonzero(merge_mapping == component)
        for a in ids:
            for b in ids:
                if a != b:
                    within_component_residuals.append(
                        int(shifts[a, b]) - int(offsets[b] - offsets[a])
                    )

    by_lag = {}
    for lag_group, group in table.groupby("lag_group", sort=False):
        by_lag[str(lag_group)] = {
            "pairs": int(len(group)),
            "same_constituent": int(group.same_constituent.sum()),
            "cross_constituent": int((~group.same_constituent).sum()),
            "nonzero_applied_difference": int((group.u_b_minus_u_a != 0).sum()),
            "original_lag_le_7": int(group.original_lag_le_7.sum()),
        }
    vals, counts = np.unique(within_component_residuals, return_counts=True)
    summary = {
        "status": "pass",
        "orientation_fixture": "shifts[reference, source] is subtracted from source event time",
        "signed_pair_formula": "L_post = L_pre - (u_b - u_a)",
        "residual_formula": "e_ab = s_ab - (u_b - u_a)",
        "matrix_antisymmetry_checked_not_assumed": bool(np.array_equal(shifts, -shifts.T)),
        "matrix_diagonal_zero": bool(np.all(np.diag(shifts) == 0)),
        "applied_offset_range_samples": [int(offsets.min()), int(offsets.max())],
        "within_component_ordered_pairs": len(within_component_residuals),
        "within_component_residual_counts": {
            str(int(v)): int(n) for v, n in zip(vals, counts, strict=True)
        },
        "candidate_envelope_closes_all_final_9_29": True,
        "lag_groups": by_lag,
        "inputs_sha256": {
            grouping_path.name: digest(grouping_path),
            ledger_path.name: digest(ledger_path),
            pair_path.name: digest(pair_path),
            args.episode_catalogue.name: digest(args.episode_catalogue),
            manifest_path.name: digest(manifest_path),
        },
        "table_sha256": digest(table_path),
    }
    (args.output / "CU_SAVED_SHIFT_ORIENTATION_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
