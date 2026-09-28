#!/usr/bin/env python3
"""Independent audit of the frozen CU target/reference selection."""

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
    ap.add_argument("packet", type=Path)
    ap.add_argument("post_tmm", type=Path)
    ap.add_argument("cs_pairs", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()

    frozen = pd.read_csv(args.packet / "FROZEN_TARGET_PAIRS.csv")
    full = pd.read_csv(args.packet / "FULL_ELIGIBLE_TARGET_PAIRS.csv")
    refs = pd.read_csv(args.packet / "FROZEN_REFERENCE_EVENTS.csv")
    comparisons = pd.read_csv(args.packet / "FROZEN_COMPARISON_PAIRS.csv")
    selection = json.loads((args.packet / "FROZEN_SELECTION.json").read_text())
    cs = pd.read_csv(args.cs_pairs)
    cs = cs[cs.final_consecutive_9_29].copy()
    with np.load(args.post_tmm, allow_pickle=False) as z:
        event_rows = z["event_row_ids"]
        times = z["times_samples"]
        channels = z["channels"]
        labels = z["labels"]

    failures: list[str] = []

    def check(name: str, condition: bool) -> None:
        if not condition:
            failures.append(name)

    check("full_eligible_is_all_cs_targets", set(zip(full.row_a, full.row_b)) == set(zip(cs.row_a, cs.row_b)))
    check("target_count", len(frozen) == 96)
    check("target_ids_dense", np.array_equal(frozen.target_id, np.arange(96)))
    check(
        "category_quotas",
        frozen.category.value_counts().to_dict()
        == {"same_constituent": 32, "cross_zero_offset": 32, "cross_nonzero_offset": 32},
    )
    check(
        "category_definitions",
        bool(
            (
                (frozen.category.eq("same_constituent") == frozen.same_constituent)
                & (
                    frozen.category.eq("cross_zero_offset")
                    == ((~frozen.same_constituent) & frozen.offset_difference_samples.eq(0))
                )
                & (
                    frozen.category.eq("cross_nonzero_offset")
                    == ((~frozen.same_constituent) & frozen.offset_difference_samples.ne(0))
                )
            ).all()
        ),
    )
    check("all_targets_survive", bool((frozen.a_survives_dedup & frozen.b_survives_dedup).all()))
    check("all_targets_final_9_29", bool(frozen.final_consecutive_9_29.all()))
    check(
        "category_ranks_dense",
        all(
            np.array_equal(np.sort(g.selection_rank_within_category), np.arange(32))
            for _, g in frozen.groupby("category")
        ),
    )
    check("sole_originally_close_included", int(frozen.original_close_displaced_beyond_radius.sum()) == 1)
    check(
        "source_frames_exact",
        bool(
            (
                frozen.row_a_source_pre_frame
                == frozen.source_frame_origin + frozen.row_a_local_pre_frame
            ).all()
            and (
                frozen.row_b_source_pre_frame
                == frozen.source_frame_origin + frozen.row_b_local_pre_frame
            ).all()
        ),
    )
    frozen_pairs = set(zip(frozen.row_a, frozen.row_b))
    check("targets_subset_of_full", frozen_pairs <= set(zip(full.row_a, full.row_b)))

    check("reference_count", len(refs) == 1500)
    check("reference_rows_unique", not refs.row.duplicated().any())
    check("reference_max_30_per_constituent", int(refs.groupby("constituent").size().max()) <= 30)
    target_rows = set(frozen.row_a) | set(frozen.row_b)
    check("reference_target_rows_disjoint", not (set(refs.row) & target_rows))
    check("post_tmm_rows_dense", np.array_equal(event_rows, np.arange(event_rows.size)))
    rr = refs.row.to_numpy(dtype=np.int64)
    check("reference_constituent_exact", np.array_equal(labels[rr], refs.constituent.to_numpy()))
    check("reference_channel_exact", np.array_equal(channels[rr], refs.detected_main_channel_index.to_numpy()))
    check("reference_source_frame_exact", bool((refs.source_pre_frame == selection["source_frame_origin"] + refs.local_pre_frame).all()))

    order = np.argsort(times, kind="stable")
    sorted_times = times[order]
    isolation_failures = []
    for rec in refs.itertuples(index=False):
        source_row = int(rec.row)
        t = int(times[source_row])
        ch = int(channels[source_row])
        lo = np.searchsorted(sorted_times, t - 30, side="left")
        hi = np.searchsorted(sorted_times, t + 30, side="right")
        neighbors = order[lo:hi]
        neighbors = neighbors[
            (neighbors != source_row)
            & (labels[neighbors] >= 0)
            & (np.abs(channels[neighbors] - ch) <= 8)
        ]
        if neighbors.size:
            isolation_failures.append(source_row)
    check("reference_isolation_proxy_exact", not isolation_failures)

    check("comparison_count", len(comparisons) == 32)
    check("comparison_ids_dense", np.array_equal(comparisons.comparison_id, np.arange(32)))
    min_lag = int(np.ceil(0.005 * selection["sampling_frequency_hz"]))
    max_lag = int(np.floor(0.020 * selection["sampling_frequency_hz"]))
    check("comparison_lag_5_20ms", bool(comparisons.lag_samples.between(min_lag, max_lag).all()))
    targets_by_id = frozen.set_index("target_id")
    comparison_metadata_exact = True
    for rec in comparisons.itertuples(index=False):
        target = targets_by_id.loc[int(rec.matched_target_id)]
        comparison_metadata_exact &= (
            int(rec.postscore_unit) == int(target.postscore_unit)
            and int(rec.constituent_a) == int(target.constituent_a)
            and int(rec.constituent_b) == int(target.constituent_b)
            and str(rec.state) == str(target.state)
        )
    check("comparison_matches_target_stratum", bool(comparison_metadata_exact))

    result = {
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "counts": {
            "full_eligible": len(full),
            "targets": len(frozen),
            "references": len(refs),
            "reference_constituents": int(refs.constituent.nunique()),
            "reference_max_per_constituent": int(refs.groupby("constituent").size().max()),
            "comparisons": len(comparisons),
            "comparison_lag_min_samples": int(comparisons.lag_samples.min()),
            "comparison_lag_max_samples": int(comparisons.lag_samples.max()),
            "target_states": frozen.state.value_counts().to_dict(),
            "target_categories": frozen.category.value_counts().to_dict(),
            "target_unique_parents_by_category": frozen.groupby("category").postscore_unit.nunique().to_dict(),
        },
        "checks": {"failure_count": len(failures)},
        "selection_created_utc": selection["created_utc"],
        "selection_sha256": digest(args.packet / "FROZEN_SELECTION.json"),
        "input_sha256": {
            "POST_TMM_STATE.npz": digest(args.post_tmm),
            "CE3000_PREDEDUP_CANDIDATE_PAIRS.csv": digest(args.cs_pairs),
        },
        "interpretation": "selection integrity only; no waveform outcome was inspected",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
