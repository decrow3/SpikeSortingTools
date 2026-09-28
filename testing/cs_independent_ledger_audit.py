#!/usr/bin/env python3
"""Bounded independent checks of the saved CS event-stage lineage ledger."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("product_dir", type=Path)
    ap.add_argument("output_json", type=Path)
    args = ap.parse_args()

    ledger_path = args.product_dir / "CE3000_EVENT_STAGE_LEDGER.npz"
    summary_path = args.product_dir / "CE3000_STAGE_LEDGER_SUMMARY.json"
    summary = json.loads(summary_path.read_text())
    with np.load(ledger_path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}

    source = arrays["source_row_id"]
    tpre = arrays["time_pre_samples"]
    offset = arrays["applied_offset_samples"]
    tpost = arrays["time_post_alignment_samples"]
    lalign = arrays["label_post_alignment"]
    lscore = arrays["label_post_score"]
    ldedup = arrays["label_post_dedup"]
    lfinal = arrays["label_final_depth_reorder"]
    dropped = (lscore >= 0) & (ldedup < 0)
    retained = ldedup >= 0
    radius = int(summary["dedup_radius_samples"])

    # Find a retained row in the same pre-reorder component near each dropped
    # row. This deliberately avoids comparing post-score component IDs with
    # depth-reordered final IDs.
    nearest = np.full(source.size, -1, dtype=np.int64)
    nearest_dist = np.full(source.size, -1, dtype=np.int64)
    ambiguous = np.zeros(source.size, dtype=bool)
    for label in np.unique(lscore[dropped]):
        di = np.flatnonzero(dropped & (lscore == label))
        ri = np.flatnonzero(retained & (lscore == label))
        if not ri.size:
            continue
        order = np.argsort(tpost[ri], kind="stable")
        ri = ri[order]
        rt = tpost[ri]
        for row in di:
            lo = np.searchsorted(rt, tpost[row] - radius, side="left")
            hi = np.searchsorted(rt, tpost[row] + radius, side="right")
            candidates = ri[lo:hi]
            if not candidates.size:
                continue
            distances = np.abs(tpost[candidates] - tpost[row])
            md = distances.min()
            best = candidates[distances == md]
            nearest[row] = int(best[0])
            nearest_dist[row] = int(md)
            ambiguous[row] = candidates.size > 1 or best.size > 1

    expected_relation = np.zeros(source.size, dtype=np.int8)
    expected_relation[dropped & (nearest >= 0) & ~ambiguous] = 1
    expected_relation[dropped & ambiguous] = 2
    expected_relation[dropped & (nearest < 0)] = 3
    published_relation = arrays.get("dedup_winner_relation")
    published_winner = arrays.get("unique_dedup_winner_row")
    correction_path = args.product_dir / "CE3000_DEDUP_RELATION_CORRECTION.npz"
    relation_source = "base_ledger"
    if correction_path.exists():
        with np.load(correction_path, allow_pickle=False) as correction:
            assert np.array_equal(correction["source_row_id"], source)
            published_relation = correction["direct_retained_relation"]
            published_winner = correction["unique_direct_retained_winner_row"]
        relation_source = "authoritative_correction_overlay"

    failures: list[str] = []
    checks = {
        "source_rows_dense_unique": bool(
            np.array_equal(source, np.arange(source.size, dtype=source.dtype))
        ),
        "signed_time_formula_all_rows": bool(np.array_equal(tpost, tpre - offset)),
        "score_preserves_assignment_state": bool(
            np.array_equal(lalign >= 0, lscore >= 0)
        ),
        "dedup_only_drops": bool(np.all(ldedup[retained] >= 0)),
        "final_preserves_dedup_assignment_state": bool(
            np.array_equal(ldedup >= 0, lfinal >= 0)
        ),
        "summary_ledger_hash_matches": sha256(ledger_path)
        == summary["ledger_sha256"],
        "dedup_relation_preserves_bridged_unknowns": bool(
            published_relation is not None
            and np.array_equal(published_relation, expected_relation)
        ),
        "unique_winners_are_direct_neighbors": bool(
            published_winner is not None
            and np.all(published_winner[expected_relation != 1] == -1)
            and np.all(published_winner[expected_relation == 1] >= 0)
            and np.all(
                np.abs(
                    tpost[published_winner[expected_relation == 1]]
                    - tpost[expected_relation == 1]
                )
                <= radius
            )
        ),
    }
    for key, value in checks.items():
        if not value:
            failures.append(key)

    result = {
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "checks": checks,
        "ledger_sha256": sha256(ledger_path),
        "dedup_relation_source": relation_source,
        "counts": {
            "source_rows": int(source.size),
            "premerge_assigned": int(np.count_nonzero(lalign >= 0)),
            "postscore_assigned": int(np.count_nonzero(lscore >= 0)),
            "score_reassigned_rows": int(
                np.count_nonzero((lalign >= 0) & (lscore >= 0) & (lalign != lscore))
            ),
            "dedup_dropped_rows": int(np.count_nonzero(dropped)),
            "postdedup_assigned": int(np.count_nonzero(retained)),
            "final_assigned": int(np.count_nonzero(lfinal >= 0)),
            "dropped_with_near_retained_same_prereorder_component": int(
                np.count_nonzero(nearest[dropped] >= 0)
            ),
            "dropped_without_near_retained_same_prereorder_component": int(
                np.count_nonzero(nearest[dropped] < 0)
            ),
            "dropped_with_ambiguous_near_retained_rows": int(
                np.count_nonzero(ambiguous[dropped])
            ),
        },
        "interpretation": (
            "A nearby retained row is deterministic algorithmic suppression "
            "evidence, not necessarily a unique physical duplicate partner."
        ),
    }

    pair_path = args.product_dir / "CE3000_PREDEDUP_CANDIDATE_PAIRS.csv"
    null_path = args.product_dir / "CE3000_NONWRAPPING_EVENT_TIME_NULL.csv"
    null_summary_path = args.product_dir / "CE3000_NULL_ROUTE_SUMMARY.json"
    if pair_path.exists() and null_path.exists() and null_summary_path.exists():
        null_summary = json.loads(null_summary_path.read_text())
        pair_rows = list(csv.DictReader(pair_path.open()))
        final_rows = [r for r in pair_rows if r["final_consecutive_9_29"] == "True"]
        route_checks = {
            "candidate_formula_exact": all(
                int(r["predicted_post_lag_samples"])
                == int(r["observed_post_lag_samples"])
                for r in pair_rows
            ),
            "candidate_pre_envelope_complete": max(
                abs(int(r["pre_lag_samples"])) for r in pair_rows
            )
            <= int(null_summary["candidate_pre_lag_envelope_samples"]),
            "all_final_9_29_inside_candidate_ledger": len(final_rows)
            == int(null_summary["stage_pair_counts"]["final_consecutive_pairs_9_29"]),
            "route_products_hashes_match": all(
                sha256(args.product_dir / name) == digest
                for name, digest in null_summary["products_sha256"].items()
            ),
        }
        null_rows = list(csv.DictReader(null_path.open()))
        frozen = set(map(float, null_summary["null_offsets_s_frozen"]))
        offsets_by_group: dict[tuple[str, str, str], set[float]] = {}
        for row in null_rows:
            key = (row["postscore_unit"], row["constituent_a"], row["constituent_b"])
            offsets_by_group.setdefault(key, set()).add(float(row["offset_s"]))
        route_checks["null_all_groups_have_all_frozen_offsets"] = all(
            offsets == frozen for offsets in offsets_by_group.values()
        )
        result["route_null_checks"] = route_checks
        result["route_null_counts"] = {
            "candidate_rows": len(pair_rows),
            "final_consecutive_9_29_rows": len(final_rows),
            "null_groups": len(offsets_by_group),
            "null_rows": len(null_rows),
        }
        for key, value in route_checks.items():
            if not value:
                failures.append(key)
        result["failures"] = failures
        result["status"] = "pass" if not failures else "fail"
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
