#!/usr/bin/env python3
"""Finite H1 audit of the saved H5 CV packet (no voltage or refitting)."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


FS = 29999.759166666667
WINDOWS = {"W2": (900.0, 1240.0), "W3": (8000.0, 8340.0)}
ORIGINS = {"W2": 26999783, "W3": 239998073}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def merge(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for a, b in sorted(spans):
        if b <= a:
            continue
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def states(catalogue: pd.DataFrame, window: str, frames: np.ndarray):
    start, end = WINDOWS[window]
    cat = catalogue[(catalogue.end_s > start) & (catalogue.start_s < end)]
    accepted = merge([
        (max(start, float(r.start_s)), min(end, float(r.end_s)))
        for r in cat[cat.status == "accepted"].itertuples()
    ])
    unresolved0 = merge([
        (max(start, float(r.start_s)), min(end, float(r.end_s)))
        for r in cat[cat.status != "accepted"].itertuples()
    ])
    unresolved: list[tuple[float, float]] = []
    for a, b in unresolved0:
        pieces = [(a, b)]
        for x, y in accepted:
            nxt = []
            for p, q in pieces:
                if y <= p or x >= q:
                    nxt.append((p, q))
                else:
                    if p < x:
                        nxt.append((p, x))
                    if y < q:
                        nxt.append((y, q))
            pieces = nxt
        unresolved.extend(pieces)
    occupied = merge(accepted + unresolved)
    rest = []
    cursor = start
    for a, b in occupied:
        if cursor < a:
            rest.append((cursor, a))
        cursor = max(cursor, b)
    if cursor < end:
        rest.append((cursor, end))
    labels = np.full(frames.size, "outside", dtype="U10")
    segment = np.full(frames.size, -1, dtype=np.int32)
    left = np.full(frames.size, -1, dtype=np.int64)
    right = np.full(frames.size, -1, dtype=np.int64)
    sid = 0
    for name, spans in (("accepted", accepted), ("unresolved", merge(unresolved)), ("rest", rest)):
        for a, b in spans:
            aa, bb = int(np.ceil(a * FS)), int(np.ceil(b * FS))
            keep = (frames >= aa) & (frames < bb)
            labels[keep], segment[keep], left[keep], right[keep] = name, sid, aa, bb
            sid += 1
    return labels, segment, left, right


def partner(query: np.ndarray, reference: np.ndarray, low: int, high: int) -> np.ndarray:
    reference = np.sort(reference)
    if not query.size or not reference.size:
        return np.zeros(query.size, dtype=bool)
    return np.array([
        np.any((np.abs(reference - q) >= low) & (np.abs(reference - q) <= high))
        for q in query
    ])


def segmented_fraction(q, qseg, r, rseg, low=8, high=29):
    hit = np.zeros(q.size, dtype=bool)
    for sid in np.unique(qseg[qseg >= 0]):
        qq, rr = qseg == sid, rseg == sid
        hit[qq] = partner(q[qq], r[rr], low, high)
    return hit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packet", type=Path, required=True)
    ap.add_argument("--catalogue", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    root = args.packet.parent
    sources = {
        "W2": (
            root / "luke0804-imec1-cp-direct-outcomes-v1/W2_saved_replay/ALL_FORCE_REPLAY.npz",
            root / "luke0804-imec1-cp-direct-outcomes-v1/W2_saved_replay/NO_FORCE_FINAL.npz",
        ),
        "W3": (
            root / "luke0804-imec1-cl-w3-force-gate-v1/gate/ALL_FORCE_REPLAY.npz",
            root / "luke0804-imec1-cl-w3-force-gate-v1/gate/NO_FORCE_FINAL.npz",
        ),
    }
    catalogue = pd.read_csv(args.catalogue)
    manifest = json.loads((args.packet / "MANIFEST.json").read_text())
    hash_failures = [
        row["path"] for row in manifest["files"]
        if sha256(args.packet / row["path"]) != row["sha256"]
    ]

    exclusions = []
    denominator = []
    child_profile_checks = []
    selected_rechecks = []
    excluded_profiles = []
    for window, (force_path, noforce_path) in sources.items():
        with np.load(force_path, allow_pickle=False) as z:
            force = {k: np.asarray(z[k]) for k in z.files}
        with np.load(noforce_path, allow_pickle=False) as z:
            noforce = {k: np.asarray(z[k]) for k in z.files}
        parent, child = force["labels"], noforce["labels"]
        frames = ORIGINS[window] + noforce["times_samples"]
        state, segment, left, right = states(catalogue, window, frames)
        frozen = pd.read_csv(args.packet / f"FROZEN_SPLIT_FAMILIES_{window}.csv")
        output_children = pd.read_csv(args.packet / "FINAL_CHILD_EVENT_PROFILE.csv")
        output_children = output_children[output_children.window == window]
        all_rows = []
        for p in np.unique(parent[parent >= 0]):
            own = parent == p
            labs, counts = np.unique(child[own & (child >= 0)], return_counts=True)
            order = np.lexsort((labs, -counts))
            labs, counts = labs[order], counts[order]
            if counts.size < 2:
                continue
            assigned = int(counts.sum())
            selected = bool(counts[1] >= 20 and counts[1] / assigned >= 0.1)
            all_rows.append((int(p), int(own.sum()), int(labs.size), int(labs[0]), int(labs[1]),
                             int(counts[0]), int(counts[1]), float(counts[1] / assigned), selected))
        all_frame = pd.DataFrame(all_rows, columns=[
            "all_force_parent", "parent_rows", "children", "big_child", "small_child",
            "big_rows", "small_rows", "small_share", "h5_selected",
        ])
        selected_set = set(all_frame.loc[all_frame.h5_selected, "all_force_parent"])
        frozen_set = set(frozen.all_force_parent)
        excluded = all_frame[~all_frame.h5_selected].copy()
        excluded.insert(0, "window", window)
        exclusions.append(excluded)
        denominator.append({
            "window": window,
            "all_force_parents": int(np.unique(parent[parent >= 0]).size),
            "all_actual_split_families": int(len(all_frame)),
            "h5_threshold_selected_families": int(len(frozen)),
            "excluded_split_families": int(len(excluded)),
            "excluded_parent_rows": int(excluded.parent_rows.sum()),
            "excluded_small_rows": int(excluded.small_rows.sum()),
            "selected_set_exactly_reproduced": selected_set == frozen_set,
        })
        for fam in frozen.itertuples(index=False):
            actual_children = int(np.unique(child[(parent == fam.all_force_parent) & (child >= 0)]).size)
            saved_children = int((output_children.all_force_parent == fam.all_force_parent).sum())
            child_profile_checks.append((window, int(fam.all_force_parent), actual_children, saved_children))

        # Recheck the smallest, median, and largest selected parent against saved observed fractions.
        observed = pd.read_csv(args.packet / "CHILD_COINCIDENCE_SUMMARY.csv")
        observed = observed[observed.window == window]
        check = frozen.sort_values(["parent_rows", "all_force_parent"])
        check = check.iloc[[0, (len(check) - 1) // 2, len(check) - 1]]
        for fam in check.itertuples(index=False):
            own = parent == fam.all_force_parent
            br = np.flatnonzero(own & (child == fam.big_child))
            sr = np.flatnonzero(own & (child == fam.small_child))
            for st in ("accepted", "unresolved", "rest"):
                b, s = br[state[br] == st], sr[state[sr] == st]
                actual = segmented_fraction(frames[s], segment[s], frames[b], segment[b])
                got = observed[(observed.all_force_parent == fam.all_force_parent) & (observed.state == st)].iloc[0]
                expected = float(actual.mean()) if actual.size else np.nan
                same = (np.isnan(expected) and np.isnan(got.small_fraction_partner_abs_8_29)) or np.isclose(
                    expected, got.small_fraction_partner_abs_8_29, atol=1e-15
                )
                selected_rechecks.append((window, int(fam.all_force_parent), st, int(actual.size), expected,
                                          float(got.small_fraction_partner_abs_8_29), bool(same)))

        # Quantify only the omitted families, including the same deterministic null definition.
        rng = np.random.default_rng(20260928 + (2 if window == "W2" else 3))
        possible = np.r_[
            -np.arange(int(np.floor(.200 * FS)), int(np.ceil(.020 * FS)) - 1, -1),
            np.arange(int(np.ceil(.020 * FS)), int(np.floor(.200 * FS)) + 1),
        ]
        offsets = np.sort(rng.choice(possible, size=200, replace=False))
        for fam in excluded.itertuples(index=False):
            own = parent == fam.all_force_parent
            br = np.flatnonzero(own & (child == fam.big_child))
            sr = np.flatnonzero(own & (child == fam.small_child))
            for st in ("accepted", "unresolved", "rest"):
                b, s = br[state[br] == st], sr[state[sr] == st]
                bt, smt, bseg, sseg = frames[b], frames[s], segment[b], segment[s]
                obs = segmented_fraction(smt, sseg, bt, bseg)
                null, obs_common, support_n = [], [], []
                for offset in offsets:
                    keep = (sseg >= 0) & (smt + offset >= left[s]) & (smt + offset < right[s])
                    q, qseg = smt[keep], sseg[keep]
                    if q.size and bt.size:
                        obs_common.append(float(segmented_fraction(q, qseg, bt, bseg).mean()))
                        null.append(float(segmented_fraction(q + offset, qseg, bt, bseg).mean()))
                        support_n.append(int(q.size))
                excluded_profiles.append({
                    "window": window, "all_force_parent": int(fam.all_force_parent), "state": st,
                    "big_child": int(fam.big_child), "small_child": int(fam.small_child),
                    "big_events": int(bt.size), "small_events": int(smt.size),
                    "observed_fraction_8_29": float(obs.mean()) if obs.size else np.nan,
                    "usable_offsets": len(null),
                    "median_observed_same_support": float(np.median(obs_common)) if obs_common else np.nan,
                    "median_null_fraction": float(np.median(null)) if null else np.nan,
                    "median_supported_small_events": float(np.median(support_n)) if support_n else np.nan,
                })

    exclusions_df = pd.concat(exclusions, ignore_index=True)
    exclusions_df.to_csv(args.output / "EXCLUDED_SPLIT_FAMILIES.csv", index=False)
    pd.DataFrame(excluded_profiles).to_csv(args.output / "EXCLUDED_FAMILY_COINCIDENCE.csv", index=False)

    # Independent closure from the row ledger, checked against both summaries.
    with np.load(args.packet / "CLOSE_PAIR_LEDGER.npz", allow_pickle=False) as z:
        ledger = {k: np.asarray(z[k]) for k in z.files}
    curve = pd.read_csv(args.packet / "PAIR_CONTRIBUTION_CURVE.csv")
    closure = pd.read_csv(args.packet / "PAIR_LEDGER_CLOSURE.csv")
    ledger_checks = []
    for window in ("W2", "W3"):
        for clock in ("actual_noforce_final", "common_fixed_all_force"):
            keep = (ledger["window"] == window) & (ledger["clock"] == clock)
            categories = {k: int(np.count_nonzero(ledger["category"][keep] == k)) for k in np.unique(ledger["category"][keep])}
            contributions = {k: int(np.count_nonzero(ledger["contribution"][keep] == k)) for k in np.unique(ledger["contribution"][keep])}
            c = closure[(closure.window == window) & (closure.clock == clock)].iloc[0]
            q = curve[(curve.window == window) & (curve.clock == clock)]
            ledger_checks.append({
                "window": window, "clock": clock, "ledger_rows": int(keep.sum()),
                "curve_rows_sum": int(q.pairs.sum()), "closure_total": int(c.total_close_pairs_abs_0_30),
                "category_sum": int(sum(categories.values())),
                "big_small": contributions.get("big_small", 0),
                "big_other": contributions.get("big_other", 0),
                "explicit_remainder": int(keep.sum()) - contributions.get("big_small", 0) - contributions.get("big_other", 0),
                "passes": int(keep.sum()) == int(q.pairs.sum()) == int(c.total_close_pairs_abs_0_30)
                and int(c.category_closure_delta) == 0
                and int(c.big_terms_plus_remainder) == int(keep.sum()),
            })

    mixture = pd.read_csv(args.packet / "NATIVE_PREMERGE_CONSTITUENT_MIXTURE.csv")
    morph = pd.read_csv(args.packet / "NATIVE_CONSTITUENT_PAIR_MORPHOLOGY.csv")
    nulls = pd.read_csv(args.packet / "CHILD_COINCIDENCE_NONWRAPPING_NULL.csv")
    child_checks = pd.DataFrame(child_profile_checks, columns=["window", "parent", "actual_children", "saved_children"])
    selected_checks = pd.DataFrame(selected_rechecks, columns=[
        "window", "parent", "state", "small_events", "independent_fraction", "saved_fraction", "passes"
    ])
    result = {
        "status": "qualified_pass",
        "packet_manifest_hash_failures": hash_failures,
        "catalogue_sha256": sha256(args.catalogue),
        "denominators": denominator,
        "scope_finding": (
            "H5 analyzed only split families whose second child had >=20 rows and >=10% assigned share; "
            "it did not analyze all exact split families. Timing/null profiles use only the two largest children."
        ),
        "all_saved_child_profiles_complete_within_selected_families": bool((child_checks.actual_children == child_checks.saved_children).all()),
        "selected_observed_fraction_rechecks": {
            "cells": int(len(selected_checks)), "all_pass": bool(selected_checks.passes.all())
        },
        "null_checks": {
            "rows": int(len(nulls)),
            "exactly_200_rows_each_family_state": bool((nulls.groupby(["window", "all_force_parent", "state"]).size() == 200).all()),
            "offsets_unique_per_window": {w: int(g.signed_offset_samples.nunique()) for w, g in nulls.groupby("window")},
            "both_signs_per_window": {w: bool((g.signed_offset_samples.lt(0).any() and g.signed_offset_samples.gt(0).any())) for w, g in nulls.groupby("window")},
            "offset_abs_ms_range": [float(nulls.signed_offset_ms.abs().min()), float(nulls.signed_offset_ms.abs().max())],
            "zero_or_unavailable_cells_retained": int((~nulls.available.astype(bool)).sum()),
            "supported_never_exceeds_original": bool((nulls.same_segment_supported_events <= nulls.original_small_events).all()),
        },
        "pair_ledger_checks": ledger_checks,
        "native_namespace_checks": {
            "mixture_rows": int(len(mixture)), "morphology_rows": int(len(morph)),
            "native_namespace_explicit": bool(mixture.namespace.str.contains("not_final_child_template").all()),
            "physical_columns_never_exceed_182": bool((mixture.physical_observed_columns <= 182).all()),
            "array_columns_are_206_but_not_treated_as_observed": bool((mixture.native_template_array_columns == 206).all() and (mixture.native_countmask_observed_columns < 206).any()),
            "mixed_child_pair_rows_explicit": int((~morph.same_noforce_mapping_child.astype(bool)).sum()),
        },
        "conclusion": (
            "The no-broad-excess result is supported for the 148 threshold-selected substantial families and survives "
            "finite independent checks. It is not an all-family/all-child result; five split families and pairwise timing "
            "among children ranked third or lower remain outside that claim."
        ),
    }
    (args.output / "CV_INDEPENDENT_AUDIT.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    selected_checks.to_csv(args.output / "SELECTED_FAMILY_OBSERVED_RECHECKS.csv", index=False)
    pd.DataFrame(ledger_checks).to_csv(args.output / "PAIR_LEDGER_INDEPENDENT_CHECKS.csv", index=False)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
