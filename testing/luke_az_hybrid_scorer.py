"""Outcome-blind scorer primitives for AZ's pitch-exact hybrid benchmark."""

from __future__ import annotations

import hashlib
from collections import defaultdict

import numpy as np


TOLERANCE_MS = 0.4


def tolerance_samples(fs_hz: float, tolerance_ms: float = TOLERANCE_MS) -> int:
    if fs_hz <= 0 or tolerance_ms < 0:
        raise ValueError("sampling frequency must be positive and tolerance nonnegative")
    return int(round(float(tolerance_ms) * float(fs_hz) / 1000.0))


def _arrays(truth_samples, donor_ids, output_samples, output_labels):
    truth = np.asarray(truth_samples, dtype=np.int64)
    donors = np.asarray(donor_ids, dtype=np.int64)
    output = np.asarray(output_samples, dtype=np.int64)
    labels = np.asarray(output_labels, dtype=np.int64)
    if truth.ndim != 1 or donors.ndim != 1 or truth.shape != donors.shape:
        raise ValueError("truth samples and donor IDs must be matching 1D arrays")
    if output.ndim != 1 or labels.ndim != 1 or output.shape != labels.shape:
        raise ValueError("output samples and labels must be matching 1D arrays")
    if np.unique(np.c_[donors, truth], axis=0).shape[0] != truth.size:
        raise ValueError("a donor truth train cannot contain duplicate samples")
    return truth, donors, output, labels


def exclusive_pairs(truth_samples, output_samples, tolerance: int):
    """Maximum-cardinality ordered 1:1 matching, inclusive at the tolerance."""
    truth = np.asarray(truth_samples, dtype=np.int64)
    output = np.asarray(output_samples, dtype=np.int64)
    torder = np.argsort(truth, kind="stable")
    oorder = np.argsort(output, kind="stable")
    ts = truth[torder]
    os = output[oorder]
    ti = []
    oi = []
    i = j = 0
    while i < ts.size and j < os.size:
        delta = int(os[j] - ts[i])
        if delta < -tolerance:
            j += 1
        elif delta > tolerance:
            i += 1
        else:
            ti.append(int(torder[i]))
            oi.append(int(oorder[j]))
            i += 1
            j += 1
    return np.asarray(ti, dtype=np.int64), np.asarray(oi, dtype=np.int64)


def pair_table(truth_samples, donor_ids, output_samples, output_labels, tolerance: int):
    truth, donors, output, labels = _arrays(
        truth_samples, donor_ids, output_samples, output_labels
    )
    rows = []
    for donor in np.unique(donors):
        dt = truth[donors == donor]
        for label in np.unique(labels):
            lo = output[labels == label]
            ti, oi = exclusive_pairs(dt, lo, tolerance)
            tp = int(ti.size)
            precision = tp / lo.size if lo.size else 0.0
            recall = tp / dt.size if dt.size else 0.0
            rows.append(
                {
                    "donor_id": int(donor),
                    "label": int(label),
                    "truth_count": int(dt.size),
                    "output_count": int(lo.size),
                    "exclusive_tp": tp,
                    "recall": float(recall),
                    "precision": float(precision),
                }
            )
    return rows


def associate_full_train(truth_samples, donor_ids, output_samples, output_labels, tolerance: int):
    """Associate each donor to one output label before any region split.

    Ranking is fixed: more exclusive matches, higher precision, higher recall,
    then lower integer label. The complete table and top-margin ambiguity are
    retained; label reuse across donors is explicitly reported as a false-merge
    candidate rather than silently resolved.
    """
    rows = pair_table(truth_samples, donor_ids, output_samples, output_labels, tolerance)
    by_donor = defaultdict(list)
    for row in rows:
        by_donor[row["donor_id"]].append(row)
    associations = []
    for donor in sorted(map(int, np.unique(np.asarray(donor_ids, dtype=np.int64)))):
        ranked = sorted(
            by_donor[donor],
            key=lambda r: (-r["exclusive_tp"], -r["precision"], -r["recall"], r["label"]),
        )
        positive = [row for row in ranked if row["exclusive_tp"] > 0]
        if not positive:
            associations.append(
                {
                    "donor_id": donor,
                    "association_status": "unmatched",
                    "primary_label": None,
                    "exclusive_tp": 0,
                    "full_train_recall": 0.0,
                    "full_train_precision": None,
                    "runner_up_label": None,
                    "tp_margin": None,
                    "exact_rank_tie": False,
                }
            )
            continue
        ranked = positive
        best = ranked[0]
        second = ranked[1] if len(ranked) > 1 else None
        associations.append(
            {
                "donor_id": donor,
                "association_status": "matched",
                "primary_label": best["label"],
                "exclusive_tp": best["exclusive_tp"],
                "full_train_recall": best["recall"],
                "full_train_precision": best["precision"],
                "runner_up_label": None if second is None else second["label"],
                "tp_margin": best["exclusive_tp"] - (0 if second is None else second["exclusive_tp"]),
                "exact_rank_tie": bool(
                    second is not None
                    and best["exclusive_tp"] == second["exclusive_tp"]
                    and best["precision"] == second["precision"]
                    and best["recall"] == second["recall"]
                ),
            }
        )
    labels_to_donors = defaultdict(list)
    for row in associations:
        if row["primary_label"] is not None:
            labels_to_donors[row["primary_label"]].append(row["donor_id"])
    false_merge_candidates = [
        {"label": int(label), "donor_ids": sorted(ds), "donor_count": len(ds)}
        for label, ds in sorted(labels_to_donors.items())
        if len(ds) > 1
    ]
    return {
        "associations": associations,
        "pair_table": rows,
        "false_merge_candidates": false_merge_candidates,
    }


def score_with_frozen_association(
    truth_samples,
    donor_ids,
    truth_is_episode,
    output_samples,
    output_labels,
    output_is_episode,
    associations,
    tolerance: int,
):
    """Globally exclusive scoring after frozen full-train association."""
    truth, donors, output, labels = _arrays(
        truth_samples, donor_ids, output_samples, output_labels
    )
    te = np.asarray(truth_is_episode, dtype=bool)
    oe = np.asarray(output_is_episode, dtype=bool)
    if te.shape != truth.shape or oe.shape != output.shape:
        raise ValueError("episode flags must align to their event arrays")
    primary = {
        int(r["donor_id"]): (
            None if r["primary_label"] is None else int(r["primary_label"])
        )
        for r in associations
    }
    if set(primary) != set(map(int, np.unique(donors))):
        raise ValueError("association must cover every donor exactly once")

    edges = []
    pre_counts = np.zeros(truth.size, dtype=np.int64)
    any_label_counts = np.zeros(truth.size, dtype=np.int64)
    any_label_sets = []
    for ti, (sample, donor) in enumerate(zip(truth, donors)):
        all_ix = np.flatnonzero(np.abs(output - sample) <= tolerance)
        any_label_counts[ti] = all_ix.size
        any_label_sets.append(sorted(set(map(int, labels[all_ix]))))
        label = primary[int(donor)]
        compatible = (
            np.empty(0, dtype=np.int64)
            if label is None
            else all_ix[labels[all_ix] == label]
        )
        pre_counts[ti] = compatible.size
        for oi in compatible:
            edges.append(
                (
                    abs(int(output[oi] - sample)),
                    int(sample),
                    int(donor),
                    int(output[oi]),
                    int(oi),
                    int(ti),
                )
            )
    matched_truth = set()
    matched_output = set()
    matches = []
    for error, _, donor, _, oi, ti in sorted(edges):
        if ti in matched_truth or oi in matched_output:
            continue
        matched_truth.add(ti)
        matched_output.add(oi)
        matches.append((ti, oi, error, donor))
    match_by_truth = {ti: (oi, error) for ti, oi, error, _ in matches}

    rows = []
    merge_labels = defaultdict(list)
    for donor, label in primary.items():
        if label is not None:
            merge_labels[label].append(donor)
    for donor in sorted(primary):
        for region, flag in (("episode", True), ("rest", False)):
            truth_ix = np.flatnonzero((donors == donor) & (te == flag))
            label = primary[donor]
            output_ix = (
                np.empty(0, dtype=np.int64)
                if label is None
                else np.flatnonzero((labels == label) & (oe == flag))
            )
            tp_truth = [i for i in truth_ix if int(i) in match_by_truth]
            tp_output = {
                match_by_truth[int(i)][0]
                for i in tp_truth
                if oe[match_by_truth[int(i)][0]] == flag
            }
            tp = len(tp_output)
            rows.append(
                {
                    "donor_id": donor,
                    "primary_label": primary[donor],
                    "region": region,
                    "truth_count": int(truth_ix.size),
                    "output_count": int(output_ix.size),
                    "tp": tp,
                    "fn": int(truth_ix.size - tp),
                    "fp": int(output_ix.size - tp),
                    "recall": float(tp / truth_ix.size) if truth_ix.size else None,
                    "precision": float(tp / output_ix.size) if output_ix.size else None,
                    "preexclusive_compatible_candidates": int(pre_counts[truth_ix].sum()),
                    "preexclusive_duplicate_candidates": int(np.maximum(pre_counts[truth_ix] - 1, 0).sum()),
                    "preexclusive_any_label_candidates": int(any_label_counts[truth_ix].sum()),
                    "ambiguous_truth_events": int(sum(len(any_label_sets[i]) > 1 for i in truth_ix)),
                    "primary_label_shared_by_donors": (
                        0 if label is None else len(merge_labels[label])
                    ),
                }
            )
    match_rows = [
        {
            "truth_index": int(ti),
            "output_index": int(oi),
            "donor_id": int(donors[ti]),
            "label": int(labels[oi]),
            "truth_sample": int(truth[ti]),
            "output_sample": int(output[oi]),
            "error_samples": int(error),
            "truth_is_episode": bool(te[ti]),
            "output_is_episode": bool(oe[oi]),
        }
        for ti, oi, error, _ in matches
    ]
    duplicate_rows = [
        {
            "truth_index": i,
            "donor_id": int(donors[i]),
            "truth_sample": int(truth[i]),
            "compatible_candidate_count": int(pre_counts[i]),
            "any_label_candidate_count": int(any_label_counts[i]),
            "candidate_labels": any_label_sets[i],
        }
        for i in range(truth.size)
        if pre_counts[i] > 1 or len(any_label_sets[i]) > 1
    ]
    return {"scores": rows, "matches": match_rows, "preexclusive_candidates": duplicate_rows}


def circular_shift_chance_controls(
    truth_samples,
    donor_ids,
    output_samples,
    output_labels,
    *,
    tolerance: int,
    start_sample: int,
    end_sample: int,
    shifts: int = 20,
    seed: int = 20260926,
):
    """Outcome-blind circular-shift control for split/merge association strength."""
    truth, donors, output, labels = _arrays(
        truth_samples, donor_ids, output_samples, output_labels
    )
    span = int(end_sample - start_sample)
    if span <= 4 * tolerance:
        raise ValueError("control interval is too short")
    rng = np.random.default_rng(seed)
    offsets = rng.integers(max(2 * tolerance + 1, 1), span - max(2 * tolerance + 1, 1), size=shifts)
    rows = []
    for replicate, offset in enumerate(offsets):
        shifted = start_sample + ((truth - start_sample + int(offset)) % span)
        association = associate_full_train(shifted, donors, output, labels, tolerance)
        for row in association["associations"]:
            rows.append(
                {
                    "replicate": replicate,
                    "offset_samples": int(offset),
                    "donor_id": row["donor_id"],
                    "best_label": row["primary_label"],
                    "exclusive_tp": row["exclusive_tp"],
                    "recall": row["full_train_recall"],
                }
            )
    return rows


def immutable_input_hash(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        h.update(f"{value.dtype.str}|{value.shape}".encode("ascii"))
        h.update(value.tobytes())
    return h.hexdigest()
