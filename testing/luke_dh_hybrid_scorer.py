"""DH hybrid scorer v3: exact permitted-edge matching plus provenance.

V2 remains preserved in ``luke_az_hybrid_scorer.py``.  This module changes the
global matching implementation and makes injection false positives conditional
on explicit output provenance.  It does not infer biological lineage.
"""
from __future__ import annotations

from collections import defaultdict
import numpy as np
import networkx as nx

from testing.luke_az_hybrid_scorer import _arrays

PROVENANCE_VALUES = {"injection_candidate", "background_supported", "ambiguous", "novel", "unknown", "noise"}


def exact_permitted_matching(truth_samples, donor_ids, output_samples, output_labels,
                             associations, tolerance, *, output_is_noise=None,
                             output_eligible=None, noise_labels=(-1,)):
    """Maximum-cardinality matching; then minimum total |timing error|.

    Permitted edges connect a truth row only to its donor's frozen primary
    label, inclusively within ``tolerance``.  Remaining timing ties use the sum
    of stable canonical edge ranks, followed by NetworkX's deterministic
    canonical insertion order.  Returned rows are sorted by canonical truth
    key, so input order cannot change the scientific pairing.
    """
    truth, donors, output, labels = _arrays(truth_samples, donor_ids, output_samples, output_labels)
    noise = np.zeros(output.size, bool) if output_is_noise is None else np.asarray(output_is_noise, bool)
    if noise.shape != output.shape:
        raise ValueError("output_is_noise must align to output arrays")
    if noise_labels is not None:
        noise |= np.isin(labels, np.asarray(tuple(noise_labels), dtype=np.int64))
    eligible = ~noise if output_eligible is None else np.asarray(output_eligible, bool) & ~noise
    if eligible.shape != output.shape:
        raise ValueError("output_eligible must align to output arrays")
    primary = {int(r["donor_id"]): None if r["primary_label"] is None else int(r["primary_label"])
               for r in associations}
    if set(primary) != set(map(int, np.unique(donors))):
        raise ValueError("association must cover every donor exactly once")

    matches = []
    edge_count = 0
    component_count = 0
    maximum_component_nodes = 0
    for label in sorted(set(x for x in primary.values() if x is not None)):
        tis = np.flatnonzero(np.array([primary[int(d)] == label for d in donors]))
        ois = np.flatnonzero((labels == label) & eligible)
        tis = tis[np.lexsort((tis, donors[tis], truth[tis]))]
        ois = ois[np.lexsort((ois, labels[ois], output[ois]))]
        candidates = []
        left = 0
        for trank, ti in enumerate(tis):
            while left < len(ois) and output[ois[left]] < truth[ti] - tolerance:
                left += 1
            j = left
            while j < len(ois) and output[ois[j]] <= truth[ti] + tolerance:
                candidates.append((trank, j, int(ti), int(ois[j]), abs(int(output[ois[j]] - truth[ti]))))
                j += 1
        edge_count += len(candidates)
        if not candidates:
            continue
        # Error is primary; stable rank sum is the declared deterministic tie.
        max_rank_sum = max(1, min(len(tis), len(ois)) * (len(tis) * (len(ois) + 1) + len(ois) + 1))
        scale = max_rank_sum + 1
        graph = nx.Graph()
        for trank, orank, ti, oi, error in candidates:
            rank = trank * (len(ois) + 1) + orank
            graph.add_edge(("t", ti), ("o", oi), weight=error * scale + rank,
                           error=error, stable_rank=rank)
        # Temporal gaps split the graph. Solve each exact independent component
        # instead of presenting a whole long unit to the blossom algorithm.
        for nodes in nx.connected_components(graph):
            component_count += 1; maximum_component_nodes = max(maximum_component_nodes, len(nodes))
            got = nx.min_weight_matching(graph.subgraph(nodes), weight="weight")
            for a, b in got:
                tnode, onode = (a, b) if a[0] == "t" else (b, a)
                ti, oi = int(tnode[1]), int(onode[1])
                matches.append((ti, oi, abs(int(output[oi] - truth[ti])), int(donors[ti])))
    matches.sort(key=lambda r: (int(truth[r[0]]), int(donors[r[0]]), r[0],
                                int(output[r[1]]), int(labels[r[1]]), r[1]))
    return matches, {
        "algorithm": "per-frozen-label NetworkX minimum-weight maximum-cardinality matching",
        "timing_objective": "minimum summed absolute sample error after maximum cardinality",
        "tie_policy": "minimum summed canonical edge rank; canonical graph insertion; canonical returned order",
        "tolerance_inclusive": True,
        "permitted_edge_count": edge_count,
        "temporal_component_count": component_count,
        "maximum_component_nodes": maximum_component_nodes,
        "noise_outputs_excluded": int(noise.sum()),
        "other_ineligible_outputs_excluded": int((~eligible & ~noise).sum()),
    }


def score_with_frozen_association_v3(truth_samples, donor_ids, truth_is_episode,
                                     output_samples, output_labels, output_is_episode,
                                     associations, tolerance, *, output_provenance=None,
                                     noise_labels=(-1,)):
    """Score injected truth without charging background or noise as injection FP."""
    truth, donors, output, labels = _arrays(truth_samples, donor_ids, output_samples, output_labels)
    te = np.asarray(truth_is_episode, bool); oe = np.asarray(output_is_episode, bool)
    provenance = (np.full(output.size, "unknown", dtype="U32") if output_provenance is None
                  else np.asarray(output_provenance, dtype="U32"))
    if te.shape != truth.shape or oe.shape != output.shape or provenance.shape != output.shape:
        raise ValueError("flags/provenance must align to event arrays")
    unknown_values = set(map(str, np.unique(provenance))) - PROVENANCE_VALUES
    if unknown_values:
        raise ValueError(f"unknown output provenance: {sorted(unknown_values)}")
    noise = (provenance == "noise") | np.isin(labels, np.asarray(tuple(noise_labels), dtype=np.int64))
    primary = {int(r["donor_id"]): None if r["primary_label"] is None else int(r["primary_label"])
               for r in associations}
    eligible = np.isin(provenance, ["injection_candidate", "ambiguous", "unknown"])
    matches, policy = exact_permitted_matching(truth, donors, output, labels, associations,
                                                tolerance, output_is_noise=noise,
                                                output_eligible=eligible, noise_labels=noise_labels)
    match_by_truth = {ti: (oi, error) for ti, oi, error, _ in matches}
    matched_output = {oi for _, oi, _, _ in matches}
    if len(match_by_truth) != len(matches) or len(matched_output) != len(matches):
        raise AssertionError("exact matching reused a truth or output row")
    pre_counts = np.zeros(truth.size, np.int64); any_counts = np.zeros(truth.size, np.int64)
    any_sets = []
    for ti, (sample, donor) in enumerate(zip(truth, donors)):
        all_ix = np.flatnonzero((np.abs(output - sample) <= tolerance) & ~noise)
        any_counts[ti] = all_ix.size; any_sets.append(sorted(set(map(int, labels[all_ix]))))
        label = primary[int(donor)]
        pre_counts[ti] = 0 if label is None else int((labels[all_ix] == label).sum())

    merge_labels = defaultdict(list)
    for donor, label in primary.items():
        if label is not None: merge_labels[label].append(donor)
    rows = []
    for donor in sorted(primary):
        for region, flag in (("episode", True), ("rest", False)):
            tx = np.flatnonzero((donors == donor) & (te == flag)); label = primary[donor]
            ox = np.empty(0, int) if label is None else np.flatnonzero((labels == label) & (oe == flag) & ~noise)
            eligible_ox = np.empty(0, int) if label is None else np.flatnonzero((labels == label) & (oe == flag) & eligible & ~noise)
            matched_here = [(ti, match_by_truth[int(ti)][0]) for ti in tx if int(ti) in match_by_truth
                            and oe[match_by_truth[int(ti)][0]] == flag]
            cross_truth = [(ti, match_by_truth[int(ti)][0]) for ti in tx if int(ti) in match_by_truth
                           and oe[match_by_truth[int(ti)][0]] != flag]
            cross_output = [(ti, oi) for ti, oi, _, _ in matches if int(oi) in set(map(int, ox)) and te[ti] != flag]
            globally_matched = len(matched_here) + len(cross_truth)
            certain_tp = sum(provenance[oi] == "injection_candidate" for _, oi in matched_here)
            ambiguous_tp = sum(provenance[oi] == "ambiguous" for _, oi in matched_here)
            unknown_tp = sum(provenance[oi] == "unknown" for _, oi in matched_here)
            matched_candidates = len(matched_here)
            unmatched = np.array([oi for oi in ox if int(oi) not in matched_output], int)
            def count(kind): return int((provenance[unmatched] == kind).sum()) if unmatched.size else 0
            rows.append({
                "donor_id": donor, "primary_label": label, "region": region,
                "truth_count": int(tx.size), "assigned_output_count": int(ox.size),
                "eligible_output_count": int(eligible_ox.size), "tp": certain_tp,
                "matched_candidates": matched_candidates, "ambiguous_tp": ambiguous_tp,
                "unknown_tp": unknown_tp,
                "globally_matched_truth_count": globally_matched,
                "global_unmatched_truth_count": int(tx.size - globally_matched),
                "fn_without_candidate": int(tx.size - globally_matched),
                "same_region_unmatched_truth_count": int(tx.size - matched_candidates),
                "regional_outflow_matches": len(cross_truth),
                "regional_inflow_matches": len(cross_output),
                "cross_boundary_truth_matches": len(cross_truth),
                "cross_boundary_output_consumed": len(cross_output),
                "injection_fp": count("injection_candidate"),
                "background_supported_unmatched": count("background_supported"),
                "ambiguous_unmatched": count("ambiguous"), "novel_unmatched": count("novel"),
                "unknown_unmatched": count("unknown"),
                "noise_excluded_for_label_region": (0 if label is None else int(((labels == label) & (oe == flag) & noise).sum())),
                "certain_recall": None if not tx.size else float(certain_tp / tx.size),
                "candidate_recall": None if not tx.size else float(globally_matched / tx.size),
                "same_region_candidate_recall": None if not tx.size else float(matched_candidates / tx.size),
                "injection_precision": None if certain_tp + count("injection_candidate") == 0 else float(certain_tp / (certain_tp + count("injection_candidate"))),
                "preexclusive_compatible_candidates": int(pre_counts[tx].sum()),
                "preexclusive_duplicate_candidates": int(np.maximum(pre_counts[tx] - 1, 0).sum()),
                "preexclusive_any_label_candidates": int(any_counts[tx].sum()),
                "ambiguous_truth_events": int(sum(len(any_sets[i]) > 1 for i in tx)),
                "primary_label_shared_by_donors": 0 if label is None else len(merge_labels[label]),
            })
    match_rows = [{
        "truth_index": ti, "output_index": oi, "donor_id": int(donors[ti]),
        "label": int(labels[oi]), "truth_sample": int(truth[ti]), "output_sample": int(output[oi]),
        "error_samples": error, "truth_is_episode": bool(te[ti]), "output_is_episode": bool(oe[oi]),
        "output_provenance": str(provenance[oi]),
        "match_certainty": "certain" if provenance[oi] == "injection_candidate" else "unresolved",
    } for ti, oi, error, _ in matches]
    duplicate_rows = [{
        "truth_index": i, "donor_id": int(donors[i]), "truth_sample": int(truth[i]),
        "compatible_candidate_count": int(pre_counts[i]), "any_label_candidate_count": int(any_counts[i]),
        "candidate_labels": any_sets[i],
    } for i in range(truth.size) if pre_counts[i] > 1 or len(any_sets[i]) > 1]
    cross_rows = [row for row in match_rows if row["truth_is_episode"] != row["output_is_episode"]]
    return {"scores": rows, "matches": match_rows, "cross_boundary_matches": cross_rows,
            "preexclusive_candidates": duplicate_rows, "matching_policy": policy,
            "provenance_mode": ("all_assigned_outputs_unknown" if output_provenance is None else "caller_supplied_oracle_fixture")}
