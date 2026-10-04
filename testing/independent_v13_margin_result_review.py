#!/usr/bin/env python3
"""Independent, read-only checks for the v13 H5 correspondence-margin result."""

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


RESULT = Path("/mnt/NPX/Luke/DARTsort_motion_experiments/en_first_medium_v13_correspondence_margin_result_20261003_v1_h5")
SEALED = Path("/mnt/NPX/Luke/DARTsort_motion_experiments/en_first_medium_v13_phase3_all61_descriptive_decomposition_20261003_v1_h5/inputs/PHASE3_AUDIT_UNIT_ROWS.json")
EXISTING = Path("/mnt/NPX/Luke/DARTsort_motion_experiments/en_first_medium_evaluator_post_qc_execution_result_20261002_v1_h5/results/pairs/REF384__existing_corrected_B384/correspondence_edges.csv")
TRANSITIONS = [176, 186, 191, 192, 196, 205, 208, 210, 220, 222]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_edges(path):
    integer = {"baseline_cluster", "candidate_cluster", "matched_events", "baseline_events", "candidate_events", "unmatched_baseline_events", "unmatched_candidate_events"}
    floating = {"baseline_retention", "candidate_retention", "jaccard"}
    rows = []
    with path.open(newline="") as stream:
        for raw in csv.DictReader(stream):
            row = {}
            for key, value in raw.items():
                if key in integer:
                    row[key] = int(value)
                elif key in floating:
                    row[key] = float(value)
                elif key in {"primary_match", "reference_in_cohort"}:
                    row[key] = value == "True"
                else:
                    row[key] = value
            rows.append(row)
    return rows


def independent_summary(edges, cohort, arm, sealed_counts):
    by_ref = defaultdict(list)
    by_candidate = defaultdict(list)
    for edge in edges:
        by_ref[edge["baseline_cluster"]].append(edge)
        by_candidate[edge["candidate_cluster"]].append(edge)
    rows = []
    related = set()
    for unit in cohort:
        candidates = sorted(by_ref[unit], key=lambda x: (-x["jaccard"], x["candidate_cluster"]))
        if not candidates:
            rows.append({
                "reference_unit": unit, "arm": arm, "overlap_filter_status": "UNMEASURED",
                "accepted_candidate": None, "best_candidate": None, "runner_up_candidate": None,
                "failed_primary_predicates": ["NO_EDGE_SURVIVES_OVERLAP_FILTER"],
                "accepted_status": "unmatched_zero", "accepted_numerator": 0,
                "accepted_reference_events": sealed_counts[unit], "accepted_retention": 0.0,
                "reference_best_gap": None, "reference_best_tie_count": None,
                "candidate_best_gap": None, "candidate_best_tie_count": None,
                "reciprocal": None, "reference_best_unique": None, "candidate_best_unique": None,
            })
            continue
        best = candidates[0]
        related.add(best["candidate_cluster"])
        reference_ties = [x for x in candidates if x["jaccard"] == best["jaccard"]]
        reference_runner = candidates[len(reference_ties)] if len(candidates) > len(reference_ties) else None
        competitors = sorted(by_candidate[best["candidate_cluster"]], key=lambda x: (-x["jaccard"], x["baseline_cluster"]))
        candidate_best = competitors[0]
        candidate_ties = [x for x in competitors if x["jaccard"] == candidate_best["jaccard"]]
        candidate_runner = competitors[len(candidate_ties)] if len(competitors) > len(candidate_ties) else None
        reference_best = best["jaccard"] == candidates[0]["jaccard"]
        reference_unique = len(reference_ties) == 1
        candidate_is_best = best["jaccard"] == candidate_best["jaccard"]
        candidate_unique = len(candidate_ties) == 1
        predicate_values = [
            (reference_best, "REFERENCE_NOT_BEST"),
            (reference_unique, "REFERENCE_BEST_TIED"),
            (candidate_is_best, "CANDIDATE_NOT_BEST"),
            (candidate_unique, "CANDIDATE_BEST_TIED"),
            (best["baseline_retention"] >= 0.5, "REFERENCE_RETENTION_BELOW_0_5"),
            (best["candidate_retention"] >= 0.5, "CANDIDATE_RETENTION_BELOW_0_5"),
        ]
        failures = [name for passed, name in predicate_values if not passed]
        independently_primary = not failures
        if independently_primary != best["primary_match"]:
            raise AssertionError(("primary predicate disagreement", unit, best))
        accepted = best["candidate_cluster"] if independently_primary else None
        rows.append({
            "reference_unit": unit, "arm": arm, "overlap_filter_status": "MEASURED",
            "accepted_candidate": accepted,
            "accepted_status": "primary_partner" if accepted is not None else "unmatched_zero",
            "accepted_numerator": best["matched_events"] if accepted is not None else 0,
            "accepted_reference_events": sealed_counts[unit],
            "accepted_retention": best["baseline_retention"] if accepted is not None else 0.0,
            "best_candidate": best["candidate_cluster"],
            "runner_up_candidate": None if reference_runner is None else reference_runner["candidate_cluster"],
            "matched_events": best["matched_events"], "reference_events": best["baseline_events"],
            "candidate_events": best["candidate_events"], "reference_retention": best["baseline_retention"],
            "candidate_retention": best["candidate_retention"], "jaccard": best["jaccard"],
            "reference_retention_margin": best["baseline_retention"] - 0.5,
            "candidate_retention_margin": best["candidate_retention"] - 0.5,
            "reference_best_gap": None if reference_runner is None else best["jaccard"] - reference_runner["jaccard"],
            "reference_best_tie_count": len(reference_ties),
            "candidate_best_gap": None if candidate_runner is None else candidate_best["jaccard"] - candidate_runner["jaccard"],
            "candidate_best_tie_count": len(candidate_ties),
            "candidate_best_reference": candidate_best["baseline_cluster"],
            "candidate_runner_up_reference": None if candidate_runner is None else candidate_runner["baseline_cluster"],
            "reciprocal": reference_best and candidate_is_best,
            "reference_best_unique": reference_unique, "candidate_best_unique": candidate_unique,
            "failed_primary_predicates": failures,
        })
    relevant = [edge for edge in edges if edge["baseline_cluster"] in cohort or edge["candidate_cluster"] in related]
    return rows, related, relevant


def same_value(actual, expected):
    if isinstance(expected, float):
        return isinstance(actual, (float, int)) and float(actual).hex() == expected.hex()
    return actual == expected


def compare_records(actual, expected, label):
    errors = []
    if len(actual) != len(expected):
        errors.append({"field": "row_count", "actual": len(actual), "expected": len(expected)})
    for index, (got, want) in enumerate(zip(actual, expected)):
        if set(got) != set(want):
            errors.append({"row": index, "field": "keys", "actual": sorted(got), "expected": sorted(want)})
            continue
        for key in want:
            if not same_value(got[key], want[key]):
                errors.append({"row": index, "field": key, "actual": got[key], "expected": want[key]})
    if errors:
        raise AssertionError((label, errors[:20]))


def validate_graph(edges, label):
    errors = []
    primary_refs = []
    primary_candidates = []
    for edge in edges:
        n, rb, cc = edge["matched_events"], edge["baseline_events"], edge["candidate_events"]
        expected = (n / rb, n / cc, n / (rb + cc - n))
        for key, value in zip(("baseline_retention", "candidate_retention", "jaccard"), expected):
            if edge[key].hex() != value.hex():
                errors.append((edge["baseline_cluster"], edge["candidate_cluster"], key, edge[key], value))
        if n < math.ceil(0.1 * min(rb, cc)):
            errors.append((edge["baseline_cluster"], edge["candidate_cluster"], "overlap_filter", n, min(rb, cc)))
        if edge["primary_match"]:
            primary_refs.append(edge["baseline_cluster"])
            primary_candidates.append(edge["candidate_cluster"])
    if errors:
        raise AssertionError((label, "graph arithmetic/filter", errors[:20]))
    if len(primary_refs) != len(set(primary_refs)) or len(primary_candidates) != len(set(primary_candidates)):
        raise AssertionError((label, "primary graph is not one-to-one"))
    return {
        "edges": len(edges), "primary_edges": len(primary_refs),
        "baseline_units": len({x["baseline_cluster"] for x in edges}),
        "candidate_units": len({x["candidate_cluster"] for x in edges}),
        "arithmetic_filter_errors": 0, "primary_one_to_one": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()
    sealed_doc = json.loads(SEALED.read_text())
    sealed_rows = sealed_doc["rows"]
    cohort = [row["reference_unit"] for row in sealed_rows]
    if cohort != sorted(cohort) or len(cohort) != 61 or len(set(cohort)) != 61:
        raise AssertionError("sealed cohort is not 61 unique sorted units")
    counts = {row["reference_unit"]: row["repaired_B384"]["reference_event_denominator"] for row in sealed_rows}

    repaired_edges = load_edges(RESULT / "output/REF__repaired_correspondence_edges.csv")
    if repaired_edges != sorted(repaired_edges, key=lambda x: (x["baseline_cluster"], x["candidate_cluster"])):
        raise AssertionError("repaired graph is not in canonical order")
    # Arithmetic checks are performed from integer event counts, not trusted derived columns.
    repaired_graph_stats = validate_graph(repaired_edges, "repaired")

    reconstructed, related, relevant = independent_summary(repaired_edges, cohort, "repaired_B384", counts)
    saved_rows = json.loads((RESULT / "output/ALL61_CORRESPONDENCE_ELIGIBILITY_MARGINS.json").read_text())
    compare_records(reconstructed, saved_rows, "independent margin reconstruction")

    six_fields = []
    six_errors = []
    for index, (saved, sealed) in enumerate(zip(saved_rows, sealed_rows)):
        arm = sealed["repaired_B384"]
        want = {
            "reference_unit": sealed["reference_unit"],
            "accepted_candidate": arm["candidate_primary_partner"],
            "accepted_status": arm["match_status"],
            "accepted_numerator": arm["matched_numerator"],
            "accepted_reference_events": arm["reference_event_denominator"],
            "accepted_retention": arm["retained_fraction"],
        }
        got = {key: saved[key] for key in want}
        row_errors = [key for key in want if not same_value(got[key], want[key])]
        if row_errors:
            six_errors.append({"row": index, "unit": sealed["reference_unit"], "fields": row_errors})
        six_fields.append({"row": index, "unit": sealed["reference_unit"], "status": "PASS" if not row_errors else "FAIL"})
    if six_errors:
        raise AssertionError(("sealed six-field reconciliation", six_errors))

    saved_competitors = load_edges(RESULT / "output/RELEVANT_NONCOHORT_COMPETITORS.csv")
    expected_competitors = []
    for edge in sorted(relevant, key=lambda x: (x["candidate_cluster"], x["baseline_cluster"])):
        expected_competitors.append({**edge, "reference_in_cohort": edge["baseline_cluster"] in set(cohort)})
    compare_records(saved_competitors, expected_competitors, "competitor extraction")

    existing_edges = load_edges(EXISTING)
    if existing_edges != sorted(existing_edges, key=lambda x: (x["baseline_cluster"], x["candidate_cluster"])):
        raise AssertionError("existing graph is not in canonical order")
    existing_graph_stats = validate_graph(existing_edges, "existing")
    existing_rows, _, _ = independent_summary(existing_edges, cohort, "existing_corrected_B384", counts)
    existing_six_errors = []
    for index, (saved, sealed) in enumerate(zip(existing_rows, sealed_rows)):
        arm = sealed["existing_corrected_B384"]
        want = {
            "reference_unit": sealed["reference_unit"],
            "accepted_candidate": arm["candidate_primary_partner"],
            "accepted_status": arm["match_status"],
            "accepted_numerator": arm["matched_numerator"],
            "accepted_reference_events": arm["reference_event_denominator"],
            "accepted_retention": arm["retained_fraction"],
        }
        got = {key: saved[key] for key in want}
        row_errors = [key for key in want if not same_value(got[key], want[key])]
        if row_errors:
            existing_six_errors.append({"row": index, "unit": sealed["reference_unit"], "fields": row_errors})
    if existing_six_errors:
        raise AssertionError(("existing sealed six-field reconciliation", existing_six_errors))
    existing_by = {row["reference_unit"]: row for row in existing_rows}
    repaired_by = {row["reference_unit"]: row for row in reconstructed}
    transition_rows = []
    for unit in TRANSITIONS:
        e, r = existing_by[unit], repaired_by[unit]
        transition_rows.append({
            "reference_unit": unit,
            "sealed_transition": f"{next(x for x in sealed_rows if x['reference_unit'] == unit)['existing_corrected_B384']['match_status']}->{next(x for x in sealed_rows if x['reference_unit'] == unit)['repaired_B384']['match_status']}",
            "existing": {key: e.get(key) for key in (
                "overlap_filter_status", "accepted_candidate", "best_candidate", "matched_events", "reference_events",
                "candidate_events", "reference_retention", "candidate_retention", "jaccard", "reference_best_gap",
                "candidate_best_gap", "reciprocal", "reference_best_unique", "candidate_best_unique", "failed_primary_predicates")},
            "repaired": {key: r.get(key) for key in (
                "overlap_filter_status", "accepted_candidate", "best_candidate", "matched_events", "reference_events",
                "candidate_events", "reference_retention", "candidate_retention", "jaccard", "reference_best_gap",
                "candidate_best_gap", "reciprocal", "reference_best_unique", "candidate_best_unique", "failed_primary_predicates")},
        })

    repaired_losses = [row for row in transition_rows if row["sealed_transition"] == "primary_partner->unmatched_zero"]
    repaired_gains = [row for row in transition_rows if row["sealed_transition"] == "unmatched_zero->primary_partner"]
    metrics = ("matched_events", "reference_retention", "candidate_retention", "jaccard")

    def delta_summary(rows):
        return {
            metric: {
                "sum_existing": sum(row["existing"][metric] for row in rows),
                "sum_repaired": sum(row["repaired"][metric] for row in rows),
                "repaired_minus_existing": sum(row["repaired"][metric] - row["existing"][metric] for row in rows),
                "units_decreased": sum(row["repaired"][metric] < row["existing"][metric] for row in rows),
                "units_increased": sum(row["repaired"][metric] > row["existing"][metric] for row in rows),
            }
            for metric in metrics
        }

    loss_failures = defaultdict(int)
    gain_prior_failures = defaultdict(int)
    for row in repaired_losses:
        for failure in row["repaired"]["failed_primary_predicates"]:
            loss_failures[failure] += 1
    for row in repaired_gains:
        for failure in row["existing"]["failed_primary_predicates"]:
            gain_prior_failures[failure] += 1

    report = {
        "status": "PASS",
        "repaired_graph": {
            "sha256": sha256(RESULT / "output/REF__repaired_correspondence_edges.csv"),
            **repaired_graph_stats,
        },
        "existing_graph": {"sha256": sha256(EXISTING), **existing_graph_stats},
        "all61": {
            "rows": len(saved_rows), "repaired_six_field_errors": len(six_errors),
            "existing_six_field_errors": len(existing_six_errors), "full_repaired_margin_field_errors": 0,
        },
        "competitors": {"saved_rows": len(saved_competitors), "expected_rows": len(expected_competitors), "exact": True},
        "transition_summary": {
            "transition_units": len(transition_rows),
            "measured_both_arms": sum(
                row["existing"]["overlap_filter_status"] == "MEASURED"
                and row["repaired"]["overlap_filter_status"] == "MEASURED"
                for row in transition_rows
            ),
            "repaired_losses": len(repaired_losses), "repaired_gains": len(repaired_gains),
            "repaired_loss_failed_predicates": dict(sorted(loss_failures.items())),
            "repaired_gain_prior_failed_predicates": dict(sorted(gain_prior_failures.items())),
            "repaired_loss_deltas": delta_summary(repaired_losses),
            "repaired_gain_deltas": delta_summary(repaired_gains),
            "all_transition_deltas": delta_summary(transition_rows),
        },
        "transition_rows": transition_rows,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(rendered)
    else:
        sys.stdout.write(rendered)


if __name__ == "__main__":
    main()
