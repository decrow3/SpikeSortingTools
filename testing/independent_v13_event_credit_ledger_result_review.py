#!/usr/bin/env python3
"""Independent ledger-output reconstruction without access to H5 event arrays."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


RESULT = Path("/mnt/NPX/Luke/DARTsort_motion_experiments/en_first_medium_v13_event_credit_ledger_result_20261003_v1_h5")
FREEZE = Path("/mnt/NPX/Luke/DARTsort_motion_experiments/en_first_medium_v13_event_credit_ledger_freeze_20261003_v1_h1/FROZEN_SELECTED_EDGES.json")
PAIR_FIELDS = ["arm", "edge_id", "pair_ordinal", "ref_input_row", "ref_frame", "ref_cluster", "candidate_input_row", "candidate_frame", "candidate_cluster", "lag_frames", "primary_credit"]


def sha_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def set_hash(values):
    return hashlib.sha256("".join(f"{int(value)}\n" for value in sorted(values)).encode("ascii")).hexdigest()


def pair_hash(rows):
    def value(item):
        return "true" if item is True else "false" if item is False else str(item)
    payload = "".join("\t".join(value(row[field]) for field in PAIR_FIELDS) + "\n" for row in rows)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def load_pairs(path):
    rows = []
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != PAIR_FIELDS:
            raise ValueError("pair header differs")
        for raw in reader:
            row = dict(raw)
            for field in ("pair_ordinal", "ref_input_row", "ref_frame", "ref_cluster", "candidate_input_row", "candidate_frame", "candidate_cluster", "lag_frames"):
                row[field] = int(row[field])
            if row["primary_credit"] not in {"true", "false"}:
                raise ValueError("noncanonical boolean")
            row["primary_credit"] = row["primary_credit"] == "true"
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    freeze = json.loads(FREEZE.read_text())
    frozen_edges = freeze["frozen_edges"]
    edge_by_id = {edge["edge_id"]: edge for edge in frozen_edges}
    if len(edge_by_id) != 40:
        raise ValueError("frozen edge identity")
    ledger = load_pairs(RESULT / "output/EVENT_CREDIT_LEDGER.tsv")
    by_edge = defaultdict(list)
    edge_order_seen = []
    for row in ledger:
        if row["edge_id"] not in edge_by_id:
            raise ValueError("unknown edge")
        if not edge_order_seen or edge_order_seen[-1] != row["edge_id"]:
            if row["edge_id"] in edge_order_seen:
                raise ValueError("noncontiguous edge")
            edge_order_seen.append(row["edge_id"])
        by_edge[row["edge_id"]].append(row)
    if edge_order_seen != [edge["edge_id"] for edge in frozen_edges]:
        raise ValueError("edge order differs")

    saved_reconciliation_rows = json.loads((RESULT / "output/EDGE_RECONCILIATION.json").read_text())
    saved_reconciliation = {row["edge_id"]: row for row in saved_reconciliation_rows}
    if len(saved_reconciliation_rows) != 40 or len(saved_reconciliation) != 40 or set(saved_reconciliation) != set(edge_by_id):
        raise ValueError("edge reconciliation identity")
    edge_checks = []
    ref_sets = {}
    candidate_sets = {}
    for edge in frozen_edges:
        edge_id = edge["edge_id"]
        rows = by_edge[edge_id]
        if [row["pair_ordinal"] for row in rows] != list(range(len(rows))):
            raise ValueError(edge_id + " ordinal")
        if len(rows) != int(edge["matched_events"]):
            raise ValueError(edge_id + " numerator")
        matched = len(rows)
        baseline_events = int(edge["baseline_events"])
        candidate_events = int(edge["candidate_events"])
        derived = {
            "unmatched_baseline_events": baseline_events - matched,
            "unmatched_candidate_events": candidate_events - matched,
            "baseline_retention": matched / baseline_events,
            "candidate_retention": matched / candidate_events,
            "jaccard": matched / (baseline_events + candidate_events - matched),
        }
        for field in ("unmatched_baseline_events", "unmatched_candidate_events"):
            if derived[field] != int(edge[field]):
                raise ValueError(edge_id + " " + field)
        for field in ("baseline_retention", "candidate_retention", "jaccard"):
            if float(derived[field]).hex() != float(edge[field]).hex():
                raise ValueError(edge_id + " " + field)
        ref = {row["ref_input_row"] for row in rows}
        candidate = {row["candidate_input_row"] for row in rows}
        if len(ref) != len(rows) or len(candidate) != len(rows):
            raise ValueError(edge_id + " nonunique")
        for row in rows:
            if row["arm"] != edge["arm"] or row["ref_cluster"] != int(edge["baseline_cluster"]) or row["candidate_cluster"] != int(edge["candidate_cluster"]):
                raise ValueError(edge_id + " identity fields")
            if row["candidate_frame"] - row["ref_frame"] != row["lag_frames"] or abs(row["lag_frames"]) > 15:
                raise ValueError(edge_id + " lag")
            if row["primary_credit"] != (edge["primary_match"] == "True"):
                raise ValueError(edge_id + " primary flag")
        saved = saved_reconciliation[edge_id]
        expected = {
            "edge_id": edge_id, "pairs": len(rows), "ref_set_sha256": set_hash(ref),
            "candidate_set_sha256": set_hash(candidate), "graph_record_sha256": edge["edge_record_sha256"], "status": "PASS",
        }
        if saved != expected:
            raise ValueError(edge_id + " reconciliation receipt")
        ref_sets[edge_id] = ref
        candidate_sets[edge_id] = candidate
        edge_checks.append(expected)

    ref_saved = json.loads((RESULT / "output/REF_ROW_PARTITIONS.json").read_text())
    ref_by_unit = {row["reference_unit"]: row for row in ref_saved}
    if len(ref_saved) != 10 or len(ref_by_unit) != 10 or set(ref_by_unit) != set(freeze["transition_units_ordered"]):
        raise ValueError("REF partition identity")
    reconstructed_ref = []
    f_sets = {}
    for unit in freeze["transition_units_ordered"]:
        sets = {}
        for arm in freeze["arm_order"]:
            selected = [edge for edge in frozen_edges if edge["arm"] == arm and int(edge["baseline_cluster"]) == unit]
            sets[arm] = set().union(*(ref_sets[edge["edge_id"]] for edge in selected))
            f_sets[(arm, unit)] = sets[arm]
        existing, repaired = sets[freeze["arm_order"][0]], sets[freeze["arm_order"][1]]
        saved = ref_by_unit[unit]
        universe = set().union(*(set(part["row_ids"]) for part in saved["partitions"].values()))
        expected_parts = {
            "common": existing & repaired,
            "existing_only": existing - repaired,
            "repaired_only": repaired - existing,
            "neither_selected_edge_credit": universe - (existing | repaired),
        }
        if len(universe) != saved["universe_count"] or set_hash(universe) != saved["universe_sha256"]:
            raise ValueError(f"REF {unit} universe")
        if len(universe) != int(next(edge for edge in frozen_edges if int(edge["baseline_cluster"]) == unit)["baseline_events"]):
            raise ValueError(f"REF {unit} denominator")
        for name, values in expected_parts.items():
            part = saved["partitions"][name]
            if values != set(part["row_ids"]) or len(values) != part["count"] or set_hash(values) != part["sha256"]:
                raise ValueError(f"REF {unit} {name}")
        reconstructed_ref.append(saved)

    candidate_saved = json.loads((RESULT / "output/CANDIDATE_ROW_PARTITIONS.json").read_text())
    candidate_by = {(row["arm"], row["candidate_cluster"]): row for row in candidate_saved}
    if len(candidate_by) != len(candidate_saved):
        raise ValueError("duplicate candidate partition identity")
    focal = set(freeze["transition_units_ordered"])
    reconstructed_candidate = []
    for arm in freeze["arm_order"]:
        candidates = sorted({int(edge["candidate_cluster"]) for edge in frozen_edges if edge["arm"] == arm})
        for candidate in candidates:
            relevant = [edge for edge in frozen_edges if edge["arm"] == arm and int(edge["candidate_cluster"]) == candidate]
            focal_sets = [candidate_sets[edge["edge_id"]] for edge in relevant if int(edge["baseline_cluster"]) in focal]
            competitor_sets = [candidate_sets[edge["edge_id"]] for edge in relevant if int(edge["baseline_cluster"]) not in focal]
            focal_rows = set().union(*focal_sets) if focal_sets else set()
            competitor_rows = set().union(*competitor_sets) if competitor_sets else set()
            saved = candidate_by[(arm, candidate)]
            universe = set().union(*(set(part["row_ids"]) for part in saved["partitions"].values()))
            expected_parts = {
                "focal_ref_only": focal_rows - competitor_rows,
                "competitor_only": competitor_rows - focal_rows,
                "shared_support": focal_rows & competitor_rows,
                "neither_selected_edge_support": universe - (focal_rows | competitor_rows),
            }
            denominator = int(relevant[0]["candidate_events"])
            if len(universe) != denominator or len(universe) != saved["universe_count"] or set_hash(universe) != saved["universe_sha256"]:
                raise ValueError(f"candidate {arm} {candidate} universe")
            for name, values in expected_parts.items():
                part = saved["partitions"][name]
                if values != set(part["row_ids"]) or len(values) != part["count"] or set_hash(values) != part["sha256"]:
                    raise ValueError(f"candidate {arm} {candidate} {name}")
            reconstructed_candidate.append(saved)
    if len(reconstructed_candidate) != len(candidate_saved) or set(candidate_by) != {(row["arm"], row["candidate_cluster"]) for row in reconstructed_candidate}:
        raise ValueError("candidate partition identity")

    hashes = json.loads((RESULT / "output/CANONICAL_HASHES.json").read_text())
    expected_hashes = {
        "pair_rows": len(ledger),
        "pair_set_sha256": pair_hash(ledger),
        "ref_partition_sha256": hashlib.sha256(json.dumps(reconstructed_ref, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "candidate_partition_sha256": hashlib.sha256(json.dumps(reconstructed_candidate, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
    }
    if hashes != expected_hashes:
        raise ValueError("canonical hashes")

    # Frozen A/B/C rules, evaluated independently and without precedence.
    repaired_arm = "repaired_B384"
    existing_arm = "existing_corrected_B384"
    abc = []
    for unit in freeze["transition_units_ordered"]:
        existing, repaired = f_sets[(existing_arm, unit)], f_sets[(repaired_arm, unit)]
        common = existing & repaired
        direct_repaired = [edge for edge in frozen_edges if edge["arm"] == repaired_arm and int(edge["baseline_cluster"]) == unit]
        a_candidates = []
        for edge in direct_repaired:
            candidate = int(edge["candidate_cluster"])
            target = candidate_sets[edge["edge_id"]]
            other_edges = [other for other in frozen_edges if other["arm"] == repaired_arm and int(other["candidate_cluster"]) == candidate and int(other["baseline_cluster"]) != unit]
            other = set().union(*(candidate_sets[item["edge_id"]] for item in other_edges)) if other_edges else set()
            other_minus_target = other - target
            if common and other_minus_target:
                a_candidates.append({"candidate": candidate, "common_ref_count": len(common), "other_minus_target_candidate_rows": len(other_minus_target)})
        a_pass = bool(a_candidates)
        b_count = len(existing - repaired)
        union = existing | repaired
        ref_jaccard = len(common) / len(union) if union else 1.0
        leaders = {}
        for arm in (existing_arm, repaired_arm):
            direct = [edge for edge in frozen_edges if edge["arm"] == arm and int(edge["baseline_cluster"]) == unit]
            leaders[arm] = max(direct, key=lambda edge: float(edge["jaccard"]))
        base_cross = (float(leaders[existing_arm]["baseline_retention"]) >= 0.5) != (float(leaders[repaired_arm]["baseline_retention"]) >= 0.5)
        candidate_cross = (float(leaders[existing_arm]["candidate_retention"]) >= 0.5) != (float(leaders[repaired_arm]["candidate_retention"]) >= 0.5)
        c_pass = ref_jaccard >= 0.95 and (base_cross or candidate_cross)
        abc.append({
            "reference_unit": unit,
            "A_redistribution_broader_grouping": {"pass": a_pass, "qualifying_candidates": a_candidates},
            "B_reduced_selected_edge_support": {"pass": b_count > 0, "existing_only_ref_rows": b_count},
            "C_threshold_sensitivity": {"pass": c_pass, "ref_set_jaccard": ref_jaccard, "baseline_retention_crossing": base_cross, "candidate_retention_crossing": candidate_cross},
            "counts": {"existing_selected_ref": len(existing), "repaired_selected_ref": len(repaired), "common": len(common), "existing_only": b_count, "repaired_only": len(repaired - existing)},
        })

    output_manifest = json.loads((RESULT / "output/OUTPUT_MANIFEST.json").read_text())
    manifest_errors = []
    for name, expected in output_manifest.items():
        observed = sha_file(RESULT / "output" / name)
        if observed != expected:
            manifest_errors.append({"name": name, "expected": expected, "observed": observed})
    if manifest_errors:
        raise ValueError(manifest_errors)

    report = {
        "schema": "h1-v13-event-credit-ledger-independent-result-check-v1",
        "status": "PASS_IMPLEMENTATION_GATE",
        "checks": {
            "pair_rows": len(ledger), "edges": len(edge_checks), "edge_errors": 0,
            "ref_partitions": len(reconstructed_ref), "candidate_partitions": len(reconstructed_candidate),
            "canonical_hashes": expected_hashes, "inner_manifest_errors": 0,
        },
        "frozen_abc": abc,
        "abc_summary": {
            "A_pass_units": [row["reference_unit"] for row in abc if row["A_redistribution_broader_grouping"]["pass"]],
            "B_pass_units": [row["reference_unit"] for row in abc if row["B_reduced_selected_edge_support"]["pass"]],
            "C_pass_units": [row["reference_unit"] for row in abc if row["C_threshold_sensitivity"]["pass"]],
        },
        "event_arrays_directly_opened_by_h1": False,
        "scope_note": "H1 independently reconstructed ledger identities, sets, partitions and graph denominators from sealed outputs; H5-local source arrays were not mounted on H1.",
    }
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
