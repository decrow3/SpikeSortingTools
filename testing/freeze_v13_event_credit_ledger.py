#!/usr/bin/env python3
"""Freeze selected graph edges for the v13 event-credit ledger without event reads."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


TRANSITIONS = [176, 186, 191, 192, 196, 205, 208, 210, 220, 222]
ARMS = [
    (
        "existing_corrected_B384",
        Path("/mnt/NPX/Luke/DARTsort_motion_experiments/en_first_medium_evaluator_post_qc_execution_result_20261002_v1_h5/results/pairs/REF384__existing_corrected_B384/correspondence_edges.csv"),
    ),
    (
        "repaired_B384",
        Path("/mnt/NPX/Luke/DARTsort_motion_experiments/en_first_medium_v13_correspondence_margin_result_20261003_v1_h5/output/REF__repaired_correspondence_edges.csv"),
    ),
]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_graph(path):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or ())
        rows = []
        for zero_index, raw in enumerate(reader):
            canonical = {field: raw[field] for field in fields}
            payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
            rows.append({
                "source_data_row_zero_based": zero_index,
                "source_file_line_one_based": zero_index + 2,
                "raw": canonical,
                "baseline_cluster": int(raw["baseline_cluster"]),
                "candidate_cluster": int(raw["candidate_cluster"]),
                "jaccard": float(raw["jaccard"]),
                "primary_match": raw["primary_match"] == "True",
                "edge_record_sha256": hashlib.sha256(payload.encode("ascii")).hexdigest(),
            })
    return fields, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--csv-output", required=True)
    args = parser.parse_args()
    frozen_edges = []
    graph_bindings = {}
    for arm_rank, (arm, path) in enumerate(ARMS):
        fields, rows = read_graph(path)
        graph_bindings[arm] = {"path": str(path), "sha256": sha256(path), "rows": len(rows)}
        by_ref = defaultdict(list)
        by_candidate = defaultdict(list)
        for row in rows:
            by_ref[row["baseline_cluster"]].append(row)
            by_candidate[row["candidate_cluster"]].append(row)
        selected = {}

        def add(row, role, transition_unit):
            key = row["source_data_row_zero_based"]
            if key not in selected:
                selected[key] = {
                    "arm": arm,
                    "arm_rank": arm_rank,
                    "transition_units": [],
                    "selection_roles": [],
                    **row,
                }
            if transition_unit not in selected[key]["transition_units"]:
                selected[key]["transition_units"].append(transition_unit)
            if role not in selected[key]["selection_roles"]:
                selected[key]["selection_roles"].append(role)

        for transition_rank, unit in enumerate(TRANSITIONS):
            unit_rows = sorted(by_ref[unit], key=lambda row: (-row["jaccard"], row["candidate_cluster"]))
            accepted = [row for row in unit_rows if row["primary_match"]]
            if len(accepted) > 1:
                raise RuntimeError(f"multiple accepted primary edges for {arm} REF {unit}")
            for row in accepted:
                add(row, f"accepted_primary:REF={unit}", unit)
            rejected = [row for row in unit_rows if not row["primary_match"]]
            if not rejected:
                continue
            best_rejected = rejected[0]["jaccard"]
            for row in rejected:
                if row["jaccard"] == best_rejected:
                    add(row, f"reference_leading_rejected_tie:REF={unit}", unit)

        # Close the frozen selection over exact candidate-side leaders, including all ties.
        seed_rows = list(selected.values())
        for seed in seed_rows:
            candidate_rows = sorted(
                by_candidate[seed["candidate_cluster"]],
                key=lambda row: (-row["jaccard"], row["baseline_cluster"]),
            )
            leader_score = candidate_rows[0]["jaccard"]
            for row in candidate_rows:
                if row["jaccard"] == leader_score:
                    for transition_unit in seed["transition_units"]:
                        add(
                            row,
                            f"candidate_leader_tie_closure:seed_REF={transition_unit}:candidate={seed['candidate_cluster']}",
                            transition_unit,
                        )

        for row in selected.values():
            raw = row["raw"]
            edge_id = (
                f"{arm}|row={row['source_data_row_zero_based']}|"
                f"ref={row['baseline_cluster']}|candidate={row['candidate_cluster']}"
            )
            frozen_edges.append({
                "arm": arm,
                "arm_rank": arm_rank,
                "transition_units": sorted(row["transition_units"], key=TRANSITIONS.index),
                "edge_id": edge_id,
                "edge_record_sha256": row["edge_record_sha256"],
                "source_data_row_zero_based": row["source_data_row_zero_based"],
                "source_file_line_one_based": row["source_file_line_one_based"],
                "selection_roles": sorted(row["selection_roles"]),
                **raw,
            })

    frozen_edges.sort(key=lambda row: (
        row["arm_rank"], min(TRANSITIONS.index(unit) for unit in row["transition_units"]),
        row["source_data_row_zero_based"], row["edge_id"],
    ))
    # Exact selection-set identity, independent of pretty-printing.
    set_lines = "".join(
        f"{row['edge_id']}\t{row['edge_record_sha256']}\t{','.join(row['selection_roles'])}\n"
        for row in frozen_edges
    )
    selection_sha256 = hashlib.sha256(set_lines.encode("ascii")).hexdigest()
    document = {
        "schema": "en-first-medium-v13-event-credit-ledger-edge-freeze-v1",
        "status": "FROZEN_PRE_EVENT_INSPECTION",
        "transition_units_ordered": TRANSITIONS,
        "arm_order": [arm for arm, _ in ARMS],
        "graph_bindings": graph_bindings,
        "selection_rule": [
            "for each arm and transition REF unit include its accepted primary edge when present",
            "include the highest-Jaccard non-primary edge and every exact-Jaccard tie at that rank",
            "for every selected candidate include every exact-Jaccard-tied candidate-side graph leader",
            "deduplicate by source graph row while retaining all selection roles"
        ],
        "selection_set_sha256": selection_sha256,
        "selection_set_canonical_encoding": "UTF-8 edge_id TAB edge_record_sha256 TAB comma-sorted-selection-roles NEWLINE in frozen_edges order",
        "selected_edges": len(frozen_edges),
        "frozen_edges": frozen_edges,
        "event_arrays_opened_by_freezer": False,
    }
    Path(args.json_output).write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    csv_fields = [
        "arm", "transition_units", "edge_id", "edge_record_sha256",
        "source_data_row_zero_based", "source_file_line_one_based", "selection_roles",
        "baseline_cluster", "candidate_cluster", "matched_events", "baseline_events",
        "candidate_events", "unmatched_baseline_events", "unmatched_candidate_events",
        "baseline_retention", "candidate_retention", "jaccard", "primary_match",
    ]
    with Path(args.csv_output).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields)
        writer.writeheader()
        for row in frozen_edges:
            writer.writerow({
                key: (
                    ";".join(row[key]) if key == "selection_roles"
                    else ";".join(str(unit) for unit in row[key]) if key == "transition_units"
                    else row[key]
                )
                for key in csv_fields
            })


if __name__ == "__main__":
    main()
