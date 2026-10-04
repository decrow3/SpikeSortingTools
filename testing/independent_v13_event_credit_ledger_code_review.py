#!/usr/bin/env python3
"""Independent source/selection review; deliberately does not open event arrays."""

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path


PACKET = Path("/mnt/NPX/Luke/DARTsort_motion_experiments/en_first_medium_v13_event_credit_ledger_contract_20261003_v1_h5")
H1_FREEZE = Path("testing/outputs/en_first_medium_v13_event_credit_ledger_freeze_20261003_v1_h1/FROZEN_SELECTED_EDGES.json")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_graph(path):
    rows = []
    with path.open(newline="") as stream:
        for index, row in enumerate(csv.DictReader(stream)):
            rows.append((index, row))
    return rows


def equal_graph_value(saved, raw, key):
    if key in {"baseline_cluster", "candidate_cluster", "matched_events", "baseline_events", "candidate_events", "unmatched_baseline_events", "unmatched_candidate_events"}:
        return int(saved) == int(raw)
    if key in {"baseline_retention", "candidate_retention", "jaccard"}:
        return float(saved).hex() == float(raw).hex()
    if key == "primary_match":
        return bool(saved) == (raw == "True")
    raise KeyError(key)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest_errors = []
    for line in (PACKET / "MANIFEST.sha256").read_text().splitlines():
        expected, relative = line.split(None, 1)
        relative = relative.strip().removeprefix("./")
        observed = sha(PACKET / relative)
        if observed != expected:
            manifest_errors.append({"path": relative, "expected": expected, "observed": observed})

    h5 = json.loads((PACKET / "FROZEN_EDGES.json").read_text())
    h1 = json.loads(H1_FREEZE.read_text())
    graph_errors = []
    h5_keys = set()
    for arm_doc in h5["arms"]:
        arm = arm_doc["arm"]
        graph = read_graph(Path(arm_doc["graph_path"]))
        by_key = {(int(row["baseline_cluster"]), int(row["candidate_cluster"])): (index, row) for index, row in graph}
        for edge in arm_doc["edges"]:
            key = (arm, int(edge["baseline_cluster"]), int(edge["candidate_cluster"]))
            h5_keys.add(key)
            source = by_key.get(key[1:])
            if source is None:
                graph_errors.append({"edge": key, "error": "missing_from_graph"})
                continue
            for field in ("baseline_cluster", "candidate_cluster", "matched_events", "baseline_events", "candidate_events", "unmatched_baseline_events", "unmatched_candidate_events", "baseline_retention", "candidate_retention", "jaccard", "primary_match"):
                if not equal_graph_value(edge[field], source[1][field], field):
                    graph_errors.append({"edge": key, "field": field, "saved": edge[field], "graph": source[1][field]})

    h1_keys = {
        ("existing" if edge["arm"].startswith("existing") else "repaired", int(edge["baseline_cluster"]), int(edge["candidate_cluster"]))
        for edge in h1["frozen_edges"]
    }

    source_text = (PACKET / "source/run_event_ledger.py").read_text()
    spec = importlib.util.spec_from_file_location("reviewed_ledger", PACKET / "source/run_event_ledger.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import numpy as np
    fixture_cases = [
        ([10, 30], [0, 1], [25, 45], [2, 3], [(0, 2, 10, 25), (1, 3, 30, 45)]),
        ([10, 10, 10], [4, 5, 6], [10, 10], [8, 9], [(4, 8, 10, 10), (5, 9, 10, 10)]),
        ([10, 11], [0, 1], [10], [2], [(0, 2, 10, 10)]),
        ([10], [0], [26], [1], []),
    ]
    fixture_errors = []
    for index, (at, ai, bt, bi, expected) in enumerate(fixture_cases):
        observed = module.pairs(np.array(at), np.array(ai), np.array(bt), np.array(bi))
        if observed != expected:
            fixture_errors.append({"case": index, "observed": observed, "expected": expected})

    report = {
        "schema": "h1-independent-v13-event-credit-ledger-code-check-v1",
        "status": "BLOCK_REPAIR_REQUIRED",
        "subject": {
            "manifest_sha256": sha(PACKET / "MANIFEST.sha256"),
            "complete_sha256": sha(PACKET / "COMPLETE.json"),
            "ledger_source_sha256": sha(PACKET / "source/run_event_ledger.py"),
            "frozen_edges_sha256": sha(PACKET / "FROZEN_EDGES.json"),
        },
        "manifest_errors": manifest_errors,
        "frozen_edge_graph_value_errors": graph_errors,
        "pairing_fixture_errors": fixture_errors,
        "selection": {
            "h5_edges": len(h5_keys), "required_h1_edges": len(h1_keys),
            "h5_extra": [list(key) for key in sorted(h5_keys - h1_keys)],
            "h5_missing_required": [list(key) for key in sorted(h1_keys - h5_keys)],
            "h5_is_strict_subset": h5_keys < h1_keys,
        },
        "source_static_findings": {
            "uses_assert_for_load_bearing_pair_count": "assert len(ps)==int(e['matched_events'])" in source_text,
            "checks_frozen_denominators_against_array_cluster_counts": False,
            "validates_array_integer_alignment_global_order_and_clock": False,
            "emits_per_edge_and_partition_set_hashes": False,
            "emits_output_manifest_and_resource_receipt": False,
            "binds_h1_interpretation_acceptance_freeze": False,
            "full_execute_and_serialization_fixture_present": False,
        },
        "event_arrays_opened_by_review": False,
    }
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
