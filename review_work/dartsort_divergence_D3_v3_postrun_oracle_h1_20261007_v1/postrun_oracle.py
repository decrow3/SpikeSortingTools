#!/usr/bin/env python3
"""Pre-outcome D3 v3 output oracle, independent of the production runner."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path


DERIVATIVE = Path("/mnt/NPX/Luke/DARTsort_motion_experiments/dartsort_divergence_D3_v3_executable_derivative_prep_h5_20261007_v2")
REVIEW = Path("/mnt/NPX/Luke/DARTsort_motion_experiments/dartsort_divergence_D3_v3_executable_derivative_review_h5_20261007_v2")

EXPECTED = {
    "derivative_manifest": "10cdbcd20afdd665840496fa3549b1e5119b4c21f8e32ffb519e03ea49b0dee7",
    "derivative_complete": "24228709fe0d032112f6224d4fe69ba0117d74c9da631e096ea1aebe4c89eb90",
    "review_manifest": "7874adc42c7aadf9da6bd93be0bb4d635ed7458874ab27c98a49195c3387e3fe",
    "review_complete": "71be7fbab5e36892d9882cfaba850ff3a11a3c910d2af14f52072512626d37b6",
    "contract": "c41c9c521accf0fcd3c95778afd0511a4e4b799394491a6c5e3134302fd29dd5",
    "runner": "725d811d2793a2d9a0fabd5ae20880a5cfaab80057dab0b8b5562e25657b5bb3",
    "request": "1d8ba4ebf7576b4c8aa1f481ef4901adc28599e613e029d874e4aa53106c40a4",
    "release": "d830db10c9841046ac26183015086c502dedbdd13ca6d0f07b469245545ca684",
    "adapter": "60ecddfdd01d1fbcdeffaa77642d6e4a88bf689dc7a7b1d9275b303efd37526b",
    "matcher": "989198578116ec53dd409bca96e544bca983f60f05bde051e0321810f21126c2",
}

MEMBERS = {
    "DARTSORT_TIME_RELATION.json", "DARTSORT_MATCHER_ORDERING.json",
    "STAGE_VALIDATION.json", "KILOSORT_PARENT_RELATION.jsonl",
    "DARTSORT_CANDIDATE_RANKING.json", "PAIR_EVENT_RELATION.jsonl",
    "PAIR_SUMMARY.json", "DARTSORT_STAGE_EVENTS.jsonl",
    "RESOURCE_ACCOUNTING.json", "PROVENANCE.json",
}
SUPPORT = [0, 314204094]
ANCHORS = {
    "A:351:high_impact_displaced_only_low_amplitude",
    "REF:375:high_impact_displaced_only_low_amplitude",
    "A:230:ordinary_noise_finite_fail", "REF:73:ordinary_noise_finite_fail",
    "A:399:srp_undefined_but_zero_pass_possible", "REF:530:srp_undefined_but_zero_pass_possible",
    "A:243:strict_pass_control", "REF:497:strict_pass_control",
}
GROUPS = {x.split(":", 2)[2] for x in ANCHORS}


class OracleError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise OracleError(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        raise OracleError(f"invalid JSON {path.name}: {exc}") from exc


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as stream:
        for line_no, line in enumerate(stream, 1):
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                raise OracleError(f"invalid JSONL {path.name}:{line_no}: {exc}") from exc
    return rows


def verify_manifest(root: Path, expected_members: set[str] | None = None) -> dict:
    manifest = load_json(root / "MANIFEST.json")
    require(manifest.get("schema") == "dartsort-divergence-d3-production-output-manifest-v1", "wrong output MANIFEST schema")
    members = manifest.get("members")
    require(isinstance(members, dict), "MANIFEST members missing")
    if expected_members is not None:
        require(set(members) == expected_members, "MANIFEST member set differs")
    for name, binding in members.items():
        path = root / name
        require(path.is_file(), f"missing member {name}")
        require(path.stat().st_size == binding.get("bytes"), f"member size drift {name}")
        require(sha256(path) == binding.get("sha256"), f"member hash drift {name}")
    return manifest


def verify_predecessors() -> dict:
    require(sha256(DERIVATIVE / "MANIFEST.json") == EXPECTED["derivative_manifest"], "derivative MANIFEST drift")
    require(sha256(DERIVATIVE / "COMPLETE.json") == EXPECTED["derivative_complete"], "derivative COMPLETE drift")
    require(sha256(REVIEW / "MANIFEST.json") == EXPECTED["review_manifest"], "review MANIFEST drift")
    require(sha256(REVIEW / "COMPLETE.json") == EXPECTED["review_complete"], "review COMPLETE drift")
    for root in (DERIVATIVE, REVIEW):
        manifest = load_json(root / "MANIFEST.json")
        for name, binding in manifest["members"].items():
            path = root / name
            require(path.stat().st_size == binding["bytes"], f"predecessor size drift {path}")
            require(sha256(path) == binding["sha256"], f"predecessor hash drift {path}")
        complete = load_json(root / "COMPLETE.json")
        require(complete["manifest_sha256"] == sha256(root / "MANIFEST.json"), f"predecessor COMPLETE binding drift {root}")
    return {"derivative_members": 11, "review_members": 4, "verified": True}


def validate_failure(root: Path) -> dict:
    require(not (root / "COMPLETE.json").exists(), "failure namespace must not contain COMPLETE")
    failure = load_json(root / "FAILURE.json")
    require(failure.get("schema") == "dartsort-divergence-d3-production-failure-v1", "wrong FAILURE schema")
    require(failure.get("status") == "failed_preserved", "wrong FAILURE status")
    require(failure.get("raw_voltage_bytes_read") == 0, "failure raw-voltage accounting nonzero")
    require(failure.get("gpu_work") is False, "failure GPU accounting nonzero")
    return {"verdict": "FAILURE_CLOSED_NO_SCIENTIFIC_RESULT", "error_type": failure.get("error_type")}


def validate_rankings(rankings: object) -> dict[str, list[int]]:
    require(isinstance(rankings, list) and len(rankings) == 8, "rankings must contain exactly eight anchors")
    require({x.get("anchor_id") for x in rankings} == ANCHORS, "ranking anchor set differs")
    passing = {}
    for item in rankings:
        require(item.get("candidate_universe") == "all final labels >=0 with >=100 events in common support", "candidate universe drift")
        require(item.get("reporting_cap") is None and item.get("multiple_candidates_retained") is True, "candidate cap or retention drift")
        require(item.get("minimum_support") == {"unit_events": 100, "exclusive_matches": 25, "smaller_train_retention": 0.01}, "candidate support drift")
        rows = item.get("rows")
        require(isinstance(rows, list), "ranking rows missing")
        keys = [(-r["exclusive_matches"], -r["smaller_train_retention"], -r["exclusive_f1"], r["dartsort_final_label"]) for r in rows]
        require(keys == sorted(keys), "ranking rows are not deterministically ordered")
        prior_metric = None
        dense = 0
        labels = []
        for row in rows:
            metric = (row["exclusive_matches"], row["smaller_train_retention"], row["exclusive_f1"])
            if metric != prior_metric:
                dense += 1
                prior_metric = metric
            require(row["dense_rank"] == dense, "dense rank mismatch")
            expected_pass = row["exclusive_matches"] >= 25 and row["smaller_train_retention"] >= 0.01
            require(row["passes_minimum_support"] is expected_pass, "candidate pass rule mismatch")
            if expected_pass:
                labels.append(row["dartsort_final_label"])
        require(item.get("passing_candidate_labels") == labels, "passing candidate serialization mismatch")
        passing[item["anchor_id"]] = labels
    return passing


def context_key(row: dict) -> tuple:
    return (row.get("pair"), row.get("group"), row.get("anchor_arm"), row.get("anchor_unit"), row.get("dartsort_final_label"))


def validate_success(root: Path) -> dict:
    actual = {p.name for p in root.iterdir() if p.is_file()}
    require(actual == MEMBERS | {"MANIFEST.json", "COMPLETE.json"}, "success namespace file set differs")
    require("FAILURE.json" not in actual, "success namespace contains FAILURE")
    verify_manifest(root, MEMBERS)
    complete = load_json(root / "COMPLETE.json")
    require(complete.get("schema") == "dartsort-divergence-d3-production-output-complete-v1", "wrong COMPLETE schema")
    require(complete.get("status") == "complete_saved_output_diagnostic", "wrong COMPLETE status")
    require(complete.get("manifest_sha256") == sha256(root / "MANIFEST.json"), "COMPLETE does not bind MANIFEST")
    require(complete.get("raw_voltage_bytes_read") == 0 and complete.get("gpu_work") is False, "COMPLETE prohibited-work accounting differs")

    relation = load_json(root / "DARTSORT_TIME_RELATION.json")
    require(relation.get("clock_for_final_correspondence") == "final_npz.times_samples", "wrong final-event clock")
    require(relation.get("coverage_half_open_samples") == SUPPORT, "final-event support differs")
    require(relation.get("global_chronological_order_required") is False, "global-order semantics differ")
    require(relation.get("shifted_final_label_rows_outside_support") == 0, "final events outside support")
    require(relation.get("row_ancestry_established") is False, "oracle requires unavailable intermediate ancestry")
    require(relation.get("matching_stage_assignment_attribution") == "unavailable", "intermediate attribution must be unavailable")
    require(relation.get("numerical_coincidence_used_as_ancestry_evidence") is False, "numerical coincidence used as ancestry")
    require(bool(relation.get("unavailability_reason")), "unavailable ancestry lacks reason")
    stage_validation = load_json(root / "STAGE_VALIDATION.json")
    require(stage_validation.get("dartsort_full_row_binding") == relation, "stage validation binding differs")

    ordering = load_json(root / "DARTSORT_MATCHER_ORDERING.json")
    require(ordering.get("ordering_rule") == "stable lexicographic sort by (final_time, source_row_id)", "matcher ordering rule differs")
    require(ordering.get("source_row_ids_retained") is True, "matcher source IDs not retained")
    for field in ("eligible_final_units", "units_reordered_at_matcher_boundary", "within_unit_input_inversions"):
        require(isinstance(ordering.get(field), int) and ordering[field] >= 0, f"invalid ordering count {field}")

    ranking_data = load_json(root / "DARTSORT_CANDIDATE_RANKING.json")
    passing = validate_rankings(ranking_data)
    summaries = load_json(root / "PAIR_SUMMARY.json")
    require(isinstance(summaries, list), "PAIR_SUMMARY must be a list")
    expected_contexts = {("REF-A", group, None, None, None) for group in GROUPS}
    for anchor_id, labels in passing.items():
        arm, unit, group = anchor_id.split(":", 2)
        pair = f"{arm}-DARTsort"
        expected_contexts.update((pair, group, arm, int(unit), label) for label in labels)
    require({context_key(x) for x in summaries} == expected_contexts, "pair summary contexts differ from rankings/frozen groups")
    summary_by_context = {context_key(x): x for x in summaries}
    require(len(summary_by_context) == len(summaries), "duplicate pair summary context")
    for summary in summaries:
        require(summary.get("ordering_rule") == "stable lexicographic sort by (sample_time, source_row_id)", "pair ordering rule differs")
        require(summary.get("source_row_ids_retained") is True and summary.get("transitive_identifier") is None, "pair identity semantics differ")

    pair_rows = load_jsonl(root / "PAIR_EVENT_RELATION.jsonl")
    by_context_side = defaultdict(list)
    for row in pair_rows:
        require(SUPPORT[0] <= row["sample_time"] < SUPPORT[1], "pair event outside support")
        by_context_side[(context_key(row), row.get("side"))].append(row)
    require({key[0] for key in by_context_side} == expected_contexts, "pair-event contexts differ")
    for (context, side), rows in by_context_side.items():
        require(side in {"a", "b"}, "invalid pair side")
        require([r["pair_local_row_index"] for r in rows] == list(range(len(rows))), "pair local row indexes not consecutive")
        keys = [(r["sample_time"], r["source_row_index"]) for r in rows]
        require(keys == sorted(keys), "pair relation rows reordered")
        require(len({r["source_row_index"] for r in rows}) == len(rows), "duplicate source row within pair side")
        for row in rows:
            require(row.get("all_neighbor_degree") == len(row.get("all_neighbor_candidate_source_row_indices", [])), "neighbor degree/list mismatch")
    for context, summary in summary_by_context.items():
        a_rows = by_context_side.get((context, "a"), [])
        b_rows = by_context_side.get((context, "b"), [])
        require(len(a_rows) == summary["a_events"] and len(b_rows) == summary["b_events"], "pair event total differs from summary")
        matched_a = [r for r in a_rows if not r["unmatched"]]
        matched_b = [r for r in b_rows if not r["unmatched"]]
        require(len(matched_a) == len(matched_b) == summary["exclusive_matches"], "exclusive match count differs from summary")
        require(summary["unmatched_a"] == len(a_rows) - len(matched_a) and summary["unmatched_b"] == len(b_rows) - len(matched_b), "unmatched count differs from summary")
        for row in matched_a:
            other = b_rows[row["matched_other_pair_local_row_index"]]
            require(other["matched_other_pair_local_row_index"] == row["pair_local_row_index"], "exclusive match is not reciprocal")
            require(other["source_row_index"] == row["matched_other_source_row_index"], "matched source ID differs")
    ranking_by_anchor = {item["anchor_id"]: item for item in ranking_data}
    for context, summary in summary_by_context.items():
        pair, group, arm, unit, label = context
        if pair == "REF-A":
            continue
        anchor_id = f"{arm}:{unit}:{group}"
        row = next(r for r in ranking_by_anchor[anchor_id]["rows"] if r["dartsort_final_label"] == label)
        require(summary["exclusive_matches"] == row["exclusive_matches"], "pair summary differs from candidate ranking")

    stage_rows = load_jsonl(root / "DARTSORT_STAGE_EVENTS.jsonl")
    final_ids = [r.get("final_source_row_id") for r in stage_rows]
    require(final_ids == sorted(set(final_ids)), "DARTsort stage rows not unique/increasing")
    for row in stage_rows:
        require(SUPPORT[0] <= row["sample_time"] < SUPPORT[1], "stage event outside support")
        require(row.get("parent_row_id") is None and row.get("matching_label") is None, "false parent attribution")
        require(row.get("matching_stage_assignment_attribution") == "unavailable", "stage attribution unexpectedly available")
        require(isinstance(row.get("selected_by_anchor_ids"), list) and row["selected_by_anchor_ids"] == sorted(set(row["selected_by_anchor_ids"])), "stage anchor IDs not canonical")

    resource = load_json(root / "RESOURCE_ACCOUNTING.json")
    for field in ("elapsed_seconds", "peak_rss_bytes_linux", "bound_input_bytes_hashed", "logical_processing_input_bytes_upper_bound", "output_bytes_before_seal"):
        require(isinstance(resource.get(field), (int, float)) and resource[field] >= 0, f"invalid resource field {field}")
    require(resource["peak_rss_bytes_linux"] <= 8589934592, "RSS bound exceeded")
    require(resource["elapsed_seconds"] <= 14400, "runtime bound exceeded")
    require(resource["logical_processing_input_bytes_upper_bound"] <= 14000000000, "logical read bound exceeded")
    require(resource["bound_input_bytes_hashed"] + resource["logical_processing_input_bytes_upper_bound"] <= 51539607552, "aggregate read bound exceeded")
    require(resource["output_bytes_before_seal"] <= 4294967296, "output bound exceeded")
    require(resource.get("raw_voltage_bytes_read") == 0 and resource.get("gpu_work") is False and resource.get("fit_replay_sort_rf_holdout") is False, "prohibited resource work reported")

    provenance = load_json(root / "PROVENANCE.json")
    for field, expected in (("contract_sha256", EXPECTED["contract"]), ("externally_pinned_contract_sha256", EXPECTED["contract"]),
                            ("request_sha256", EXPECTED["request"]), ("review_release_sha256", EXPECTED["release"]),
                            ("runner_sha256", EXPECTED["runner"]), ("adapter_sha256", EXPECTED["adapter"]),
                            ("matcher_sha256", EXPECTED["matcher"])):
        require(provenance.get(field) == expected, f"provenance binding differs {field}")
    require(provenance.get("dartsort_row_ancestry_established") is False, "provenance asserts ancestry")
    require(provenance.get("matching_stage_assignment_attribution") == "unavailable", "provenance asserts intermediate attribution")
    require(provenance.get("final_event_clock") == "final_npz.times_samples", "provenance final clock differs")
    require(provenance.get("production_saved_output_execution") is True and provenance.get("raw_voltage_bytes_read") == 0, "provenance execution scope differs")

    return {
        "verdict": "GO_FINAL_REF_A_TO_DARTSORT_CORRESPONDENCE_ONLY",
        "intermediate_matching_to_final_attribution": "unavailable",
        "anchors": 8,
        "pair_contexts": len(expected_contexts),
        "stage_rows": len(stage_rows),
    }


def validate_output(root: Path) -> dict:
    require(root.is_dir(), "output namespace missing")
    if (root / "FAILURE.json").exists():
        return validate_failure(root)
    return validate_success(root)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_output(args.output), indent=2, sort_keys=True))
