from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import postrun_oracle as oracle


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seal(root: Path) -> None:
    members = {}
    for name in sorted(oracle.MEMBERS):
        path = root / name
        members[name] = {"bytes": path.stat().st_size, "sha256": sha(path)}
    write_json(root / "MANIFEST.json", {
        "schema": "dartsort-divergence-d3-production-output-manifest-v1",
        "members": members,
    })
    write_json(root / "COMPLETE.json", {
        "schema": "dartsort-divergence-d3-production-output-complete-v1",
        "status": "complete_saved_output_diagnostic",
        "manifest_sha256": sha(root / "MANIFEST.json"),
        "raw_voltage_bytes_read": 0,
        "gpu_work": False,
    })


def build_success(root: Path) -> Path:
    root.mkdir()
    relation = {
        "rows": 4,
        "clock_for_final_correspondence": "final_npz.times_samples",
        "clock_for_matching_stage_attribution": "matching1.h5/times_samples on the same source row",
        "coverage_half_open_samples": oracle.SUPPORT,
        "global_chronological_order_required": False,
        "parent_path_matches": False,
        "row_counts_equal": True,
        "feature_rows_equal": True,
        "channels_equal_every_row": True,
        "source_row_preservation_verified_by_frozen_hashes": True,
        "row_ancestry_established": False,
        "matching_stage_assignment_attribution": "unavailable",
        "shifted_final_label_rows_outside_support": 0,
        "negative_final_label_rows": 0,
        "numerical_coincidence_used_as_ancestry_evidence": False,
        "unavailability_reason": "exact parent path did not hold",
    }
    write_json(root / "DARTSORT_TIME_RELATION.json", relation)
    write_json(root / "STAGE_VALIDATION.json", {
        "dartsort_full_row_binding": relation,
        "kilosort_empirical_export_validation": {},
        "ref_semantic_limit": "empirical only",
    })
    write_json(root / "DARTSORT_MATCHER_ORDERING.json", {
        "eligible_final_units": 1,
        "units_reordered_at_matcher_boundary": 1,
        "within_unit_input_inversions": 1,
        "ordering_rule": "stable lexicographic sort by (final_time, source_row_id)",
        "source_row_ids_retained": True,
    })

    rankings = []
    for anchor_id in sorted(oracle.ANCHORS):
        arm, unit, group = anchor_id.split(":", 2)
        row = {
            "dartsort_final_label": 7,
            "candidate_events": 100,
            "anchor_events": 25,
            "exclusive_matches": 25,
            "smaller_train_retention": 1.0,
            "exclusive_f1": 0.4,
            "passes_minimum_support": True,
            "dense_rank": 1,
        }
        rankings.append({
            "anchor_id": anchor_id,
            "arm": arm,
            "unit": int(unit),
            "group": group,
            "rows": [row],
            "passing_candidate_labels": [7],
            "candidate_universe": "all final labels >=0 with >=100 events in common support",
            "ranking": "matches desc, smaller-train retention desc, F1 desc; exact ties share dense rank; label asc serialization only",
            "minimum_support": {"unit_events": 100, "exclusive_matches": 25, "smaller_train_retention": 0.01},
            "reporting_cap": None,
            "multiple_candidates_retained": True,
        })
    write_json(root / "DARTSORT_CANDIDATE_RANKING.json", rankings)

    contexts = [("REF-A", group, None, None, None) for group in sorted(oracle.GROUPS)]
    for anchor_id in sorted(oracle.ANCHORS):
        arm, unit, group = anchor_id.split(":", 2)
        contexts.append((f"{arm}-DARTsort", group, arm, int(unit), 7))
    summaries = []
    pair_rows = []
    for pair, group, arm, unit, label in contexts:
        context = {"pair": pair, "group": group, "anchor_arm": arm,
                   "anchor_unit": unit, "dartsort_final_label": label}
        is_final_pair = pair != "REF-A"
        a_events = 25
        b_events = 100 if is_final_pair else 25
        summaries.append({
            **context, "a_events": a_events, "b_events": b_events, "exclusive_matches": 25,
            "unmatched_a": 0, "unmatched_b": b_events - 25, "ambiguous_a_degree_gt1": 0,
            "ambiguous_b_degree_gt1": 0, "a_input_reordered_at_matcher_boundary": False,
            "b_input_reordered_at_matcher_boundary": False,
            "ordering_rule": "stable lexicographic sort by (sample_time, source_row_id)",
            "source_row_ids_retained": True, "transitive_identifier": None,
        })
        for index in range(a_events):
            pair_rows.append({
                **context, "side": "a", "pair_local_row_index": index,
                "source_row_index": 1 + index, "sample_time": 100 + index * 100,
                "matched_other_pair_local_row_index": index,
                "matched_other_source_row_index": 1001 + index,
                "all_neighbor_candidate_source_row_indices": [1001 + index],
                "all_neighbor_degree": 1, "unmatched": False,
            })
        for index in range(b_events):
            matched = index < 25
            time = 100 + index * 100 if matched else 10000 + index * 20
            pair_rows.append({
                **context, "side": "b", "pair_local_row_index": index,
                "source_row_index": 1001 + index, "sample_time": time,
                "matched_other_pair_local_row_index": index if matched else None,
                "matched_other_source_row_index": 1 + index if matched else None,
                "all_neighbor_candidate_source_row_indices": [1 + index] if matched else [],
                "all_neighbor_degree": 1 if matched else 0, "unmatched": not matched,
            })
    write_json(root / "PAIR_SUMMARY.json", summaries)
    write_jsonl(root / "PAIR_EVENT_RELATION.jsonl", pair_rows)
    write_jsonl(root / "KILOSORT_PARENT_RELATION.jsonl", [])
    write_jsonl(root / "DARTSORT_STAGE_EVENTS.jsonl", [
        {"final_source_row_id": 100, "parent_row_id": None, "sample_time": 100,
         "matching_label": None, "matching_stage_assignment_attribution": "unavailable",
         "final_label": 7, "selected_by_anchor_ids": sorted(oracle.ANCHORS)},
        {"final_source_row_id": 101, "parent_row_id": None, "sample_time": 110,
         "matching_label": None, "matching_stage_assignment_attribution": "unavailable",
         "final_label": 7, "selected_by_anchor_ids": sorted(oracle.ANCHORS)},
    ])
    write_json(root / "RESOURCE_ACCOUNTING.json", {
        "elapsed_seconds": 1.0, "peak_rss_bytes_linux": 1024,
        "bound_input_bytes_hashed": 1000,
        "logical_processing_input_bytes_upper_bound": 2000,
        "output_bytes_before_seal": 3000,
        "raw_voltage_bytes_read": 0, "gpu_work": False,
        "fit_replay_sort_rf_holdout": False,
    })
    write_json(root / "PROVENANCE.json", {
        "contract_sha256": oracle.EXPECTED["contract"],
        "externally_pinned_contract_sha256": oracle.EXPECTED["contract"],
        "request_sha256": oracle.EXPECTED["request"],
        "review_release_sha256": oracle.EXPECTED["release"],
        "runner_sha256": oracle.EXPECTED["runner"],
        "adapter_sha256": oracle.EXPECTED["adapter"],
        "matcher_sha256": oracle.EXPECTED["matcher"],
        "dartsort_row_ancestry_established": False,
        "matching_stage_assignment_attribution": "unavailable",
        "final_event_clock": "final_npz.times_samples",
        "production_saved_output_execution": True,
        "raw_voltage_bytes_read": 0,
    })
    seal(root)
    return root


def expect_reject(root: Path, pattern: str) -> None:
    with pytest.raises(oracle.OracleError, match=pattern):
        oracle.validate_output(root)


def test_bound_predecessors_are_exact_and_complete():
    assert oracle.verify_predecessors()["verified"] is True


def test_positive_final_only_fixture(tmp_path):
    result = oracle.validate_output(build_success(tmp_path / "positive"))
    assert result["verdict"] == "GO_FINAL_REF_A_TO_DARTSORT_CORRESPONDENCE_ONLY"
    assert result["intermediate_matching_to_final_attribution"] == "unavailable"


def test_missing_field_fixture_fails_closed(tmp_path):
    root = build_success(tmp_path / "missing")
    relation = json.loads((root / "DARTSORT_TIME_RELATION.json").read_text())
    relation.pop("clock_for_final_correspondence")
    write_json(root / "DARTSORT_TIME_RELATION.json", relation)
    stage = json.loads((root / "STAGE_VALIDATION.json").read_text())
    stage["dartsort_full_row_binding"] = relation
    write_json(root / "STAGE_VALIDATION.json", stage)
    seal(root)
    expect_reject(root, "wrong final-event clock")


def test_stale_member_hash_fixture_fails_closed(tmp_path):
    root = build_success(tmp_path / "stale")
    with (root / "RESOURCE_ACCOUNTING.json").open("a") as stream:
        stream.write(" ")
    expect_reject(root, "member size drift|member hash drift")


def test_reordered_relation_rows_fixture_fails_closed(tmp_path):
    root = build_success(tmp_path / "reordered")
    rows = oracle.load_jsonl(root / "PAIR_EVENT_RELATION.jsonl")
    rows[0], rows[1] = rows[1], rows[0]
    write_jsonl(root / "PAIR_EVENT_RELATION.jsonl", rows)
    seal(root)
    expect_reject(root, "pair local row indexes not consecutive|pair relation rows reordered")


def test_false_parent_attribution_fixture_fails_closed(tmp_path):
    root = build_success(tmp_path / "false_parent")
    rows = oracle.load_jsonl(root / "DARTSORT_STAGE_EVENTS.jsonl")
    rows[0]["parent_row_id"] = 100
    rows[0]["matching_label"] = 9
    write_jsonl(root / "DARTSORT_STAGE_EVENTS.jsonl", rows)
    seal(root)
    expect_reject(root, "false parent attribution")


def test_preserved_failure_closure_fixture(tmp_path):
    root = tmp_path / "failure"
    root.mkdir()
    write_json(root / "FAILURE.json", {
        "schema": "dartsort-divergence-d3-production-failure-v1",
        "status": "failed_preserved", "error_type": "RuntimeError",
        "error": "synthetic", "raw_voltage_bytes_read": 0, "gpu_work": False,
    })
    assert oracle.validate_output(root)["verdict"] == "FAILURE_CLOSED_NO_SCIENTIFIC_RESULT"


def test_mixed_failure_and_complete_is_rejected(tmp_path):
    root = tmp_path / "mixed"
    root.mkdir()
    write_json(root / "FAILURE.json", {
        "schema": "dartsort-divergence-d3-production-failure-v1",
        "status": "failed_preserved", "raw_voltage_bytes_read": 0, "gpu_work": False,
    })
    write_json(root / "COMPLETE.json", {})
    expect_reject(root, "failure namespace must not contain COMPLETE")


def test_far_apart_reciprocal_matches_are_rejected(tmp_path):
    root = build_success(tmp_path / "far_match")
    rows = oracle.load_jsonl(root / "PAIR_EVENT_RELATION.jsonl")
    first_context = oracle.context_key(rows[0])
    for row in rows:
        if oracle.context_key(row) == first_context and row["side"] == "b":
            row["sample_time"] += 1000
    write_jsonl(root / "PAIR_EVENT_RELATION.jsonl", rows)
    seal(root)
    expect_reject(root, "neighbor window differs|exclusive match exceeds inclusive tolerance")


def test_incomplete_eligible_label_universe_is_rejected(tmp_path):
    root = build_success(tmp_path / "incomplete_universe")
    ordering = json.loads((root / "DARTSORT_MATCHER_ORDERING.json").read_text())
    ordering["eligible_final_units"] = 2
    write_json(root / "DARTSORT_MATCHER_ORDERING.json", ordering)
    seal(root)
    expect_reject(root, "ranked universe count differs from eligible_final_units")


def test_inconsistent_ranking_metrics_are_rejected(tmp_path):
    root = build_success(tmp_path / "bad_metrics")
    rankings = json.loads((root / "DARTSORT_CANDIDATE_RANKING.json").read_text())
    rankings[0]["rows"][0]["smaller_train_retention"] = 0.5
    write_json(root / "DARTSORT_CANDIDATE_RANKING.json", rankings)
    seal(root)
    expect_reject(root, "smaller-train retention differs from counts")


def test_duplicate_eligible_label_rows_are_rejected(tmp_path):
    root = build_success(tmp_path / "duplicate_label")
    rankings = json.loads((root / "DARTSORT_CANDIDATE_RANKING.json").read_text())
    for item in rankings:
        item["rows"].append(dict(item["rows"][0]))
        item["passing_candidate_labels"].append(7)
    ordering = json.loads((root / "DARTSORT_MATCHER_ORDERING.json").read_text())
    ordering["eligible_final_units"] = 2
    write_json(root / "DARTSORT_MATCHER_ORDERING.json", ordering)
    write_json(root / "DARTSORT_CANDIDATE_RANKING.json", rankings)
    seal(root)
    expect_reject(root, "duplicate eligible final label row")


def test_negative_matched_index_is_rejected(tmp_path):
    root = build_success(tmp_path / "negative_index")
    rows = oracle.load_jsonl(root / "PAIR_EVENT_RELATION.jsonl")
    first_context = oracle.context_key(rows[0])
    target = next(row for row in rows if oracle.context_key(row) == first_context and row["side"] == "a" and row["pair_local_row_index"] == 24)
    target["matched_other_pair_local_row_index"] = -1
    write_jsonl(root / "PAIR_EVENT_RELATION.jsonl", rows)
    seal(root)
    expect_reject(root, "matched index is null or out of range")
