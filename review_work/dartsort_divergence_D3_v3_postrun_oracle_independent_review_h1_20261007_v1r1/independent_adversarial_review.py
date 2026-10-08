#!/usr/bin/env python3
"""Independent adversarial checks for the frozen D3 v3 post-run oracle."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path


PACKET = Path(
    "/mnt/NPX/Luke/DARTsort_motion_experiments/"
    "dartsort_divergence_D3_v3_postrun_oracle_h1_20261007_v1"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


oracle = load_module("postrun_oracle", PACKET / "postrun_oracle.py")
fixtures = load_module("frozen_oracle_fixtures", PACKET / "test_postrun_oracle.py")


def accepted(root: Path) -> dict:
    return oracle.validate_output(root)


def rejected(root: Path) -> str:
    try:
        oracle.validate_output(root)
    except oracle.OracleError as exc:
        return f"REJECTED: {exc}"
    return "FALSE_ACCEPT"


def main() -> int:
    results: dict[str, object] = {}
    with tempfile.TemporaryDirectory(prefix="d3_oracle_independent_") as temp:
        base = Path(temp)

        positive = fixtures.build_success(base / "positive")
        results["positive"] = accepted(positive)

        missing = fixtures.build_success(base / "missing_field")
        relation = json.loads((missing / "DARTSORT_TIME_RELATION.json").read_text())
        relation.pop("clock_for_final_correspondence")
        fixtures.write_json(missing / "DARTSORT_TIME_RELATION.json", relation)
        stage = json.loads((missing / "STAGE_VALIDATION.json").read_text())
        stage["dartsort_full_row_binding"] = relation
        fixtures.write_json(missing / "STAGE_VALIDATION.json", stage)
        fixtures.seal(missing)
        results["missing_field"] = rejected(missing)

        stale = fixtures.build_success(base / "stale_hash")
        with (stale / "RESOURCE_ACCOUNTING.json").open("a") as stream:
            stream.write(" ")
        results["stale_hash"] = rejected(stale)

        reordered = fixtures.build_success(base / "reordered_rows")
        rows = oracle.load_jsonl(reordered / "PAIR_EVENT_RELATION.jsonl")
        rows[0], rows[1] = rows[1], rows[0]
        fixtures.write_jsonl(reordered / "PAIR_EVENT_RELATION.jsonl", rows)
        fixtures.seal(reordered)
        results["reordered_rows"] = rejected(reordered)

        false_parent = fixtures.build_success(base / "false_parent")
        rows = oracle.load_jsonl(false_parent / "DARTSORT_STAGE_EVENTS.jsonl")
        rows[0]["parent_row_id"] = 100
        rows[0]["matching_label"] = 9
        fixtures.write_jsonl(false_parent / "DARTSORT_STAGE_EVENTS.jsonl", rows)
        fixtures.seal(false_parent)
        results["false_parent"] = rejected(false_parent)

        failure = base / "failure_closure"
        failure.mkdir()
        fixtures.write_json(failure / "FAILURE.json", {
            "schema": "dartsort-divergence-d3-production-failure-v1",
            "status": "failed_preserved",
            "error_type": "RuntimeError",
            "error": "synthetic",
            "raw_voltage_bytes_read": 0,
            "gpu_work": False,
        })
        results["failure_closure"] = accepted(failure)

        far = fixtures.build_success(base / "far_apart_matches")
        rows = oracle.load_jsonl(far / "PAIR_EVENT_RELATION.jsonl")
        for row in rows:
            if row["side"] == "b":
                row["sample_time"] += 1000
        fixtures.write_jsonl(far / "PAIR_EVENT_RELATION.jsonl", rows)
        fixtures.seal(far)
        results["false_accept_matches_1000_samples_apart"] = accepted(far)

        incomplete = fixtures.build_success(base / "incomplete_candidate_universe")
        ordering = json.loads((incomplete / "DARTSORT_MATCHER_ORDERING.json").read_text())
        ordering["eligible_final_units"] = 2
        fixtures.write_json(incomplete / "DARTSORT_MATCHER_ORDERING.json", ordering)
        fixtures.seal(incomplete)
        results["false_accept_one_ranked_label_with_two_eligible_units"] = accepted(incomplete)

        inconsistent = fixtures.build_success(base / "inconsistent_ranking_metrics")
        rankings = json.loads((inconsistent / "DARTSORT_CANDIDATE_RANKING.json").read_text())
        for item in rankings:
            item["rows"][0]["candidate_events"] = 1000
            item["rows"][0]["anchor_events"] = 2000
            # Keep the originally claimed 25 matches, 0.25 retention, and 0.25 F1.
            # The correct values would be 0.025 and 1/60 respectively.
        fixtures.write_json(inconsistent / "DARTSORT_CANDIDATE_RANKING.json", rankings)
        fixtures.seal(inconsistent)
        results["false_accept_internally_inconsistent_ranking_metrics"] = accepted(inconsistent)

    print(json.dumps(results, indent=2, sort_keys=True))
    required_ok = (
        results["positive"]["verdict"] == "GO_FINAL_REF_A_TO_DARTSORT_CORRESPONDENCE_ONLY"
        and str(results["missing_field"]).startswith("REJECTED:")
        and str(results["stale_hash"]).startswith("REJECTED:")
        and str(results["reordered_rows"]).startswith("REJECTED:")
        and str(results["false_parent"]).startswith("REJECTED:")
        and results["failure_closure"]["verdict"] == "FAILURE_CLOSED_NO_SCIENTIFIC_RESULT"
    )
    return 1 if required_ok and any(key.startswith("false_accept") for key in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
