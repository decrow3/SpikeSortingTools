#!/usr/bin/env python3
"""Focused independent adversarial review of D3 v3 post-run oracle v2."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path


PACKET = Path(
    "/mnt/NPX/Luke/DARTsort_motion_experiments/"
    "dartsort_divergence_D3_v3_postrun_oracle_h1_20261007_v2"
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
fixtures = load_module("oracle_v2_fixtures", PACKET / "test_postrun_oracle.py")


def classify(root: Path) -> str:
    try:
        result = oracle.validate_output(root)
    except (oracle.OracleError, IndexError, KeyError, TypeError, ValueError) as exc:
        return f"REJECTED: {type(exc).__name__}: {exc}"
    return result["verdict"]


def main() -> int:
    results = {}
    with tempfile.TemporaryDirectory(prefix="d3_oracle_v2_delta_") as temp:
        base = Path(temp)

        positive = fixtures.build_success(base / "positive")
        results["positive"] = classify(positive)

        far = fixtures.build_success(base / "far")
        rows = oracle.load_jsonl(far / "PAIR_EVENT_RELATION.jsonl")
        context = oracle.context_key(rows[0])
        for row in rows:
            if oracle.context_key(row) == context and row["side"] == "b":
                row["sample_time"] += 1000
        fixtures.write_jsonl(far / "PAIR_EVENT_RELATION.jsonl", rows)
        fixtures.seal(far)
        results["original_far_apart_counterexample"] = classify(far)

        incomplete = fixtures.build_success(base / "incomplete")
        ordering = json.loads((incomplete / "DARTSORT_MATCHER_ORDERING.json").read_text())
        ordering["eligible_final_units"] = 2
        fixtures.write_json(incomplete / "DARTSORT_MATCHER_ORDERING.json", ordering)
        fixtures.seal(incomplete)
        results["original_incomplete_universe_counterexample"] = classify(incomplete)

        inconsistent = fixtures.build_success(base / "inconsistent")
        rankings = json.loads((inconsistent / "DARTSORT_CANDIDATE_RANKING.json").read_text())
        rankings[0]["rows"][0]["smaller_train_retention"] = 0.5
        fixtures.write_json(inconsistent / "DARTSORT_CANDIDATE_RANKING.json", rankings)
        fixtures.seal(inconsistent)
        results["original_inconsistent_metric_counterexample"] = classify(inconsistent)

        duplicate = fixtures.build_success(base / "duplicate_labels")
        ordering = json.loads((duplicate / "DARTSORT_MATCHER_ORDERING.json").read_text())
        ordering["eligible_final_units"] = 2
        fixtures.write_json(duplicate / "DARTSORT_MATCHER_ORDERING.json", ordering)
        rankings = json.loads((duplicate / "DARTSORT_CANDIDATE_RANKING.json").read_text())
        for item in rankings:
            second = dict(item["rows"][0])
            second["dense_rank"] = 1
            item["rows"].append(second)
            item["passing_candidate_labels"].append(second["dartsort_final_label"])
        fixtures.write_json(duplicate / "DARTSORT_CANDIDATE_RANKING.json", rankings)
        fixtures.seal(duplicate)
        results["duplicate_label_count_false_accept"] = classify(duplicate)

        negative = fixtures.build_success(base / "negative_match_index")
        rows = oracle.load_jsonl(negative / "PAIR_EVENT_RELATION.jsonl")
        context = oracle.context_key(rows[0])
        for row in rows:
            if (oracle.context_key(row) == context and row["side"] == "a"
                    and row["pair_local_row_index"] == 24):
                row["matched_other_pair_local_row_index"] = -1
        fixtures.write_jsonl(negative / "PAIR_EVENT_RELATION.jsonl", rows)
        fixtures.seal(negative)
        results["negative_match_index_false_accept"] = classify(negative)

    print(json.dumps(results, indent=2, sort_keys=True))
    original_repairs_hold = all(
        str(results[key]).startswith("REJECTED:")
        for key in (
            "original_far_apart_counterexample",
            "original_incomplete_universe_counterexample",
            "original_inconsistent_metric_counterexample",
        )
    )
    false_accept_remains = any(
        results[key] == "GO_FINAL_REF_A_TO_DARTSORT_CORRESPONDENCE_ONLY"
        for key in ("duplicate_label_count_false_accept", "negative_match_index_false_accept")
    )
    return 1 if original_repairs_hold and false_accept_remains else 2


if __name__ == "__main__":
    raise SystemExit(main())
