#!/usr/bin/env python3
"""Independent fixture review for the frozen D3 v3 post-run oracle v3."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path


PACKET = Path(
    "/mnt/NPX/Luke/DARTsort_motion_experiments/"
    "dartsort_divergence_D3_v3_postrun_oracle_h1_20261007_v3"
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
fixtures = load_module("oracle_v3_fixtures", PACKET / "test_postrun_oracle.py")


def classify(root: Path) -> str:
    try:
        return oracle.validate_output(root)["verdict"]
    except (oracle.OracleError, IndexError, KeyError, TypeError, ValueError) as exc:
        return f"REJECTED: {type(exc).__name__}: {exc}"


def main() -> int:
    results = {}
    with tempfile.TemporaryDirectory(prefix="d3_oracle_v3_review_") as temp:
        base = Path(temp)
        results["positive"] = classify(fixtures.build_success(base / "positive"))

        duplicate = fixtures.build_success(base / "duplicate")
        rankings = json.loads((duplicate / "DARTSORT_CANDIDATE_RANKING.json").read_text())
        for item in rankings:
            item["rows"].append(dict(item["rows"][0]))
            item["passing_candidate_labels"].append(7)
        ordering = json.loads((duplicate / "DARTSORT_MATCHER_ORDERING.json").read_text())
        ordering["eligible_final_units"] = 2
        fixtures.write_json(duplicate / "DARTSORT_MATCHER_ORDERING.json", ordering)
        fixtures.write_json(duplicate / "DARTSORT_CANDIDATE_RANKING.json", rankings)
        fixtures.seal(duplicate)
        results["v2_duplicate_label"] = classify(duplicate)

        negative = fixtures.build_success(base / "negative")
        rows = oracle.load_jsonl(negative / "PAIR_EVENT_RELATION.jsonl")
        context = oracle.context_key(rows[0])
        target = next(row for row in rows if oracle.context_key(row) == context and row["side"] == "a" and row["pair_local_row_index"] == 24)
        target["matched_other_pair_local_row_index"] = -1
        fixtures.write_jsonl(negative / "PAIR_EVENT_RELATION.jsonl", rows)
        fixtures.seal(negative)
        results["v2_negative_match_index"] = classify(negative)

        substituted = fixtures.build_success(base / "substituted_universe")
        rankings = json.loads((substituted / "DARTSORT_CANDIDATE_RANKING.json").read_text())
        for item in rankings:
            item["rows"][0]["dartsort_final_label"] = 999
            item["passing_candidate_labels"] = [999]
        fixtures.write_json(substituted / "DARTSORT_CANDIDATE_RANKING.json", rankings)
        summaries = json.loads((substituted / "PAIR_SUMMARY.json").read_text())
        for row in summaries:
            if row["pair"] != "REF-A":
                row["dartsort_final_label"] = 999
        fixtures.write_json(substituted / "PAIR_SUMMARY.json", summaries)
        pair_rows = oracle.load_jsonl(substituted / "PAIR_EVENT_RELATION.jsonl")
        for row in pair_rows:
            if row["pair"] != "REF-A":
                row["dartsort_final_label"] = 999
        fixtures.write_jsonl(substituted / "PAIR_EVENT_RELATION.jsonl", pair_rows)
        stage_rows = oracle.load_jsonl(substituted / "DARTSORT_STAGE_EVENTS.jsonl")
        for row in stage_rows:
            row["final_label"] = 999
        fixtures.write_jsonl(substituted / "DARTSORT_STAGE_EVENTS.jsonl", stage_rows)
        fixtures.seal(substituted)
        results["canonical_label_substitution_false_accept"] = classify(substituted)

    print(json.dumps(results, indent=2, sort_keys=True))
    literal_repairs_hold = str(results["v2_duplicate_label"]).startswith("REJECTED:") and str(results["v2_negative_match_index"]).startswith("REJECTED:")
    canonical_false_accept = results["canonical_label_substitution_false_accept"] == "GO_FINAL_REF_A_TO_DARTSORT_CORRESPONDENCE_ONLY"
    return 1 if literal_repairs_hold and canonical_false_accept else 2


if __name__ == "__main__":
    raise SystemExit(main())
