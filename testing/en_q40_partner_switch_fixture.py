#!/usr/bin/env python3
"""Known-answer controls for +40 endpoint partner classification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from testing.en_q40_partner_switch import classify_pair


def edge(ref: int, cand: int, primary: bool, matched: int, ref_events: int,
         cand_events: int) -> dict[str, object]:
    union = ref_events + cand_events - matched
    return {
        "reference_cluster": ref, "candidate_cluster": cand, "matched_events": matched,
        "reference_events": ref_events, "candidate_events": cand_events,
        "reference_retention": matched / ref_events, "candidate_retention": matched / cand_events,
        "f1": 2 * matched / (ref_events + cand_events), "jaccard": matched / union,
        "primary_match": primary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    edges = pd.DataFrame([
        edge(1, 10, True, 90, 100, 100),
        edge(2, 20, True, 80, 100, 90),
        edge(30, 11, True, 70, 80, 90),
        edge(2, 11, False, 40, 100, 90),
    ])
    counts_ref, counts_cand = {1: 100, 2: 100, 30: 80}, {10: 100, 11: 90, 20: 90}
    retained = classify_pair(pd.Series({"reference_cluster": 1, "candidate_cluster": 10}),
                             edges, counts_ref, counts_cand)
    if not retained["old_pair_is_state_primary"] or retained["reference_endpoint_switched"]:
        raise RuntimeError("retained old pair classification failed")
    switched = classify_pair(pd.Series({"reference_cluster": 2, "candidate_cluster": 11}),
                             edges, counts_ref, counts_cand)
    if (not switched["both_endpoints_switched"] or switched["reference_state_primary_candidate"] != 20
            or switched["candidate_state_primary_reference"] != 30
            or not switched["old_pair_has_qualifying_edge"]
            or switched["reference_best_other_unit"] != 20):
        raise RuntimeError("two-ended switch or secondary-edge classification failed")
    absent = classify_pair(pd.Series({"reference_cluster": 3, "candidate_cluster": 12}),
                           edges, counts_ref, counts_cand)
    if (absent["reference_state_events"] != 0 or absent["candidate_state_events"] != 0
            or absent["old_pair_has_qualifying_edge"] or absent["reference_best_other_unit"] is not None):
        raise RuntimeError("zero-event/no-edge classification failed")
    result = {
        "schema": "en-q40-partner-switch-fixture-v1", "status": "pass",
        "retained_old_primary": True, "both_endpoint_switches": True,
        "secondary_old_edge_preserved": True, "best_qualifying_edge_reported": True,
        "zero_event_no_edge_preserved": True,
    }
    if args.output:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
