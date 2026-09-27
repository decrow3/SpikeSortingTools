#!/usr/bin/env python3
"""Tiny independent BI correspondence fixtures; not a production matcher."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


UNMATCHED = -1
UNAVAILABLE = -2


def _events(rows):
    out = []
    for row in rows:
        row = dict(row)
        row.setdefault("available", True)
        out.append(row)
    ids = [int(row["row_id"]) for row in out]
    if len(ids) != len(set(ids)):
        raise ValueError("row_id must be immutable and unique within each side")
    return out


def exclusive_reference(source, target, *, policy, use_z=True, excluded_intervals=()):
    """Independent tiny-array reference for two declared greedy policies.

    This intentionally is not maximum-cardinality matching and is not offered
    as a replacement for any frozen production policy.
    """
    source, target = _events(source), _events(target)

    def event_available(row):
        if not row["available"]:
            return False
        if any(lo <= int(row["frame"]) < hi for lo, hi in excluded_intervals):
            return False
        return not use_z or np.isfinite(float(row["z_um"]))

    sa = {int(x["row_id"]): bool(event_available(x)) for x in source}
    ta = {int(x["row_id"]): bool(event_available(x)) for x in target}
    source_by_id = {int(x["row_id"]): x for x in source}
    target_by_id = {int(x["row_id"]): x for x in target}

    edges = []
    for sid, s in source_by_id.items():
        if not sa[sid]:
            continue
        for tid, t in target_by_id.items():
            if not ta[tid]:
                continue
            dt = abs(int(s["frame"]) - int(t["frame"]))
            dz = abs(float(s["z_um"]) - float(t["z_um"])) if use_z else 0.0
            if dt > 12 or (use_z and dz > 30.0):
                continue
            a, b = sorted((int(s["frame"]), int(t["frame"])))
            if any(a < hi and b >= lo for lo, hi in excluded_intervals):
                continue
            edges.append((dt, dz, sid, tid))

    if policy == "edge_order":
        ordered = sorted(edges)
    elif policy == "source_order_nearest":
        ordered = []
        for sid in sorted(source_by_id):
            ordered.extend(sorted(edge for edge in edges if edge[2] == sid))
    else:
        raise ValueError(policy)

    source_match = {sid: UNAVAILABLE if not sa[sid] else UNMATCHED for sid in source_by_id}
    target_match = {tid: UNAVAILABLE if not ta[tid] else UNMATCHED for tid in target_by_id}
    used_s, used_t = set(), set()
    for _, _, sid, tid in ordered:
        if sid in used_s or tid in used_t:
            continue
        source_match[sid], target_match[tid] = tid, sid
        used_s.add(sid)
        used_t.add(tid)

    nmatch = len(used_s)
    savail = sum(sa.values())
    tavail = sum(ta.values())
    accounting = {
        "matches": nmatch,
        "source_available": savail,
        "source_unmatched": savail - nmatch,
        "source_unavailable": len(source) - savail,
        "target_available": tavail,
        "target_unmatched": tavail - nmatch,
        "target_unavailable": len(target) - tavail,
        "exclusive": len(set(used_s)) == nmatch == len(set(used_t)),
    }
    assert accounting["matches"] + accounting["source_unmatched"] == accounting["source_available"]
    assert accounting["matches"] + accounting["target_unmatched"] == accounting["target_available"]
    return {
        "source_to_target": {str(k): int(v) for k, v in sorted(source_match.items())},
        "target_to_source": {str(k): int(v) for k, v in sorted(target_match.items())},
        "accounting": accounting,
    }


def shifted_common_support(source, target, *, support, shift):
    """Trim both sides to the same observable support after a target shift."""
    lo, hi = support
    source_ids = [int(x["row_id"]) for x in source if lo <= int(x["frame"]) + shift <= hi]
    target_ids = [int(x["row_id"]) for x in target if lo + shift <= int(x["frame"]) <= hi]
    return {"source_row_ids": source_ids, "target_row_ids": target_ids}


def fixture_contract():
    return {
        "schema": "bi-h1-event-correspondence-fixtures-v1",
        "codes": {"unmatched": UNMATCHED, "unavailable": UNAVAILABLE},
        "boundaries": {"time_samples_inclusive": 12, "observed_z_um_inclusive": 30.0},
        "policy_note": (
            "Two deterministic exclusive greedy policies are reported. Policy-sensitive assignment "
            "differences are valid differences, not automatically bugs. Neither policy is maximum-cardinality."
        ),
        "fixtures": [
            {
                "id": "duplicate_timestamps",
                "source": [{"row_id": 101, "frame": 10, "z_um": 0}, {"row_id": 100, "frame": 10, "z_um": 0}],
                "target": [{"row_id": 201, "frame": 10, "z_um": 0}, {"row_id": 200, "frame": 10, "z_um": 0}],
                "expected": {"source_to_target": {"100": 200, "101": 201}, "matches": 2},
            },
            {
                "id": "inclusive_boundaries",
                "source": [{"row_id": 110, "frame": 100, "z_um": 0}, {"row_id": 111, "frame": 200, "z_um": 0}],
                "target": [{"row_id": 210, "frame": 112, "z_um": 30}, {"row_id": 211, "frame": 212, "z_um": -30}],
                "expected": {"source_to_target": {"110": 210, "111": 211}, "matches": 2},
            },
            {
                "id": "competing_candidates_policy_sensitive",
                "source": [{"row_id": 120, "frame": 0, "z_um": 0}, {"row_id": 121, "frame": 1, "z_um": 0}],
                "target": [{"row_id": 220, "frame": 1, "z_um": 0}, {"row_id": 221, "frame": 12, "z_um": 0}],
                "expected_by_policy": {
                    "source_order_nearest": {"120": 220, "121": 221},
                    "edge_order": {"120": 221, "121": 220},
                },
                "expected_matches": 2,
            },
            {
                "id": "finite_nonfinite_coordinates",
                "source": [
                    {"row_id": 130, "frame": 0, "z_um": float("nan")},
                    {"row_id": 131, "frame": 100, "z_um": 0},
                    {"row_id": 132, "frame": 200, "z_um": 0},
                ],
                "target": [
                    {"row_id": 230, "frame": 0, "z_um": 0},
                    {"row_id": 231, "frame": 100, "z_um": float("nan")},
                    {"row_id": 232, "frame": 250, "z_um": 0},
                ],
                "expected_with_z": {
                    "source_to_target": {"130": -2, "131": -1, "132": -1},
                    "target_to_source": {"230": -1, "231": -2, "232": -1},
                    "matches": 0,
                },
                "expected_without_z": {"source_to_target": {"130": 230, "131": 231, "132": -1}, "matches": 2},
            },
            {
                "id": "excluded_interval_breaks_same_state_endpoints",
                "excluded_intervals": [[100, 110]],
                "source": [
                    {"row_id": 140, "frame": 98, "z_um": 0},
                    {"row_id": 141, "frame": 120, "z_um": 100},
                    {"row_id": 142, "frame": 105, "z_um": 0},
                ],
                "target": [
                    {"row_id": 240, "frame": 112, "z_um": 0},
                    {"row_id": 241, "frame": 132, "z_um": 100},
                    {"row_id": 242, "frame": 105, "z_um": 0},
                ],
                "expected": {
                    "source_to_target": {"140": -1, "141": 241, "142": -2},
                    "target_to_source": {"240": -1, "241": 141, "242": -2},
                    "matches": 1,
                },
            },
        ],
        "shifted_common_support": {
            "support": [0, 100], "shift": 12,
            "source": [{"row_id": 150, "frame": 0}, {"row_id": 151, "frame": 88}, {"row_id": 152, "frame": 89}, {"row_id": 153, "frame": 100}],
            "target": [{"row_id": 250, "frame": 12}, {"row_id": 251, "frame": 100}, {"row_id": 252, "frame": 11}, {"row_id": 253, "frame": 101}],
            "expected": {"source_row_ids": [150, 151], "target_row_ids": [250, 251]},
        },
    }


def run_contract(contract):
    results = {"schema": contract["schema"], "fixtures": [], "all_pass": True}
    for fixture in contract["fixtures"]:
        kwargs = {"use_z": True, "excluded_intervals": fixture.get("excluded_intervals", ())}
        by_policy = {}
        for policy in ("source_order_nearest", "edge_order"):
            got = exclusive_reference(fixture["source"], fixture["target"], policy=policy, **kwargs)
            by_policy[policy] = got
            if "expected" in fixture:
                exp = fixture["expected"]
                assert got["source_to_target"] == exp["source_to_target"]
                if "target_to_source" in exp:
                    assert got["target_to_source"] == exp["target_to_source"]
                assert got["accounting"]["matches"] == exp["matches"]
            if "expected_by_policy" in fixture:
                assert got["source_to_target"] == fixture["expected_by_policy"][policy]
                assert got["accounting"]["matches"] == fixture["expected_matches"]
        # Input reordering must recover the identical immutable-row mapping.
        reordered = exclusive_reference(list(reversed(fixture["source"])), list(reversed(fixture["target"])),
                                        policy="source_order_nearest", **kwargs)
        assert reordered == by_policy["source_order_nearest"]
        if fixture["id"] == "finite_nonfinite_coordinates":
            no_z = exclusive_reference(fixture["source"], fixture["target"], policy="source_order_nearest", use_z=False)
            assert no_z["source_to_target"] == fixture["expected_without_z"]["source_to_target"]
            assert no_z["accounting"]["matches"] == fixture["expected_without_z"]["matches"]
            by_policy["source_order_nearest_without_z"] = no_z
        results["fixtures"].append({"id": fixture["id"], "pass": True, "results": by_policy})
    sc = contract["shifted_common_support"]
    got = shifted_common_support(sc["source"], sc["target"], support=sc["support"], shift=sc["shift"])
    assert got == sc["expected"]
    results["shifted_common_support"] = {"pass": True, "result": got}
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    contract = fixture_contract()
    results = run_contract(contract)
    (args.output / "fixture_contract.json").write_text(json.dumps(contract, indent=2, allow_nan=True) + "\n")
    (args.output / "fixture_results.json").write_text(json.dumps(results, indent=2) + "\n")
    readme = """# BI h1 event-correspondence QA fixtures

Small independent fixtures for immutable row IDs, inclusive +/-12-sample and
30-um boundaries, exclusive accounting, unmatched (-1) versus unavailable
(-2), input reordering, excluded intervals, and shifted-control common support.

Two valid deterministic greedy policies are recorded for the deliberately
policy-sensitive competing-candidate case. This packet does not prescribe or
replace h5's frozen production policy and does not use maximum-cardinality
matching. Compare an implementation only after identifying its declared edge
ordering and tie breaks.
"""
    (args.output / "README.md").write_text(readme)
    files = [args.output / "fixture_contract.json", args.output / "fixture_results.json", args.output / "README.md"]
    with (args.output / "SHA256SUMS").open("w") as f:
        for path in files:
            f.write(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n")
    print(json.dumps({"all_pass": results["all_pass"], "fixtures": len(results["fixtures"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
