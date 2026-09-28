"""Post-primary unresolved-default graphs and stationary support forecasts."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from testing.ci_force_gate import gated_force_union
from testing.cj_force_gate_w2 import event_offsets_by_unit_block
from testing.cj_force_gate_w2_v2 import lag_counts


FS = 29999.759166666667
CENTRAL_LAGS = tuple(range(-29, -8)) + tuple(range(9, 30))
RATE_SCENARIOS = (0.5, 1.0, 2.0)
FULL_SESSION_S = 10473.55
CANONICAL_MASK_S = 2263.5
FULL_REST_S = FULL_SESSION_S - CANONICAL_MASK_S


def read_csv(path):
    return list(csv.DictReader(open(path)))


def exact_lag_exposure(block_length, lags):
    lower, upper = 89, block_length - 89
    return sum(max(0, min(upper, upper - lag) - max(lower, lower - lag))
               for lag in lags)


def graph_summary(direct, qda, passed, unresolved):
    primary = gated_force_union(
        direct_force_mask=direct, secondary_pass_mask=passed,
        qda_accept_mask=qda)
    sensitivity = gated_force_union(
        direct_force_mask=direct, secondary_pass_mask=(passed | unresolved),
        qda_accept_mask=qda)
    all_force = gated_force_union(
        direct_force_mask=direct, secondary_pass_mask=direct,
        qda_accept_mask=qda)
    no_force = gated_force_union(
        direct_force_mask=direct, secondary_pass_mask=np.zeros_like(direct),
        qda_accept_mask=qda)
    upper = np.triu_indices_from(direct, 1)
    result = {}
    for name, value in (("primary_withhold", primary),
                        ("sensitivity_accept_unresolved", sensitivity),
                        ("all_force", all_force), ("no_force", no_force)):
        result[name] = {
            key: value[key] for key in (
                "accepted_direct_edges", "gated_force_relations",
                "indirect_relations", "rejected_direct_edges_reconnected")}
        result[name]["final_union_relations"] = int(
            value["final_union_mask"][upper].sum())
        result[name]["force_qda_overlap_relations"] = int(
            (value["gated_force_mask"] & qda)[upper].sum())
    result["sensitivity_equals_all_force"] = bool(np.array_equal(
        sensitivity["final_union_mask"], all_force["final_union_mask"]))
    result["sensitivity_equals_no_force"] = bool(np.array_equal(
        sensitivity["final_union_mask"], no_force["final_union_mask"]))
    result["sensitivity_equals_primary"] = bool(np.array_equal(
        sensitivity["final_union_mask"], primary["final_union_mask"]))
    return result


def analyze_window(name, post_path, route_path, blocks_path, edge_path):
    with np.load(post_path, allow_pickle=False) as data:
        post = {key: data[key] for key in data.files}
    with np.load(route_path, allow_pickle=False) as data:
        route = {key: data[key] for key in data.files}
    blocks = []
    for row in read_csv(blocks_path):
        blocks.append({key: int(row[key]) for key in (
            "block_id", "segment_id", "within_segment", "start_sample", "end_sample")})
    saved_edges = read_csv(edge_path)
    if not np.array_equal(post["event_row_ids"], route["event_row_ids"]):
        raise ValueError(f"{name}: row mismatch")
    if not np.array_equal(post["labels"], route["source_labels"]):
        raise ValueError(f"{name}: label mismatch")
    if not np.array_equal(post["times_samples"], route["fixed_times_samples"]):
        raise ValueError(f"{name}: clock mismatch")
    direct = route["direct_force_mask"].copy()
    np.fill_diagonal(direct, False)
    passed = np.zeros_like(direct)
    unresolved = np.zeros_like(direct)
    status_by_pair = {}
    for row in saved_edges:
        i, j = int(row["unit_i"]), int(row["unit_j"])
        status_by_pair[(i, j)] = row["status"]
        target = passed if row["status"] == "secondary_pass" else unresolved
        if row["status"] in ("secondary_pass", "unresolved"):
            target[i, j] = target[j, i] = True
    graph = graph_summary(direct, route["qda_accept_mask"], passed, unresolved)
    events = event_offsets_by_unit_block(post["times_samples"], post["labels"], blocks)
    block_lengths = {b["block_id"]: b["end_sample"] - b["start_sample"] for b in blocks}
    total_support_s = sum(length - 178 for length in block_lengths.values()) / FS
    exact_exposure = sum(exact_lag_exposure(length, CENTRAL_LAGS)
                         for length in block_lengths.values()) / FS ** 2
    unit_counts = {}
    for (unit, _), times in events.items():
        unit_counts[unit] = unit_counts.get(unit, 0) + len(times)

    ii, jj = np.triu_indices_from(direct, 1)
    selected = np.flatnonzero(direct[ii, jj])
    rows = []
    for ordinal, index in enumerate(selected):
        unit_i, unit_j = int(ii[index]), int(jj[index])
        common = [b for b in blocks if (unit_i, b["block_id"]) in events
                  and (unit_j, b["block_id"]) in events]
        by_segment = {}
        for block in common:
            by_segment.setdefault(block["segment_id"], []).append(block)
        by_segment = {key: value for key, value in by_segment.items() if len(value) >= 2}
        scoring_blocks = [b for key in sorted(by_segment) for b in by_segment[key]]
        events_i_scoring = sum(len(events[(unit_i, b["block_id"])]) for b in scoring_blocks)
        events_j_scoring = sum(len(events[(unit_j, b["block_id"])]) for b in scoring_blocks)
        central = shoulder = 0
        for block in scoring_blocks:
            c, s = lag_counts(events[(unit_i, block["block_id"])],
                              events[(unit_j, block["block_id"])])
            central += c; shoulder += s
        shoulder_expected = shoulder * 42 / 90
        count_i, count_j = unit_counts.get(unit_i, 0), unit_counts.get(unit_j, 0)
        rate_i, rate_j = count_i / total_support_s, count_j / total_support_s
        independent_expected = rate_i * rate_j * exact_exposure
        row = {
            "window": name, "edge_ordinal": ordinal,
            "unit_i": unit_i, "unit_j": unit_j,
            "status": status_by_pair[(unit_i, unit_j)],
            "all_rest_events_i": count_i, "all_rest_events_j": count_j,
            "all_rest_rate_i_hz": rate_i, "all_rest_rate_j_hz": rate_j,
            "common_positive_blocks": len(scoring_blocks),
            "scoring_events_i": events_i_scoring,
            "scoring_events_j": events_j_scoring,
            "central_count": central, "shoulder_count": shoulder,
            "shoulder_expected_central": shoulder_expected,
            "independent_rate_expected_central": independent_expected,
            "saved_reason": next((x["reason"] for x in saved_edges
                                  if int(x["unit_i"]) == unit_i and int(x["unit_j"]) == unit_j), ""),
        }
        mean_block_s = total_support_s / len(blocks)
        for factor in RATE_SCENARIOS:
            tag = str(factor).replace(".", "p")
            event_multiplier = max(
                100 / (count_i * factor) if count_i else np.inf,
                100 / (count_j * factor) if count_j else np.inf)
            probability_common = ((1 - np.exp(-rate_i * factor * mean_block_s))
                                  * (1 - np.exp(-rate_j * factor * mean_block_s)))
            block_multiplier = (20 / (len(blocks) * probability_common)
                                if probability_common > 0 else np.inf)
            independent_multiplier = (20 / (independent_expected * factor ** 2)
                                      if independent_expected > 0 else np.inf)
            shoulder_multiplier = (20 / (shoulder_expected * factor ** 2)
                                   if shoulder_expected > 0 else np.inf)
            row[f"support_multiplier_independent_{tag}"] = max(
                1.0, event_multiplier, block_multiplier, independent_multiplier)
            row[f"support_multiplier_shoulder_{tag}"] = max(
                1.0, event_multiplier, block_multiplier, shoulder_multiplier)
        rows.append(row)
    rest_multiplier = FULL_REST_S / total_support_s
    summary = {
        "window": name, "direct_edges": len(rows),
        "status_counts": {status: sum(row["status"] == status for row in rows)
                          for status in ("secondary_pass", "secondary_fail", "unresolved")},
        "unresolved_reasons": {reason: sum(row["status"] == "unresolved" and row["saved_reason"] == reason for row in rows)
                               for reason in sorted({row["saved_reason"] for row in rows if row["status"] == "unresolved"})},
        "complete_rest_blocks": len(blocks),
        "exact_retained_rest_support_s": total_support_s,
        "full_session_canonical_rest_s": FULL_REST_S,
        "full_to_window_rest_multiplier": rest_multiplier,
        "graph": graph,
    }
    for factor in RATE_SCENARIOS:
        tag = str(factor).replace(".", "p")
        for basis in ("independent", "shoulder"):
            key = f"support_multiplier_{basis}_{tag}"
            summary[f"edges_potentially_meeting_support_{basis}_{tag}"] = sum(
                np.isfinite(row[key]) and row[key] <= rest_multiplier for row in rows)
    return rows, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--w2-root", type=Path, required=True)
    parser.add_argument("--w2-ce-root", type=Path, required=True)
    parser.add_argument("--w3-root", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    w2_rows, w2_summary = analyze_window(
        "W2", args.w2_ce_root / "POST_TMM_STATE.npz",
        args.w2_ce_root / "supplemental_saved_array_audit_v1" / "3000_ROUTES_AND_CLOCKS.npz",
        args.w2_root / "v2" / "REST_BLOCKS.csv", args.w2_root / "v2" / "EDGE_RESULTS.csv")
    w3_rows, w3_summary = analyze_window(
        "W3", args.w3_root / "POST_TMM_STATE.npz",
        args.w3_root / "gate" / "FORCE_ROUTE_ARRAYS.npz",
        args.w3_root / "gate" / "REST_BLOCKS.csv", args.w3_root / "gate" / "EDGE_RESULTS.csv")
    rows = w2_rows + w3_rows
    with open(args.output / "EDGE_EXPOSURE.csv", "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (args.output / "SUMMARY.json").write_text(json.dumps({
        "status": "complete_post_primary_sensitivity_and_stationary_support_forecast",
        "primary_semantics": "pass only; unresolved withheld",
        "sensitivity_semantics": "pass or unresolved; resolved failures withheld",
        "rate_scenarios": list(RATE_SCENARIOS),
        "forecast_claim": "potential support minima only; not pass probability or identity",
        "full_session_s": FULL_SESSION_S,
        "canonical_mask_s": CANONICAL_MASK_S,
        "canonical_rest_s": FULL_REST_S,
        "windows": {"W2": w2_summary, "W3": w3_summary},
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
