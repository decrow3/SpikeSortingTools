"""Corrected saved-state W2 refractory-dip gate using exact mask intervals."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from testing.ci_force_gate import gated_force_union
from testing.cj_force_gate_w2 import (
    event_offsets_by_unit_block,
    random_derangement,
    sha256,
    write_csv,
)


CENTRAL_MIN = 9
CENTRAL_MAX = 29
SHOULDER_MIN = 45
SHOULDER_MAX = 89
CENTRAL_LAG_VALUES = 42
SHOULDER_LAG_VALUES = 90
EXPECTED_MASK_SHA256 = (
    "86425e8a6627a10ac8932ba01b799cb7be7141f731ebb4ecc01e303326b9da55")


def lag_counts(a_offsets, b_offsets):
    """Count the exact annular central and shoulder cross-lags."""
    a = np.asarray(a_offsets, dtype=np.int64)
    b = np.sort(np.asarray(b_offsets, dtype=np.int64))
    central = shoulder = 0
    for value in a:
        central += np.searchsorted(b, value - CENTRAL_MIN, side="right")
        central -= np.searchsorted(b, value - CENTRAL_MAX, side="left")
        central += np.searchsorted(b, value + CENTRAL_MAX, side="right")
        central -= np.searchsorted(b, value + CENTRAL_MIN, side="left")
        shoulder += np.searchsorted(b, value - SHOULDER_MIN, side="right")
        shoulder -= np.searchsorted(b, value - SHOULDER_MAX, side="left")
        shoulder += np.searchsorted(b, value + SHOULDER_MAX, side="right")
        shoulder -= np.searchsorted(b, value + SHOULDER_MIN, side="left")
    return int(central), int(shoulder)


def session_time_to_local_frame(value_s, fs, source_start_frame):
    """First local frame whose session time is at least value_s."""
    return int(np.ceil(float(value_s) * fs)) - int(source_start_frame)


def load_exact_rest_blocks(mask_path, metadata_path):
    metadata = json.loads(metadata_path.read_text())
    mask_hash = sha256(mask_path)
    if mask_hash != EXPECTED_MASK_SHA256 or mask_hash != metadata["mask_sha256"]:
        raise ValueError("canonical mask hash mismatch")
    fs = float(metadata["sampling_frequency_hz"])
    source_start, source_end = map(
        int, metadata["exact_source_frame_interval_half_open"])
    session_start = float(metadata["session_time_offset_s"])
    session_end = float(metadata["exact_realized_session_end_s"])
    n_frames = int(metadata["local_frame_count"])
    if source_end - source_start != n_frames:
        raise ValueError("frame-count metadata mismatch")

    mask = []
    with open(mask_path, newline="") as stream:
        for row in csv.DictReader(stream):
            left = max(session_start, float(row["start_s"]))
            right = min(session_end, float(row["end_s"]))
            if left < right:
                mask.append((left, right, row["source"]))
    mask.sort()
    for previous, current in zip(mask, mask[1:], strict=False):
        if current[0] < previous[1]:
            raise ValueError("canonical mask intervals overlap")

    rest_segments = []
    cursor = session_start
    for left, right, _ in mask:
        if cursor < left:
            rest_segments.append((cursor, left))
        cursor = max(cursor, right)
    if cursor < session_end:
        rest_segments.append((cursor, session_end))

    blocks = []
    for segment_id, (left, right) in enumerate(rest_segments):
        count = int(np.floor((right - left) / 5.0 + 1e-12))
        for within_segment in range(count):
            start_s = left + 5.0 * within_segment
            end_s = start_s + 5.0
            start_sample = session_time_to_local_frame(start_s, fs, source_start)
            end_sample = session_time_to_local_frame(end_s, fs, source_start)
            if not (0 <= start_sample < end_sample <= n_frames):
                raise ValueError("block outside exact W2 frame bounds")
            blocks.append({
                "block_id": len(blocks),
                "segment_id": segment_id,
                "within_segment": within_segment,
                "start_sample": start_sample,
                "end_sample": end_sample,
                "start_s": start_s,
                "end_s": end_s,
            })
    return blocks, mask, metadata


def score_edge(unit_i, unit_j, blocks, events, *, n_bootstrap, n_null, seed):
    common = [b for b in blocks if (unit_i, b["block_id"]) in events
              and (unit_j, b["block_id"]) in events]
    by_segment = {}
    for block in common:
        by_segment.setdefault(block["segment_id"], []).append(block)
    by_segment = {key: value for key, value in by_segment.items() if len(value) >= 2}
    common = [block for key in sorted(by_segment) for block in by_segment[key]]
    result = {
        "unit_i": int(unit_i), "unit_j": int(unit_j), "status": "unresolved",
        "secondary_pass": False, "common_blocks": len(common),
        "events_i": int(sum(len(events[(unit_i, b["block_id"])]) for b in common)),
        "events_j": int(sum(len(events[(unit_j, b["block_id"])]) for b in common)),
        "central_count": 0, "shoulder_count": 0, "expected_central": 0.0,
        "dip_ratio": np.nan, "bootstrap_upper": np.nan, "null_q05": np.nan,
        "null_lower_tail_p": np.nan, "accepted_bootstrap": 0,
        "accepted_null": 0,
    }
    if len(common) < 20:
        result["reason"] = "insufficient_common_blocks"; return result
    if min(result["events_i"], result["events_j"]) < 100:
        result["reason"] = "insufficient_events"; return result

    diagonal_counts = []
    matrices = {}
    for segment, segment_blocks in by_segment.items():
        n_blocks = len(segment_blocks)
        central_matrix = np.zeros((n_blocks, n_blocks), dtype=np.int64)
        shoulder_matrix = np.zeros((n_blocks, n_blocks), dtype=np.int64)
        for target_index, target in enumerate(segment_blocks):
            a = events[(unit_i, target["block_id"])]
            for source_index, source in enumerate(segment_blocks):
                b = events[(unit_j, source["block_id"])]
                central_matrix[target_index, source_index], shoulder_matrix[
                    target_index, source_index] = lag_counts(a, b)
        matrices[segment] = (central_matrix, shoulder_matrix)
        diagonal_counts.extend(zip(
            np.diag(central_matrix), np.diag(shoulder_matrix), strict=True))
    block_central = np.asarray([value[0] for value in diagonal_counts], dtype=np.int64)
    block_shoulder = np.asarray([value[1] for value in diagonal_counts], dtype=np.int64)
    central = int(block_central.sum())
    shoulder = int(block_shoulder.sum())
    expected = shoulder * CENTRAL_LAG_VALUES / SHOULDER_LAG_VALUES
    result.update(central_count=central, shoulder_count=shoulder,
                  expected_central=float(expected))
    if expected < 20:
        result["reason"] = "insufficient_expected_central"; return result

    ratio = (central + 0.5) / (expected + 0.5)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(block_central),
                         size=(n_bootstrap, len(block_central)))
    bootstrap_central = block_central[draws].sum(axis=1)
    bootstrap_shoulder = block_shoulder[draws].sum(axis=1)
    bootstrap = (bootstrap_central + 0.5) / (
        bootstrap_shoulder * CENTRAL_LAG_VALUES / SHOULDER_LAG_VALUES + 0.5)
    if not np.isfinite(bootstrap).all():
        result["reason"] = "nonfinite_bootstrap"; return result

    null = np.empty(n_null, dtype=float)
    for draw in range(n_null):
        null_central = null_shoulder = 0
        for segment in sorted(matrices):
            central_matrix, shoulder_matrix = matrices[segment]
            permutation = random_derangement(len(central_matrix), rng)
            indices = np.arange(len(central_matrix))
            null_central += int(central_matrix[indices, permutation].sum())
            null_shoulder += int(shoulder_matrix[indices, permutation].sum())
        null[draw] = (null_central + 0.5) / (
            null_shoulder * CENTRAL_LAG_VALUES / SHOULDER_LAG_VALUES + 0.5)
    if not np.isfinite(null).all():
        result["reason"] = "nonfinite_null"; return result
    upper = float(np.quantile(bootstrap, 0.975))
    null_q05 = float(np.quantile(null, 0.05))
    passed = upper <= 0.50 and ratio <= null_q05
    result.update(
        status="secondary_pass" if passed else "secondary_fail",
        reason="", secondary_pass=bool(passed), dip_ratio=float(ratio),
        bootstrap_upper=upper, null_q05=null_q05,
        null_lower_tail_p=float(
            (1 + np.count_nonzero(null <= ratio)) / (n_null + 1)),
        accepted_bootstrap=n_bootstrap, accepted_null=n_null,
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ce-root", type=Path, required=True)
    parser.add_argument("--interval-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", default="3000")
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--null", type=int, default=1000)
    parser.add_argument("--method-commit", required=True)
    args = parser.parse_args()
    started = time.time()
    args.output.mkdir(parents=True, exist_ok=False)

    post_path = args.ce_root / "POST_TMM_STATE.npz"
    route_path = (args.ce_root / "supplemental_saved_array_audit_v1"
                  / f"{args.arm}_ROUTES_AND_CLOCKS.npz")
    mask_path = args.interval_root / "censor_mask_v1.csv"
    metadata_path = args.interval_root / "W2_INTERVAL_METADATA.json"
    with np.load(post_path, allow_pickle=False) as data:
        post = {key: data[key] for key in data.files}
    with np.load(route_path, allow_pickle=False) as data:
        route = {key: data[key] for key in data.files}
    if not np.array_equal(post["event_row_ids"], route["event_row_ids"]):
        raise ValueError("row namespace mismatch")
    if not np.array_equal(post["labels"], route["source_labels"]):
        raise ValueError("label namespace mismatch")
    if not np.array_equal(post["times_samples"], route["fixed_times_samples"]):
        raise ValueError("fixed clock mismatch")

    blocks, mask_intervals, metadata = load_exact_rest_blocks(mask_path, metadata_path)
    retained = np.zeros(len(post["times_samples"]), dtype=bool)
    for block in blocks:
        retained |= ((post["times_samples"] >= block["start_sample"])
                     & (post["times_samples"] < block["end_sample"]))
    if np.any(route["fixed_state_episode"][retained]):
        raise ValueError("retained block contains a canonically censored row")
    fs = float(metadata["sampling_frequency_hz"])
    events = event_offsets_by_unit_block(post["times_samples"], post["labels"], blocks)
    direct = route["direct_force_mask"].copy()
    np.fill_diagonal(direct, False)
    unit_i, unit_j = np.triu_indices_from(direct, 1)
    selected = np.flatnonzero(direct[unit_i, unit_j])
    edge_results = []
    secondary = np.zeros_like(direct)
    for ordinal, index in enumerate(selected):
        result = score_edge(
            int(unit_i[index]), int(unit_j[index]), blocks, events,
            n_bootstrap=args.bootstrap, n_null=args.null, seed=args.seed + ordinal)
        result["edge_ordinal"] = ordinal
        result["qda_requested"] = bool(
            route["qda_requested_mask"][unit_i[index], unit_j[index]])
        result["qda_accepted"] = bool(
            route["qda_accept_mask"][unit_i[index], unit_j[index]])
        edge_results.append(result)
        if result["secondary_pass"]:
            secondary[unit_i[index], unit_j[index]] = True
            secondary[unit_j[index], unit_i[index]] = True

    gated = gated_force_union(
        direct_force_mask=direct, secondary_pass_mask=secondary,
        qda_accept_mask=route["qda_accept_mask"])
    all_force = gated_force_union(
        direct_force_mask=direct, secondary_pass_mask=direct,
        qda_accept_mask=route["qda_accept_mask"])
    no_force = gated_force_union(
        direct_force_mask=direct, secondary_pass_mask=np.zeros_like(direct),
        qda_accept_mask=route["qda_accept_mask"])
    qda_only = route["qda_accept_mask"].copy()
    np.fill_diagonal(qda_only, True)
    summary = {
        "status": "complete_saved_fixed_clock_hard_partition_only_v2",
        "arm": args.arm,
        "source_rows": len(post["event_row_ids"]),
        "source_units": int(post["labels"].max() + 1),
        "direct_force_edges": len(selected),
        "exact_rest_segments_with_blocks": len({b["segment_id"] for b in blocks}),
        "complete_rest_blocks": len(blocks),
        "retained_rows_before_edge_trim": int(retained.sum()),
        "retained_censored_rows": int(route["fixed_state_episode"][retained].sum()),
        "secondary_pass_edges": int(sum(x["secondary_pass"] for x in edge_results)),
        "secondary_fail_edges": int(sum(x["status"] == "secondary_fail" for x in edge_results)),
        "unresolved_edges": int(sum(x["status"] == "unresolved" for x in edge_results)),
        "unresolved_reasons": {
            reason: sum(x.get("reason") == reason for x in edge_results)
            for reason in sorted({x.get("reason") for x in edge_results if x.get("reason")})},
        "gated_graph": {key: gated[key] for key in (
            "accepted_direct_edges", "gated_force_relations", "indirect_relations",
            "rejected_direct_edges_reconnected")},
        "all_force_reproduces_saved": bool(np.array_equal(
            all_force["gated_force_mask"], route["expanded_force_mask"])),
        "no_force_equals_qda_only": bool(np.array_equal(
            no_force["final_union_mask"], qda_only)),
        "gated_equals_no_force": bool(np.array_equal(
            gated["final_union_mask"], no_force["final_union_mask"])),
        "final_union_relations": int(np.triu(gated["final_union_mask"], 1).sum()),
        "accepted_qda_relations": int(np.triu(route["qda_accept_mask"], 1).sum()),
        "qda_requested_relations": int(np.triu(route["qda_requested_mask"], 1).sum()),
        "clock": "fixed pre-recluster sample clock",
        "sampling_frequency_hz": fs,
        "session_time_offset_s": metadata["session_time_offset_s"],
        "exact_local_frame_interval_half_open": metadata[
            "exact_local_frame_interval_half_open"],
        "canonical_mask_intervals_overlapping_w2": len(mask_intervals),
        "canonical_mask_sha256": metadata["mask_sha256"],
        "aggregation_dedup_outcome": "not run",
        "event_filter": "none beyond saved hard partition, exact canonical rest blocks, and boundary support",
        "central_lags_abs_samples": [CENTRAL_MIN, CENTRAL_MAX],
        "excluded_central_lags_abs_samples": [0, CENTRAL_MIN - 1],
        "shoulder_lags_abs_samples": [SHOULDER_MIN, SHOULDER_MAX],
        "lag_exposure_ratio": CENTRAL_LAG_VALUES / SHOULDER_LAG_VALUES,
        "bootstrap_draws": args.bootstrap,
        "null_draws": args.null,
        "null_scope_caveat": "same-segment nonwrapping shared-support control; nonstationary rates and inherited missingness remain",
        "elapsed_s": time.time() - started,
    }
    write_csv(args.output / "EDGE_RESULTS.csv", edge_results)
    write_csv(args.output / "REST_BLOCKS.csv", blocks)
    np.savez_compressed(
        args.output / "GATED_ARRAYS.npz",
        secondary_pass_mask=secondary,
        accepted_direct_mask=gated["accepted_direct_mask"],
        gated_force_mask=gated["gated_force_mask"],
        final_union_mask=gated["final_union_mask"],
        component_ids=gated["component_ids"],
    )
    (args.output / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    receipt = {
        "post_tmm_sha256": sha256(post_path),
        "routes_sha256": sha256(route_path),
        "canonical_mask_sha256": sha256(mask_path),
        "interval_metadata_sha256": sha256(metadata_path),
        "helper_sha256": sha256(Path(__file__).with_name("ci_force_gate.py")),
        "script_sha256": sha256(Path(__file__)),
        "seed": args.seed,
        "method_commit_required_before_outcomes": args.method_commit,
        "scientific_scope": "saved fixed-clock hard-partition graph diagnostic only",
    }
    (args.output / "RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n")
    products = []
    for name in ("EDGE_RESULTS.csv", "REST_BLOCKS.csv", "GATED_ARRAYS.npz",
                 "SUMMARY.json", "RECEIPT.json"):
        path = args.output / name
        products.append({
            "path": name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    (args.output / "COMPLETE.json").write_text(json.dumps({
        "status": "complete", "products": products, "written_last": True,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
