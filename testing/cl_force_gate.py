"""Portable frozen CJ-v2 force gate for same-run saved state arrays."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from testing.ci_force_gate import gated_force_union
from testing.cj_force_gate_w2 import event_offsets_by_unit_block, write_csv
from testing.cj_force_gate_w2_v2 import score_edge


METHOD_VERSION = "cj-v2-annulus9-29-exact-mask-v1"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rest_blocks(mask_path, metadata_path, expected_mask_sha256):
    """Build complete 5-s blocks from an exact session-time mask complement."""
    metadata = json.loads(Path(metadata_path).read_text())
    actual_hash = sha256(mask_path)
    if actual_hash != expected_mask_sha256:
        raise ValueError("mask does not match the caller-frozen hash")
    if metadata.get("mask_sha256") != actual_hash:
        raise ValueError("mask does not match interval metadata")
    fs = float(metadata["sampling_frequency_hz"])
    source_start, source_end = map(
        int, metadata["exact_source_frame_interval_half_open"])
    local_start, local_end = map(
        int, metadata["exact_local_frame_interval_half_open"])
    if local_start != 0 or source_end - source_start != local_end:
        raise ValueError("inconsistent exact frame intervals")
    session_start = float(metadata["session_time_offset_s"])
    session_end = float(metadata["exact_realized_session_end_s"])

    masked = []
    with open(mask_path, newline="") as stream:
        for row in csv.DictReader(stream):
            left = max(session_start, float(row["start_s"]))
            right = min(session_end, float(row["end_s"]))
            if left < right:
                masked.append((left, right, row.get("source", "")))
    masked.sort()
    for previous, current in zip(masked, masked[1:], strict=False):
        if current[0] < previous[1]:
            raise ValueError("mask intervals overlap")
    rest = []
    cursor = session_start
    for left, right, _ in masked:
        if cursor < left:
            rest.append((cursor, left))
        cursor = max(cursor, right)
    if cursor < session_end:
        rest.append((cursor, session_end))

    blocks = []
    for segment_id, (left, right) in enumerate(rest):
        for within_segment in range(int(np.floor((right - left) / 5 + 1e-12))):
            start_s = left + 5 * within_segment
            end_s = start_s + 5
            start_sample = int(np.ceil(start_s * fs)) - source_start
            end_sample = int(np.ceil(end_s * fs)) - source_start
            if not (0 <= start_sample < end_sample <= local_end):
                raise ValueError("block outside exact local interval")
            blocks.append({
                "block_id": len(blocks), "segment_id": segment_id,
                "within_segment": within_segment,
                "start_sample": start_sample, "end_sample": end_sample,
                "start_s": start_s, "end_s": end_s,
            })
    return blocks, masked, metadata


def validate_saved_state(post, routes, blocks):
    for post_key, route_key in (
        ("event_row_ids", "event_row_ids"),
        ("labels", "source_labels"),
        ("times_samples", "fixed_times_samples"),
    ):
        if not np.array_equal(post[post_key], routes[route_key]):
            raise ValueError(f"saved state mismatch: {post_key}/{route_key}")
    retained = np.zeros(len(post["times_samples"]), dtype=bool)
    for block in blocks:
        retained |= ((post["times_samples"] >= block["start_sample"])
                     & (post["times_samples"] < block["end_sample"]))
    if np.any(routes["fixed_state_episode"][retained]):
        raise ValueError("retained exact-rest row is censored")
    return retained


def run_gate(post, routes, blocks, *, si_accept_mask=None, seed=20260927,
             n_bootstrap=1000, n_null=1000):
    """Apply the frozen statistic, direct-edge gate, connectivity, and unions."""
    retained = validate_saved_state(post, routes, blocks)
    events = event_offsets_by_unit_block(post["times_samples"], post["labels"], blocks)
    direct = np.asarray(routes["direct_force_mask"], bool).copy()
    np.fill_diagonal(direct, False)
    unit_i, unit_j = np.triu_indices_from(direct, 1)
    selected = np.flatnonzero(direct[unit_i, unit_j])
    secondary = np.zeros_like(direct)
    edge_results = []
    for ordinal, index in enumerate(selected):
        result = score_edge(
            int(unit_i[index]), int(unit_j[index]), blocks, events,
            n_bootstrap=n_bootstrap, n_null=n_null, seed=seed + ordinal)
        result["edge_ordinal"] = ordinal
        result["qda_requested"] = bool(
            routes["qda_requested_mask"][unit_i[index], unit_j[index]])
        result["qda_accepted"] = bool(
            routes["qda_accept_mask"][unit_i[index], unit_j[index]])
        edge_results.append(result)
        if result["secondary_pass"]:
            secondary[unit_i[index], unit_j[index]] = True
            secondary[unit_j[index], unit_i[index]] = True
    gated = gated_force_union(
        direct_force_mask=direct, secondary_pass_mask=secondary,
        qda_accept_mask=routes["qda_accept_mask"], si_accept_mask=si_accept_mask)
    all_force = gated_force_union(
        direct_force_mask=direct, secondary_pass_mask=direct,
        qda_accept_mask=routes["qda_accept_mask"], si_accept_mask=si_accept_mask)
    no_force = gated_force_union(
        direct_force_mask=direct, secondary_pass_mask=np.zeros_like(direct),
        qda_accept_mask=routes["qda_accept_mask"], si_accept_mask=si_accept_mask)
    return {
        "edges": edge_results, "secondary_pass_mask": secondary,
        "gated": gated, "all_force": all_force, "no_force": no_force,
        "retained_rows": int(retained.sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--post-state", type=Path, required=True)
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--interval-metadata", type=Path, required=True)
    parser.add_argument("--expected-mask-sha256", required=True)
    parser.add_argument("--si-status", choices=("absent", "array"), required=True)
    parser.add_argument("--si-array", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260927)
    args = parser.parse_args()
    if (args.si_status == "array") != (args.si_array is not None):
        raise ValueError("SI array is required iff --si-status=array")
    args.output.mkdir(parents=True, exist_ok=False)
    with np.load(args.post_state, allow_pickle=False) as data:
        post = {key: data[key] for key in data.files}
    with np.load(args.routes, allow_pickle=False) as data:
        routes = {key: data[key] for key in data.files}
    si = None
    if args.si_array is not None:
        with np.load(args.si_array, allow_pickle=False) as data:
            si = data["si_accept_mask"]
    blocks, masked, metadata = load_rest_blocks(
        args.mask, args.interval_metadata, args.expected_mask_sha256)
    result = run_gate(post, routes, blocks, si_accept_mask=si, seed=args.seed)
    write_csv(args.output / "EDGE_RESULTS.csv", result["edges"])
    write_csv(args.output / "REST_BLOCKS.csv", blocks)
    np.savez_compressed(
        args.output / "GATED_ARRAYS.npz",
        secondary_pass_mask=result["secondary_pass_mask"],
        accepted_direct_mask=result["gated"]["accepted_direct_mask"],
        gated_force_mask=result["gated"]["gated_force_mask"],
        final_union_mask=result["gated"]["final_union_mask"],
        component_ids=result["gated"]["component_ids"],
    )
    summary = {
        "method_version": METHOD_VERSION,
        "seed_rule": "base seed plus strict-upper direct-edge ordinal",
        "seed": args.seed,
        "bootstrap_draws": 1000, "null_draws": 1000,
        "si_status": args.si_status,
        "direct_force_edges": len(result["edges"]),
        "secondary_pass_edges": sum(x["secondary_pass"] for x in result["edges"]),
        "secondary_fail_edges": sum(x["status"] == "secondary_fail" for x in result["edges"]),
        "unresolved_edges": sum(x["status"] == "unresolved" for x in result["edges"]),
        "gated_graph": {key: result["gated"][key] for key in (
            "accepted_direct_edges", "gated_force_relations", "indirect_relations",
            "rejected_direct_edges_reconnected")},
        "complete_rest_blocks": len(blocks),
        "mask_intervals_overlapping_window": len(masked),
        "retained_rows": result["retained_rows"],
        "all_force_relations": result["all_force"]["gated_force_relations"],
        "no_force_relations": result["no_force"]["gated_force_relations"],
        "mask_sha256": args.expected_mask_sha256,
        "post_state_sha256": sha256(args.post_state),
        "routes_sha256": sha256(args.routes),
        "interval_metadata_sha256": sha256(args.interval_metadata),
        "session_time_offset_s": metadata["session_time_offset_s"],
    }
    (args.output / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
