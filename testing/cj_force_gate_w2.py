"""Saved-state W2 refractory-dip force gate; no voltage or grouping replay."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from testing.ci_force_gate import gated_force_union


CENTRAL_MAX = 29
SHOULDER_MIN = 45
SHOULDER_MAX = 89
CENTRAL_LAG_VALUES = 59
SHOULDER_LAG_VALUES = 90
EDGE_TRIM = SHOULDER_MAX


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def lag_counts(a_offsets, b_offsets):
    """Count exact discrete central and shoulder cross-lags."""
    a = np.asarray(a_offsets, dtype=np.int64)
    b = np.sort(np.asarray(b_offsets, dtype=np.int64))
    central = shoulder = 0
    for value in a:
        central += np.searchsorted(b, value + CENTRAL_MAX, side="right")
        central -= np.searchsorted(b, value - CENTRAL_MAX, side="left")
        shoulder += np.searchsorted(b, value - SHOULDER_MIN, side="right")
        shoulder -= np.searchsorted(b, value - SHOULDER_MAX, side="left")
        shoulder += np.searchsorted(b, value + SHOULDER_MAX, side="right")
        shoulder -= np.searchsorted(b, value + SHOULDER_MIN, side="left")
    return int(central), int(shoulder)


def random_derangement(size, rng):
    if size < 2:
        raise ValueError("derangement requires at least two blocks")
    if size == 2:
        return np.array([1, 0])
    base = np.arange(size)
    for _ in range(100):
        candidate = rng.permutation(size)
        if np.all(candidate != base):
            return candidate
    # Deterministic nonidentity fallback; still no fixed points.
    return np.roll(base, 1)


def infer_complete_rest_blocks(times_samples, segment_ids, episode_mask, fs):
    """Infer quarter-second segment bounds and tile complete 5-s blocks."""
    rows = []
    for segment in np.unique(segment_ids):
        selected = segment_ids == segment
        states = np.unique(episode_mask[selected])
        if states.size != 1:
            raise ValueError(f"mixed state in segment {segment}")
        if states[0]:
            continue
        seconds = times_samples[selected] / fs
        start_s = np.floor(seconds.min() * 4.0 + 1e-9) / 4.0
        end_s = np.ceil(seconds.max() * 4.0 - 1e-9) / 4.0
        n_complete = int(np.floor((end_s - start_s) / 5.0 + 1e-9))
        for within_segment in range(n_complete):
            left_s = start_s + 5.0 * within_segment
            right_s = left_s + 5.0
            left = int(np.ceil(left_s * fs))
            right = int(np.ceil(right_s * fs))
            rows.append({"block_id": len(rows), "segment_id": int(segment),
                         "within_segment": within_segment, "start_sample": left,
                         "end_sample": right, "start_s": left_s, "end_s": right_s})
    return rows


def event_offsets_by_unit_block(times, labels, blocks):
    values = {}
    for block in blocks:
        left, right = block["start_sample"], block["end_sample"]
        selected = ((times >= left + EDGE_TRIM) & (times < right - EDGE_TRIM)
                    & (labels >= 0))
        indices = np.flatnonzero(selected)
        for unit in np.unique(labels[indices]):
            unit_times = times[indices[labels[indices] == unit]] - left
            values[(int(unit), block["block_id"])] = np.sort(unit_times)
    return values


def score_edge(unit_i, unit_j, blocks, events, *, n_bootstrap, n_null, seed):
    common = [b for b in blocks if (unit_i, b["block_id"]) in events
              and (unit_j, b["block_id"]) in events]
    by_segment = {}
    for block in common:
        by_segment.setdefault(block["segment_id"], []).append(block)
    # Nonwrapping block derangement requires at least two target/source blocks.
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
        n = len(segment_blocks)
        cm = np.zeros((n, n), dtype=np.int64)
        sm = np.zeros((n, n), dtype=np.int64)
        for ai, target in enumerate(segment_blocks):
            a = events[(unit_i, target["block_id"])]
            for bj, source in enumerate(segment_blocks):
                b = events[(unit_j, source["block_id"])]
                cm[ai, bj], sm[ai, bj] = lag_counts(a, b)
        matrices[segment] = (cm, sm)
        diagonal_counts.extend(zip(np.diag(cm), np.diag(sm), strict=True))
    block_c = np.asarray([x[0] for x in diagonal_counts], dtype=np.int64)
    block_s = np.asarray([x[1] for x in diagonal_counts], dtype=np.int64)
    central, shoulder = int(block_c.sum()), int(block_s.sum())
    expected = shoulder * CENTRAL_LAG_VALUES / SHOULDER_LAG_VALUES
    result.update(central_count=central, shoulder_count=shoulder,
                  expected_central=float(expected))
    if expected < 20:
        result["reason"] = "insufficient_expected_central"; return result

    ratio = (central + 0.5) / (expected + 0.5)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(block_c), size=(n_bootstrap, len(block_c)))
    bc = block_c[draws].sum(axis=1)
    bs = block_s[draws].sum(axis=1)
    bootstrap = (bc + 0.5) / (bs * CENTRAL_LAG_VALUES / SHOULDER_LAG_VALUES + 0.5)
    if not np.isfinite(bootstrap).all():
        result["reason"] = "nonfinite_bootstrap"; return result

    null = np.empty(n_null, dtype=float)
    for draw in range(n_null):
        nc = ns = 0
        for segment in sorted(matrices):
            cm, sm = matrices[segment]
            permutation = random_derangement(len(cm), rng)
            rows = np.arange(len(cm))
            nc += int(cm[rows, permutation].sum())
            ns += int(sm[rows, permutation].sum())
        null[draw] = (nc + 0.5) / (
            ns * CENTRAL_LAG_VALUES / SHOULDER_LAG_VALUES + 0.5)
    if not np.isfinite(null).all():
        result["reason"] = "nonfinite_null"; return result
    upper = float(np.quantile(bootstrap, 0.975))
    null_q05 = float(np.quantile(null, 0.05))
    passed = upper <= 0.50 and ratio <= null_q05
    result.update(status="secondary_pass" if passed else "secondary_fail",
                  reason="", secondary_pass=bool(passed), dip_ratio=float(ratio),
                  bootstrap_upper=upper, null_q05=null_q05,
                  null_lower_tail_p=float((1 + np.count_nonzero(null <= ratio)) / (n_null + 1)),
                  accepted_bootstrap=n_bootstrap, accepted_null=n_null)
    return result


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ce-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", default="3000")
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--null", type=int, default=1000)
    args = parser.parse_args()
    started = time.time(); args.output.mkdir(parents=True, exist_ok=False)

    post_path = args.ce_root / "POST_TMM_STATE.npz"
    route_path = (args.ce_root / "supplemental_saved_array_audit_v1"
                  / f"{args.arm}_ROUTES_AND_CLOCKS.npz")
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

    fs = 29999.759166666667
    blocks = infer_complete_rest_blocks(
        post["times_samples"], route["fixed_segment"], route["fixed_state_episode"], fs)
    events = event_offsets_by_unit_block(post["times_samples"], post["labels"], blocks)
    direct = route["direct_force_mask"].copy(); np.fill_diagonal(direct, False)
    ii, jj = np.triu_indices_from(direct, 1)
    selected = np.flatnonzero(direct[ii, jj])
    edge_results = []
    secondary = np.zeros_like(direct)
    for ordinal, index in enumerate(selected):
        result = score_edge(int(ii[index]), int(jj[index]), blocks, events,
                            n_bootstrap=args.bootstrap, n_null=args.null,
                            seed=args.seed + ordinal)
        result["edge_ordinal"] = ordinal
        result["qda_requested"] = bool(route["qda_requested_mask"][ii[index], jj[index]])
        result["qda_accepted"] = bool(route["qda_accept_mask"][ii[index], jj[index]])
        edge_results.append(result)
        if result["secondary_pass"]:
            secondary[ii[index], jj[index]] = secondary[jj[index], ii[index]] = True

    gated = gated_force_union(direct_force_mask=direct,
        secondary_pass_mask=secondary, qda_accept_mask=route["qda_accept_mask"])
    all_force = gated_force_union(direct_force_mask=direct,
        secondary_pass_mask=direct, qda_accept_mask=route["qda_accept_mask"])
    no_force = gated_force_union(direct_force_mask=direct,
        secondary_pass_mask=np.zeros_like(direct), qda_accept_mask=route["qda_accept_mask"])
    qda_only = route["qda_accept_mask"].copy(); np.fill_diagonal(qda_only, True)
    summary = {
        "status": "complete_saved_fixed_clock_hard_partition_only",
        "arm": args.arm, "source_rows": len(post["event_row_ids"]),
        "source_units": int(post["labels"].max() + 1),
        "direct_force_edges": len(selected), "complete_rest_blocks": len(blocks),
        "secondary_pass_edges": int(sum(x["secondary_pass"] for x in edge_results)),
        "secondary_fail_edges": int(sum(x["status"] == "secondary_fail" for x in edge_results)),
        "unresolved_edges": int(sum(x["status"] == "unresolved" for x in edge_results)),
        "unresolved_reasons": {reason: sum(x.get("reason") == reason for x in edge_results)
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
        "aggregation_dedup_outcome": "not run",
        "event_filter": "none beyond saved hard partition, rest state, complete block and boundary support",
        "central_lags_samples": [-29, 29],
        "shoulder_lags_abs_samples": [45, 89],
        "lag_exposure_ratio": CENTRAL_LAG_VALUES / SHOULDER_LAG_VALUES,
        "bootstrap_draws": args.bootstrap, "null_draws": args.null,
        "elapsed_s": time.time() - started,
    }
    write_csv(args.output / "EDGE_RESULTS.csv", edge_results)
    write_csv(args.output / "REST_BLOCKS.csv", blocks)
    np.savez_compressed(args.output / "GATED_ARRAYS.npz",
        secondary_pass_mask=secondary, accepted_direct_mask=gated["accepted_direct_mask"],
        gated_force_mask=gated["gated_force_mask"], final_union_mask=gated["final_union_mask"],
        component_ids=gated["component_ids"])
    (args.output / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    receipt = {
        "post_tmm_sha256": sha256(post_path), "routes_sha256": sha256(route_path),
        "helper_sha256": sha256(Path(__file__).with_name("ci_force_gate.py")),
        "script_sha256": sha256(Path(__file__)), "seed": args.seed,
        "method_commit_required_before_outcomes": "07811bc",
        "scientific_scope": "saved fixed-clock hard-partition graph diagnostic only",
    }
    (args.output / "RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n")
    products = []
    for name in ("EDGE_RESULTS.csv", "REST_BLOCKS.csv", "GATED_ARRAYS.npz",
                 "SUMMARY.json", "RECEIPT.json"):
        path = args.output / name
        products.append({"path": name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    (args.output / "COMPLETE.json").write_text(json.dumps(
        {"status": "complete", "products": products, "written_last": True}, indent=2) + "\n")


if __name__ == "__main__":
    main()
