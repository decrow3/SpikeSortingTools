"""Qualify CN support forecasts against the frozen scoring domain."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from testing.cj_force_gate_w2 import event_offsets_by_unit_block


CENTRAL_LAGS = tuple(range(-29, -8)) + tuple(range(9, 30))
SCENARIOS = (0.5, 1.0, 2.0)
MASK_SHA256 = "86425e8a6627a10ac8932ba01b799cb7be7141f731ebb4ecc01e303326b9da55"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def parse_meta(path: Path) -> dict:
    values = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    size = int(values["fileSizeBytes"])
    channels = int(values["nSavedChans"])
    fs = float(values["imSampRate"])
    if size % (2 * channels):
        raise ValueError("AP byte count is not integral int16 frames")
    frames = size // (2 * channels)
    duration = frames / fs
    if not np.isclose(duration, float(values["fileTimeSecs"]), rtol=0, atol=1e-9):
        raise ValueError("AP frame-derived duration disagrees with metadata")
    return {
        "frames": frames, "sampling_frequency_hz": fs, "duration_s": duration,
        "file_size_bytes": size, "saved_channels": channels,
        "first_sample": int(values["firstSample"]), "file_sha1": values["fileSHA1"],
    }


def merge_mask(path: Path, duration_s: float) -> list[tuple[float, float]]:
    if sha256(path) != MASK_SHA256:
        raise ValueError("canonical mask hash mismatch")
    values = []
    for row in read_rows(path):
        left = max(0.0, float(row["start_s"]))
        right = min(duration_s, float(row["end_s"]))
        if left < right:
            values.append((left, right))
    values.sort()
    merged = []
    for left, right in values:
        if merged and left <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], right))
        else:
            merged.append((left, right))
    return merged


def rest_segments(mask, duration_s):
    result = []
    cursor = 0.0
    for left, right in mask:
        if cursor < left:
            result.append((cursor, left))
        cursor = max(cursor, right)
    if cursor < duration_s:
        result.append((cursor, duration_s))
    return result


def ceil_frame(seconds: float, fs: float) -> int:
    return int(np.ceil(seconds * fs))


def full_blocks(segments, fs, frames):
    blocks = []
    for segment_id, (left, right) in enumerate(segments):
        count = int(np.floor((right - left) / 5.0 + 1e-12))
        for within in range(count):
            start_s = left + 5.0 * within
            end_s = start_s + 5.0
            start, end = ceil_frame(start_s, fs), ceil_frame(end_s, fs)
            if not (0 <= start < end <= frames):
                raise ValueError("full-session rest block outside AP frames")
            blocks.append({"segment_id": segment_id, "within_segment": within,
                           "start_sample": start, "end_sample": end})
    return blocks


def lag_exposure_samples(length: int) -> int:
    lower, upper = 89, length - 89
    return int(sum(max(0, min(upper, upper - lag) - max(lower, lower - lag))
                   for lag in CENTRAL_LAGS))


def expected_full_support(blocks, rate_i, rate_j, factor, fs):
    by_segment = {}
    for block in blocks:
        by_segment.setdefault(block["segment_id"], []).append(block)
    common = events_i = events_j = central = eligible_exposure = 0.0
    for segment in by_segment.values():
        if len(segment) < 2:
            continue
        lengths = np.asarray([(b["end_sample"] - b["start_sample"]) / fs
                              for b in segment], dtype=float)
        p_i = 1.0 - np.exp(-rate_i * factor * lengths)
        p_j = 1.0 - np.exp(-rate_j * factor * lengths)
        q = p_i * p_j
        for index, (block, seconds) in enumerate(zip(segment, lengths, strict=True)):
            other = np.delete(q, index)
            p_other = 1.0 - float(np.prod(1.0 - other))
            common += q[index] * p_other
            events_i += rate_i * factor * seconds * p_j[index] * p_other
            events_j += rate_j * factor * seconds * p_i[index] * p_other
            exposure = lag_exposure_samples(block["end_sample"] - block["start_sample"])
            eligible_exposure += exposure * p_other
            central += (rate_i * factor / fs) * (rate_j * factor / fs) * exposure * p_other
    return {
        "expected_common_positive_blocks": common,
        "expected_scoring_events_i": events_i,
        "expected_scoring_events_j": events_j,
        "expected_independent_central": central,
        "eligible_lag_exposure_samples": eligible_exposure,
    }


def window_context(post_path: Path, blocks_path: Path):
    with np.load(post_path, allow_pickle=False) as data:
        times, labels = data["times_samples"], data["labels"]
    blocks = []
    for row in read_rows(blocks_path):
        blocks.append({key: int(row[key]) for key in
                       ("block_id", "segment_id", "within_segment", "start_sample", "end_sample")})
    events = event_offsets_by_unit_block(times, labels, blocks)
    lengths = {b["block_id"]: b["end_sample"] - b["start_sample"] for b in blocks}
    return blocks, events, lengths


def scoring_blocks(unit_i, unit_j, blocks, events):
    common = [b for b in blocks if (unit_i, b["block_id"]) in events
              and (unit_j, b["block_id"]) in events]
    by_segment = {}
    for block in common:
        by_segment.setdefault(block["segment_id"], []).append(block)
    return [b for key in sorted(by_segment) if len(by_segment[key]) >= 2
            for b in by_segment[key]]


def summarize_state_pairs(pair_path: Path, occupancy_path: Path):
    pairs = read_rows(pair_path)
    occupancy = read_rows(occupancy_path)
    rates = {(r["policy"], int(r["final_unit"])):
             (float(r["rest_rate_hz"]), float(r["episode_rate_hz"])) for r in occupancy}
    result = {}
    for policy in sorted({row["policy"] for row in pairs}):
        rows = [row for row in pairs if row["policy"] == policy]
        rest_rates = np.asarray([rates[(policy, int(row["rest_child"]))][0] for row in rows])
        episode_rates = np.asarray([rates[(policy, int(row["episode_child"]))][1] for row in rows])
        low = [int(row["rest_child_spikes"]) < 20 or int(row["episode_child_spikes"]) < 20
               for row in rows]
        result[policy] = {
            "pairs": len(rows),
            "all_force_parents": len({int(row["all_force_parent"]) for row in rows}),
            "pairs_with_lt20_spikes_in_either_child": int(sum(low)),
            "fraction_with_lt20_spikes_in_either_child": float(np.mean(low)),
            "rest_child_rest_rate_hz_median": float(np.median(rest_rates)),
            "episode_child_episode_rate_hz_median": float(np.median(episode_rates)),
            "rate_note": "saved per-state rates, pair-weighted; repeated children retain repeated weight",
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--cn", type=Path, required=True)
    parser.add_argument("--w2-post", type=Path, required=True)
    parser.add_argument("--w2-blocks", type=Path, required=True)
    parser.add_argument("--w3-post", type=Path, required=True)
    parser.add_argument("--w3-blocks", type=Path, required=True)
    parser.add_argument("--h5-cn", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    source_paths = [args.meta, args.mask, args.cn / "EDGE_EXPOSURE.csv",
                    args.cn / "SUMMARY.json", args.w2_post, args.w2_blocks,
                    args.w3_post, args.w3_blocks,
                    args.h5_cn / "STATE_COMPLEMENTARY_CHILDREN.csv",
                    args.h5_cn / "ACTUAL_STATE_OCCUPANCY.csv"]
    receipt = {str(path): {"bytes": path.stat().st_size, "sha256": sha256(path)}
               for path in source_paths}
    meta = parse_meta(args.meta)
    mask = merge_mask(args.mask, meta["duration_s"])
    segments = rest_segments(mask, meta["duration_s"])
    blocks = full_blocks(segments, meta["sampling_frequency_hz"], meta["frames"])
    wctx = {
        "W2": window_context(args.w2_post, args.w2_blocks),
        "W3": window_context(args.w3_post, args.w3_blocks),
    }
    original = read_rows(args.cn / "EDGE_EXPOSURE.csv")
    rows = []
    for edge in original:
        window = edge["window"]
        unit_i, unit_j = int(edge["unit_i"]), int(edge["unit_j"])
        old_blocks, events, lengths = wctx[window]
        observed_blocks = scoring_blocks(unit_i, unit_j, old_blocks, events)
        observed_exposure = sum(lag_exposure_samples(lengths[b["block_id"]])
                                for b in observed_blocks)
        for factor in SCENARIOS:
            values = expected_full_support(
                blocks, float(edge["all_rest_rate_i_hz"]),
                float(edge["all_rest_rate_j_hz"]), factor,
                meta["sampling_frequency_hz"])
            shoulder_now = float(edge["shoulder_expected_central"])
            shoulder_full = (shoulder_now * factor * factor
                             * values["eligible_lag_exposure_samples"] / observed_exposure
                             if shoulder_now > 0 and observed_exposure > 0 else np.inf)
            common_ok = values["expected_common_positive_blocks"] >= 20
            events_ok = min(values["expected_scoring_events_i"],
                            values["expected_scoring_events_j"]) >= 100
            row = {
                "window": window, "edge_ordinal": int(edge["edge_ordinal"]),
                "unit_i": unit_i, "unit_j": unit_j, "status": edge["status"],
                "rate_factor": factor, **values,
                "expected_shoulder_central": shoulder_full,
                "qualified_independent_expected_support": bool(
                    common_ok and events_ok and values["expected_independent_central"] >= 20),
                "qualified_shoulder_expected_support": bool(
                    common_ok and events_ok and np.isfinite(shoulder_full) and shoulder_full >= 20),
            }
            rows.append(row)
    with (args.output / "QUALIFIED_EDGE_FORECAST.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    state = summarize_state_pairs(args.h5_cn / "STATE_COMPLEMENTARY_CHILDREN.csv",
                                  args.h5_cn / "ACTUAL_STATE_OCCUPANCY.csv")
    exact = {
        **meta,
        "mask_sha256": sha256(args.mask),
        "merged_mask_intervals": len(mask),
        "merged_mask_s": float(sum(right - left for left, right in mask)),
        "rest_segments": len(segments),
        "rest_s": float(sum(right - left for left, right in segments)),
        "complete_5s_blocks": len(blocks),
        "segments_with_at_least_2_complete_blocks": int(sum(
            np.floor((right-left)/5.0 + 1e-12) >= 2 for left, right in segments)),
        "complete_block_duration_s": float(sum(
            b["end_sample"] - b["start_sample"] for b in blocks) / meta["sampling_frequency_hz"]),
        "trimmed_block_duration_s": float(sum(
            max(0, b["end_sample"] - b["start_sample"] - 178) for b in blocks)
            / meta["sampling_frequency_hz"]),
    }
    counts = {}
    for window in ("W2", "W3"):
        wr = [row for row in rows if row["window"] == window]
        counts[window] = {}
        for factor in SCENARIOS:
            fr = [row for row in wr if row["rate_factor"] == factor]
            counts[window][str(factor)] = {
                "direct_edges": len(fr),
                "qualified_independent_expected_support": sum(
                    row["qualified_independent_expected_support"] for row in fr),
                "qualified_shoulder_expected_support": sum(
                    row["qualified_shoulder_expected_support"] for row in fr),
            }
    (args.output / "SOURCE_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (args.output / "EXACT_FULL_REST.json").write_text(json.dumps(exact, indent=2) + "\n")
    (args.output / "STATE_COMPLEMENTARY_QUALIFICATION.json").write_text(
        json.dumps(state, indent=2) + "\n")
    (args.output / "SUMMARY.json").write_text(json.dumps({
        "status": "complete_saved_data_qualification",
        "original_cn_preserved": True,
        "original_cn_interpretation": "optimistic all-rest support screen",
        "qualified_interpretation": "expected scoring-domain support under stationary independent Poisson rates; not realization probability, gate power, identity, recovery, or yield",
        "exact_full_rest": exact,
        "qualified_counts": counts,
        "state_complementary": state,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
