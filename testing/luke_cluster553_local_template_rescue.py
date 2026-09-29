"""Run the frozen target-local template rescue diagnostic for cluster 553."""

from __future__ import annotations

import argparse
import bisect
import json
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

from testing.luke_cluster452_identity_replay import exclusive_pairs, interval, sha256


def extract_waves(raw: np.memmap, times: np.ndarray, channels: np.ndarray, pre: int,
                  post: int, baseline: int, gain: float) -> np.ndarray:
    valid = np.asarray(times, dtype=np.int64)
    valid = valid[(valid >= pre) & (valid < raw.shape[0] - post)]
    waves = np.empty((len(valid), pre + post, len(channels)), dtype=np.float32)
    for index, time in enumerate(valid):
        wave = np.asarray(raw[time - pre:time + post, channels], dtype=np.float32) * gain
        wave -= np.median(wave[:baseline], axis=0, keepdims=True)
        waves[index] = wave
    return waves


def waveform_scores(waves: np.ndarray, template: np.ndarray, noise: np.ndarray,
                    max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Best-lag cosine and noise-standardized matched-filter projection."""
    cosines = np.full(len(waves), -np.inf)
    projections = np.full(len(waves), -np.inf)
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            w, t = waves[:, -lag:], template[:lag]
        elif lag > 0:
            w, t = waves[:, :-lag], template[lag:]
        else:
            w, t = waves, template
        flat_w = w.reshape(len(w), -1).astype(np.float64)
        flat_t = t.reshape(-1).astype(np.float64)
        denominator = np.linalg.norm(flat_w, axis=1) * np.linalg.norm(flat_t)
        cosine = np.divide(flat_w @ flat_t, denominator, out=np.zeros(len(w)), where=denominator > 0)
        weighted_w = (w / noise[None, None, :]).reshape(len(w), -1).astype(np.float64)
        weighted_t = (t / noise[None, :]).reshape(-1).astype(np.float64)
        projection = (weighted_w @ weighted_t) / max(np.linalg.norm(weighted_t), 1e-12)
        cosines = np.maximum(cosines, cosine)
        projections = np.maximum(projections, projection)
    return cosines, projections


def deduplicate_proposals(times: np.ndarray, strengths: np.ndarray, refractory: int) -> np.ndarray:
    """Keep the strongest proposal within each refractory neighbourhood."""
    if not len(times):
        return np.empty(0, dtype=np.int64)
    order = np.lexsort((times, -strengths))
    accepted: list[int] = []
    for index in order:
        time = int(times[index])
        position = bisect.bisect_left(accepted, time)
        left_ok = position == 0 or time - accepted[position - 1] > refractory
        right_ok = position == len(accepted) or accepted[position] - time > refractory
        if left_ok and right_ok:
            accepted.insert(position, time)
    return np.asarray(accepted, dtype=np.int64)


def propose(raw: np.memmap, start: int, stop: int, channels: np.ndarray, noise_counts: np.ndarray,
            sigma: float, distance: int, chunk: int, maximum: int) -> np.ndarray:
    output = []
    pad = distance
    for core_start in range(start, stop, chunk):
        core_stop = min(core_start + chunk, stop)
        read_start, read_stop = max(start, core_start - pad), min(stop, core_stop + pad)
        traces = np.asarray(raw[read_start:read_stop, channels], dtype=np.float32)
        times, strengths = [], []
        for column in range(traces.shape[1]):
            peaks, properties = find_peaks(
                -traces[:, column], height=sigma * noise_counts[column], distance=distance
            )
            absolute = peaks + read_start
            keep = (absolute >= core_start) & (absolute < core_stop)
            times.extend(absolute[keep].tolist())
            strengths.extend((properties["peak_heights"][keep] / noise_counts[column]).tolist())
        selected = deduplicate_proposals(np.asarray(times), np.asarray(strengths), distance)
        output.append(selected)
        if sum(map(len, output)) > maximum:
            raise RuntimeError("proposal resource cap exceeded")
    return np.concatenate(output) if output else np.empty(0, dtype=np.int64)


def refractory_fraction(times: np.ndarray, threshold: int) -> float:
    times = np.unique(np.asarray(times, dtype=np.int64))
    return float(np.mean(np.diff(times) < threshold)) if len(times) > 1 else 0.0


def match_fraction(first: np.ndarray, second: np.ndarray, tolerance: int) -> float:
    hit, _ = exclusive_pairs(first, second, tolerance)
    return float(len(hit) / max(len(first), 1))


def run(config_path: Path, output: Path) -> dict:
    if output.exists():
        raise RuntimeError(f"output exists; preserve prior evidence: {output}")
    config = json.loads(config_path.read_text())
    inputs, operation, evaluation = config["inputs"], config["operation"], config["evaluation"]
    proposal_cfg, gate_cfg = operation["proposal"], operation["gate"]
    fs = float(inputs["sampling_frequency_hz"])
    pre, post = int(operation["pre_samples"]), int(operation["post_samples"])
    channels = np.asarray(operation["channels"], dtype=int)
    gain = float(operation["gain_uv_per_count"])
    manifest = json.loads(Path(inputs["binary_manifest"]).read_text())["kwargs"]
    dtype, n_channels = np.dtype(manifest["dtype"]), int(manifest["num_channels"])
    raw_path = Path(inputs["raw_recording"])
    frames = raw_path.stat().st_size // (dtype.itemsize * n_channels)
    raw = np.memmap(raw_path, mode="r", dtype=dtype, shape=(frames, n_channels))

    windows = {
        name: (int(round(value["start_s"] * fs)), int(round(value["stop_s"] * fs)))
        for name, value in config["intervals"].items()
    }
    curated = Path(inputs["curated_output"])
    spike_times = np.load(curated / "spike_times.npy", mmap_mode="r").reshape(-1)
    spike_labels = np.load(curated / "spike_clusters.npy", mmap_mode="r").reshape(-1)
    target_times = np.array(spike_times[spike_labels == int(inputs["target_cluster"])], dtype=np.int64)
    target_times.sort()
    cal_start, cal_stop = windows["template_and_calibration"]
    cal_target = target_times[interval(target_times, cal_start, cal_stop)]
    target_waves = extract_waves(raw, cal_target, channels, pre, post,
                                 int(operation["baseline_samples"]), gain)
    template = np.median(target_waves, axis=0)

    stride = int(proposal_cfg["noise_stride_samples"])
    noise_sample = np.asarray(raw[cal_start:cal_stop:stride, channels], dtype=np.float32)
    center = np.median(noise_sample, axis=0)
    noise_counts = np.median(np.abs(noise_sample - center), axis=0) / 0.6744897501960817
    noise_uv = noise_counts * gain
    if np.any(noise_counts <= 0):
        raise RuntimeError("nonpositive channel noise estimate")

    shifts = [int(round(float(value) * fs)) for value in gate_cfg["circular_shift_background_s"]]
    background_times = np.concatenate([
        cal_start + ((cal_target - cal_start + shift) % (cal_stop - cal_start)) for shift in shifts
    ])
    background_waves = extract_waves(raw, background_times, channels, pre, post,
                                     int(operation["baseline_samples"]), gain)
    target_cos, target_score = waveform_scores(
        target_waves, template, noise_uv, int(operation["maximum_alignment_lag_samples"])
    )
    bg_cos, bg_score = waveform_scores(
        background_waves, template, noise_uv, int(operation["maximum_alignment_lag_samples"])
    )
    score_threshold = max(float(np.quantile(bg_score, 0.999)), float(np.quantile(target_score, 0.10)))
    cosine_threshold = float(gate_cfg["minimum_waveform_cosine"])

    candidate_by_window, proposal_counts = {}, {}
    for name, (start, stop) in windows.items():
        proposals = propose(
            raw, start, stop, channels, noise_counts,
            float(proposal_cfg["negative_peak_threshold_sigma"]),
            int(proposal_cfg["proposal_refractory_samples"]),
            int(round(float(proposal_cfg["chunk_duration_s"]) * fs)),
            int(proposal_cfg["maximum_total_proposals"]),
        )
        waves = extract_waves(raw, proposals, channels, pre, post,
                              int(operation["baseline_samples"]), gain)
        cosines, scores = waveform_scores(
            waves, template, noise_uv, int(operation["maximum_alignment_lag_samples"])
        )
        accepted = (cosines >= cosine_threshold) & (scores >= score_threshold)
        candidate_by_window[name] = (proposals, proposals[accepted], cosines[accepted], scores[accepted])
        proposal_counts[name] = {"proposed": int(len(proposals)), "accepted": int(np.count_nonzero(accepted))}

    tolerance = int(round(float(evaluation["event_tolerance_ms"]) * 1e-3 * fs))
    legacy = Path(inputs["legacy_curated"])
    lt = np.load(legacy / "spike_times.npy", mmap_mode="r").reshape(-1)
    lc = np.load(legacy / "spike_clusters.npy", mmap_mode="r").reshape(-1)
    anchor = np.array(lt[lc == int(inputs["legacy_anchor_cluster_for_evaluation_only"])], dtype=np.int64)
    anchor.sort()
    sorter = Path(inputs["sorter_output"])
    full = np.load(sorter / "full_st.npy", mmap_mode="r")
    from testing.luke_cluster21_detection_lineage import detection_template_depths
    from testing.luke_cluster452_identity_replay import template_depths
    depth = detection_template_depths(np.load(sorter / "ops.npy", allow_pickle=True).item())
    target_depth = template_depths(curated)[int(inputs["target_cluster"])]
    ids = np.flatnonzero(np.abs(depth - target_depth) <= 100.0)
    local_full = np.sort(np.asarray(full[np.isin(np.asarray(full[:, 1], int), ids), 0], dtype=np.int64))

    summaries = {}
    all_added = []
    for name, (start, stop) in windows.items():
        proposals, candidates, cosines, _ = candidate_by_window[name]
        existing = target_times[interval(target_times, start, stop)]
        anchor_window = anchor[interval(anchor, start, stop)]
        local = local_full[interval(local_full, start, stop)]
        anchor_local_hit, _ = exclusive_pairs(anchor_window, local, tolerance)
        local_mask = np.zeros(len(anchor_window), bool); local_mask[anchor_local_hit] = True
        unmatched = anchor_window[~local_mask]
        _, candidate_existing_hit = exclusive_pairs(existing, candidates, tolerance)
        is_existing = np.zeros(len(candidates), bool); is_existing[candidate_existing_hit] = True
        added = candidates[~is_existing]
        added_cos = cosines[~is_existing]
        all_added.append(added)
        summaries[name] = {
            **proposal_counts[name], "existing_target_events": int(len(existing)),
            "proposal_match_fraction_of_existing_target": match_fraction(existing, proposals, tolerance),
            "accepted_match_fraction_of_existing_target": match_fraction(existing, candidates, tolerance),
            "independent_anchor_events": int(len(anchor_window)),
            "previously_local_full_st_unmatched": int(len(unmatched)),
            "recovery_fraction_of_previously_unmatched_anchor": match_fraction(unmatched, added, tolerance),
            "added_events": int(len(added)),
            "median_added_event_template_cosine": float(np.median(added_cos)) if len(added_cos) else None,
        }

    added = np.concatenate(all_added)
    existing_in_windows = np.concatenate([
        target_times[interval(target_times, start, stop)] for start, stop in windows.values()
    ])
    baseline_rv = refractory_fraction(existing_in_windows, tolerance)
    union_rv = refractory_fraction(np.concatenate([existing_in_windows, added]), tolerance)
    ref = summaries["template_and_calibration"]
    fail = summaries["failing_evaluation"]
    background_acceptance = float(np.mean((bg_cos >= cosine_threshold) & (bg_score >= score_threshold)))
    gates = {
        "reference_sensitivity": ref["accepted_match_fraction_of_existing_target"] >= float(evaluation["minimum_reference_target_sensitivity"]),
        "background_acceptance": background_acceptance <= float(evaluation["maximum_calibration_background_acceptance_fraction"]),
        "failing_unmatched_recovery": fail["recovery_fraction_of_previously_unmatched_anchor"] >= float(evaluation["minimum_failing_unmatched_anchor_recovery_fraction"]),
        "refractory_preservation": union_rv - baseline_rv <= float(evaluation["maximum_added_refractory_violation_fraction"]),
        "added_waveform_similarity": fail["median_added_event_template_cosine"] is not None and fail["median_added_event_template_cosine"] >= float(evaluation["minimum_added_event_template_cosine"]),
    }
    result = {
        "schema": config["schema"], "status": "complete",
        "verdict": "advance_local_template_rescue" if all(gates.values()) else "reject_local_template_rescue",
        "config_sha256": sha256(config_path), "score_threshold": score_threshold,
        "template_target_spikes": int(len(cal_target)), "noise_uv": noise_uv.tolist(),
        "calibration_target_acceptance_fraction": float(np.mean(
            (target_cos >= cosine_threshold) & (target_score >= score_threshold)
        )),
        "calibration_background_acceptance_fraction": background_acceptance,
        "windows": summaries, "baseline_refractory_fraction": baseline_rv,
        "candidate_union_refractory_fraction": union_rv,
        "candidate_refractory_increase": union_rv - baseline_rv, "gates": gates,
        "sort_launched": False, "production_changed": False,
    }
    output.mkdir(parents=True)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    np.save(output / "template.npy", template)
    for name, (proposals, times, cosines, scores) in candidate_by_window.items():
        np.savez(output / f"{name}_accepted.npz", times=times, cosines=cosines, scores=scores)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/luke_cluster553_local_template_rescue.v1.json"))
    parser.add_argument("--output", type=Path, default=Path("testing/outputs/luke_cluster553_local_template_rescue_v1"))
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output), indent=2))


if __name__ == "__main__":
    main()
