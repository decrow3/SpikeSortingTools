"""Audit cluster 553 single-event template scores before another detector arm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from testing.luke_cluster452_identity_replay import interval, sha256
from testing.luke_cluster553_local_template_rescue import extract_waves, waveform_scores


def quantiles(values: np.ndarray) -> dict[str, float]:
    return {str(q): float(np.quantile(values, q)) for q in (0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999)}


def run(config_path: Path, output: Path) -> dict:
    if output.exists():
        raise RuntimeError(f"output exists; preserve prior evidence: {output}")
    config = json.loads(config_path.read_text())
    inputs, operation = config["inputs"], config["operation"]
    fs = float(inputs["sampling_frequency_hz"])
    manifest = json.loads(Path(inputs["binary_manifest"]).read_text())["kwargs"]
    dtype, n_channels = np.dtype(manifest["dtype"]), int(manifest["num_channels"])
    raw_path = Path(inputs["raw_recording"])
    frames = raw_path.stat().st_size // (dtype.itemsize * n_channels)
    raw = np.memmap(raw_path, mode="r", dtype=dtype, shape=(frames, n_channels))
    channels = np.asarray(operation["channels"], dtype=int)
    gain = float(operation["gain_uv_per_count"])
    pre, post, baseline = (int(operation[key]) for key in ("pre_samples", "post_samples", "baseline_samples"))
    window = config["intervals"]["template_and_calibration"]
    start, stop = int(round(window["start_s"] * fs)), int(round(window["stop_s"] * fs))
    curated = Path(inputs["curated_output"])
    times = np.load(curated / "spike_times.npy", mmap_mode="r").reshape(-1)
    labels = np.load(curated / "spike_clusters.npy", mmap_mode="r").reshape(-1)
    target = np.sort(np.array(times[labels == int(inputs["target_cluster"])], dtype=np.int64))
    target = target[interval(target, start, stop)]
    target_waves = extract_waves(raw, target, channels, pre, post, baseline, gain)
    template = np.median(target_waves, axis=0)
    proposal = operation["proposal"]
    sample = np.asarray(raw[start:stop:int(proposal["noise_stride_samples"]), channels], dtype=np.float32)
    center = np.median(sample, axis=0)
    noise_uv = np.median(np.abs(sample - center), axis=0) / 0.6744897501960817 * gain
    shifts = [int(round(float(value) * fs)) for value in operation["gate"]["circular_shift_background_s"]]
    background = np.concatenate([start + ((target - start + shift) % (stop - start)) for shift in shifts])
    background_waves = extract_waves(raw, background, channels, pre, post, baseline, gain)
    target_cos, target_score = waveform_scores(target_waves, template, noise_uv,
                                                int(operation["maximum_alignment_lag_samples"]))
    bg_cos, bg_score = waveform_scores(background_waves, template, noise_uv,
                                        int(operation["maximum_alignment_lag_samples"]))
    score_threshold = max(float(np.quantile(bg_score, 0.999)), float(np.quantile(target_score, 0.1)))
    cosine_threshold = float(operation["gate"]["minimum_waveform_cosine"])
    result = {
        "schema": "luke-cluster553-template-calibration-audit-v1", "status": "complete",
        "config_sha256": sha256(config_path), "target_events": int(len(target)),
        "background_events": int(len(background)), "score_threshold": score_threshold,
        "target_cosine_quantiles": quantiles(target_cos), "background_cosine_quantiles": quantiles(bg_cos),
        "target_score_quantiles": quantiles(target_score), "background_score_quantiles": quantiles(bg_score),
        "target_score_only_acceptance": float(np.mean(target_score >= score_threshold)),
        "background_score_only_acceptance": float(np.mean(bg_score >= score_threshold)),
        "target_cosine_only_acceptance": float(np.mean(target_cos >= cosine_threshold)),
        "background_cosine_only_acceptance": float(np.mean(bg_cos >= cosine_threshold)),
        "target_joint_acceptance": float(np.mean((target_score >= score_threshold) & (target_cos >= cosine_threshold))),
        "background_joint_acceptance": float(np.mean((bg_score >= score_threshold) & (bg_cos >= cosine_threshold))),
        "sort_launched": False, "production_changed": False,
    }
    output.mkdir(parents=True)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/luke_cluster553_local_template_rescue.v1.json"))
    parser.add_argument("--output", type=Path, default=Path("testing/outputs/luke_cluster553_template_calibration_audit_v1"))
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output), indent=2))


if __name__ == "__main__":
    main()
