"""Test whether bounded waveform-alignment error explains template-score failure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from testing.luke_cluster452_identity_replay import interval, sha256
from testing.luke_cluster553_local_template_rescue import extract_waves, waveform_scores
from testing.luke_cluster553_template_calibration_audit import quantiles


def alignment_verdict(rows: list[dict], minimum: float) -> tuple[str, list[int]]:
    passing = [int(row["maximum_lag_samples"]) for row in rows
               if row["target_acceptance"] >= minimum]
    return ("alignment_followup_supported" if passing else "alignment_followup_not_supported"), passing


def run(config_path: Path, output: Path) -> dict:
    if output.exists():
        raise RuntimeError(f"output exists; preserve prior evidence: {output}")
    audit = json.loads(config_path.read_text())
    source_path = Path(audit["source_candidate_config"])
    source = json.loads(source_path.read_text())
    inputs, operation = source["inputs"], source["operation"]
    fs = float(inputs["sampling_frequency_hz"])
    manifest = json.loads(Path(inputs["binary_manifest"]).read_text())["kwargs"]
    dtype, n_channels = np.dtype(manifest["dtype"]), int(manifest["num_channels"])
    raw_path = Path(inputs["raw_recording"])
    frames = raw_path.stat().st_size // (dtype.itemsize * n_channels)
    raw = np.memmap(raw_path, mode="r", dtype=dtype, shape=(frames, n_channels))
    channels = np.asarray(operation["channels"], dtype=int)
    gain = float(operation["gain_uv_per_count"])
    pre, post, baseline = (int(operation[key]) for key in ("pre_samples", "post_samples", "baseline_samples"))
    window = source["intervals"]["template_and_calibration"]
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
    q = float(audit["background_threshold_quantile"])
    rows = []
    for lag in map(int, audit["maximum_alignment_lag_samples"]):
        _, target_score = waveform_scores(target_waves, template, noise_uv, lag)
        _, background_score = waveform_scores(background_waves, template, noise_uv, lag)
        threshold = float(np.quantile(background_score, q))
        rows.append({
            "maximum_lag_samples": lag, "score_threshold": threshold,
            "target_acceptance": float(np.mean(target_score >= threshold)),
            "background_acceptance": float(np.mean(background_score >= threshold)),
            "target_score_quantiles": quantiles(target_score),
            "background_score_quantiles": quantiles(background_score),
        })
    verdict, passing = alignment_verdict(rows, float(audit["minimum_target_acceptance_for_followup"]))
    result = {
        "schema": audit["schema"], "status": "complete", "verdict": verdict,
        "config_sha256": sha256(config_path), "source_config_sha256": sha256(source_path),
        "target_events": int(len(target)), "background_events": int(len(background)),
        "rows": rows, "passing_lags_samples": passing,
        "continuous_voltage_scanned": False, "sort_launched": False,
        "production_changed": False,
    }
    output.mkdir(parents=True)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/luke_cluster553_template_alignment_audit.v1.json"))
    parser.add_argument("--output", type=Path, default=Path("testing/outputs/luke_cluster553_template_alignment_audit_v1"))
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output), indent=2))


if __name__ == "__main__":
    main()
