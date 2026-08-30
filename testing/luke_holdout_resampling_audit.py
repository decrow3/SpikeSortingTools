"""Paired sorter-free motion-resampling audit on the sealed Luke holdout.

The candidate coordinates are joined to the blinded strata only for automated
metric calculation; no review labels or sorter outputs are read.  The audit
compares the conditioned voltage with the historical p=1/zero-border warp and
SpikeInterface's DREDGE-like p=2/extrapolating variants.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from testing.luke_interpolation_implementation_audit import (
    VARIANTS,
    build_variants,
)
from testing.luke_upstream_stage_ablation import max_channel_shift_correlation, robust_sigma


RAW_ROOT = Path("/mnt/NPX/Luke/20250804/Luke0804_V2V1_g0")
PIPELINE_BASE = Path("/mnt/NPX/Luke/20250804")
BLINDED = Path("testing/outputs/luke_prospective_holdout/holdout_candidates_v2.csv")
KEY = Path("testing/outputs/luke_prospective_holdout/holdout_candidate_key_v2.csv")
DRAW_SUMMARY = Path("testing/outputs/luke_prospective_holdout/event_draw_summary_v2.json")
OUTPUT = Path("testing/outputs/luke_holdout_resampling_audit")
VARIANT_NAMES = (
    "pipeline_p1_zero_int16",
    "official_rigid_gain025_p2_extrapolate_int16",
    "official_rigid_p2_extrapolate_int16",
    "official_p2_extrapolate_float",
    "official_p2_extrapolate_int16",
)


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_events(blinded_path: Path, key_path: Path, draw_summary_path: Path) -> pd.DataFrame:
    summary = json.loads(draw_summary_path.read_text())
    if not summary.get("complete_draw"):
        raise RuntimeError("Holdout draw is incomplete")
    if sha256_file(blinded_path) != summary["blinded_candidates_sha256"]:
        raise RuntimeError("Blinded holdout hash mismatch")
    if sha256_file(key_path) != summary["candidate_key_sha256"]:
        raise RuntimeError("Holdout key hash mismatch")
    blinded = pd.read_csv(blinded_path)
    key = pd.read_csv(key_path)
    if blinded.candidate_id.duplicated().any() or key.candidate_id.duplicated().any():
        raise RuntimeError("candidate_id is not unique")
    events = blinded.merge(key, on=["candidate_id", "probe", "window_id"], validate="one_to_one")
    if len(events) != len(blinded) or len(events) != len(key):
        raise RuntimeError("Blind/key join lost candidates")
    return events.sort_values(["probe", "sample_index"]).reset_index(drop=True)


def build_conditioned(raw, probe: str):
    from spikeinterface.preprocessing import (
        blank_staturation,
        common_reference,
        filter,
        interpolate_bad_channels,
        phase_shift,
    )

    pipeline = PIPELINE_BASE / f"dredge_pipeline_results_Luke0804_V2V1_g0_{probe}"
    gain_values = np.unique(raw.get_property("gain_to_uV"))
    if len(gain_values) != 1:
        raise ValueError(f"Expected one gain for {probe}")
    gain = float(gain_values[0])
    shifted = phase_shift(raw) if np.any(raw.get_property("inter_sample_shift")) else raw
    blanked = blank_staturation(shifted, 500.0 / gain, direction="both")
    similarity, noise = np.load(pipeline / "conditioning/channel_metrics.npy")
    bad = (similarity < -0.5) | (noise > 0.3)
    bad_ids = raw.get_channel_ids()[bad]
    interpolated = interpolate_bad_channels(blanked, bad_ids)
    bandpassed = filter(
        interpolated,
        band=[300.0, 6000.0],
        btype="bandpass",
        filter_order=12,
        ftype="butter",
        direction="forward-backward",
    )
    conditioned = common_reference(
        bandpassed,
        reference="local",
        operator="median",
        local_radius=(40, 140),
    )
    return conditioned, bad_ids, gain


def polarity_anchored_metrics(
    traces: np.ndarray,
    depths_um: np.ndarray,
    fs: float,
    anchor_depth_um: float,
    polarity: str,
) -> tuple[dict, np.ndarray]:
    traces = np.asarray(traces, dtype=np.float32)
    center = traces.shape[0] // 2
    local = np.flatnonzero(np.abs(depths_um - anchor_depth_um) <= 150.0)
    search_half = int(round(0.6e-3 * fs))
    search = traces[center - search_half : center + search_half + 1][:, local]
    flat_index = int(np.argmin(search) if polarity == "negative" else np.argmax(search))
    local_time, local_channel = np.unravel_index(flat_index, search.shape)
    aligned = center - search_half + int(local_time)
    channel = int(local[local_channel])
    signed = float(search[local_time, local_channel])
    amplitude = -signed if polarity == "negative" else signed
    baseline = np.ones(len(traces), dtype=bool)
    exclusion = int(round(2e-3 * fs))
    baseline[center - exclusion : center + exclusion + 1] = False
    noise = robust_sigma(traces[baseline], axis=0)
    floor = max(float(np.median(noise[local])) * 0.1, np.finfo(float).eps)
    peak_noise = max(float(noise[channel]), floor)
    core_half = int(round(1.5e-3 * fs))
    core = traces[aligned - core_half : aligned + core_half + 1].copy()
    if core.shape[0] != 2 * core_half + 1:
        raise ValueError("Truncated event core")
    return {
        "anchor_peak_channel": channel,
        "anchor_peak_depth_um": float(depths_um[channel]),
        "anchor_peak_depth_error_um": float(depths_um[channel] - anchor_depth_um),
        "anchor_peak_amplitude_counts": max(0.0, amplitude),
        "anchor_peak_snr": max(0.0, amplitude) / peak_noise,
        "snippet_rms_counts": float(np.sqrt(np.mean(np.square(traces, dtype=np.float64)))),
        "local_snippet_rms_counts": float(np.sqrt(np.mean(np.square(traces[:, local], dtype=np.float64)))),
        "zero_fraction": float(np.mean(traces == 0)),
        "local_zero_fraction": float(np.mean(traces[:, local] == 0)),
    }, core


def audit_probe(probe: str, events: pd.DataFrame, variants: tuple[str, ...]) -> tuple[pd.DataFrame, dict]:
    import spikeinterface.extractors as se
    from spikeinterface.core.motion import Motion
    from spikeinterface.preprocessing import astype
    from spikeinterface.sortingcomponents.motion import interpolate_motion

    raw = se.read_spikeglx(folder_path=RAW_ROOT, load_sync_channel=False, stream_id=f"{probe}.ap")
    conditioned, bad_ids, gain = build_conditioned(raw, probe)
    motion_dir = PIPELINE_BASE / f"dredge_pipeline_results_Luke0804_V2V1_g0_{probe}/motion/dredge-motion"
    displacement = np.load(motion_dir / "motion.npy")
    temporal_bins = np.load(motion_dir / "time_bins.npy")
    spatial_bins = np.load(motion_dir / "depth_bins.npy")
    motion = Motion(
        displacement=displacement,
        temporal_bins_s=temporal_bins,
        spatial_bins_um=spatial_bins,
    )
    standard_variants = [name for name in variants if name in VARIANTS]
    recordings = build_variants(conditioned, motion, standard_variants)
    rigid_variants = {
        "official_rigid_gain025_p2_extrapolate_int16": 0.25,
        "official_rigid_p2_extrapolate_int16": 1.0,
    }
    for rigid_name, gain_scale in rigid_variants.items():
        if rigid_name not in variants:
            continue
        rigid_motion = Motion(
            displacement=gain_scale
            * np.nanmedian(displacement, axis=1, keepdims=True),
            temporal_bins_s=temporal_bins,
            spatial_bins_um=np.asarray([np.nanmedian(spatial_bins)]),
        )
        recordings[rigid_name] = astype(
            interpolate_motion(
                astype(conditioned, "float32"),
                rigid_motion,
                border_mode="force_extrapolate",
                spatial_interpolation_method="kriging",
                sigma_um=20.0,
                p=2,
            ),
            "int16",
        )
    depths = np.asarray(conditioned.get_channel_locations())[:, 1]
    fs = float(raw.get_sampling_frequency())
    half = int(round(5e-3 * fs))
    rows = []
    baseline_waves = {}
    selected = events[events.probe == probe]
    for variant, recording in recordings.items():
        print(f"{probe}: extracting {len(selected)} events from {variant}", flush=True)
        for event in selected.itertuples(index=False):
            traces = recording.get_traces(
                start_frame=int(event.sample_index) - half,
                end_frame=int(event.sample_index) + half + 1,
                return_scaled=False,
            )
            metrics, core = polarity_anchored_metrics(
                traces, depths, fs, float(event.depth_um), str(event.polarity)
            )
            if variant == "conditioned_baseline":
                baseline_waves[event.candidate_id] = core
                correlation = 1.0
            else:
                correlation = max_channel_shift_correlation(
                    baseline_waves[event.candidate_id], core
                )
            rows.append({
                "candidate_id": event.candidate_id,
                "probe": probe,
                "window_id": event.window_id,
                "motion_stratum": event.motion_stratum,
                "depth_third": int(event.depth_third),
                "polarity": event.polarity,
                "amplitude_stratum": event.amplitude_stratum,
                "variant": variant,
                "correlation_to_conditioned_baseline": correlation,
                **metrics,
            })
    return pd.DataFrame(rows), {
        "sampling_rate_hz": fs,
        "bad_channel_ids": [str(value) for value in bad_ids],
        "gain_uv_per_bit": gain,
        "motion_dir": str(motion_dir),
    }


def summarize(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = metrics[metrics.variant == "conditioned_baseline"].set_index(
        "candidate_id"
    )
    if baseline.index.duplicated().any():
        raise RuntimeError("Baseline candidate IDs are not unique")
    paired = metrics.copy()
    for field in (
        "anchor_peak_amplitude_counts",
        "anchor_peak_snr",
        "snippet_rms_counts",
        "local_snippet_rms_counts",
    ):
        reference = paired.candidate_id.map(baseline[field])
        paired[f"ratio_{field}"] = paired[field] / reference.replace(0, np.nan)
    reference_depth_error = paired.candidate_id.map(
        baseline["anchor_peak_depth_error_um"]
    )
    paired["delta_anchor_peak_depth_error_um"] = (
        paired.anchor_peak_depth_error_um - reference_depth_error
    )
    fields = [
        "ratio_anchor_peak_amplitude_counts",
        "ratio_anchor_peak_snr",
        "correlation_to_conditioned_baseline",
        "delta_anchor_peak_depth_error_um",
        "zero_fraction",
        "local_zero_fraction",
    ]
    groups = ["probe", "motion_stratum", "polarity", "amplitude_stratum", "variant"]
    grouped = paired.groupby(groups, observed=True)
    summary = grouped[fields].agg(["count", "median", lambda x: x.quantile(0.1)]).reset_index()
    summary.columns = ["_".join(str(v) for v in col if v).replace("_<lambda_0>", "_p10") for col in summary.columns]
    return paired, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-events-per-probe", type=int)
    parser.add_argument(
        "--variant",
        action="append",
        dest="variants",
        choices=VARIANT_NAMES,
        help="Repeat to restrict resampling branches; conditioned baseline is always included",
    )
    parser.add_argument(
        "--events-per-cell",
        type=int,
        help="Use the lowest candidate IDs within each sealed probe/window/depth/polarity/amplitude cell",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/luke-holdout-resampling-numba")
    events = load_events(BLINDED, KEY, DRAW_SUMMARY)
    if args.events_per_cell is not None:
        if args.events_per_cell < 1:
            raise ValueError("--events-per-cell must be positive")
        cell = [
            "probe",
            "window_id",
            "depth_third",
            "polarity",
            "amplitude_stratum",
        ]
        events = (
            events.sort_values("candidate_id")
            .groupby(cell, observed=True, group_keys=False)
            .head(args.events_per_cell)
        )
    if args.max_events_per_probe:
        events = events.groupby("probe", group_keys=False).head(args.max_events_per_probe)
    frames, probe_info = [], {}
    variants = tuple(args.variants or VARIANT_NAMES)
    for probe in ("imec0", "imec1"):
        frame, info = audit_probe(probe, events, variants)
        frames.append(frame)
        probe_info[probe] = info
    metrics, summary = summarize(pd.concat(frames, ignore_index=True))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "paired_event_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "paired_event_summary.csv", index=False)
    decision = {
        "events": int(metrics.candidate_id.nunique()),
        "complete_holdout": (
            args.max_events_per_probe is None and args.events_per_cell is None
        ),
        "events_per_cell": args.events_per_cell,
        "variants": list(variants),
        "automatic_motion_promotion": False,
        "promotion_rule": "No resampling branch advances to sorting until polarity-, amplitude-, motion- and probe-stratified waveform preservation is reviewed; yield is not used.",
        "sorter_outputs_accessed": False,
        "review_labels_accessed": False,
        "probe_info": probe_info,
    }
    (args.output_dir / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
