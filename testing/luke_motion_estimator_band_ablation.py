"""Paired short-window test of Luke DREDGE estimator bandwidth.

The production-like conditioning graph is split only at its terminal AP
filter: 300--3000 Hz (the historical motion branch) versus 300--6000 Hz (the
sorting branch).  DREDGE is fit independently to a deterministic half of the
peaks from each band.  Both fields are then:

1. compared with the same held-out 300--6000 Hz peak raster; and
2. applied at identical gains to the same 300--6000 Hz voltage.

This is discovery material.  It does not access the prospective event holdout
or run a sorter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from testing.luke_direct_motion_scale_audit import (
    RasterSpec,
    amplitude_edges,
    build_base_rasters,
    estimate_pair_shift,
    prepare_raster,
)
from testing.luke_interpolation_implementation_audit import anchored_event_metrics
from testing.luke_motion_scale_characterization import correlation, interpolate_field
from testing.luke_upstream_stage_ablation import max_channel_shift_correlation


LUKE_ROOT = Path("/mnt/NPX/Luke/20250804")
RAW_ROOT = LUKE_ROOT / "Luke0804_V2V1_g0"
PIPELINE_ROOT = LUKE_ROOT / "dredge_pipeline_results_Luke0804_V2V1_g0_imec1"
DEFAULT_OUTPUT = PIPELINE_ROOT / "motion_estimator_band_ablation"
STREAM_ID = "imec1.ap"


@dataclass(frozen=True)
class Window:
    name: str = "registration_outlier_first_minute"
    start_s: float = 8160.0
    duration_s: float = 60.0


WINDOW = Window()
BANDS = {"ap_300_3000": (300.0, 3000.0), "ap_300_6000": (300.0, 6000.0)}
RASTER_SPEC = RasterSpec("amp_depth_dz2_smooth10", 2.0, 10.0, True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start-s", type=float, default=WINDOW.start_s)
    parser.add_argument("--duration-s", type=float, default=WINDOW.duration_s)
    parser.add_argument("--maximum-events", type=int, default=160)
    parser.add_argument("--gains", type=float, nargs="+", default=[0.25, 1.0])
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def deterministic_half(sample_index: np.ndarray, channel_index: np.ndarray) -> np.ndarray:
    """Assign every channel from one sample time to the same split half."""
    sample = np.asarray(sample_index, dtype=np.uint64)
    # channel_index is accepted to keep the call site explicit, but must not
    # enter the hash: Luke's shared transients can produce many simultaneous
    # channel detections and must not leak across estimator/evaluation halves.
    np.broadcast_arrays(sample, np.asarray(channel_index))
    mixed = sample * np.uint64(11400714819323198485) + np.uint64(0x9E3779B97F4A7C15)
    return (mixed >> np.uint64(63)).astype(np.int8)


def array_digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        contiguous = np.ascontiguousarray(value)
        digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def build_conditioned_branches(raw):
    """Reconstruct Luke conditioning and vary only the terminal upper cutoff."""
    from spikeinterface.preprocessing import (
        blank_staturation,
        common_reference,
        filter,
        interpolate_bad_channels,
        phase_shift,
    )

    gain_values = np.unique(raw.get_property("gain_to_uV"))
    if len(gain_values) != 1:
        raise ValueError(f"Expected one AP gain, found {gain_values}")
    shifted = phase_shift(raw) if np.any(raw.get_property("inter_sample_shift")) else raw
    blanked = blank_staturation(shifted, 500.0 / float(gain_values[0]), direction="both")
    similarity, noise = np.load(PIPELINE_ROOT / "conditioning/channel_metrics.npy")
    bad = (similarity < -0.5) | (noise > 0.3)
    interpolated = interpolate_bad_channels(blanked, raw.get_channel_ids()[bad])
    branches = {}
    for name, band in BANDS.items():
        filtered = filter(
            interpolated,
            band=list(band),
            btype="bandpass",
            filter_order=12,
            ftype="butter",
            direction="forward-backward",
        )
        branches[name] = common_reference(
            filtered,
            reference="local",
            operator="median",
            local_radius=(40, 140),
        )
    return branches, raw.get_channel_ids()[bad]


def cache_spec(args: argparse.Namespace, fs: float) -> dict:
    return {
        "stream_id": STREAM_ID,
        "window": {"start_s": args.start_s, "duration_s": args.duration_s},
        "bands_hz": {name: list(value) for name, value in BANDS.items()},
        "conditioning": {
            "phase_shift": True,
            "blank_saturation_uv": 500.0,
            "bad_channel_metrics": str(PIPELINE_ROOT / "conditioning/channel_metrics.npy"),
            "local_reference_radius_um": [40, 140],
            "filter_order": 12,
            "filter_type": "butter",
            "filter_direction": "forward-backward",
        },
        "peak_detection": {
            "method": "locally_exclusive",
            "radius_um": 50.0,
            "detect_threshold": 5.0,
            "estimator_half": 0,
            "evaluation_half": 1,
            "split_unit": "sample_index_all_channels_grouped",
        },
        "peak_localization": {"method": "monopolar_triangulation"},
        "dredge": {
            "method": "dredge_ap",
            "rigid": False,
            "win_shape": "gaussian",
            "win_step_um": 200.0,
            "win_scale_um": 300.0,
            "win_margin_um": 50.0,
            "bin_um": 1.0,
            "bin_s": 1.0,
            "histogram_time_smooth_s": 1.0,
            "histogram_depth_smooth_um": 1.0,
            "time_horizon_s": 60.0,
            "max_disp_um": 80.0,
            "mincorr": 0.1,
            "device": "cuda",
        },
        "application": {
            "target_band_hz": [300.0, 6000.0],
            "gains": list(args.gains),
            "border_mode": "force_extrapolate",
            "spatial_interpolation_method": "kriging",
            "sigma_um": 20.0,
            "p": 2,
            "output_dtype": "int16",
        },
        "sampling_frequency_hz": fs,
        "maximum_events": args.maximum_events,
        "prospective_holdout_accessed": False,
        "sorter_run": False,
    }


def save_motion(target: Path, motion, peaks: np.ndarray, locations: np.ndarray) -> None:
    target.mkdir(parents=True, exist_ok=True)
    np.save(target / "peaks.npy", peaks)
    np.save(target / "peak_locations.npy", locations)
    np.save(target / "motion.npy", motion.displacement[0])
    np.save(target / "time_bins.npy", motion.temporal_bins_s[0])
    np.save(target / "depth_bins.npy", motion.spatial_bins_um)


def load_motion(target: Path):
    from spikeinterface.core.motion import Motion

    return Motion(
        np.load(target / "motion.npy"),
        np.load(target / "time_bins.npy"),
        np.load(target / "depth_bins.npy"),
    )


def estimate_fields(
    branches: dict,
    args: argparse.Namespace,
    output: Path,
    *,
    reuse_peak_cache: bool,
) -> dict:
    from spikeinterface.sortingcomponents.motion import estimate_motion
    from spikeinterface.sortingcomponents.peak_detection import detect_peaks
    from spikeinterface.sortingcomponents.peak_localization import localize_peaks

    results = {}
    for name, recording in branches.items():
        target = output / "estimators" / name
        complete = all((target / leaf).exists() for leaf in ("peaks.npy", "peak_locations.npy", "motion.npy", "time_bins.npy", "depth_bins.npy"))
        if complete and not args.force and reuse_peak_cache:
            print(f"Reusing {name} estimator", flush=True)
            results[name] = load_motion(target)
            continue
        peak_cache = reuse_peak_cache and (target / "peaks.npy").exists() and (target / "peak_locations.npy").exists()
        if peak_cache:
            print(f"Reusing {name} peak detection/localization", flush=True)
            peaks = np.load(target / "peaks.npy")
            locations = np.load(target / "peak_locations.npy")
        else:
            print(f"Detecting {name} peaks", flush=True)
            peaks = detect_peaks(
                recording,
                method="locally_exclusive",
                radius_um=50.0,
                detect_threshold=5.0,
                n_jobs=4,
                chunk_duration="1s",
                progress_bar=True,
            )
            print(f"Localizing {len(peaks):,} {name} peaks", flush=True)
            locations = localize_peaks(
                recording,
                peaks,
                method="monopolar_triangulation",
                n_jobs=4,
                chunk_duration="1s",
                progress_bar=True,
            )
        halves = deterministic_half(peaks["sample_index"], peaks["channel_index"])
        estimator = halves == 0
        print(f"Estimating {name} DREDGE from {int(np.sum(estimator)):,} peaks", flush=True)
        motion = estimate_motion(
            recording=recording,
            peaks=peaks[estimator],
            peak_locations=locations[estimator],
            direction="y",
            rigid=False,
            win_shape="gaussian",
            win_step_um=200.0,
            win_scale_um=300.0,
            win_margin_um=50.0,
            method="dredge_ap",
            extra_outputs=False,
            progress_bar=True,
            verbose=False,
            bin_um=1.0,
            bin_s=1.0,
            histogram_time_smooth_s=1.0,
            histogram_depth_smooth_um=1.0,
            time_horizon_s=60.0,
            max_disp_um=80.0,
            mincorr=0.1,
            device="cuda",
        )
        if target.exists():
            for leaf in target.iterdir():
                if leaf.is_file():
                    leaf.unlink()
        save_motion(target, motion, peaks, locations)
        results[name] = motion
    return results


def common_fields(motions: dict) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    starts = [float(value.temporal_bins_s[0][0]) for value in motions.values()]
    stops = [float(value.temporal_bins_s[0][-1]) for value in motions.values()]
    depth_starts = [float(value.spatial_bins_um[0]) for value in motions.values()]
    depth_stops = [float(value.spatial_bins_um[-1]) for value in motions.values()]
    times = np.arange(max(starts), min(stops) + 1e-9, 1.0)
    depths = np.arange(max(depth_starts), min(depth_stops) + 1e-9, 200.0)
    fields = {
        name: interpolate_field(
            value.displacement[0], value.temporal_bins_s[0], value.spatial_bins_um, times, depths
        )
        for name, value in motions.items()
    }
    return times, depths, fields


def compare_fields(motions: dict) -> pd.DataFrame:
    _, _, fields = common_fields(motions)
    names = list(fields)
    rows = []
    for name, field in fields.items():
        rigid = np.nanmedian(field, axis=1)
        residual = field - rigid[:, None]
        rows.append(
            {
                "estimator_band": name,
                "rigid_excursion_p95_p5_um": float(np.quantile(rigid, 0.95) - np.quantile(rigid, 0.05)),
                "median_nonrigid_spread_um": float(np.median(np.quantile(field, 0.95, axis=1) - np.quantile(field, 0.05, axis=1))),
                "p95_abs_displacement_um": float(np.quantile(np.abs(field), 0.95)),
                "rigid_correlation_to_other_band": correlation(rigid, np.nanmedian(fields[names[1 - names.index(name)]], axis=1)),
                "residual_correlation_to_other_band": correlation(residual, fields[names[1 - names.index(name)]] - np.nanmedian(fields[names[1 - names.index(name)]], axis=1)[:, None]),
            }
        )
    return pd.DataFrame(rows)


def raster_scale_metrics(
    motions: dict,
    wide_target: Path,
    fs: float,
    duration_s: float,
    recording_t_start_s: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    peaks = np.load(wide_target / "peaks.npy")
    locations = np.load(wide_target / "peak_locations.npy")
    evaluation = deterministic_half(peaks["sample_index"], peaks["channel_index"]) == 1
    peaks, locations = peaks[evaluation], locations[evaluation]
    edges = amplitude_edges(peaks["amplitude"])
    base = build_base_rasters(
        peaks,
        locations,
        time_bin_s=5.0,
        duration_s=duration_s,
        depth_bin_um=2.0,
        depth_stop_um=3840.0,
        amplitude_bin_edges=edges,
    )[0]
    raster = prepare_raster(base, RASTER_SPEC, 2.0)
    centers = np.arange(raster.shape[0]) * 5.0 + 2.5
    pairs = [(i, i + 4) for i in range(len(centers) - 4)]
    observed = []
    quality = []
    for first, second in pairs:
        value = estimate_pair_shift(
            raster[first], raster[second], depth_bin_um=2.0, maximum_shift_um=60.0
        )
        observed.append(value["observed_shift_um"])
        quality.append(
            (not value["hit_search_boundary"])
            and value["peak_score"] >= 0.5
            and value["score_margin_vs_distant_peak"] >= 0.002
        )
    rows = []
    pair_rows = []
    observed = np.asarray(observed)
    quality = np.asarray(quality, dtype=bool)
    for name, motion in motions.items():
        rigid = np.nanmedian(motion.displacement[0], axis=1)
        relative_bins = motion.temporal_bins_s[0] - recording_t_start_s
        predicted = np.interp(centers, relative_bins, rigid)
        predicted = np.asarray([predicted[b] - predicted[a] for a, b in pairs])
        usable = quality & np.isfinite(observed) & np.isfinite(predicted)
        for pair_index, ((first, second), obs, pred, keep) in enumerate(
            zip(pairs, observed, predicted, usable)
        ):
            pair_rows.append(
                {
                    "estimator_band": name,
                    "pair_index": pair_index,
                    "first_time_s": centers[first],
                    "second_time_s": centers[second],
                    "observed_shift_um": obs,
                    "predicted_shift_um": pred,
                    "qualified": bool(keep),
                }
            )
        for gain in (0.25, 1.0):
            error = np.abs(observed[usable] - gain * predicted[usable])
            rows.append(
                {
                    "estimator_band": name,
                    "gain": gain,
                    "evaluation_peaks": int(len(peaks)),
                    "candidate_pairs": len(pairs),
                    "qualified_pairs": int(np.sum(usable)),
                    "median_absolute_raster_error_um": float(np.median(error)) if len(error) else np.nan,
                    "mean_absolute_raster_error_um": float(np.mean(error)) if len(error) else np.nan,
                    "predicted_observed_correlation": correlation(predicted[usable], observed[usable]) if np.sum(usable) >= 2 else np.nan,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(pair_rows)


def select_separated_indices(
    sample_index: np.ndarray,
    amplitude: np.ndarray,
    maximum_events: int,
    minimum_separation_samples: int,
) -> np.ndarray:
    """Greedily retain strongest events separated in time."""
    order = np.argsort(np.abs(amplitude), kind="stable")[::-1]
    selected: list[int] = []
    selected_samples: list[int] = []
    for index in order:
        sample = int(sample_index[index])
        if any(abs(sample - other) < minimum_separation_samples for other in selected_samples):
            continue
        selected.append(int(index))
        selected_samples.append(sample)
        if len(selected) >= maximum_events:
            break
    return np.asarray(selected, dtype=int)


def choose_evaluation_events(
    target: Path,
    maximum_events: int,
    duration_samples: int,
    minimum_separation_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    peaks = np.load(target / "peaks.npy")
    locations = np.load(target / "peak_locations.npy")
    mask = deterministic_half(peaks["sample_index"], peaks["channel_index"]) == 1
    mask &= (peaks["sample_index"] >= 300) & (peaks["sample_index"] < duration_samples - 300)
    indices = np.flatnonzero(mask)
    if len(indices) > maximum_events:
        keep = select_separated_indices(
            peaks["sample_index"][indices],
            peaks["amplitude"][indices],
            maximum_events,
            minimum_separation_samples,
        )
        indices = np.sort(indices[keep])
    return peaks[indices], locations[indices]


def voltage_metrics(wide, motions: dict, target: Path, fs: float, gains: list[float], maximum_events: int) -> pd.DataFrame:
    from spikeinterface.core.motion import Motion
    from spikeinterface.preprocessing import astype
    from spikeinterface.sortingcomponents.motion import interpolate_motion

    recordings = {"no_motion": wide}
    for name, motion in motions.items():
        for gain in gains:
            scaled = Motion(
                motion.displacement[0] * gain,
                motion.temporal_bins_s[0],
                motion.spatial_bins_um,
            )
            recordings[f"{name}_g{gain:g}"] = astype(
                interpolate_motion(
                    astype(wide, "float32"),
                    scaled,
                    border_mode="force_extrapolate",
                    spatial_interpolation_method="kriging",
                    sigma_um=20.0,
                    p=2,
                ),
                "int16",
            )
    peaks, locations = choose_evaluation_events(
        target,
        maximum_events,
        wide.get_num_samples(),
        max(1, int(round(1e-3 * fs))),
    )
    depths = np.asarray(wide.get_channel_locations())[:, 1]
    half = int(round(5e-3 * fs))
    baseline_waves = {}
    rows = []
    for variant, recording in recordings.items():
        print(f"Scoring {len(peaks)} held-out events from {variant}", flush=True)
        for index, (peak, location) in enumerate(zip(peaks, locations)):
            sample = int(peak["sample_index"])
            traces = recording.get_traces(start_frame=sample - half, end_frame=sample + half + 1)
            metrics, wave = anchored_event_metrics(traces, depths, fs, float(location["y"]))
            if variant == "no_motion":
                baseline_waves[index] = wave
                waveform_correlation = 1.0
                baseline_peak = metrics["anchor_peak_amplitude_counts"]
            else:
                waveform_correlation = max_channel_shift_correlation(baseline_waves[index], wave)
                baseline_peak = rows[index]["anchor_peak_amplitude_counts"]
            rows.append(
                {
                    "variant": variant,
                    "event_index": index,
                    "sample_index": sample,
                    "baseline_depth_um": float(location["y"]),
                    "baseline_amplitude": float(peak["amplitude"]),
                    "waveform_correlation_to_no_motion": waveform_correlation,
                    "peak_amplitude_ratio_to_no_motion": metrics["anchor_peak_amplitude_counts"] / baseline_peak if baseline_peak else np.nan,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def summarize_voltage(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby("variant", sort=False)
        .agg(
            events=("event_index", "size"),
            median_peak_amplitude_ratio=("peak_amplitude_ratio_to_no_motion", "median"),
            p10_peak_amplitude_ratio=("peak_amplitude_ratio_to_no_motion", lambda x: x.quantile(0.1)),
            median_waveform_correlation=("waveform_correlation_to_no_motion", "median"),
            p10_waveform_correlation=("waveform_correlation_to_no_motion", lambda x: x.quantile(0.1)),
            median_anchor_depth_error_um=("anchor_peak_depth_error_um", "median"),
            median_local_zero_fraction=("local_zero_fraction", "median"),
        )
        .reset_index()
    )


def paired_band_voltage(metrics: pd.DataFrame, seed: int = 20250804) -> pd.DataFrame:
    """Paired wide-minus-narrow effects with event bootstrap intervals."""
    rng = np.random.default_rng(seed)
    rows = []
    gains = sorted(
        {
            float(value.rsplit("g", 1)[1])
            for value in metrics.variant.unique()
            if value.startswith("ap_300_")
        }
    )
    for gain in gains:
        narrow = metrics[metrics.variant == f"ap_300_3000_g{gain:g}"].set_index("event_index")
        wide = metrics[metrics.variant == f"ap_300_6000_g{gain:g}"].set_index("event_index")
        common = narrow.index.intersection(wide.index)
        comparisons = {
            "waveform_correlation_wide_minus_narrow": (
                wide.loc[common, "waveform_correlation_to_no_motion"].to_numpy()
                - narrow.loc[common, "waveform_correlation_to_no_motion"].to_numpy()
            ),
            "absolute_peak_error_wide_minus_narrow": (
                np.abs(wide.loc[common, "peak_amplitude_ratio_to_no_motion"].to_numpy() - 1.0)
                - np.abs(narrow.loc[common, "peak_amplitude_ratio_to_no_motion"].to_numpy() - 1.0)
            ),
        }
        for metric, values in comparisons.items():
            values = values[np.isfinite(values)]
            boot = np.empty(2000, dtype=float)
            for index in range(len(boot)):
                boot[index] = np.median(values[rng.integers(0, len(values), len(values))])
            rows.append(
                {
                    "gain": gain,
                    "metric": metric,
                    "events": len(values),
                    "median_paired_difference": float(np.median(values)),
                    "mean_paired_difference": float(np.mean(values)),
                    "ci95_low": float(np.quantile(boot, 0.025)),
                    "ci95_high": float(np.quantile(boot, 0.975)),
                }
            )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> dict:
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/luke-band-ablation-numba")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/luke-band-ablation-mpl")
    import spikeinterface.extractors as se

    raw = se.read_spikeglx(folder_path=RAW_ROOT, load_sync_channel=False, stream_id=STREAM_ID)
    fs = float(raw.get_sampling_frequency())
    spec = cache_spec(args, fs)
    plan = {**spec, "output_dir": str(args.output_dir)}
    if args.plan_only and not args.run:
        print(json.dumps(plan, indent=2))
        return plan
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    same_spec = False
    if manifest_path.exists() and not args.force:
        saved = json.loads(manifest_path.read_text())
        if saved != spec:
            raise RuntimeError("Output manifest differs; choose a new output directory or use --force")
        same_spec = True
    elif manifest_path.exists():
        same_spec = json.loads(manifest_path.read_text()) == spec
    manifest_path.write_text(json.dumps(spec, indent=2) + "\n")
    branches, bad_ids = build_conditioned_branches(raw)
    start = int(round(args.start_s * fs))
    stop = int(round((args.start_s + args.duration_s) * fs))
    windowed = {name: value.frame_slice(start_frame=start, end_frame=stop) for name, value in branches.items()}
    motions = estimate_fields(
        windowed,
        args,
        args.output_dir,
        reuse_peak_cache=same_spec,
    )
    field_summary = compare_fields(motions)
    raster_summary, raster_pairs = raster_scale_metrics(
        motions,
        args.output_dir / "estimators/ap_300_6000",
        fs,
        args.duration_s,
        float(windowed["ap_300_6000"].get_time_info()["t_start"] or 0.0),
    )
    voltage = voltage_metrics(
        windowed["ap_300_6000"],
        motions,
        args.output_dir / "estimators/ap_300_6000",
        fs,
        list(args.gains),
        args.maximum_events,
    )
    voltage_summary = summarize_voltage(voltage)
    paired_voltage = paired_band_voltage(voltage)
    field_summary.to_csv(args.output_dir / "field_summary.csv", index=False)
    raster_summary.to_csv(args.output_dir / "heldout_raster_summary.csv", index=False)
    raster_pairs.to_csv(args.output_dir / "heldout_raster_pairs.csv", index=False)
    voltage.to_csv(args.output_dir / "heldout_event_voltage_metrics.csv", index=False)
    voltage_summary.to_csv(args.output_dir / "heldout_event_voltage_summary.csv", index=False)
    paired_voltage.to_csv(args.output_dir / "paired_band_voltage_comparison.csv", index=False)
    result = {
        **plan,
        "bad_channel_ids": [str(value) for value in bad_ids],
        "detected_peaks": {
            name: int(len(np.load(args.output_dir / "estimators" / name / "peaks.npy", mmap_mode="r")))
            for name in BANDS
        },
        "files": [
            "field_summary.csv",
            "heldout_raster_summary.csv",
            "heldout_raster_pairs.csv",
            "heldout_event_voltage_metrics.csv",
            "heldout_event_voltage_summary.csv",
            "paired_band_voltage_comparison.csv",
        ],
        "interpretation_guardrail": (
            "The held-out raster is independent of each estimator half but remains AP peak based; "
            "voltage metrics quantify resampling damage, not biological spike recall."
        ),
    }
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print("\nField summary\n", field_summary.to_string(index=False))
    print("\nHeld-out raster summary\n", raster_summary.to_string(index=False))
    print("\nHeld-out voltage summary\n", voltage_summary.to_string(index=False))
    print("\nPaired estimator-band voltage comparison\n", paired_voltage.to_string(index=False))
    return result


def main() -> None:
    args = parse_args()
    if not args.plan_only and not args.run:
        raise SystemExit("Choose --plan-only or --run")
    run(args)


if __name__ == "__main__":
    main()
