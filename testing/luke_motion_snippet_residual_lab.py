"""Rapid cross-validated residual screen for Luke motion candidates.

This is a discovery-only, sorter-free laboratory.  It reuses provisional unit
labels and spike times from the completed no-motion 96-channel depth-strip
sort, but never treats those labels as biological ground truth.  For each
candidate motion field it extracts a few previously used 2-second snippets,
applies the same Kilosort CAR/high-pass approximation, learns unit templates on
one half of the snippets, and scores held-out events in the other half.  The
two folds are then reversed.

The primary endpoint is cross-validated, noise-scaled local residual energy.
Candidate-specific templates may adapt to a deterministic transform, but may
not see the event or snippet they are scored on.  The remaining 648 sealed
holdout events are not opened by this program.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from testing.luke_conditioning_stage_audit import ks_highpass, robust_sigma
from testing.luke_rigid025_depth_strip import relative_motion_bins


ROOT = Path(
    "/media/huklab/Data/NPX/Ryansorting/Luke/"
    "Luke0804_two_axis_pilot_imec1"
)
SOURCE = ROOT / "recordings/core_depth_strip"
SORTER = ROOT / "sorts/core_depth_strip/single_ks_preprocessing_claim_off/sorter_output"
MOTION_DIR = Path(
    "/mnt/NPX/Luke/20250804/"
    "dredge_pipeline_results_Luke0804_V2V1_g0_imec1/motion/dredge-motion"
)
CONTINUITY = Path(
    "testing/outputs/luke_full_strip_diagnostic_audit/unit_continuity_metrics.csv"
)
OUTPUT = Path("testing/outputs/luke_motion_snippet_residual_lab")


@dataclass(frozen=True)
class Snippet:
    name: str
    start_s: float
    duration_s: float
    motion_class: str


# All intervals were used before the prospective holdout was drawn.  They are
# discovery material and deliberately avoid the six sealed holdout epochs.
SNIPPETS = (
    Snippet("quiet_3951", 3951.0, 2.0, "quiet"),
    Snippet("good_7095", 7095.0, 2.0, "quiet"),
    Snippet("neutral_7215", 7215.0, 2.0, "intermediate"),
    Snippet("shared_7274", 7274.0, 2.0, "high_motion"),
    Snippet("calibration_7800", 7800.0, 2.0, "intermediate"),
    Snippet("pathological_8160", 8160.0, 2.0, "high_motion"),
    Snippet("pathological_8218", 8218.0, 2.0, "high_motion"),
    Snippet("calibration_8398", 8398.0, 2.0, "intermediate"),
)


@dataclass(frozen=True)
class Variant:
    name: str
    rigid_gain: float
    residual_gain: float
    sigma_um: float


VARIANTS = (
    Variant("no_motion", 0.0, 0.0, 20.0),
    Variant("rigid025_p2_sigma20", 0.25, 0.0, 20.0),
    Variant("rigid025_nr010_p2_sigma20", 0.25, 0.10, 20.0),
    Variant("rigid025_nr025_p2_sigma20", 0.25, 0.25, 20.0),
    Variant("rigid025_nr010_p2_sigma10", 0.25, 0.10, 10.0),
    Variant("rigid025_nr025_p2_sigma10", 0.25, 0.25, 10.0),
)


def decompose_motion(displacement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a time-depth field into its depth median and residual."""
    values = np.asarray(displacement, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).any():
        raise ValueError("displacement must be a finite time-by-depth array")
    rigid = np.nanmedian(values, axis=1, keepdims=True)
    return rigid, values - rigid


def variant_displacement(displacement: np.ndarray, variant: Variant) -> np.ndarray:
    rigid, residual = decompose_motion(displacement)
    return variant.rigid_gain * rigid + variant.residual_gain * residual


def choose_units(
    continuity: pd.DataFrame,
    spike_times: np.ndarray,
    spike_clusters: np.ndarray,
    fs: float,
    snippets: tuple[Snippet, ...],
    *,
    spike_detection_templates: np.ndarray | None = None,
    maximum_units: int,
    minimum_events_per_fold: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Choose stable provisional units represented in both snippet folds."""
    eligible = continuity.loc[
        continuity.ks_good.astype(bool)
        & (continuity.presence_fraction_300s >= 0.75)
        & (continuity.first_last_pc_cosine >= 0.8)
        & (continuity.edge_spike_fraction <= 0.25)
    ].copy()
    rows = []
    for snippet_index, snippet in enumerate(snippets):
        lo = int(round(snippet.start_s * fs))
        hi = int(round((snippet.start_s + snippet.duration_s) * fs))
        keep = (spike_times >= lo) & (spike_times < hi)
        detections = (
            np.full(int(np.sum(keep)), -1, dtype=int)
            if spike_detection_templates is None
            else np.asarray(spike_detection_templates[keep], dtype=int)
        )
        for sample, unit, detection_template in zip(
            spike_times[keep], spike_clusters[keep], detections
        ):
            rows.append(
                {
                    "snippet_index": snippet_index,
                    "snippet": snippet.name,
                    "motion_class": snippet.motion_class,
                    "fold": snippet_index % 2,
                    "sample_index": int(sample),
                    "unit_id": int(unit),
                    "detection_template": int(detection_template),
                }
            )
    events = pd.DataFrame(rows)
    if events.empty:
        raise RuntimeError("No sorted spikes occur in the requested snippets")
    tolerance = int(round(0.5e-3 * fs))
    isolated_parts = []
    for _, snippet_events in events.groupby("snippet_index", sort=False):
        part = snippet_events.sort_values("sample_index").copy()
        times = part.sample_index.to_numpy()
        units = part.unit_id.to_numpy()
        isolated = np.ones(len(part), dtype=bool)
        left = 0
        for index, sample in enumerate(times):
            while times[left] < sample - tolerance:
                left += 1
            right = np.searchsorted(times, sample + tolerance, side="right")
            isolated[index] = not np.any(units[left:right] != units[index])
        part["cross_unit_isolated"] = isolated
        isolated_parts.append(part)
    events = pd.concat(isolated_parts, ignore_index=True)
    events = events[events.cross_unit_isolated]
    events = events[events.unit_id.isin(eligible.unit_id)]
    if spike_detection_templates is not None:
        dominant = eligible.set_index("unit_id")["dominant_detection_template"]
        events = events[
            events.detection_template
            == events.unit_id.map(dominant).astype(int)
        ]
    counts = (
        events.groupby(["unit_id", "fold"]).size().unstack(fill_value=0)
        .reindex(columns=[0, 1], fill_value=0)
    )
    supported = counts.index[(counts[0] >= minimum_events_per_fold) & (counts[1] >= minimum_events_per_fold)]
    ranking = (
        eligible[eligible.unit_id.isin(supported)]
        .assign(
            snippet_spikes=lambda x: x.unit_id.map(events.unit_id.value_counts()),
            selection_amplitude=lambda x: (
                x["median_amplitude"]
                if "median_amplitude" in x
                else np.zeros(len(x), dtype=float)
            ),
        )
        .sort_values(
            ["selection_amplitude", "first_last_pc_cosine", "snippet_spikes", "unit_id"],
            ascending=[False, False, False, True],
        )
        .head(maximum_units)
    )
    if ranking.empty:
        raise RuntimeError("No stable unit has enough events in both snippet folds")
    chosen = events[events.unit_id.isin(ranking.unit_id)].copy()
    # Bound dense units without changing snippet/fold coverage.
    chosen = (
        chosen.sort_values(["unit_id", "snippet_index", "sample_index"])
        .groupby(["unit_id", "snippet_index"], group_keys=False)
        .head(24)
    )
    return ranking, chosen


def best_scaled_residual(
    waveform: np.ndarray, template: np.ndarray, maximum_shift: int = 2
) -> dict[str, float | int]:
    """Fit one nonnegative, temporally shifted template to a held-out event."""
    observed = np.asarray(waveform, dtype=float)
    model = np.asarray(template, dtype=float)
    if observed.shape != model.shape or observed.ndim != 2:
        raise ValueError("waveform and template must be matching 2-D arrays")
    denominator = float(np.sum(observed**2))
    if denominator <= 0:
        return {
            "residual_fraction": np.nan,
            "template_cosine": np.nan,
            "coefficient": np.nan,
            "time_shift_samples": 0,
            "peak_residual": np.nan,
        }
    candidates = []
    for shift in range(-maximum_shift, maximum_shift + 1):
        shifted = np.zeros_like(model)
        if shift < 0:
            shifted[:shift] = model[-shift:]
        elif shift > 0:
            shifted[shift:] = model[:-shift]
        else:
            shifted[:] = model
        vector = shifted.ravel()
        norm = float(vector @ vector)
        coefficient = max(0.0, float(vector @ observed.ravel()) / norm) if norm else 0.0
        residual = observed - coefficient * shifted
        residual_fraction = float(np.sum(residual**2) / denominator)
        cosine_denominator = np.linalg.norm(vector) * np.linalg.norm(observed.ravel())
        cosine = float(vector @ observed.ravel() / cosine_denominator) if cosine_denominator else np.nan
        candidates.append((residual_fraction, -cosine, shift, coefficient, residual))
    residual_fraction, negative_cosine, shift, coefficient, residual = min(
        candidates, key=lambda item: (item[0], item[1], abs(item[2]))
    )
    return {
        "residual_fraction": residual_fraction,
        "template_cosine": -negative_cosine,
        "coefficient": coefficient,
        "time_shift_samples": int(shift),
        "peak_residual": float(np.max(np.abs(residual))),
    }


def ks_center_car_lower_median(values: np.ndarray) -> np.ndarray:
    """Match Kilosort's channel centering and Torch lower-median CAR."""
    traces = np.asarray(values, dtype=np.float32)
    centered = traces - traces.mean(axis=0, keepdims=True)
    lower_middle = (centered.shape[1] - 1) // 2
    channel_median = np.partition(centered, lower_middle, axis=1)[:, lower_middle]
    return centered - channel_median[:, None]


def qualify_against_saved_template(
    events: pd.DataFrame,
    waves: dict[tuple[int, int], np.ndarray],
    templates: np.ndarray,
    locations: np.ndarray,
    noise_scale: np.ndarray,
    minimum_cosine: float,
    minimum_events_per_fold: int,
    radius_um: float = 160.0,
) -> pd.DataFrame:
    """Conservatively retain events resembling their saved no-motion template.

    This is intentionally baseline-favoring and is only an event-quality gate;
    candidate templates remain cross-validated on disjoint snippets.
    """
    rows = []
    for row in events.itertuples(index=False):
        unit_id = int(row.unit_id)
        key = (unit_id, int(row.sample_index))
        if key not in waves:
            continue
        saved = np.asarray(templates[unit_id])
        peak = np.unravel_index(np.argmax(np.abs(saved)), saved.shape)[1]
        local = np.flatnonzero(np.abs(locations[:, 1] - locations[peak, 1]) <= radius_um)
        metrics = best_scaled_residual(
            waves[key][:, local] / noise_scale[local],
            saved[:, local] / noise_scale[local],
        )
        rows.append(
            {
                "unit_id": unit_id,
                "sample_index": int(row.sample_index),
                "saved_template_cosine": metrics["template_cosine"],
                "saved_template_residual_fraction": metrics["residual_fraction"],
            }
        )
    qualified = events.merge(
        pd.DataFrame(rows), on=["unit_id", "sample_index"], validate="one_to_one"
    )
    qualified = qualified[qualified.saved_template_cosine >= minimum_cosine]
    counts = qualified.groupby(["unit_id", "fold"]).size().unstack(fill_value=0).reindex(columns=[0, 1], fill_value=0)
    supported = counts.index[(counts[0] >= minimum_events_per_fold) & (counts[1] >= minimum_events_per_fold)]
    return qualified[qualified.unit_id.isin(supported)].copy()


def qualify_coherent_waveform_families(
    events: pd.DataFrame,
    waves: dict[tuple[int, int], np.ndarray],
    templates: np.ndarray,
    locations: np.ndarray,
    noise_scale: np.ndarray,
    minimum_family_cosine: float,
    minimum_peak_snr: float,
    minimum_events_per_fold: int,
    radius_um: float = 160.0,
) -> pd.DataFrame:
    """Keep a direct cosine neighborhood around each unit's baseline medoid."""
    retained = []
    for unit_id, group in events.groupby("unit_id"):
        saved = np.asarray(templates[int(unit_id)])
        peak = np.unravel_index(np.argmax(np.abs(saved)), saved.shape)[1]
        local = np.flatnonzero(np.abs(locations[:, 1] - locations[peak, 1]) <= radius_um)
        available = []
        for row in group.itertuples(index=False):
            key = (int(unit_id), int(row.sample_index))
            if key not in waves:
                continue
            waveform = waves[key][:, local] / noise_scale[local]
            peak_snr = float(np.max(np.abs(waveform)))
            if peak_snr >= minimum_peak_snr:
                available.append((row, waveform, peak_snr))
        if len(available) < 2 * minimum_events_per_fold:
            continue
        similarities = np.eye(len(available), dtype=float)
        for first in range(len(available)):
            for second in range(first + 1, len(available)):
                forward = best_scaled_residual(
                    available[first][1], available[second][1]
                )["template_cosine"]
                reverse = best_scaled_residual(
                    available[second][1], available[first][1]
                )["template_cosine"]
                value = float(max(forward, reverse))
                similarities[first, second] = similarities[second, first] = value
        neighbor_counts = np.sum(similarities >= minimum_family_cosine, axis=1)
        medoid = int(np.argmax(neighbor_counts))
        for index, (row, _, peak_snr) in enumerate(available):
            if similarities[medoid, index] < minimum_family_cosine:
                continue
            retained.append(
                {
                    **row._asdict(),
                    "baseline_family_medoid_sample": int(
                        available[medoid][0].sample_index
                    ),
                    "baseline_family_cosine": float(similarities[medoid, index]),
                    "baseline_peak_snr": peak_snr,
                }
            )
    if not retained:
        return events.iloc[0:0].copy()
    qualified = pd.DataFrame(retained)
    counts = qualified.groupby(["unit_id", "fold"]).size().unstack(fill_value=0).reindex(columns=[0, 1], fill_value=0)
    supported = counts.index[(counts[0] >= minimum_events_per_fold) & (counts[1] >= minimum_events_per_fold)]
    return qualified[qualified.unit_id.isin(supported)].copy()


def build_recordings(source, variants: tuple[Variant, ...]):
    from spikeinterface.core.motion import Motion
    from spikeinterface.sortingcomponents.motion import interpolate_motion

    displacement = np.load(MOTION_DIR / "motion.npy")
    temporal_bins = np.load(MOTION_DIR / "time_bins.npy")
    spatial_bins = np.load(MOTION_DIR / "depth_bins.npy")
    relative_bins, acquisition_start_s = relative_motion_bins(temporal_bins)
    recordings = {"no_motion": source}
    for variant in variants:
        if variant.name == "no_motion":
            continue
        motion = Motion(
            variant_displacement(displacement, variant),
            relative_bins,
            spatial_bins,
        )
        recordings[variant.name] = interpolate_motion(
            source.astype("float32"),
            motion,
            border_mode="force_extrapolate",
            spatial_interpolation_method="kriging",
            sigma_um=variant.sigma_um,
            p=2,
        ).astype("int16")
    return recordings, acquisition_start_s


def extract_processed_snippets(recording, snippets: tuple[Snippet, ...], fs: float) -> dict[int, np.ndarray]:
    padding = int(round(0.10 * fs))
    result = {}
    for index, snippet in enumerate(snippets):
        lo = int(round(snippet.start_s * fs))
        hi = int(round((snippet.start_s + snippet.duration_s) * fs))
        traces = recording.get_traces(
            start_frame=lo - padding,
            end_frame=hi + padding,
            return_scaled=False,
        ).astype(np.float32)
        processed = ks_highpass(ks_center_car_lower_median(traces), fs)
        result[index] = processed[padding:-padding]
    return result


def event_waveforms(
    processed: dict[int, np.ndarray],
    events: pd.DataFrame,
    snippets: tuple[Snippet, ...],
    fs: float,
    nt: int,
    nt0min: int,
) -> dict[tuple[int, int], np.ndarray]:
    waves = {}
    for row in events.itertuples(index=False):
        snippet = snippets[int(row.snippet_index)]
        local = int(row.sample_index - round(snippet.start_s * fs))
        start = local - nt0min
        stop = start + nt
        waveform = processed[int(row.snippet_index)][start:stop]
        if waveform.shape[0] == nt:
            waves[(int(row.unit_id), int(row.sample_index))] = waveform
    return waves


def score_variant(
    variant: str,
    waves: dict[tuple[int, int], np.ndarray],
    events: pd.DataFrame,
    templates: np.ndarray,
    locations: np.ndarray,
    noise_scale: np.ndarray,
    radius_um: float = 160.0,
) -> pd.DataFrame:
    rows = []
    for unit_id, group in events.groupby("unit_id"):
        peak = np.unravel_index(np.argmax(np.abs(templates[int(unit_id)])), templates[int(unit_id)].shape)[1]
        local_channels = np.flatnonzero(np.abs(locations[:, 1] - locations[peak, 1]) <= radius_um)
        for score_fold in (0, 1):
            training = group[group.fold != score_fold]
            testing = group[group.fold == score_fold]
            train_waves = [
                waves[(int(unit_id), int(row.sample_index))][:, local_channels]
                / noise_scale[local_channels]
                for row in training.itertuples(index=False)
                if (int(unit_id), int(row.sample_index)) in waves
            ]
            if not train_waves:
                continue
            template = np.median(np.asarray(train_waves), axis=0)
            for row in testing.itertuples(index=False):
                key = (int(unit_id), int(row.sample_index))
                if key not in waves:
                    continue
                waveform = waves[key][:, local_channels] / noise_scale[local_channels]
                metrics = best_scaled_residual(waveform, template)
                rows.append(
                    {
                        "variant": variant,
                        "unit_id": int(unit_id),
                        "sample_index": int(row.sample_index),
                        "snippet": row.snippet,
                        "motion_class": row.motion_class,
                        "fold": int(row.fold),
                        "n_training_events": len(train_waves),
                        "n_local_channels": len(local_channels),
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def summarize(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = scores[scores.variant == "no_motion"].set_index(["unit_id", "sample_index"])
    paired = scores.copy()
    keys = pd.MultiIndex.from_frame(paired[["unit_id", "sample_index"]])
    paired["delta_residual_vs_no_motion"] = paired.residual_fraction.to_numpy() - baseline.residual_fraction.reindex(keys).to_numpy()
    paired["delta_cosine_vs_no_motion"] = paired.template_cosine.to_numpy() - baseline.template_cosine.reindex(keys).to_numpy()
    summary = (
        paired.groupby(["variant", "motion_class"], observed=True)
        .agg(
            events=("sample_index", "size"),
            units=("unit_id", "nunique"),
            median_residual_fraction=("residual_fraction", "median"),
            median_delta_residual_vs_no_motion=("delta_residual_vs_no_motion", "median"),
            p90_residual_fraction=("residual_fraction", lambda x: x.quantile(0.9)),
            median_template_cosine=("template_cosine", "median"),
            median_delta_cosine_vs_no_motion=("delta_cosine_vs_no_motion", "median"),
            median_peak_residual=("peak_residual", "median"),
        )
        .reset_index()
    )
    return paired, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--maximum-units", type=int, default=12)
    parser.add_argument("--minimum-events-per-fold", type=int, default=3)
    parser.add_argument("--minimum-saved-template-cosine", type=float, default=-1.0)
    parser.add_argument("--minimum-family-cosine", type=float, default=0.6)
    parser.add_argument("--minimum-peak-snr", type=float, default=4.0)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict:
    import spikeinterface.core as sc

    source = sc.load(SOURCE)
    fs = float(source.get_sampling_frequency())
    spike_times = np.load(SORTER / "spike_times.npy", mmap_mode="r").reshape(-1)
    spike_clusters = np.load(SORTER / "spike_clusters.npy", mmap_mode="r").reshape(-1)
    spike_detection_templates = np.load(
        SORTER / "spike_detection_templates.npy", mmap_mode="r"
    ).reshape(-1)
    continuity = pd.read_csv(CONTINUITY)
    # Smoke restricts the variant count, not temporal coverage.  Keeping both
    # folds represented across all eight discovery snippets favors fewer,
    # higher-amplitude units over one high-rate marginal cluster.
    snippets = SNIPPETS
    variants = VARIANTS[:3] if args.smoke else VARIANTS
    units, events = choose_units(
        continuity,
        spike_times,
        spike_clusters,
        fs,
        snippets,
        spike_detection_templates=spike_detection_templates,
        maximum_units=min(args.maximum_units, 4) if args.smoke else args.maximum_units,
        minimum_events_per_fold=2 if args.smoke else args.minimum_events_per_fold,
    )
    plan = {
        "status": "discovery_only_provisional_unit_identity",
        "snippets": [snippet.__dict__ for snippet in snippets],
        "variants": [variant.__dict__ for variant in variants],
        "selected_units": units.unit_id.astype(int).tolist(),
        "selected_events": int(len(events)),
        "remaining_confirmatory_holdout_events_accessed": False,
        "sorter_run": False,
    }
    if args.plan_only and not args.run:
        print(json.dumps(plan, indent=2))
        return plan

    recordings, acquisition_start_s = build_recordings(source, variants)
    templates = np.load(SORTER / "templates.npy", mmap_mode="r")
    locations = np.asarray(source.get_channel_locations())
    ops = np.load(SORTER / "ops.npy", allow_pickle=True).item()
    nt, nt0min = int(ops["nt"]), int(ops["nt0min"])
    all_scores = []
    baseline_processed = extract_processed_snippets(recordings["no_motion"], snippets, fs)
    baseline_stack = np.concatenate([values[::30] for values in baseline_processed.values()])
    noise_scale = robust_sigma(baseline_stack, axis=0)
    noise_scale = np.maximum(noise_scale, np.median(noise_scale) * 0.1)
    baseline_waves = event_waveforms(
        baseline_processed, events, snippets, fs, nt, nt0min
    )
    events_before_qualification = len(events)
    events = qualify_against_saved_template(
        events,
        baseline_waves,
        templates,
        locations,
        noise_scale,
        args.minimum_saved_template_cosine,
        1 if args.smoke else args.minimum_events_per_fold,
    )
    events = qualify_coherent_waveform_families(
        events,
        baseline_waves,
        templates,
        locations,
        noise_scale,
        args.minimum_family_cosine,
        args.minimum_peak_snr,
        1 if args.smoke else args.minimum_events_per_fold,
    )
    if events.empty:
        raise RuntimeError(
            "No coherent events remain after baseline qualification; inspect alignment or lower the explicit diagnostic threshold"
        )
    for variant in variants:
        print(f"Scoring {variant.name}", flush=True)
        processed = baseline_processed if variant.name == "no_motion" else extract_processed_snippets(recordings[variant.name], snippets, fs)
        waves = (
            baseline_waves
            if variant.name == "no_motion"
            else event_waveforms(processed, events, snippets, fs, nt, nt0min)
        )
        all_scores.append(
            score_variant(
                variant.name,
                waves,
                events,
                templates,
                locations,
                noise_scale,
            )
        )
    paired, summary = summarize(pd.concat(all_scores, ignore_index=True))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    units.to_csv(args.output_dir / "selected_units.csv", index=False)
    events.to_csv(args.output_dir / "selected_events.csv", index=False)
    paired.to_csv(args.output_dir / "event_residual_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "residual_summary.csv", index=False)
    result = {
        **plan,
        "inferred_acquisition_start_s": acquisition_start_s,
        "events_before_saved_template_qualification": int(events_before_qualification),
        "saved_template_cosine_threshold": args.minimum_saved_template_cosine,
        "baseline_family_cosine_threshold": args.minimum_family_cosine,
        "baseline_peak_snr_threshold": args.minimum_peak_snr,
        "qualified_events": int(len(events)),
        "events_scored_per_variant": int(paired.groupby("variant").size().min()),
        "files": [
            "selected_units.csv",
            "selected_events.csv",
            "event_residual_metrics.csv",
            "residual_summary.csv",
        ],
        "interpretation": "Cross-validated waveform stationarity under provisional unit labels; not spike recall or biological identity.",
    }
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    args = parse_args()
    if not args.plan_only and not args.run:
        raise SystemExit("Choose --plan-only or --run")
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/luke-motion-snippet-numba")
    run(args)


if __name__ == "__main__":
    main()
