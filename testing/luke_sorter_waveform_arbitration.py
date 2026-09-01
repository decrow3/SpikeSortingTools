"""Arbitrate Luke cross-sort unit families on the same accepted band voltage."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt

from testing.luke_sorter_band_comparison import DEFAULT_RESCUE_ROOT, SpikeSet
from testing.luke_sorter_unit_families import DEFAULT_OUTPUT as FAMILY_OUTPUT
from testing.luke_sorter_band_comparison import (
    _load_json,
    _validate_spikes,
    load_dartsort_band,
    load_kiasort_band,
    load_ks4_band,
)


DEFAULT_OUTPUT = Path("testing/outputs/luke_sorter_waveform_arbitration")


def best_shift_cosine(
    first: np.ndarray, second: np.ndarray, maximum_shift: int = 15
) -> tuple[float, int]:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    best = (-np.inf, 0)
    for shift in range(-maximum_shift, maximum_shift + 1):
        if shift < 0:
            left, right = first[-shift:], second[:shift]
        elif shift > 0:
            left, right = first[:-shift], second[shift:]
        else:
            left, right = first, second
        denominator = np.linalg.norm(left) * np.linalg.norm(right)
        value = float(np.sum(left * right) / denominator) if denominator else np.nan
        if np.isfinite(value) and value > best[0]:
            best = (value, shift)
    return float(best[0]), int(best[1])


def stratified_times(
    times: np.ndarray,
    maximum_events: int,
    frame_count: int,
    margin: int,
    bin_count: int = 12,
) -> np.ndarray:
    valid = np.asarray(times, dtype=np.int64)
    valid = valid[(valid >= margin) & (valid < frame_count - margin)]
    selected = []
    quota = int(np.ceil(maximum_events / bin_count))
    for time_bin in range(bin_count):
        lower = time_bin * frame_count // bin_count
        upper = (time_bin + 1) * frame_count // bin_count
        values = valid[(valid >= lower) & (valid < upper)]
        if values.size > quota:
            indices = np.linspace(0, values.size - 1, quota).round().astype(int)
            values = values[indices]
        selected.extend(map(int, values))
    result = np.asarray(sorted(set(selected)), dtype=np.int64)
    if result.size > maximum_events:
        indices = np.linspace(0, result.size - 1, maximum_events).round().astype(int)
        result = result[indices]
    return result


def shifted_control_times(
    times: np.ndarray, frame_count: int, margin: int, offset: int
) -> np.ndarray:
    result = times + offset
    overflow = result >= frame_count - margin
    result[overflow] = times[overflow] - offset
    if np.any(result < margin) or np.any(result >= frame_count - margin):
        raise RuntimeError("Control time shift left the valid recording range")
    return result


def extract_conditioned_snippets(
    recording: np.memmap,
    times: np.ndarray,
    pre_samples: int,
    post_samples: int,
    gain_uv_per_count: float,
    sos: np.ndarray,
) -> np.ndarray:
    snippets = np.stack(
        [recording[time - pre_samples : time + post_samples] for time in times]
    ).astype(np.float32)
    snippets *= gain_uv_per_count
    snippets -= np.median(snippets, axis=2, keepdims=True)
    return sosfiltfilt(sos, snippets, axis=1).astype(np.float32)


def explained_fraction(snippets: np.ndarray, template: np.ndarray) -> np.ndarray:
    snippet_margin = snippets.shape[1] // 4
    template_margin = template.shape[0] // 4
    values = snippets[:, snippet_margin:-snippet_margin]
    model = template[template_margin:-template_margin].reshape(-1)
    flattened = values.reshape(values.shape[0], -1)
    denominator = np.sum(flattened * flattened, axis=1) * np.sum(model * model)
    dot = flattened @ model
    scores = np.divide(dot * dot, denominator, out=np.zeros_like(dot), where=denominator > 0)
    scores[dot <= 0] = 0.0
    return scores


def unit_waveform_metrics(
    spikes: SpikeSet,
    recording: np.memmap,
    channel_depths: np.ndarray,
    fs: float,
    frame_count: int,
    gain_uv_per_count: float,
    maximum_events: int,
) -> tuple[pd.DataFrame, dict[tuple[str, int], np.ndarray]]:
    pre_samples, post_samples = 45, 76
    sos = butter(3, [300.0, 6000.0], btype="bandpass", fs=fs, output="sos")
    rows = []
    templates: dict[tuple[str, int], np.ndarray] = {}
    for unit in np.unique(spikes.labels):
        unit_times = spikes.times[spikes.labels == unit]
        if unit_times.size < 20:
            continue
        times = stratified_times(unit_times, maximum_events, frame_count, post_samples)
        if times.size < 10:
            continue
        snippets = extract_conditioned_snippets(
            recording, times, pre_samples, post_samples, gain_uv_per_count, sos
        )
        template = np.median(snippets, axis=0)
        templates[(spikes.name, int(unit))] = template
        even, odd = np.arange(times.size) % 2 == 0, np.arange(times.size) % 2 == 1
        even_template, odd_template = np.median(snippets[even], axis=0), np.median(snippets[odd], axis=0)
        reliability, _ = best_shift_cosine(even_template, odd_template, 3)
        half = frame_count // 2
        early, late = times < half, times >= half
        continuity = np.nan
        if early.sum() >= 5 and late.sum() >= 5:
            continuity, _ = best_shift_cosine(
                np.median(snippets[early], axis=0), np.median(snippets[late], axis=0), 5
            )
        event_scores = np.r_[
            explained_fraction(snippets[even], odd_template),
            explained_fraction(snippets[odd], even_template),
        ]
        control_times = shifted_control_times(
            times, frame_count, post_samples, int(round(0.017 * fs))
        )
        controls = extract_conditioned_snippets(
            recording, control_times, pre_samples, post_samples, gain_uv_per_count, sos
        )
        control_scores = np.r_[
            explained_fraction(controls[even], odd_template),
            explained_fraction(controls[odd], even_template),
        ]
        peak_flat = int(np.argmax(np.abs(template)))
        peak_time, peak_channel = np.unravel_index(peak_flat, template.shape)
        positive_peak = float(np.max(template))
        negative_peak = float(-np.min(template))
        rows.append(
            {
                "sorter": spikes.name,
                "unit_id": int(unit),
                "source_spike_count": int(unit_times.size),
                "sampled_event_count": int(times.size),
                "template_peak_uv": float(template[peak_time, peak_channel]),
                "template_peak_abs_uv": float(abs(template[peak_time, peak_channel])),
                "peak_channel_local": int(peak_channel),
                "peak_depth_um": float(channel_depths[peak_channel]),
                "positive_to_negative_peak_ratio": positive_peak / max(negative_peak, 1e-12),
                "positive_dominant": bool(positive_peak > negative_peak),
                "split_half_template_cosine": reliability,
                "early_late_template_cosine": continuity,
                "median_independent_template_explained_fraction": float(np.median(event_scores)),
                "median_shifted_control_explained_fraction": float(np.median(control_scores)),
                "median_explained_fraction_excess": float(
                    np.median(event_scores) - np.median(control_scores)
                ),
            }
        )
    return pd.DataFrame(rows), templates


def cross_sort_template_matches(
    templates: dict[tuple[str, int], np.ndarray], first: str, second: str
) -> pd.DataFrame:
    first_units = sorted(unit for sorter, unit in templates if sorter == first)
    second_units = sorted(unit for sorter, unit in templates if sorter == second)
    rows = []
    for source, targets in ((first, second_units), (second, first_units)):
        target_sorter = second if source == first else first
        source_units = first_units if source == first else second_units
        for unit in source_units:
            candidates = []
            for target_unit in targets:
                cosine, shift = best_shift_cosine(
                    templates[(source, unit)], templates[(target_sorter, target_unit)], 15
                )
                candidates.append((cosine, target_unit, shift))
            cosine, target_unit, shift = max(candidates)
            rows.append(
                {
                    "source_sorter": source,
                    "source_unit": unit,
                    "target_sorter": target_sorter,
                    "best_target_unit": target_unit,
                    "best_template_cosine": cosine,
                    "best_shift_samples": shift,
                }
            )
    return pd.DataFrame(rows)


def family_waveform_summary(
    candidates: pd.DataFrame, templates: dict[tuple[str, int], np.ndarray]
) -> pd.DataFrame:
    rows = []
    for family in candidates.itertuples():
        ks_units = [int(value) for value in family.ks4_units.split(";")]
        kia_units = [int(value) for value in family.kiasort_units.split(";")]
        missing = [
            f"{sorter}:{unit}"
            for sorter, units in (
                ("ks4_no_motion", ks_units),
                ("kiasort_band_pilot", kia_units),
            )
            for unit in units
            if (sorter, unit) not in templates
        ]
        cross = [
            best_shift_cosine(
                templates[("ks4_no_motion", ks)],
                templates[("kiasort_band_pilot", kia)],
                15,
            )[0]
            for ks in ks_units
            for kia in kia_units
            if ("ks4_no_motion", ks) in templates
            and ("kiasort_band_pilot", kia) in templates
        ]
        multi_sorter, multi_units = (
            ("ks4_no_motion", ks_units)
            if len(ks_units) > 1
            else ("kiasort_band_pilot", kia_units)
        )
        within = [
            best_shift_cosine(
                templates[(multi_sorter, first)], templates[(multi_sorter, second)], 15
            )[0]
            for first, second in combinations(multi_units, 2)
            if (multi_sorter, first) in templates
            and (multi_sorter, second) in templates
        ]
        rows.append(
            {
                **family._asdict(),
                "missing_template_units": ";".join(missing),
                "minimum_cross_sort_template_cosine": (
                    float(np.min(cross)) if cross else np.nan
                ),
                "median_cross_sort_template_cosine": (
                    float(np.median(cross)) if cross else np.nan
                ),
                "maximum_cross_sort_template_cosine": (
                    float(np.max(cross)) if cross else np.nan
                ),
                "median_multi_side_template_cosine": (
                    float(np.median(within)) if within else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def save_review_figure(
    output_path: Path,
    spike_sets: list[SpikeSet],
    templates: dict[tuple[str, int], np.ndarray],
    isolated_kia: pd.DataFrame,
    frame_count: int,
    fs: float,
) -> None:
    import os

    matplotlib_config = Path("/tmp/luke-sorter-matplotlib")
    matplotlib_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config))
    import matplotlib.pyplot as plt

    by_name = {spikes.name: spikes for spikes in spike_sets}
    figure, axes = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True)
    time_axis = np.arange(12) * 10 + 5

    def plot_family(
        row: int,
        ks_units: list[int],
        kia_units: list[int],
        title: tuple[str, str],
    ) -> None:
        for sorter, units, linestyle in (
            ("ks4_no_motion", ks_units, "--"),
            ("kiasort_band_pilot", kia_units, "-"),
        ):
            spikes = by_name[sorter]
            for unit in units:
                if (sorter, unit) not in templates:
                    continue
                values = spikes.times[spikes.labels == unit]
                counts = np.bincount(
                    np.minimum(values * 12 // frame_count, 11), minlength=12
                )
                axes[row, 0].plot(
                    time_axis,
                    counts / 10.0,
                    linestyle,
                    marker="o",
                    markersize=3,
                    label=f"{'KS4' if sorter == 'ks4_no_motion' else 'KIA'} {unit}",
                )
                template = templates[(sorter, unit)]
                _, peak_channel = np.unravel_index(
                    np.argmax(np.abs(template)), template.shape
                )
                trace = template[:, peak_channel]
                trace = trace / max(np.max(np.abs(trace)), 1e-12)
                milliseconds = (np.arange(trace.size) - 45) / fs * 1000.0
                axes[row, 1].plot(
                    milliseconds,
                    trace,
                    linestyle,
                    label=f"{'KS4' if sorter == 'ks4_no_motion' else 'KIA'} {unit}",
                )
        axes[row, 0].set_title(title[0])
        axes[row, 0].set(xlabel="Window time (s)", ylabel="Events/s", xlim=(0, 120))
        axes[row, 0].legend(fontsize=7, ncol=2)
        axes[row, 1].set_title(title[1])
        axes[row, 1].set(xlabel="Time from reported event (ms)", ylabel="Normalized voltage")
        axes[row, 1].legend(fontsize=7, ncol=2)

    plot_family(
        0,
        [233],
        [57, 58, 59, 60, 61],
        (
            "KS4 233 vs KIASORT 57–61: 10 s event rates",
            "Raw templates: one KS4 ↔ five KIASORT labels",
        ),
    )
    plot_family(
        1,
        [242, 246],
        [77],
        (
            "KS4 242/246 vs KIASORT 77: 10 s event rates",
            "Raw templates: best remaining merge hypothesis",
        ),
    )
    colors = np.where(isolated_kia.positive_dominant, "tab:orange", "tab:blue")
    axes[2, 0].scatter(
        isolated_kia.refractory_fraction_1p5ms,
        isolated_kia.median_explained_fraction_excess,
        c=colors,
        alpha=0.75,
    )
    shortlisted = isolated_kia[isolated_kia.waveform_shortlist]
    axes[2, 0].scatter(
        shortlisted.refractory_fraction_1p5ms,
        shortlisted.median_explained_fraction_excess,
        facecolors="none",
        edgecolors="black",
        s=100,
        linewidths=1.5,
    )
    for row in shortlisted.itertuples():
        axes[2, 0].annotate(
            str(row.unit_id),
            (row.refractory_fraction_1p5ms, row.median_explained_fraction_excess),
            xytext=(4, 4),
            textcoords="offset points",
        )
    axes[2, 0].axvline(0.02, color="0.4", linestyle=":")
    axes[2, 0].axhline(0.05, color="0.4", linestyle=":")
    axes[2, 0].set_title("KIASORT-only families: raw-waveform shortlist")
    axes[2, 0].set(
        xlabel="1.5 ms refractory fraction",
        ylabel="Median explained-fraction excess",
    )
    axes[2, 1].scatter(
        isolated_kia.split_half_template_cosine,
        isolated_kia.early_late_template_cosine,
        c=colors,
        alpha=0.75,
    )
    axes[2, 1].scatter(
        shortlisted.split_half_template_cosine,
        shortlisted.early_late_template_cosine,
        facecolors="none",
        edgecolors="black",
        s=100,
        linewidths=1.5,
    )
    for row in shortlisted.itertuples():
        axes[2, 1].annotate(
            str(row.unit_id),
            (row.split_half_template_cosine, row.early_late_template_cosine),
            xytext=(4, 4),
            textcoords="offset points",
        )
    axes[2, 1].axvline(0.65, color="0.4", linestyle=":")
    axes[2, 1].axhline(0.65, color="0.4", linestyle=":")
    axes[2, 1].set_title("Independent-template stability")
    axes[2, 1].set(xlabel="Split-half cosine", ylabel="Early/late cosine")
    figure.suptitle("Luke rapid-motion band: unit-family waveform arbitration")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rescue-root", type=Path, default=DEFAULT_RESCUE_ROOT)
    parser.add_argument("--family-output", type=Path, default=FAMILY_OUTPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--maximum-events-per-unit", type=int, default=96)
    parser.add_argument("--kiasort-output-name", default="kiasort_channels_82_114")
    parser.add_argument(
        "--kiasort-recording-output-name", default="kiasort_channels_82_114"
    )
    args = parser.parse_args()

    window_dir = args.rescue_root / "sorter_bakeoff/windows/rapid_motion-8b4978262d"
    kia_name = args.kiasort_output_name
    manifests = {
        name: _load_json(window_dir / name / "bakeoff_sort_manifest.json")
        for name in ("ks4_no_motion", "dartsort_native", kia_name)
    }
    windows = [value["window"] for value in manifests.values()]
    if len({value["request_digest"] for value in windows}) != 1:
        raise RuntimeError("Source sorter windows differ")
    if any(value.get("raw_voltage_warp") is not False for value in manifests.values()):
        raise RuntimeError("Waveform arbitration requires unwarped source voltage")
    fs = float(windows[0]["sampling_frequency_hz"])
    frame_count = int(windows[0]["frame_count"])
    spike_sets = [
        load_ks4_band(
            args.rescue_root,
            int(windows[0]["start_frame"]),
            int(windows[0]["end_frame"]),
            41,
            56,
        ),
        load_dartsort_band(window_dir, 82, 114),
        load_kiasort_band(window_dir, kia_name),
    ]
    for spikes in spike_sets:
        _validate_spikes(spikes, frame_count)

    native = window_dir / args.kiasort_recording_output_name / "native_output"
    recording_path = native / "recording.dat"
    expected_bytes = frame_count * 32 * np.dtype("<i2").itemsize
    if recording_path.stat().st_size != expected_bytes:
        raise RuntimeError("Accepted waveform band has an unexpected byte count")
    recording = np.memmap(recording_path, dtype="<i2", mode="r", shape=(frame_count, 32))
    channel_depths = np.asarray(loadmat(native / "channel_map.mat")["ycoords"]).reshape(-1)
    rescue_manifest = _load_json(args.rescue_root / "recording/rescue_recording_manifest.json")
    if rescue_manifest["request_digest"] != manifests[kia_name]["recording_request_digest"]:
        raise RuntimeError("Waveform recording receipt differs from sorter receipts")
    gain = float(rescue_manifest["gain_uv_per_count"])

    metric_frames = []
    templates: dict[tuple[str, int], np.ndarray] = {}
    for spikes in spike_sets:
        metrics, sorter_templates = unit_waveform_metrics(
            spikes,
            recording,
            channel_depths,
            fs,
            frame_count,
            gain,
            args.maximum_events_per_unit,
        )
        metric_frames.append(metrics)
        templates.update(sorter_templates)
    unit_metrics_frame = pd.concat(metric_frames, ignore_index=True)
    matches = cross_sort_template_matches(
        templates, "ks4_no_motion", "kiasort_band_pilot"
    )
    candidates = pd.read_csv(args.family_output / "ks4_kiasort_candidates.csv")
    family_summary = family_waveform_summary(candidates, templates)
    isolated = pd.read_csv(args.family_output / "isolated_units.csv")
    unit_metrics_frame = unit_metrics_frame.merge(
        isolated[["sorter", "unit_id", "family_id"]].assign(isolated_family=True),
        on=["sorter", "unit_id"],
        how="left",
    )
    unit_metrics_frame["isolated_family"] = unit_metrics_frame.isolated_family.eq(True)
    kia_matches = matches.loc[
        matches.source_sorter == "kiasort_band_pilot",
        ["source_unit", "best_target_unit", "best_template_cosine"],
    ]
    isolated_kia = unit_metrics_frame.loc[
        (unit_metrics_frame.sorter == "kiasort_band_pilot")
        & unit_metrics_frame.isolated_family
    ].merge(
        isolated[
            [
                "sorter",
                "unit_id",
                "counterpart_event_fraction",
                "refractory_fraction_1p5ms",
                "presence_fraction_10s",
            ]
        ],
        on=["sorter", "unit_id"],
    ).merge(kia_matches, left_on="unit_id", right_on="source_unit")
    isolated_kia["waveform_shortlist"] = (
        (isolated_kia.refractory_fraction_1p5ms <= 0.02)
        & (isolated_kia.split_half_template_cosine >= 0.65)
        & (isolated_kia.early_late_template_cosine >= 0.65)
        & (isolated_kia.median_explained_fraction_excess >= 0.05)
    )

    full_recording = np.memmap(
        args.rescue_root / "recording/traces_cached_seg0.raw",
        dtype="<i2",
        mode="r",
        shape=(int(rescue_manifest["num_samples"]), int(rescue_manifest["num_channels"])),
    )
    sample_checks = [0, 1, 1000, frame_count // 2, frame_count - 2, frame_count - 1]
    if not all(
        np.array_equal(
            recording[local_frame],
            full_recording[int(windows[0]["start_frame"]) + local_frame, 82:114],
        )
        for local_frame in sample_checks
    ):
        raise RuntimeError("Waveform band differs from the accepted full recording")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    unit_metrics_frame.to_csv(args.output_dir / "unit_waveform_metrics.csv", index=False)
    matches.to_csv(args.output_dir / "cross_sort_template_matches.csv", index=False)
    family_summary.to_csv(args.output_dir / "candidate_family_waveforms.csv", index=False)
    isolated_kia.to_csv(args.output_dir / "isolated_kiasort_waveform_review.csv", index=False)
    np.savez_compressed(
        args.output_dir / "unit_templates.npz",
        **{
            f"{sorter}__{unit}": template
            for (sorter, unit), template in templates.items()
        },
    )
    save_review_figure(
        args.output_dir / "family_waveform_examples.png",
        spike_sets,
        templates,
        isolated_kia,
        frame_count,
        fs,
    )
    metadata = {
        "status": "targeted_raw_waveform_arbitration_complete",
        "window_request_digest": windows[0]["request_digest"],
        "recording_request_digest": rescue_manifest["request_digest"],
        "recording_file_bytes": expected_bytes,
        "exact_full_recording_sample_checks": sample_checks,
        "gain_uv_per_count": gain,
        "conditioning": "32-channel common median then third-order 300-6000 Hz Butterworth, zero phase",
        "maximum_events_per_unit": args.maximum_events_per_unit,
        "kiasort_output_name": kia_name,
        "kiasort_recording_output_name": args.kiasort_recording_output_name,
        "independence": "alternating sampled events form and score opposing templates",
        "control": "same unit template at event times shifted by 17 ms",
        "isolated_kiasort_waveform_shortlist": {
            "maximum_refractory_fraction_1p5ms": 0.02,
            "minimum_split_half_template_cosine": 0.65,
            "minimum_early_late_template_cosine": 0.65,
            "minimum_median_explained_fraction_excess": 0.05,
            "unit_count": int(isolated_kia.waveform_shortlist.sum()),
            "spike_count": int(
                isolated_kia.loc[isolated_kia.waveform_shortlist, "source_spike_count"].sum()
            ),
        },
        "limitations": [
            "KIASORT used the 32-channel band while KS4 and DARTsort were full-probe sorts.",
            "A short-window template score is evidence of spike-like consistency, not unit identity.",
            "Residual fractions are local event-centered template projections, not full sorter residuals.",
        ],
    }
    (args.output_dir / "waveform_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(family_summary.to_string(index=False))


if __name__ == "__main__":
    main()
