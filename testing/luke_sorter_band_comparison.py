"""Compare accepted KS4, DARTsort, and KIASORT outputs on one depth band."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.io import loadmat


DEFAULT_RESCUE_ROOT = Path(
    "/mnt/NPX/Luke/20250804/"
    "rescue_pipeline_results_Luke0804_V2V1_g0_imec1"
)
DEFAULT_OUTPUT = Path("testing/outputs/luke_sorter_band_comparison")


@dataclass(frozen=True)
class SpikeSet:
    name: str
    times: np.ndarray
    labels: np.ndarray
    depths_um: np.ndarray


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _validate_spikes(spikes: SpikeSet, frame_count: int) -> None:
    n = spikes.times.size
    if spikes.labels.size != n or spikes.depths_um.size != n:
        raise RuntimeError(f"{spikes.name} arrays have different lengths")
    if np.any(spikes.times < 0) or np.any(spikes.times >= frame_count):
        raise RuntimeError(f"{spikes.name} has out-of-window times")
    if np.any(spikes.times[1:] < spikes.times[:-1]):
        raise RuntimeError(f"{spikes.name} times are not sorted")
    if not np.all(np.isfinite(spikes.depths_um)):
        raise RuntimeError(f"{spikes.name} has non-finite depths")


def _same_time_label_multiset(
    first_times: np.ndarray,
    first_labels: np.ndarray,
    second_times: np.ndarray,
    second_labels: np.ndarray,
) -> bool:
    """Compare events without assuming an ordering among equal-time spikes."""
    if first_times.size != second_times.size:
        return False
    first_order = np.lexsort((first_labels, first_times))
    second_order = np.lexsort((second_labels, second_times))
    return bool(
        np.array_equal(first_times[first_order], second_times[second_order])
        and np.array_equal(first_labels[first_order], second_labels[second_order])
    )


def load_ks4_band(
    rescue_root: Path,
    start_frame: int,
    end_frame: int,
    first_depth_row: int,
    last_depth_row: int,
) -> SpikeSet:
    native = rescue_root / "kilosort4/sorter_output"
    times = np.load(native / "spike_times.npy", mmap_mode="r").reshape(-1)
    labels = np.load(native / "spike_clusters.npy", mmap_mode="r").reshape(-1)
    positions = np.load(native / "spike_positions.npy", mmap_mode="r")
    window = (times >= start_frame) & (times < end_frame)
    local_times = np.asarray(times[window], dtype=np.int64) - start_frame
    local_labels = np.asarray(labels[window], dtype=np.int64)
    depths = np.asarray(positions[window, 1], dtype=float)
    depth_rows = np.rint(depths / 20.0).astype(int)
    band = (depth_rows >= first_depth_row) & (depth_rows <= last_depth_row)
    return SpikeSet("ks4_no_motion", local_times[band], local_labels[band], depths[band])


def load_dartsort_band(window_dir: Path, channel_start: int, channel_end: int) -> SpikeSet:
    native = np.load(window_dir / "dartsort_native/native_output/dartsort_sorting.npz")
    labels = np.asarray(native["labels"], dtype=np.int64)
    channels = np.asarray(native["channels"], dtype=np.int64)
    keep = (labels >= 0) & (channels >= channel_start) & (channels < channel_end)
    times = np.asarray(native["times_samples"][keep], dtype=np.int64)
    labels = labels[keep]
    depths = np.asarray(native["geom"], dtype=float)[channels[keep], 1]
    order = np.argsort(times, kind="stable")
    return SpikeSet("dartsort_native", times[order], labels[order], depths[order])


def load_kiasort_band(window_dir: Path, output_name: str) -> SpikeSet:
    output = window_dir / output_name
    accepted_times = np.load(output / "spike_times.npy")
    accepted_labels = np.load(output / "spike_labels.npy")
    accepted_depths = output / "spike_depths_um.npy"
    if accepted_depths.exists():
        return SpikeSet(
            "kiasort_band_pilot",
            np.asarray(accepted_times, dtype=np.int64),
            np.asarray(accepted_labels, dtype=np.int64),
            np.asarray(np.load(accepted_depths), dtype=float),
        )
    native = output / "native_output/RES_Sorted"
    with h5py.File(native / "spike_idx.h5") as handle:
        times = np.asarray(handle["/spike_idx"]).reshape(-1).astype(np.int64)
    with h5py.File(native / "unifiedLabels.h5") as handle:
        labels = np.asarray(handle["/unifiedLabels"]).reshape(-1).astype(np.int64)
    with h5py.File(native / "channelNum.h5") as handle:
        channels = np.asarray(handle["/channelNum"]).reshape(-1).astype(np.int64) - 1
    keep = labels >= 0
    times, labels, channels = times[keep], labels[keep], channels[keep]
    order = np.argsort(times, kind="stable")
    times, labels, channels = times[order], labels[order], channels[order]
    if not _same_time_label_multiset(times, labels, accepted_times, accepted_labels):
        raise RuntimeError("KIASORT native arrays do not reproduce accepted arrays")
    channel_map = loadmat(output / "native_output/channel_map.mat")
    ycoords = np.asarray(channel_map["ycoords"]).reshape(-1)
    return SpikeSet("kiasort_band_pilot", times, labels, ycoords[channels])


def unit_metrics(spikes: SpikeSet, fs: float, duration_s: float) -> pd.DataFrame:
    refractory_samples = int(round(1.5e-3 * fs))
    bin_samples = int(round(10.0 * fs))
    first_end = int(round(20.0 * fs))
    last_start = int(round((duration_s - 20.0) * fs))
    half = int(round(duration_s * fs / 2.0))
    presence_bin_count = int(np.ceil(duration_s / 10.0))
    rows = []
    for unit_id in np.unique(spikes.labels):
        mask = spikes.labels == unit_id
        times = spikes.times[mask]
        depths = spikes.depths_um[mask]
        early_depths = depths[times < half]
        late_depths = depths[times >= half]
        rows.append(
            {
                "sorter": spikes.name,
                "unit_id": int(unit_id),
                "spike_count": int(times.size),
                "firing_rate_hz": float(times.size / duration_s),
                "refractory_fraction_1p5ms": (
                    float(np.mean(np.diff(times) < refractory_samples))
                    if times.size > 1
                    else np.nan
                ),
                "presence_fraction_10s": float(
                    np.unique(times // bin_samples).size / presence_bin_count
                ),
                "present_first_20s": bool(np.any(times < first_end)),
                "present_last_20s": bool(np.any(times >= last_start)),
                "median_depth_um": float(np.median(depths)),
                "early_late_depth_delta_um": (
                    float(abs(np.median(late_depths) - np.median(early_depths)))
                    if early_depths.size >= 5 and late_depths.size >= 5
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def cross_unit_coincidence_fraction(
    times: np.ndarray,
    labels: np.ndarray,
    depths: np.ndarray,
    temporal_radius: int,
    depth_radius_um: float,
) -> float:
    order = np.argsort(times, kind="stable")
    times, labels, depths = times[order], labels[order], depths[order]
    marked = np.zeros(times.size, dtype=bool)
    for left in range(times.size):
        right = left + 1
        while right < times.size and times[right] - times[left] <= temporal_radius:
            if labels[right] != labels[left] and abs(depths[right] - depths[left]) <= depth_radius_um:
                marked[left] = True
                marked[right] = True
            right += 1
    return float(marked.mean()) if marked.size else 0.0


def shifted_coincidence_null(
    spikes: SpikeSet,
    frame_count: int,
    temporal_radius: int,
    depth_radius_um: float,
    seed: int,
    repeats: int = 5,
) -> float:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        shifted = spikes.times.copy()
        for unit_id in np.unique(spikes.labels):
            mask = spikes.labels == unit_id
            offset = int(rng.integers(20 * temporal_radius, frame_count - 20 * temporal_radius))
            shifted[mask] = (shifted[mask] + offset) % frame_count
        values.append(
            cross_unit_coincidence_fraction(
                shifted, spikes.labels, spikes.depths_um, temporal_radius, depth_radius_um
            )
        )
    return float(np.mean(values))


def event_overlap(first: SpikeSet, second: SpikeSet, time_radius: int, depth_radius: float) -> dict:
    times = np.concatenate([first.times, second.times])
    depths = np.concatenate([first.depths_um, second.depths_um])
    source = np.concatenate(
        [np.zeros(first.times.size, dtype=np.int8), np.ones(second.times.size, dtype=np.int8)]
    )
    order = np.argsort(times, kind="stable")
    times, depths, source = times[order], depths[order], source[order]
    marked = np.zeros(times.size, dtype=bool)
    for left in range(times.size):
        right = left + 1
        while right < times.size and times[right] - times[left] <= time_radius:
            if source[right] != source[left] and abs(depths[right] - depths[left]) <= depth_radius:
                marked[left] = True
                marked[right] = True
            right += 1
    first_mask, second_mask = source == 0, source == 1
    return {
        "first": first.name,
        "second": second.name,
        "first_spike_count": int(first.times.size),
        "second_spike_count": int(second.times.size),
        "first_matched_fraction": float(marked[first_mask].mean()),
        "second_matched_fraction": float(marked[second_mask].mean()),
    }


def summarize_sorter(
    spikes: SpikeSet,
    units: pd.DataFrame,
    fs: float,
    duration_s: float,
    frame_count: int,
    seed: int,
) -> dict:
    eligible = units.loc[units.spike_count >= 20]
    coincidence_radius = int(round(0.25e-3 * fs))
    observed = cross_unit_coincidence_fraction(
        spikes.times, spikes.labels, spikes.depths_um, coincidence_radius, 75.0
    )
    null = shifted_coincidence_null(
        spikes, frame_count, coincidence_radius, 75.0, seed
    )
    return {
        "sorter": spikes.name,
        "spike_count": int(spikes.times.size),
        "spikes_per_s": float(spikes.times.size / duration_s),
        "unit_count": int(units.shape[0]),
        "analysis_unit_count_ge_20_spikes": int(eligible.shape[0]),
        "median_analysis_unit_rate_hz": float(eligible.firing_rate_hz.median()),
        "median_refractory_fraction_1p5ms": float(
            eligible.refractory_fraction_1p5ms.median()
        ),
        "median_presence_fraction_10s": float(eligible.presence_fraction_10s.median()),
        "stable_unit_fraction_presence_ge_0p9": float(
            (eligible.presence_fraction_10s >= 0.9).mean()
        ),
        "first_last_20s_unit_fraction": float(
            (eligible.present_first_20s & eligible.present_last_20s).mean()
        ),
        "median_early_late_depth_delta_um": float(
            eligible.early_late_depth_delta_um.median()
        ),
        "cross_unit_coincidence_fraction_0p25ms_75um": observed,
        "shifted_coincidence_null": null,
        "coincidence_excess": observed - null,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rescue-root", type=Path, default=DEFAULT_RESCUE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--channel-start", type=int, default=82)
    parser.add_argument("--channel-count", type=int, default=32)
    parser.add_argument("--kiasort-output-name")
    args = parser.parse_args()

    window_dir = args.rescue_root / "sorter_bakeoff/windows/rapid_motion-8b4978262d"
    manifests = {
        name: _load_json(window_dir / name / "bakeoff_sort_manifest.json")
        for name in ("ks4_no_motion", "dartsort_native")
    }
    kia_name = args.kiasort_output_name or (
        f"kiasort_channels_{args.channel_start}_{args.channel_start + args.channel_count}"
    )
    manifests[kia_name] = _load_json(window_dir / kia_name / "bakeoff_sort_manifest.json")
    if not all(manifest.get("complete") is True for manifest in manifests.values()):
        raise RuntimeError("At least one source sorter manifest is incomplete")
    if any(manifest.get("raw_voltage_warp") is not False for manifest in manifests.values()):
        raise RuntimeError("Comparison requires unwarped-voltage sorter outputs")
    recording_digests = {
        manifest["recording_request_digest"] for manifest in manifests.values()
    }
    if len(recording_digests) != 1:
        raise RuntimeError("Sorter outputs use different source recordings")
    if manifests[kia_name].get("channel_selection") != {
        "start_index": args.channel_start,
        "end_index_exclusive": args.channel_start + args.channel_count,
        "count": args.channel_count,
        "full_probe": False,
    }:
        raise RuntimeError("KIASORT manifest does not match the requested channel band")
    windows = [value["window"] for value in manifests.values()]
    if len({value["request_digest"] for value in windows}) != 1:
        raise RuntimeError("Sorter outputs use different bake-off windows")
    fs = float(windows[0]["sampling_frequency_hz"])
    frame_count = int(windows[0]["frame_count"])
    duration_s = frame_count / fs
    channel_end = args.channel_start + args.channel_count
    first_depth_row, last_depth_row = args.channel_start // 2, (channel_end - 1) // 2

    spike_sets = [
        load_ks4_band(
            args.rescue_root,
            int(windows[0]["start_frame"]),
            int(windows[0]["end_frame"]),
            first_depth_row,
            last_depth_row,
        ),
        load_dartsort_band(window_dir, args.channel_start, channel_end),
        load_kiasort_band(window_dir, kia_name),
    ]
    for spikes in spike_sets:
        _validate_spikes(spikes, frame_count)

    unit_frames = [unit_metrics(spikes, fs, duration_s) for spikes in spike_sets]
    units = pd.concat(unit_frames, ignore_index=True)
    summaries = pd.DataFrame(
        [
            summarize_sorter(spikes, unit_frame, fs, duration_s, frame_count, 1701 + i)
            for i, (spikes, unit_frame) in enumerate(zip(spike_sets, unit_frames))
        ]
    )
    overlap = pd.DataFrame(
        [
            event_overlap(spike_sets[i], spike_sets[j], int(round(0.5e-3 * fs)), 60.0)
            for i, j in ((0, 1), (0, 2), (1, 2))
        ]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries.to_csv(args.output_dir / "sorter_summary.csv", index=False)
    units.to_csv(args.output_dir / "unit_metrics.csv", index=False)
    overlap.to_csv(args.output_dir / "pairwise_event_overlap.csv", index=False)
    metadata = {
        "status": "descriptive_band_comparison_complete",
        "window_request_digest": windows[0]["request_digest"],
        "duration_s": duration_s,
        "channel_selection": {
            "start_index": args.channel_start,
            "end_index_exclusive": channel_end,
            "depth_rows_inclusive": [first_depth_row, last_depth_row],
        },
        "analysis_unit_min_spikes": 20,
        "kiasort_output_name": kia_name,
        "event_overlap_tolerance_ms": 0.5,
        "event_overlap_depth_tolerance_um": 60.0,
        "duplicate_tolerance_ms": 0.25,
        "duplicate_depth_tolerance_um": 75.0,
        "limitations": [
            "KIASORT is a 32-channel pilot; KS4 and DARTsort were filtered post hoc.",
            "Event overlap does not establish biological unit identity.",
            "Waveform-family continuity and residual recovery are not available from normalized arrays.",
            "DARTsort remains upstream work in progress and is not production recommended.",
        ],
        "source_request_digests": {
            name: manifest["request_digest"] for name, manifest in manifests.items()
        },
    }
    (args.output_dir / "comparison_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    readme = """# Luke sorter architecture band comparison

This artifact compares the accepted rapid-motion outputs on channel indices
`[82, 114)`. KIASORT was run on that band; KS4 and DARTsort were restricted to
the same physical rows after sorting. Metrics are descriptive guardrails, not
evidence of biological unit identity or a production advancement decision.

- `sorter_summary.csv`: sorter-level yield and temporal guardrails.
- `unit_metrics.csv`: per-unit inputs to the summary.
- `pairwise_event_overlap.csv`: symmetric event agreement within 0.5 ms/60 um.
- `comparison_metadata.json`: definitions, provenance, and limitations.
"""
    (args.output_dir / "README.md").write_text(readme)
    print(summaries.to_string(index=False))
    print("\nPairwise event overlap:\n", overlap.to_string(index=False))


if __name__ == "__main__":
    main()
