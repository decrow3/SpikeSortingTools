#!/usr/bin/env python3
"""EM.2a CPU-only operator screen on the frozen EM.1 snippets."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from testing.em_huklaban5_voltage_equivalence import (  # noqa: E402
    DT,
    TARGET_IDS,
    dd_exact_coordinate_mapping,
    dd_source_time_origin,
    load_relative_recording,
)
from testing.em_huklaban5_voltage_equivalence_arms import (  # noqa: E402
    BASE,
    DD,
    FROZEN,
    array_sha256,
    atomic_json,
    preflight,
    read_aligned,
    sha256,
)
from npx_preprocessing.motion.lattice_remap_si import sample_stepwise_shifts  # noqa: E402


FIELD = Path(
    "/media/huklaban5/writable/DARTsort_motion_experiments/incoming/"
    "luke0804_imec1_two_layer_v1/luke0804_imec1_two_layer_motion.npz"
)
SAVED_D2L_FIELD = Path(
    "/mnt/NPX/Luke/DARTsort_motion_experiments/"
    "luke0804-imec1-ao-d2l-v2/windows/W2/runs/D2L/motion/fields.npz"
)
PEAKS = DD / "gate_voltage_v1/S_L_actual_voltage_peaks.npz"
EXPECTED_FIELD_SHA256 = "85062a37f38b3c5212393d627fe387b629fedd95068a4136a72f330b5aa4c2d9"
EXPECTED_SAVED_FIELD_SHA256 = "ee34e8a6a036b10d77a0bc8563b65d171368e8869be5e96f654ee072c5bb24cf"
EXPECTED_PEAKS_SHA256 = "ca05b7a17595a3d69515294b04611a3e93399e3b0767eba0fcf871088914b6fa"
WAVEFORM_RADIUS = 60
EVENTS_PER_STRATUM = 16


ARM_SPECS = {
    "none_s0": {"field": "none", "method": "none"},
    "rounded_exact_dd": {"field": "lattice_rounded", "method": "exact_coordinate"},
    "rounded_nearest": {
        "field": "lattice_rounded",
        "method": "nearest",
        "dtype": "float32",
        "force_extrapolate": False,
    },
    "rounded_idw": {
        "field": "lattice_rounded",
        "method": "idw",
        "num_closest": 4,
        "dtype": "float32",
        "force_extrapolate": False,
    },
    "rounded_kriging": {
        "field": "lattice_rounded",
        "method": "kriging",
        "sigma_um": 20.0,
        "p": 1.0,
        "sparse_thresh": None,
        "dtype": "float32",
        "force_extrapolate": False,
    },
    "unrounded_nearest": {
        "field": "unrounded_two_layer_rigid",
        "method": "nearest",
        "dtype": "float32",
        "force_extrapolate": False,
    },
    "unrounded_idw": {
        "field": "unrounded_two_layer_rigid",
        "method": "idw",
        "num_closest": 4,
        "dtype": "float32",
        "force_extrapolate": False,
    },
    "unrounded_kriging": {
        "field": "unrounded_two_layer_rigid",
        "method": "kriging",
        "sigma_um": 20.0,
        "p": 1.0,
        "sparse_thresh": None,
        "dtype": "float32",
        "force_extrapolate": False,
    },
}


COMPARISONS = (
    ("rounded_exact_dd", "none_s0"),
    ("rounded_nearest", "rounded_exact_dd"),
    ("rounded_idw", "rounded_exact_dd"),
    ("rounded_kriging", "rounded_exact_dd"),
    ("rounded_nearest", "rounded_idw"),
    ("rounded_nearest", "rounded_kriging"),
    ("rounded_idw", "rounded_kriging"),
    ("unrounded_nearest", "none_s0"),
    ("unrounded_idw", "none_s0"),
    ("unrounded_kriging", "none_s0"),
    ("unrounded_nearest", "unrounded_idw"),
    ("unrounded_nearest", "unrounded_kriging"),
    ("unrounded_idw", "unrounded_kriging"),
    ("unrounded_nearest", "rounded_nearest"),
    ("unrounded_idw", "rounded_idw"),
    ("unrounded_kriging", "rounded_kriging"),
)


def assign_stepwise(centers: np.ndarray, values: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Assign half-open uniform cells; exact boundaries use the later cell."""

    centers = np.asarray(centers, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    if centers.ndim != 1 or values.shape != centers.shape or centers.size < 2:
        raise ValueError("centers and values must be matching nontrivial vectors")
    spacing = np.diff(centers)
    if not np.allclose(spacing, spacing[0], rtol=0, atol=1e-12):
        raise ValueError("field centers are not uniform")
    right_edges = centers + spacing[0] / 2
    indices = np.searchsorted(right_edges, times, side="right")
    if np.any(indices >= len(values)):
        raise ValueError("times exceed field support")
    return values[indices]


def interpolation_insertion(
    parent_ids: list[str],
    full_ids: list[str],
    source_geometry: np.ndarray,
    full_geometry: np.ndarray,
    get_bad_weights,
) -> tuple[np.ndarray, dict[str, object]]:
    """Map 383 accepted channels to the full 384-site interpolated input."""

    insertion = np.zeros((len(parent_ids), len(full_ids)), dtype=np.float32)
    for full_index, channel_id in enumerate(full_ids):
        if channel_id != "imec1.ap#AP191":
            insertion[parent_ids.index(channel_id), full_index] = 1.0
    ap191 = full_ids.index("imec1.ap#AP191")
    sigma_um = float(np.min(np.diff(np.unique(full_geometry[:, 1]))))
    weights = get_bad_weights(
        source_geometry,
        full_geometry[[ap191]],
        sigma_um=sigma_um,
        p=1.3,
    )[:, 0].astype(np.float32)
    insertion[:, ap191] = weights
    if not np.isclose(weights.sum(), 1.0, rtol=0, atol=1e-6):
        raise RuntimeError("AP191 weights do not sum to one")
    return insertion, {
        "bad_channel_id": "imec1.ap#AP191",
        "method": "SpikeInterface get_kriging_channel_weights",
        "sigma_um": sigma_um,
        "p": 1.3,
        "weight_threshold": 0.005,
        "weights_sha256": array_sha256(weights),
        "nonzero_weights": int(np.count_nonzero(weights)),
    }


def effective_kernel(
    get_kernel,
    insertion: np.ndarray,
    full_geometry: np.ndarray,
    target_geometry: np.ndarray,
    displacement_um: float,
    spec: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    moved = target_geometry.copy()
    moved[:, 1] += float(displacement_um)
    kwargs = {key: value for key, value in spec.items() if key not in {"field", "method"}}
    kernel = get_kernel(full_geometry, moved, method=spec["method"], **kwargs)
    effective = insertion @ kernel
    structural_zero = np.count_nonzero(kernel, axis=0) == 0
    return effective.astype(np.float32, copy=False), structural_zero


def exact_zero_mask(
    source_geometry: np.ndarray, target_geometry: np.ndarray, states: np.ndarray
) -> np.ndarray:
    output = np.zeros((len(states), len(target_geometry)), dtype=bool)
    for state in np.unique(states):
        mapping = dd_exact_coordinate_mapping(source_geometry, target_geometry, float(state))
        output[states == state] = mapping < 0
    return output


def render(
    source: np.ndarray,
    states: np.ndarray,
    kernels: dict[float, tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    output = np.empty((len(source), next(iter(kernels.values()))[0].shape[1]), dtype=np.float32)
    zeros = np.empty(output.shape, dtype=bool)
    for state in np.unique(states):
        key = float(state)
        frames = states == state
        kernel, structural_zero = kernels[key]
        output[frames] = source[frames] @ kernel
        zeros[frames] = structural_zero
    return output, zeros


def metric_row(
    window: str,
    candidate_name: str,
    reference_name: str,
    candidate: np.ndarray,
    reference: np.ndarray,
) -> dict[str, object]:
    delta = candidate.astype(np.float64) - reference.astype(np.float64)
    return {
        "window": window,
        "candidate": candidate_name,
        "reference": reference_name,
        "samples": int(delta.size),
        "rms_uv": float(np.sqrt(np.mean(np.square(delta)))),
        "max_abs_uv": float(np.max(np.abs(delta))),
        "count_abs_gt_1e_6": int(np.count_nonzero(np.abs(delta) > 1e-6)),
        "candidate_sha256": array_sha256(candidate),
        "reference_sha256": array_sha256(reference),
    }


def select_events(
    peaks: dict[str, np.ndarray], selections: list[dict[str, object]]
) -> pd.DataFrame:
    rows = []
    times = peaks["times_samples"]
    channels = peaks["channels"]
    strata = {
        "ap191_neighborhood": (0, 18),
        "interior": (18, 174),
        "top_edge": (174, 182),
    }
    for selection in selections:
        lo = int(selection["start_frame"]) + WAVEFORM_RADIUS
        hi = int(selection["end_frame_exclusive"]) - WAVEFORM_RADIUS
        for stratum, (c0, c1) in strata.items():
            candidates = np.flatnonzero(
                (times >= lo) & (times < hi) & (channels >= c0) & (channels < c1)
            )
            if candidates.size > EVENTS_PER_STRATUM:
                positions = np.linspace(0, candidates.size - 1, EVENTS_PER_STRATUM).round().astype(int)
                candidates = candidates[positions]
            for peak_index in candidates:
                rows.append(
                    {
                        "event_id": len(rows),
                        "peak_index": int(peak_index),
                        "window": selection["label"],
                        "stratum": stratum,
                        "local_sample": int(times[peak_index]),
                        "channel_index": int(channels[peak_index]),
                        "channel_id": TARGET_IDS[int(channels[peak_index])],
                    }
                )
    return pd.DataFrame(rows)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    av = np.asarray(a, dtype=np.float64).ravel()
    bv = np.asarray(b, dtype=np.float64).ravel()
    denom = np.linalg.norm(av) * np.linalg.norm(bv)
    return float(np.dot(av, bv) / denom) if denom else float("nan")


def waveform_metrics(
    window_events: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    read_start: int,
) -> list[dict[str, object]]:
    rows = []
    for event in window_events.itertuples(index=False):
        center = int(event.local_sample) - read_start
        sample_slice = slice(center - WAVEFORM_RADIUS, center + WAVEFORM_RADIUS + 1)
        channel_slice = slice(max(0, event.channel_index - 4), min(len(TARGET_IDS), event.channel_index + 5))
        references = {
            "rounded_exact_dd": arrays["rounded_exact_dd"][sample_slice, channel_slice],
            "none_s0": arrays["none_s0"][sample_slice, channel_slice],
        }
        for arm, traces in arrays.items():
            waveform = traces[sample_slice, channel_slice]
            for reference_name, reference in references.items():
                delta = waveform.astype(np.float64) - reference.astype(np.float64)
                peak_ref = float(np.max(np.abs(reference)))
                rows.append(
                    {
                        "event_id": int(event.event_id),
                        "window": event.window,
                        "stratum": event.stratum,
                        "channel_id": event.channel_id,
                        "arm": arm,
                        "reference": reference_name,
                        "local_channel_count": waveform.shape[1],
                        "cosine": cosine(waveform, reference),
                        "rms_uv": float(np.sqrt(np.mean(np.square(delta)))),
                        "max_abs_uv": float(np.max(np.abs(delta))),
                        "peak_abs_ratio": float(np.max(np.abs(waveform)) / peak_ref) if peak_ref else np.nan,
                    }
                )
    return rows


def plot_overlays(
    output: Path,
    representatives: list[tuple[object, dict[str, np.ndarray], int]],
    sampling_frequency: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shown = (
        "none_s0",
        "rounded_exact_dd",
        "rounded_kriging",
        "unrounded_nearest",
        "unrounded_idw",
        "unrounded_kriging",
    )
    colors = dict(zip(shown, ("#777777", "#000000", "#2ca02c", "#d62728", "#ff7f0e", "#1f77b4")))
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True)
    time_ms = (np.arange(-WAVEFORM_RADIUS, WAVEFORM_RADIUS + 1) / sampling_frequency) * 1000
    for axis, item in zip(axes.ravel(), representatives):
        event, arrays, read_start = item
        center = int(event.local_sample) - read_start
        sl = slice(center - WAVEFORM_RADIUS, center + WAVEFORM_RADIUS + 1)
        for arm in shown:
            axis.plot(time_ms, arrays[arm][sl, event.channel_index], color=colors[arm], lw=1, label=arm)
        axis.axvline(0, color="#bbbbbb", lw=0.7)
        axis.set_title(f"{event.window} / {event.stratum} / {event.channel_id}", fontsize=9)
        axis.set_ylabel("µV")
    for axis in axes[-1]:
        axis.set_xlabel("Time from cached peak (ms)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle("EM.2a cached-event voltage overlays", y=0.935)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(output / "WAVEFORM_OVERLAYS.png", dpi=180)
    fig.savefig(output / "WAVEFORM_OVERLAYS.pdf")
    plt.close(fig)


def preserve_failure(error: BaseException) -> None:
    if "--output-dir" not in sys.argv:
        return
    index = sys.argv.index("--output-dir") + 1
    if index >= len(sys.argv):
        return
    output = Path(sys.argv[index])
    if output.is_dir() and not (output / "COMPLETE.json").exists():
        atomic_json(
            output / "FAILURE.json",
            {
                "status": "failed",
                "exception_type": type(error).__name__,
                "exception": str(error),
                "traceback": traceback.format_exc(),
                "sort_launched": False,
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    import spikeinterface as si
    from spikeinterface.preprocessing import get_spatial_interpolation_kernel
    from spikeinterface.preprocessing.preprocessing_tools import get_kriging_channel_weights

    started_wall = time.monotonic()
    started_cpu = time.process_time()
    state = preflight(si)
    if sha256(FIELD) != EXPECTED_FIELD_SHA256:
        raise RuntimeError("unrounded source field hash differs")
    if sha256(SAVED_D2L_FIELD) != EXPECTED_SAVED_FIELD_SHA256:
        raise RuntimeError("saved D2L field hash differs")
    if sha256(PEAKS) != EXPECTED_PEAKS_SHA256:
        raise RuntimeError("cached peak hash differs")

    with np.load(FIELD, allow_pickle=False) as saved:
        field_centers = np.asarray(saved["time_s"], dtype=np.float64)
        field_values = np.asarray(saved["displacement_um"], dtype=np.float64)[:, 0]
        field_sign = str(saved["sign_convention"])
    with np.load(SAVED_D2L_FIELD, allow_pickle=False) as saved:
        d2l_source_times = np.asarray(saved["source_time_s"], dtype=np.float64)
        d2l_values = np.asarray(saved["displacement"], dtype=np.float64)[0]
    if field_sign != "corrected = observed - displacement":
        raise RuntimeError("unrounded field sign differs")
    if not np.array_equal(np.interp(d2l_source_times, field_centers, field_values), d2l_values):
        raise RuntimeError("unrounded source does not exactly reproduce saved D2L field")

    peaks = {}
    with np.load(PEAKS, allow_pickle=False) as saved:
        for key in ("times_samples", "channels"):
            peaks[key] = np.asarray(saved[key])
    selections = state["selections"]
    events = select_events(peaks, selections)
    events.to_csv(args.output_dir / "EVENTS.csv", index=False)
    if len(events) == 0:
        raise RuntimeError("no cached events intersect frozen snippets")

    parent = state["parent"]
    parent_ids = state["parent_ids"]
    source_geometry = state["source_geometry"]
    target_geometry = state["target_geometry"]
    full_geometry = state["full_geometry"]
    full_ids = state["full_ids"]
    insertion, interpolation_receipt = interpolation_insertion(
        parent_ids,
        full_ids,
        source_geometry,
        full_geometry,
        get_kriging_channel_weights,
    )
    target_indices = np.asarray([parent_ids.index(value) for value in TARGET_IDS])
    source_origin = dd_source_time_origin(parent, state["dd_receipt"])
    fs = float(parent.get_sampling_frequency())
    dd_chunk = round(fs)
    knots = state["knots"]
    lattice_centers = knots.time_s.to_numpy(np.float64)
    lattice_values = knots.q_um.to_numpy(np.float64)
    dd_sl = load_relative_recording(si, DD / "inputs/S_L/recording.json")
    dd_s0 = load_relative_recording(si, DD / "inputs/S_0/recording.json")

    all_times = []
    for selection in selections:
        start, end = int(selection["start_frame"]), int(selection["end_frame_exclusive"])
        # Include the waveform margins because cached-event extraction renders
        # those samples with their own field states as well.
        all_times.append(
            source_origin
            + np.arange(start - WAVEFORM_RADIUS, end + WAVEFORM_RADIUS) / fs
        )
    all_times = np.concatenate(all_times)
    rounded_screen_states = np.unique(
        sample_stepwise_shifts(lattice_centers, lattice_values, all_times, cell_width_s=DT)
    )
    unrounded_screen_states = np.unique(assign_stepwise(field_centers, field_values, all_times))
    # Audit all rounded W2 states and the unrounded W2 extrema so border behavior
    # is checked even when a frozen voltage snippet does not contain that state.
    rounded_kernel_states = np.unique(lattice_values)
    unrounded_kernel_states = np.unique(
        np.r_[unrounded_screen_states, d2l_values.min(), d2l_values.max()]
    )

    kernel_cache: dict[str, dict[float, tuple[np.ndarray, np.ndarray]]] = {}
    kernel_evidence = {}
    for arm, spec in ARM_SPECS.items():
        if spec["method"] in {"none", "exact_coordinate"}:
            continue
        states_for_arm = rounded_kernel_states if spec["field"] == "lattice_rounded" else unrounded_kernel_states
        cache = {}
        rows = []
        for displacement in states_for_arm:
            kernel, structural_zero = effective_kernel(
                get_spatial_interpolation_kernel,
                insertion,
                full_geometry,
                target_geometry,
                float(displacement),
                spec,
            )
            cache[float(displacement)] = (kernel, structural_zero)
            rows.append(
                {
                    "displacement_um": float(displacement),
                    "effective_kernel_sha256": array_sha256(kernel),
                    "structural_zero_targets": int(structural_zero.sum()),
                    "column_sum_min": float(kernel.sum(axis=0).min()),
                    "column_sum_max": float(kernel.sum(axis=0).max()),
                }
            )
        kernel_cache[arm] = cache
        kernel_evidence[arm] = {
            "spec": spec,
            "states": rows,
            "combined_effective_kernel_sha256": hashlib.sha256(
                b"".join(cache[key][0].tobytes() for key in sorted(cache))
            ).hexdigest(),
        }
    atomic_json(args.output_dir / "KERNELS.json", kernel_evidence)
    outer_border_audit = {
        arm: [
            {
                "displacement_um": row["displacement_um"],
                "structural_zero_targets": row["structural_zero_targets"],
            }
            for row in evidence["states"]
            if row["structural_zero_targets"]
        ]
        for arm, evidence in kernel_evidence.items()
    }

    metric_rows = []
    zero_rows = []
    waveform_rows = []
    representative_items = []
    pair_accumulators = {
        pair: {"sum_sq": 0.0, "count": 0, "max_abs": 0.0, "different": 0}
        for pair in COMPARISONS
    }
    voltage_bytes_read = 0
    for selection in selections:
        start = int(selection["start_frame"])
        end = int(selection["end_frame_exclusive"])
        read_start = start - WAVEFORM_RADIUS
        read_end = end + WAVEFORM_RADIUS
        source, parent_bytes = read_aligned(parent, read_start, read_end, dd_chunk)
        stored_sl = dd_sl.get_traces(start_frame=read_start, end_frame=read_end)
        stored_s0 = dd_s0.get_traces(start_frame=read_start, end_frame=read_end)
        voltage_bytes_read += parent_bytes + stored_sl.nbytes + stored_s0.nbytes
        if not np.array_equal(source[:, target_indices], stored_s0):
            delta = np.max(np.abs(source[:, target_indices].astype(float) - stored_s0.astype(float)))
            if delta > 1e-6:
                raise RuntimeError(f"S0 differs from accepted parent crop by {delta}")

        times = source_origin + np.arange(read_start, read_end) / fs
        rounded = sample_stepwise_shifts(
            lattice_centers, lattice_values, times, cell_width_s=DT
        ).astype(np.float64)
        unrounded = assign_stepwise(field_centers, field_values, times)
        arrays = {"none_s0": stored_s0, "rounded_exact_dd": stored_sl}
        structural_zeros = {
            "none_s0": np.zeros(stored_s0.shape, dtype=bool),
            "rounded_exact_dd": exact_zero_mask(source_geometry, target_geometry, rounded),
        }
        for arm, spec in ARM_SPECS.items():
            if arm in arrays:
                continue
            states_for_arm = rounded if spec["field"] == "lattice_rounded" else unrounded
            arrays[arm], structural_zeros[arm] = render(source, states_for_arm, kernel_cache[arm])

        central = slice(WAVEFORM_RADIUS, WAVEFORM_RADIUS + end - start)
        for candidate, reference in COMPARISONS:
            metric_rows.append(
                metric_row(
                    selection["label"],
                    candidate,
                    reference,
                    arrays[candidate][central],
                    arrays[reference][central],
                )
            )
            delta = arrays[candidate][central].astype(np.float64) - arrays[reference][central]
            accumulator = pair_accumulators[(candidate, reference)]
            accumulator["sum_sq"] += float(np.square(delta).sum())
            accumulator["count"] += int(delta.size)
            accumulator["max_abs"] = max(accumulator["max_abs"], float(np.max(np.abs(delta))))
            accumulator["different"] += int(np.count_nonzero(np.abs(delta) > 1e-6))

        for arm, mask in structural_zeros.items():
            window_mask = mask[central]
            for channel, channel_id in enumerate(TARGET_IDS):
                zero_rows.append(
                    {
                        "window": selection["label"],
                        "arm": arm,
                        "channel_index": channel,
                        "channel_id": channel_id,
                        "structural_zero_frames": int(window_mask[:, channel].sum()),
                        "frame_count": int(window_mask.shape[0]),
                        "structural_zero_fraction": float(window_mask[:, channel].mean()),
                    }
                )

        window_events = events.loc[events.window == selection["label"]]
        waveform_rows.extend(waveform_metrics(window_events, arrays, read_start))
        for stratum in ("ap191_neighborhood", "interior", "top_edge"):
            subset = window_events.loc[window_events.stratum == stratum]
            if len(subset):
                representative_items.append((next(subset.itertuples(index=False)), arrays, read_start))

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(args.output_dir / "TRACE_METRICS.csv", index=False)
    zero_table = pd.DataFrame(zero_rows)
    zero_table.to_csv(args.output_dir / "ZERO_FILL_BY_CHANNEL.csv", index=False)
    waveform_table = pd.DataFrame(waveform_rows)
    waveform_table.to_csv(args.output_dir / "WAVEFORM_METRICS.csv", index=False)
    plot_overlays(args.output_dir, representative_items, fs)

    aggregate_comparisons = []
    for (candidate, reference), accumulator in pair_accumulators.items():
        aggregate_comparisons.append(
            {
                "candidate": candidate,
                "reference": reference,
                "rms_uv": float(np.sqrt(accumulator["sum_sq"] / accumulator["count"])),
                "max_abs_uv": accumulator["max_abs"],
                "count_abs_gt_1e_6": accumulator["different"],
                "values_compared": accumulator["count"],
                "numerically_identical_at_1e_6": accumulator["different"] == 0,
            }
        )
    aggregate_lookup = {
        (row["candidate"], row["reference"]): row for row in aggregate_comparisons
    }
    unrounded_distinct = all(
        not aggregate_lookup[pair]["numerically_identical_at_1e_6"]
        for pair in (
            ("unrounded_nearest", "unrounded_idw"),
            ("unrounded_nearest", "unrounded_kriging"),
            ("unrounded_idw", "unrounded_kriging"),
        )
    )
    rounded_nearest_idw_redundant = aggregate_lookup[
        ("rounded_nearest", "rounded_idw")
    ]["numerically_identical_at_1e_6"]
    differences_beyond_ap191 = bool(
        (
            (waveform_table.stratum == "interior")
            & waveform_table.arm.str.startswith("unrounded_")
            & (waveform_table.reference == "rounded_exact_dd")
            & (waveform_table.rms_uv > 1e-6)
        ).any()
    )
    waveform_summary = (
        waveform_table.groupby(["arm", "reference"], sort=True)
        .agg(
            events=("event_id", "count"),
            median_cosine=("cosine", "median"),
            median_rms_uv=("rms_uv", "median"),
            max_abs_uv=("max_abs_uv", "max"),
            median_peak_abs_ratio=("peak_abs_ratio", "median"),
        )
        .reset_index()
        .to_dict(orient="records")
    )
    zero_summary = (
        zero_table.groupby("arm", sort=True)
        .agg(
            structural_zero_frame_channels=("structural_zero_frames", "sum"),
            possible_frame_channels=("frame_count", "sum"),
            channels_ever_zero=("structural_zero_frames", lambda value: int(np.count_nonzero(value))),
        )
        .reset_index()
    )
    zero_summary["structural_zero_fraction"] = (
        zero_summary.structural_zero_frame_channels / zero_summary.possible_frame_channels
    )
    result = {
        "schema": "em2a-operator-screen-v1",
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": {
            "unrounded_operator_arms_materially_distinct": unrounded_distinct,
            "rounded_nearest_and_idw_redundant_at_1e_6": rounded_nearest_idw_redundant,
            "differences_are_not_only_ap191": differences_beyond_ap191,
            "recommended_nonredundant_em2b_arms": [
                "none_s0",
                "rounded_exact_dd",
                "unrounded_kriging",
                "unrounded_idw",
                "unrounded_nearest",
                "rounded_kriging_bridge",
            ],
            "primary_comparison": "rounded_exact_dd versus unrounded_kriging",
            "sort_authorized_or_launched": False,
        },
        "interpretation": (
            "Snippet-level voltage and cached-waveform distinctions establish nonredundancy and isolate field/kernel effects; "
            "they do not establish sorting benefit, cell identity, or purity."
        ),
        "environment": {
            "interpreter": sys.executable,
            "python_version": sys.version.split()[0],
            "spikeinterface_version": si.__version__,
        },
        "inputs": {
            "field": {"path": str(FIELD), "sha256": EXPECTED_FIELD_SHA256},
            "saved_d2l_field": {"path": str(SAVED_D2L_FIELD), "sha256": EXPECTED_SAVED_FIELD_SHA256},
            "saved_d2l_exact_match_at_w2_centers": True,
            "cached_peaks": {"path": str(PEAKS), "sha256": EXPECTED_PEAKS_SHA256},
            "frozen_selections_sha256": sha256(FROZEN / "SELECTIONS.json"),
            "parent_provenance_sha256": state["compact_hashes"]["provenance"],
        },
        "field_contract": {
            "unrounded_sign": field_sign,
            "unrounded_reference_subtraction": False,
            "unrounded_time_assignment": "0.25-s half-open source-field cells; exact boundary uses later cell",
            "rounded_time_assignment": "DD 0.25-s half-open lattice cells; exact boundary uses later cell",
            "unrounded_states_screened": int(len(unrounded_screen_states)),
            "rounded_states_screened": rounded_screen_states.tolist(),
            "unrounded_kernel_states_audited": int(len(unrounded_kernel_states)),
            "rounded_kernel_states_audited": rounded_kernel_states.tolist(),
        },
        "ap191_interpolation": interpolation_receipt,
        "arm_specs": ARM_SPECS,
        "aggregate_comparisons": aggregate_comparisons,
        "outer_border_kernel_audit": outer_border_audit,
        "zero_fill_summary": zero_summary.to_dict(orient="records"),
        "waveform_summary": waveform_summary,
        "cached_events": {
            "count": int(len(events)),
            "waveform_samples": 2 * WAVEFORM_RADIUS + 1,
            "selection": "up to 16 time-ordered, evenly spaced cached S_L peaks per frozen-window stratum",
            "strata": ["ap191_neighborhood", "interior", "top_edge"],
        },
        "resources": {
            "cpu_s": time.process_time() - started_cpu,
            "wall_s": time.monotonic() - started_wall,
            "logical_voltage_bytes_read": voltage_bytes_read,
        },
        "constraints": {
            "cpu_only": True,
            "sort_launched": False,
            "rf_run": False,
            "outer_holdout_accessed": False,
            "voltage_exported": False,
        },
    }
    atomic_json(args.output_dir / "RESULT.json", result)
    files = []
    for path in sorted(args.output_dir.iterdir()):
        if path.name not in {"MANIFEST.json", "COMPLETE.json"}:
            files.append({"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    atomic_json(args.output_dir / "MANIFEST.json", {"files": files})
    atomic_json(
        args.output_dir / "COMPLETE.json",
        {
            "status": "complete",
            "result_sha256": sha256(args.output_dir / "RESULT.json"),
            "manifest_sha256": sha256(args.output_dir / "MANIFEST.json"),
        },
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        preserve_failure(error)
        raise
