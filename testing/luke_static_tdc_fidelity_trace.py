"""Trace accepted KS4 events through the static SI 0.104.8 TDC gates.

This is a diagnostic of the KS4-seeded TDC interface, not a sorter benchmark.
It samples equal numbers of one-to-one replayed and missed KS4 events and asks,
on the exact static-peeler input voltage:

* whether the first-pass fast TDC detector has a nearby peak;
* whether the source KS4 template is spatially eligible at that peak;
* whether TDC's short-window nearest-template rule chooses it;
* its isolated best-fit amplitude and residual improvement; and
* whether the saved full static-peeler output contains a same- or any-label
  event nearby.

The isolated fit intentionally excludes neighboring simultaneous candidates.
Events that pass this local trace but are absent from the full peeler are
therefore classified as full-context competition/peeling failures rather than
as unexplained numerical errors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


RESCUE_ROOT = Path(
    "/mnt/NPX/Luke/20250804/"
    "rescue_pipeline_results_Luke0804_V2V1_g0_imec1"
)
PAIR_RELATIVE = Path(
    "sorter_bakeoff/windows/rapid_motion-8b4978262d/ks4_seeded_peeler_pair"
)
OUTPUT = Path("testing/outputs/luke_static_tdc_fidelity_trace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rescue-root", type=Path, default=RESCUE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--n-events", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--event-tolerance-ms", type=float, default=0.5)
    parser.add_argument("--trace-pad-ms", type=float, default=4.0)
    parser.add_argument("--peak-sign", choices=("neg", "pos", "both"), default="neg")
    parser.add_argument("--detect-threshold", type=float)
    parser.add_argument("--cluster-radius-um", type=float, default=150.0)
    return parser.parse_args()


def one_to_one_reference_mask(
    reference_times: np.ndarray,
    reference_labels: np.ndarray,
    target_times: np.ndarray,
    target_labels: np.ndarray,
    tolerance: int,
) -> np.ndarray:
    """Mark label-preserving reference events using one target at most once."""
    reference_times = np.asarray(reference_times, dtype=np.int64)
    reference_labels = np.asarray(reference_labels, dtype=np.int64)
    target_times = np.asarray(target_times, dtype=np.int64)
    target_labels = np.asarray(target_labels, dtype=np.int64)
    matched = np.zeros(reference_times.size, dtype=bool)
    for unit in np.intersect1d(np.unique(reference_labels), np.unique(target_labels)):
        ref_indices = np.flatnonzero(reference_labels == unit)
        target = target_times[target_labels == unit]
        ref_cursor = 0
        target_cursor = 0
        while ref_cursor < ref_indices.size and target_cursor < target.size:
            ref_index = ref_indices[ref_cursor]
            delta = int(target[target_cursor]) - int(reference_times[ref_index])
            if delta < -tolerance:
                target_cursor += 1
            elif delta > tolerance:
                ref_cursor += 1
            else:
                matched[ref_index] = True
                ref_cursor += 1
                target_cursor += 1
    return matched


def deterministic_balanced_sample(
    replayed: np.ndarray, n_events: int, seed: int
) -> np.ndarray:
    """Draw equal replayed/missed controls without replacement."""
    replayed = np.asarray(replayed, dtype=bool)
    if n_events < 2:
        raise ValueError("At least two events are required")
    rng = np.random.default_rng(seed)
    half = n_events // 2
    groups = [np.flatnonzero(replayed), np.flatnonzero(~replayed)]
    if any(group.size < half for group in groups):
        raise ValueError("A replay stratum is too small for the requested balanced sample")
    selected = np.concatenate(
        [rng.choice(group, size=half, replace=False) for group in groups]
    )
    if n_events % 2:
        larger = max(groups, key=len)
        remaining = np.setdiff1d(larger, selected, assume_unique=False)
        selected = np.r_[selected, rng.choice(remaining, size=1, replace=False)]
    return np.sort(selected)


def nearby_output(
    time: int,
    label: int,
    target_times: np.ndarray,
    target_labels: np.ndarray,
    tolerance: int,
) -> tuple[bool, bool]:
    left = int(np.searchsorted(target_times, time - tolerance, side="left"))
    right = int(np.searchsorted(target_times, time + tolerance, side="right"))
    any_label = right > left
    same_label = bool(any_label and np.any(target_labels[left:right] == label))
    return any_label, same_label


def classify_first_gate(
    *,
    fast_peak: bool,
    correct_template_candidate: bool,
    selected_correct_template: bool,
    fitted_amplitude: float,
    static_any_label: bool,
    static_same_label: bool,
) -> str:
    """Assign the earliest directly observed gate; avoid causal overclaiming."""
    if not fast_peak:
        return "no_fast_detector_peak"
    if not correct_template_candidate:
        return "correct_template_outside_candidate_neighborhood"
    if not selected_correct_template:
        return "nearest_template_competition"
    if not np.isfinite(fitted_amplitude) or fitted_amplitude < 0.7:
        return "below_tdc_amplitude_floor"
    if static_same_label:
        return "replayed_same_label"
    if static_any_label:
        return "full_peeler_reassigned_label"
    return "passes_isolated_gates_but_missing_in_full_peeler"


def dense_template(sparse: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = np.zeros((sparse.shape[0], mask.size), dtype=sparse.dtype)
    result[:, mask] = sparse[:, : int(mask.sum())]
    return result


def fit_template(
    traces: np.ndarray,
    sparse_template: np.ndarray,
    channel_mask: np.ndarray,
    center: int,
    nbefore: int,
    shifts: np.ndarray,
) -> dict[str, float | int]:
    """Find TDC's best shift and an isolated least-squares amplitude/fit score."""
    n_active = int(channel_mask.sum())
    template = sparse_template[:, :n_active].astype(np.float64, copy=False)
    norm = float(np.sum(template * template))
    if not n_active or norm <= 0:
        return {
            "shift": 0,
            "amplitude": 0.0,
            "fit_improvement": 0.0,
            "sse": float("nan"),
        }
    best = None
    for shift in shifts:
        start = int(center + shift - nbefore)
        stop = start + template.shape[0]
        waveform = traces[start:stop, :][:, channel_mask].astype(
            np.float64, copy=False
        )
        if waveform.shape != template.shape:
            continue
        distance = float(np.sum((waveform - template) ** 2))
        if best is None or distance < best[0]:
            best = (distance, int(shift), waveform)
    if best is None:
        return {
            "shift": 0,
            "amplitude": float("nan"),
            "fit_improvement": float("nan"),
            "sse": float("nan"),
        }
    _, shift, waveform = best
    amplitude = float(np.sum(template * waveform) / norm)
    baseline_sse = float(np.sum(waveform * waveform))
    fitted_sse = float(np.sum((waveform - amplitude * template) ** 2))
    improvement = 1.0 - fitted_sse / baseline_sse if baseline_sse > 0 else 0.0
    return {
        "shift": shift,
        "amplitude": amplitude,
        "fit_improvement": improvement,
        "sse": fitted_sse,
    }


def detector_peaks(detector, traces: np.ndarray, center: int, tolerance: int):
    (peaks,) = detector.compute(traces, None, None, 0, 0)
    keep = np.abs(peaks["sample_index"].astype(np.int64) - center) <= tolerance
    return peaks[keep]


def choose_peak_trace(
    peaks,
    traces: np.ndarray,
    peeler,
    source_cluster: int,
) -> dict[str, object] | None:
    """Choose the most favorable observed peak for the source template."""
    if peaks.size == 0:
        return None
    from spikeinterface.sortingcomponents.matching.tdc_peeler import (
        get_most_probable_cluster,
    )

    short_templates = peeler.sparse_templates_array_static[:, peeler.slice_short, :]
    rows = []
    for peak in peaks:
        sample = int(peak["sample_index"])
        channel = int(peak["channel_index"])
        candidates = peeler.possible_clusters_by_channel[channel]
        source_candidate = bool(np.any(candidates == source_cluster))
        selected = -1
        if candidates.size:
            selected = int(
                get_most_probable_cluster(
                    traces,
                    short_templates,
                    candidates,
                    sample,
                    peeler.nbefore_short,
                    peeler.nafter_short,
                    peeler.sparsity_mask,
                )
            )
        normalized_amplitude = abs(float(peak["amplitude"])) / float(
            peeler.abs_thresholds[channel]
        )
        rows.append(
            {
                "sample": sample,
                "channel": channel,
                "amplitude": float(peak["amplitude"]),
                "normalized_peak_amplitude": normalized_amplitude,
                "candidate_count": int(candidates.size),
                "source_candidate": source_candidate,
                "selected_cluster": selected,
                "selected_correct": selected == source_cluster,
            }
        )
    # Prefer a correct selection, then source eligibility, then peak strength.
    return max(
        rows,
        key=lambda row: (
            row["selected_correct"],
            row["source_candidate"],
            row["normalized_peak_amplitude"],
        ),
    )


def main() -> None:
    args = parse_args()
    if (
        args.n_events < 2
        or args.event_tolerance_ms <= 0
        or args.trace_pad_ms <= 0
        or args.cluster_radius_um <= 0
        or (args.detect_threshold is not None and args.detect_threshold <= 0)
    ):
        raise ValueError("Sample size and timing parameters must be positive")
    try:
        import spikeinterface
        from spikeinterface.core import Templates
        from spikeinterface.preprocessing import common_reference, highpass_filter
        from spikeinterface.sortingcomponents.matching.tdc_peeler import (
            TridesclousPeeler,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Run in the spike-sort-challengers environment used for SI 0.104.8"
        ) from exc
    if spikeinterface.__version__ != "0.104.8":
        raise RuntimeError(f"Expected SpikeInterface 0.104.8, got {spikeinterface.__version__}")

    from pipeline.bakeoff import _load_si_extractor

    pair = args.rescue_root / PAIR_RELATIVE
    manifest = json.loads((pair / "bakeoff_sort_manifest.json").read_text())
    window = manifest["window"]
    fs = float(window["sampling_frequency_hz"])
    tolerance = int(round(args.event_tolerance_ms * fs / 1000.0))
    pad = int(round(args.trace_pad_ms * fs / 1000.0))
    shared = pair / "shared_inputs"
    reference_times = np.load(shared / "reference_spike_times.npy")
    reference_labels = np.load(shared / "reference_spike_labels.npy")
    unit_ids = np.load(shared / "unit_ids.npy")
    sparse_templates = np.load(shared / "templates.npy")
    sparsity_mask = np.load(shared / "template_sparsity_mask.npy")
    noise_levels = np.load(shared / "noise_levels.npy")
    static_dir = pair / "ks4_seeded_static_peeler"
    static_times = np.load(static_dir / "spike_times.npy")
    static_labels = np.load(static_dir / "spike_labels.npy")
    replayed = one_to_one_reference_mask(
        reference_times,
        reference_labels,
        static_times,
        static_labels,
        tolerance,
    )
    selected = deterministic_balanced_sample(replayed, args.n_events, args.seed)

    recording = _load_si_extractor(args.rescue_root / "recording")
    recording = recording.frame_slice(
        start_frame=int(window["start_frame"]), end_frame=int(window["end_frame"])
    )
    recording.reset_times()
    recording = highpass_filter(recording, freq_min=300.0, dtype="float32")
    recording = common_reference(
        recording, reference="global", operator="median", dtype="float32"
    )
    templates = Templates(
        templates_array=sparse_templates,
        sampling_frequency=fs,
        nbefore=int(manifest["template_summary"]["nbefore"]),
        is_in_uV=False,
        sparsity_mask=sparsity_mask,
        channel_ids=recording.channel_ids,
        unit_ids=unit_ids,
        probe=recording.get_probe(),
    )
    detect_threshold = (
        float(args.detect_threshold)
        if args.detect_threshold is not None
        else float(manifest["config"]["detect_threshold"])
    )
    peeler = TridesclousPeeler(
        recording,
        templates,
        peak_sign=args.peak_sign,
        noise_levels=noise_levels,
        detect_threshold=detect_threshold,
        cluster_radius_um=float(args.cluster_radius_um),
        motion_aware=False,
    )
    if pad <= max(peeler.nbefore + 2, peeler.nafter + 2, peeler.margin):
        raise ValueError("Trace padding is shorter than TDC's required margin")
    unit_to_cluster = {int(unit): index for index, unit in enumerate(unit_ids)}

    rows = []
    for event_number, reference_index in enumerate(selected):
        time = int(reference_times[reference_index])
        label = int(reference_labels[reference_index])
        source_cluster = unit_to_cluster[label]
        source_mask = peeler.sparsity_mask[source_cluster]
        source_sparse_template = peeler.sparse_templates_array_static[source_cluster]
        source_dense_template = dense_template(source_sparse_template, source_mask)
        source_negative_peak = float(np.min(source_dense_template))
        source_positive_peak = float(np.max(source_dense_template))
        source_positive_dominant = source_positive_peak > abs(source_negative_peak)
        start = time - pad
        stop = time + pad + 1
        if start < 0 or stop > recording.get_num_samples():
            continue
        traces = recording.get_traces(
            start_frame=start, end_frame=stop, return_in_uV=False
        ).astype(np.float32, copy=False)
        center = pad
        local_source = traces[
            center - tolerance : center + tolerance + 1, :
        ][:, source_mask]
        local_thresholds = peeler.abs_thresholds[source_mask]
        negative_multiple = float(np.max(-local_source / local_thresholds[None, :]))
        positive_multiple = float(np.max(local_source / local_thresholds[None, :]))
        if args.peak_sign == "neg":
            source_threshold_multiple = negative_multiple
        elif args.peak_sign == "pos":
            source_threshold_multiple = positive_multiple
        else:
            source_threshold_multiple = max(negative_multiple, positive_multiple)
        source_raw_threshold = source_threshold_multiple >= 1.0
        fast = detector_peaks(peeler.fast_spike_detector, traces, center, tolerance)
        used_detector = "fast" if fast.size else "none"
        chosen = choose_peak_trace(fast, traces, peeler, source_cluster)
        source_fit = {
            "shift": 0,
            "amplitude": float("nan"),
            "fit_improvement": float("nan"),
            "sse": float("nan"),
        }
        selected_fit = source_fit.copy()
        if chosen is not None:
            source_fit = fit_template(
                traces,
                peeler.sparse_templates_array_static[source_cluster],
                peeler.sparsity_mask[source_cluster],
                int(chosen["sample"]),
                peeler.nbefore,
                peeler.possible_shifts,
            )
            selected_cluster = int(chosen["selected_cluster"])
            if selected_cluster >= 0:
                selected_fit = fit_template(
                    traces,
                    peeler.sparse_templates_array_static[selected_cluster],
                    peeler.sparsity_mask[selected_cluster],
                    int(chosen["sample"]),
                    peeler.nbefore,
                    peeler.possible_shifts,
                )
        static_any, static_same = nearby_output(
            time, label, static_times, static_labels, tolerance
        )
        correct_candidate = bool(chosen and chosen["source_candidate"])
        selected_correct = bool(chosen and chosen["selected_correct"])
        gate = classify_first_gate(
            fast_peak=bool(fast.size),
            correct_template_candidate=correct_candidate,
            selected_correct_template=selected_correct,
            fitted_amplitude=float(source_fit["amplitude"]),
            static_any_label=static_any,
            static_same_label=static_same,
        )
        rows.append(
            {
                "event_number": event_number,
                "reference_index": int(reference_index),
                "sample_index": time,
                "unit_id": label,
                "one_to_one_static_replayed": bool(replayed[reference_index]),
                "fast_peak_present": bool(fast.size),
                "fast_peak_count": int(fast.size),
                "used_detector": used_detector,
                "source_template_positive_dominant": source_positive_dominant,
                "source_template_negative_peak": source_negative_peak,
                "source_template_positive_peak": source_positive_peak,
                "source_active_raw_detector_threshold_crossing": source_raw_threshold,
                "source_active_max_detector_threshold_multiple": source_threshold_multiple,
                "detected_peak_channel": (
                    int(chosen["channel"]) if chosen is not None else -1
                ),
                "detected_peak_offset_samples": (
                    int(chosen["sample"]) - center if chosen is not None else np.nan
                ),
                "detected_peak_threshold_multiple": (
                    float(chosen["normalized_peak_amplitude"])
                    if chosen is not None
                    else np.nan
                ),
                "candidate_template_count": (
                    int(chosen["candidate_count"]) if chosen is not None else 0
                ),
                "correct_template_candidate": correct_candidate,
                "selected_correct_template": selected_correct,
                "selected_cluster_index": (
                    int(chosen["selected_cluster"]) if chosen is not None else -1
                ),
                "selected_unit_id": (
                    int(unit_ids[int(chosen["selected_cluster"])])
                    if chosen is not None and int(chosen["selected_cluster"]) >= 0
                    else -1
                ),
                "correct_template_best_shift": int(source_fit["shift"]),
                "correct_template_fitted_amplitude": float(source_fit["amplitude"]),
                "correct_template_fit_improvement": float(
                    source_fit["fit_improvement"]
                ),
                "selected_template_fitted_amplitude": float(
                    selected_fit["amplitude"]
                ),
                "selected_template_fit_improvement": float(
                    selected_fit["fit_improvement"]
                ),
                "static_any_label_present": static_any,
                "static_same_label_present": static_same,
                "first_observed_gate": gate,
            }
        )
    events = pd.DataFrame(rows)
    gate_counts = (
        events.groupby(["one_to_one_static_replayed", "first_observed_gate"])
        .size()
        .rename("n_events")
        .reset_index()
    )
    gate_counts["fraction_within_replay_stratum"] = gate_counts.n_events / gate_counts[
        "one_to_one_static_replayed"
    ].map(events.one_to_one_static_replayed.value_counts())
    strata = []
    for replay_status, group in events.groupby("one_to_one_static_replayed"):
        strata.append(
            {
                "one_to_one_static_replayed": bool(replay_status),
                "n_events": len(group),
                "fast_peak_fraction": float(group.fast_peak_present.mean()),
                "source_active_raw_detector_threshold_fraction": float(
                    group.source_active_raw_detector_threshold_crossing.mean()
                ),
                "correct_template_candidate_fraction": float(
                    group.correct_template_candidate.mean()
                ),
                "selected_correct_template_fraction": float(
                    group.selected_correct_template.mean()
                ),
                "below_amplitude_floor_fraction": float(
                    (group.correct_template_fitted_amplitude < 0.7).mean()
                ),
                "above_nominal_amplitude_ceiling_fraction": float(
                    (group.correct_template_fitted_amplitude > 1.4).mean()
                ),
                "median_correct_template_amplitude": float(
                    group.correct_template_fitted_amplitude.median()
                ),
                "median_correct_template_fit_improvement": float(
                    group.correct_template_fit_improvement.median()
                ),
            }
        )
    stratum_summary = pd.DataFrame(strata)
    polarity_summary = (
        events.groupby(
            ["one_to_one_static_replayed", "source_template_positive_dominant"]
        )
        .agg(
            n_events=("event_number", "size"),
            source_active_raw_detector_threshold_fraction=(
                "source_active_raw_detector_threshold_crossing",
                "mean",
            ),
            correct_template_candidate_fraction=(
                "correct_template_candidate",
                "mean",
            ),
            selected_correct_template_fraction=(
                "selected_correct_template",
                "mean",
            ),
            median_correct_template_amplitude=(
                "correct_template_fitted_amplitude",
                "median",
            ),
        )
        .reset_index()
    )
    unit_positive = {}
    for cluster_index, unit_id in enumerate(unit_ids):
        full_template = dense_template(
            peeler.sparse_templates_array_static[cluster_index],
            peeler.sparsity_mask[cluster_index],
        )
        unit_positive[int(unit_id)] = bool(
            np.max(full_template) > abs(np.min(full_template))
        )
    reference_positive = np.array(
        [unit_positive[int(unit_id)] for unit_id in reference_labels], dtype=bool
    )
    population_rows = []
    for positive in (False, True):
        mask = reference_positive == positive
        population_rows.append(
            {
                "source_template_positive_dominant": positive,
                "reference_event_count": int(mask.sum()),
                "reference_event_fraction": float(mask.mean()),
                "one_to_one_static_replayed_count": int(replayed[mask].sum()),
                "one_to_one_static_replayed_fraction": float(replayed[mask].mean()),
            }
        )
    population_polarity = pd.DataFrame(population_rows)
    request = {
        "schema_version": "static-tdc-fidelity-trace-v1",
        "source_peeler_request_digest": manifest["request_digest"],
        "window_request_digest": window["request_digest"],
        "spikeinterface_version": spikeinterface.__version__,
        "n_requested_events": args.n_events,
        "n_traced_events": len(events),
        "seed": args.seed,
        "event_tolerance_ms": args.event_tolerance_ms,
        "trace_pad_ms": args.trace_pad_ms,
        "trace_peak_sign": args.peak_sign,
        "trace_detect_threshold": detect_threshold,
        "trace_cluster_radius_um": args.cluster_radius_um,
        "source_static_peeler_peak_sign": "neg",
        "balanced_sampling": "equal one-to-one same-label replayed and missed KS4 events",
        "fit_context": "isolated source/selected template fit without neighbor regressors",
        "tdc_amplitude_behavior": (
            "amplitude <0.7 is rejected; amplitude >1.4 is retained with amplitude "
            "forced to 1 for another peeling pass in SI 0.104.8"
        ),
        "limitations": [
            "The balanced sample estimates gate enrichment, not population prevalence.",
            "Isolated amplitude fits omit simultaneous-neighbor regression used by the full peeler.",
            "The fine detector is not assigned a first-stage gate because full TDC invokes it only after earlier subtraction loops.",
            "A nearby detector peak does not prove that the biological KS4 event caused it.",
        ],
        "complete": True,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    events.to_csv(args.output_dir / "event_gate_trace.csv", index=False)
    gate_counts.to_csv(args.output_dir / "gate_counts.csv", index=False)
    stratum_summary.to_csv(args.output_dir / "replay_stratum_summary.csv", index=False)
    polarity_summary.to_csv(args.output_dir / "template_polarity_summary.csv", index=False)
    population_polarity.to_csv(
        args.output_dir / "population_template_polarity.csv", index=False
    )
    (args.output_dir / "manifest.json").write_text(json.dumps(request, indent=2) + "\n")
    print(stratum_summary.to_string(index=False))
    print("\nFirst observed gates:\n")
    print(gate_counts.to_string(index=False))
    print(f"\nWrote {args.output_dir}")


if __name__ == "__main__":
    main()
