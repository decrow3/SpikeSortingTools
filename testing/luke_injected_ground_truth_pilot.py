"""CPU-only injected-ground-truth diagnostic for Luke imec1.

This is an adapter around :mod:`luke_injected_ground_truth_benchmark`; it does
not alter that sealed prospective scaffold.  It deliberately reuses only the
previously reviewed discovery cohort and previously used discovery windows.
Consequently every result is diagnostic, never confirmatory.

The runner extracts 10 raw-domain event snippets from manually reviewed neural
events, qualifies each against a *different* reviewed event from the same
discovery unit/window, injects tapered float32 templates into paired short raw
backgrounds, and traces the injected-minus-uninjected delta through the current
CPU conditioning stages.  It neither estimates motion nor invokes a sorter.

Run in the SpikeInterface/Kilosort environment::

    python testing/luke_injected_ground_truth_pilot.py --run
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

from testing.luke_conditioning_stage_audit import (
    SATURATION_UV,
    load_raw_recordings,
    robust_sigma,
    stage_arrays,
)
from testing.luke_injected_ground_truth_benchmark import (
    InjectionEvent,
    inject_float32_raw_domain,
    template_sha256,
    validate_template,
)
from testing.luke_two_axis_pilot import N_CHANNELS


DEFAULT_KEY = Path(
    "testing/outputs/luke_multichannel_event_validation/imec1/review_key.csv"
)
DEFAULT_LABELS = Path(
    "testing/outputs/luke_multichannel_event_validation/imec1/review_labels.csv"
)
DEFAULT_OUTPUT = Path("testing/outputs/luke_injected_ground_truth_pilot")
STATUS = "diagnostic_discovery_only_not_confirmatory"


@dataclass(frozen=True)
class BackgroundWindow:
    name: str
    start_s: float
    duration_s: float
    role: str


# Both epochs were selected and repeatedly inspected before this benchmark.
BACKGROUNDS = (
    BackgroundWindow("quiet_discovery", 3951.0, 0.5, "prior quiet negative control"),
    BackgroundWindow("pathological_discovery", 8160.0, 0.5, "prior registration-outlier window"),
)

N_DONORS = 10
TEMPLATE_HALF_SAMPLES = 60
TEMPLATE_CHANNEL_RADIUS = 16
EDGE_GUARD_SAMPLES = 8
ANALYSIS_HALF_SAMPLES = 60
GUARD_S = 0.08
AMPLITUDE_SCALES = (0.5, 0.75, 1.0, 1.25, 1.5)
REPORTED_STAGES = (
    "raw",
    "phase_float32",
    "phase_clip_car",
    "phase_clip_car_highpass",
    "current_source_int16_car_highpass",
)
CONFLICTING_RAW_PATTERNS = (
    "luke_draw_prospective_holdout_events.py",
    "luke_seal_holdout_windows.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--review-key", type=Path, default=DEFAULT_KEY)
    parser.add_argument("--review-labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-donors", type=int, default=N_DONORS)
    return parser.parse_args()


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    return hashlib.sha256(
        f"{array.dtype.str}|{array.shape}".encode("ascii") + array.tobytes()
    ).hexdigest()


def conflicting_raw_processes() -> list[dict[str, str | int]]:
    """Best-effort guard against competing prospective-holdout raw scans."""
    conflicts = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                errors="replace"
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(pattern in command for pattern in CONFLICTING_RAW_PATTERNS):
            conflicts.append({"pid": int(entry.name), "command": command.strip()})
    return conflicts


def build_discovery_pairs(
    key: pd.DataFrame, labels: pd.DataFrame, n_donors: int = N_DONORS
) -> pd.DataFrame:
    """Select deterministic extraction/qualification pairs from discovery data."""
    reviewed = key.merge(labels, on="review_id", validate="one_to_one")
    neural = reviewed.loc[reviewed["review_label"].eq("neural")].copy()
    if len(neural) < n_donors * 2:
        raise ValueError("too few manually reviewed neural discovery events")
    neural["snr_rank"] = neural.groupby(["unit_id", "window"])["peak_snr"].rank(
        pct=True, method="first"
    )
    # Round-robin over the three repeated discovery families and their windows,
    # then spread extraction choices over within-family SNR rank.
    groups = [
        group.sort_values(["snr_rank", "review_id"]).reset_index(drop=True)
        for _, group in neural.groupby(["unit_id", "window"], sort=True)
        if len(group) >= 2
    ]
    candidates: list[pd.Series] = []
    target_quantiles = (0.15, 0.35, 0.55, 0.75, 0.9)
    for quantile in target_quantiles:
        for group in groups:
            index = int(np.argmin(np.abs(group["snr_rank"].to_numpy() - quantile)))
            row = group.iloc[index]
            if row["review_id"] not in {item["review_id"] for item in candidates}:
                candidates.append(row)
            if len(candidates) == n_donors:
                break
        if len(candidates) == n_donors:
            break
    if len(candidates) < n_donors:
        raise ValueError("could not form requested balanced donor set")

    rows = []
    donor_ids = {str(item["review_id"]) for item in candidates}
    used_qualifiers: set[str] = set()
    for donor in candidates:
        pool = neural.loc[
            neural["unit_id"].eq(donor["unit_id"])
            & neural["window"].eq(donor["window"])
            & neural["review_id"].ne(donor["review_id"])
            & ~neural["review_id"].isin(donor_ids)
            & ~neural["review_id"].isin(used_qualifiers)
        ].copy()
        if pool.empty:
            raise ValueError(f"no disjoint qualification event for {donor['review_id']}")
        pool["distance"] = np.abs(pool["peak_snr"] - donor["peak_snr"])
        qualifier = pool.sort_values(["distance", "review_id"]).iloc[0]
        used_qualifiers.add(str(qualifier["review_id"]))
        rows.append(
            {
                "template_id": f"T{len(rows) + 1:02d}",
                "donor_review_id": donor["review_id"],
                "donor_sample_index": int(donor["aligned_sample_index"]),
                "donor_unit_id": int(donor["unit_id"]),
                "donor_window": donor["window"],
                "donor_peak_channel": int(donor["peak_channel"]),
                "donor_review_label": donor["review_label"],
                "donor_review_confidence": donor["review_confidence"],
                "donor_peak_snr": float(donor["peak_snr"]),
                "qualifier_review_id": qualifier["review_id"],
                "qualifier_sample_index": int(qualifier["aligned_sample_index"]),
                "qualifier_peak_channel": int(qualifier["peak_channel"]),
                "qualifier_review_label": qualifier["review_label"],
                "qualifier_review_confidence": qualifier["review_confidence"],
                "split_is_disjoint": bool(
                    donor["review_id"] != qualifier["review_id"]
                    and donor["sample_index"] != qualifier["sample_index"]
                ),
            }
        )
    result = pd.DataFrame(rows)
    if not result["split_is_disjoint"].all():
        raise AssertionError("donor extraction and qualification overlap")
    return result


def prepare_template(
    snippet: np.ndarray,
    peak_channel: int,
    *,
    channel_radius: int = TEMPLATE_CHANNEL_RADIUS,
    edge_guard_samples: int = EDGE_GUARD_SAMPLES,
) -> np.ndarray:
    """Baseline-remove, spatially restrict and taper a raw event snippet."""
    values = np.asarray(snippet, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != N_CHANNELS:
        raise ValueError(f"expected (samples, {N_CHANNELS}) raw snippet")
    if not 0 <= peak_channel < N_CHANNELS:
        raise ValueError("peak channel outside AP channels")
    if 2 * edge_guard_samples >= values.shape[0]:
        raise ValueError("edge guard leaves no template interior")
    edge = np.concatenate((values[:edge_guard_samples], values[-edge_guard_samples:]))
    result = values - np.median(edge, axis=0, keepdims=True)
    keep = np.zeros(N_CHANNELS, dtype=bool)
    keep[max(0, peak_channel - channel_radius) : peak_channel + channel_radius + 1] = True
    result[:, ~keep] = 0.0
    taper = np.ones(values.shape[0], dtype=np.float32)
    ramp = np.sin(np.linspace(0, np.pi / 2, edge_guard_samples, dtype=np.float32)) ** 2
    taper[:edge_guard_samples] = ramp
    taper[-edge_guard_samples:] = ramp[::-1]
    taper[[0, -1]] = 0.0
    result *= taper[:, None]
    result[[0, -1]] = 0.0
    return validate_template(result.astype(np.float32), edge_guard_samples=1)


def centered_cosine(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64).ravel()
    second = np.asarray(second, dtype=np.float64).ravel()
    first -= first.mean()
    second -= second.mean()
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    return float(np.dot(first, second) / denominator) if denominator else float("nan")


def align_channel_waveforms(
    donor: np.ndarray, qualifier: np.ndarray, donor_channel: int, qualifier_channel: int
) -> tuple[np.ndarray, np.ndarray]:
    radius = TEMPLATE_CHANNEL_RADIUS
    donor_indices = np.arange(donor_channel - radius, donor_channel + radius + 1)
    qualifier_indices = np.arange(qualifier_channel - radius, qualifier_channel + radius + 1)
    valid = (
        (donor_indices >= 0)
        & (donor_indices < donor.shape[1])
        & (qualifier_indices >= 0)
        & (qualifier_indices < qualifier.shape[1])
    )
    return donor[:, donor_indices[valid]], qualifier[:, qualifier_indices[valid]]


def phase_float32_array(values: np.ndarray, prototype, fs: float) -> np.ndarray:
    """Apply the acquisition phase correction to one small in-memory array."""
    import spikeinterface as si
    from spikeinterface.preprocessing import phase_shift

    recording = si.NumpyRecording(values.astype(np.float32, copy=False), fs)
    recording.set_channel_locations(prototype.get_channel_locations())
    recording.set_property(
        "inter_sample_shift", np.asarray(prototype.get_property("inter_sample_shift"))
    )
    return phase_shift(recording, dtype="float32").get_traces().astype(np.float32)


def nearest_artifact_distance(sample: int, saturation_times: np.ndarray) -> int | None:
    if saturation_times.size == 0:
        return None
    index = int(np.searchsorted(saturation_times, sample))
    neighbors = saturation_times[max(0, index - 1) : index + 1]
    return int(np.min(np.abs(neighbors - sample)))


def build_schedule(
    pairs: pd.DataFrame, n_samples: int, fs: float, saturation_mask: np.ndarray
) -> list[InjectionEvent]:
    margin = max(TEMPLATE_HALF_SAMPLES + 2, int(round(GUARD_S * fs)))
    locations = np.linspace(margin, n_samples - margin - 1, len(pairs), dtype=int)
    saturation_times = np.flatnonzero(np.any(saturation_mask, axis=1))
    # Reserve up to two events for a controlled near-artifact stratum. Offset
    # by 1 ms so the injection is near a raw saturation rather than centered on
    # the same sample, and require separated supports.
    if saturation_times.size:
        near_candidates = saturation_times + int(round(1e-3 * fs))
        near_candidates = near_candidates[
            (near_candidates >= margin) & (near_candidates < n_samples - margin)
        ]
        selected_near: list[int] = []
        for sample in near_candidates:
            if all(abs(int(sample) - other) > 2 * TEMPLATE_HALF_SAMPLES for other in selected_near):
                selected_near.append(int(sample))
            if len(selected_near) == min(2, len(locations)):
                break
        if selected_near:
            locations[-len(selected_near) :] = selected_near
    events = []
    for index, (row, sample) in enumerate(zip(pairs.itertuples(index=False), locations)):
        distance = nearest_artifact_distance(int(sample), saturation_times)
        events.append(
            InjectionEvent(
                event_id=f"{row.template_id}_I{index + 1:02d}",
                template_id=row.template_id,
                sample_index=int(sample),
                amplitude_scale=AMPLITUDE_SCALES[index % len(AMPLITUDE_SCALES)],
                artifact_distance_samples=distance,
            )
        )
    return events


def retention_row(
    event: InjectionEvent,
    template: np.ndarray,
    stage_name: str,
    stage_delta: np.ndarray,
    reference_delta: np.ndarray,
    noise: np.ndarray,
    fs: float,
) -> dict:
    center = int(event.sample_index)
    start = center - ANALYSIS_HALF_SAMPLES
    stop = center + ANALYSIS_HALF_SAMPLES + 1
    observed = stage_delta[start:stop]
    reference = reference_delta[start:stop]
    reference_amplitude = float(np.max(np.abs(reference)))
    observed_amplitude = float(np.max(np.abs(observed)))
    observed_peak = np.unravel_index(int(np.argmax(np.abs(observed))), observed.shape)
    template_peak = np.unravel_index(int(np.argmax(np.abs(template))), template.shape)
    raw_peak_channel = int(template_peak[1])
    injected_peak_snr = float(
        event.amplitude_scale * np.max(np.abs(template[:, raw_peak_channel]))
        / max(float(noise[raw_peak_channel]), np.finfo(float).eps)
    )
    artifact_ms = (
        None
        if event.artifact_distance_samples is None
        else float(event.artifact_distance_samples * 1e3 / fs)
    )
    return {
        **asdict(event),
        "stage": stage_name,
        "injected_peak_snr": injected_peak_snr,
        "snr_bin": "lt6" if injected_peak_snr < 6 else ("6_to_10" if injected_peak_snr < 10 else "ge10"),
        "polarity": "negative" if abs(float(np.min(template))) >= abs(float(np.max(template))) else "positive",
        "artifact_distance_ms": artifact_ms,
        "artifact_proximity": "none_in_background" if artifact_ms is None else ("near_le_2ms" if artifact_ms <= 2 else "far_gt_2ms"),
        "reference_peak_amplitude_counts": reference_amplitude,
        "observed_peak_amplitude_counts": observed_amplitude,
        "amplitude_retention": observed_amplitude / max(reference_amplitude, np.finfo(float).eps),
        "cosine_to_reference": centered_cosine(reference, observed),
        "reference_peak_channel": raw_peak_channel,
        "observed_peak_channel": int(observed_peak[1]),
        "localization_error_channels": int(observed_peak[1] - raw_peak_channel),
    }


def extract_template(recording, sample: int, peak_channel: int) -> np.ndarray:
    snippet = recording.get_traces(
        start_frame=sample - TEMPLATE_HALF_SAMPLES,
        end_frame=sample + TEMPLATE_HALF_SAMPLES + 1,
    ).astype(np.float32)
    return prepare_template(snippet, peak_channel)


def run_pilot(args: argparse.Namespace) -> dict:
    conflicts = conflicting_raw_processes()
    if conflicts:
        raise RuntimeError(
            "prospective holdout raw scan is active; defer this diagnostic: "
            + json.dumps(conflicts)
        )
    pairs = build_discovery_pairs(
        pd.read_csv(args.review_key), pd.read_csv(args.review_labels), args.n_donors
    )
    raw, _, _, fs, gain, weights = load_raw_recordings()
    threshold_counts = SATURATION_UV / gain

    templates: dict[str, np.ndarray] = {}
    qualifications = []
    for row in pairs.itertuples(index=False):
        donor = extract_template(raw, row.donor_sample_index, row.donor_peak_channel)
        qualifier = extract_template(raw, row.qualifier_sample_index, row.qualifier_peak_channel)
        donor_local, qualifier_local = align_channel_waveforms(
            donor, qualifier, row.donor_peak_channel, row.qualifier_peak_channel
        )
        templates[row.template_id] = donor
        qualifications.append(
            {
                "template_id": row.template_id,
                "donor_review_id": row.donor_review_id,
                "qualifier_review_id": row.qualifier_review_id,
                "raw_waveform_cosine": centered_cosine(donor_local, qualifier_local),
                "donor_array_sha256": template_sha256(donor),
                "qualification_is_independent_event": True,
            }
        )
    pairs = pairs.merge(pd.DataFrame(qualifications), on=["template_id", "donor_review_id", "qualifier_review_id"])

    metric_rows: list[dict] = []
    schedule_rows: list[dict] = []
    background_rows: list[dict] = []
    for window in BACKGROUNDS:
        start_frame = int(round(window.start_s * fs))
        stop_frame = start_frame + int(round(window.duration_s * fs))
        background = raw.get_traces(start_frame=start_frame, end_frame=stop_frame).astype(np.float32)
        saturation_mask = np.abs(background) >= threshold_counts
        events = build_schedule(pairs, len(background), fs, saturation_mask)
        injected = inject_float32_raw_domain(background, templates, events, edge_guard_samples=1)
        if not np.array_equal(background, raw.get_traces(start_frame=start_frame, end_frame=stop_frame).astype(np.float32)):
            raise AssertionError("paired source background changed during injection")

        phase_background = phase_float32_array(background, raw, fs)
        phase_injected = phase_float32_array(injected, raw, fs)
        stages_background, _ = stage_arrays(
            background, phase_background, phase_background, fs, threshold_counts, weights
        )
        stages_injected, _ = stage_arrays(
            injected, phase_injected, phase_injected, fs, threshold_counts, weights
        )
        deltas = {name: stages_injected[name] - stages_background[name] for name in REPORTED_STAGES}
        noise = robust_sigma(background, axis=0)
        for event in events:
            template = templates[event.template_id] * np.float32(event.amplitude_scale)
            for stage_name in REPORTED_STAGES:
                reference = deltas[
                    "raw" if stage_name == "raw" else "phase_float32"
                ]
                row = retention_row(
                    event, template, stage_name, deltas[stage_name], reference, noise, fs
                )
                row["background"] = window.name
                metric_rows.append(row)
            schedule_rows.append({"background": window.name, **asdict(event)})
        background_rows.append(
            {
                "background": window.name,
                "role": window.role,
                "start_s": window.start_s,
                "duration_s": window.duration_s,
                "start_frame": start_frame,
                "stop_frame": stop_frame,
                "array_sha256": array_sha256(background),
                "saturated_sample_count": int(np.sum(saturation_mask)),
                "saturated_time_count": int(np.sum(np.any(saturation_mask, axis=1))),
            }
        )

    metrics = pd.DataFrame(metric_rows)
    summary = (
        metrics.groupby(["background", "stage", "snr_bin", "polarity", "artifact_proximity"], dropna=False)
        .agg(
            n=("event_id", "size"),
            median_amplitude_retention=("amplitude_retention", "median"),
            median_cosine=("cosine_to_reference", "median"),
            median_abs_localization_error_channels=("localization_error_channels", lambda x: float(np.median(np.abs(x)))),
        )
        .reset_index()
    )
    final_stage = metrics.loc[
        metrics["stage"].eq("current_source_int16_car_highpass")
    ]
    qualifier_cosines = pairs["raw_waveform_cosine"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(args.output_dir / "donor_manifest.csv", index=False)
    pd.DataFrame(schedule_rows).to_csv(args.output_dir / "injection_schedule.csv", index=False)
    pd.DataFrame(background_rows).to_csv(args.output_dir / "background_manifest.csv", index=False)
    metrics.to_csv(args.output_dir / "retention_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "retention_summary.csv", index=False)
    np.savez_compressed(args.output_dir / "donor_templates.npz", **templates)
    result = {
        "status": STATUS,
        "scope": "CPU stage-level only; no sorter; no motion estimation",
        "donor_count": len(templates),
        "background_count": len(BACKGROUNDS),
        "injection_count": len(schedule_rows),
        "retention_row_count": len(metrics),
        "sampling_frequency_hz": fs,
        "gain_uV_per_count": gain,
        "threshold_counts_500uV": threshold_counts,
        "headline_diagnostics": {
            "qualifier_raw_waveform_cosine_median": float(
                qualifier_cosines.median()
            ),
            "qualifier_raw_waveform_cosine_min": float(qualifier_cosines.min()),
            "qualifier_raw_waveform_cosine_max": float(qualifier_cosines.max()),
            "qualifiers_at_least_0_8": int((qualifier_cosines >= 0.8).sum()),
            "final_stage_median_amplitude_retention": float(
                final_stage["amplitude_retention"].median()
            ),
            "final_stage_median_cosine_to_reference": float(
                final_stage["cosine_to_reference"].median()
            ),
            "final_stage_median_abs_localization_error_channels": float(
                final_stage["localization_error_channels"].abs().median()
            ),
            "final_stage_fraction_abs_localization_error_gt_2_channels": float(
                (final_stage["localization_error_channels"].abs() > 2).mean()
            ),
        },
        "discovery_sources": {
            "review_key": str(args.review_key),
            "review_labels": str(args.review_labels),
            "backgrounds": [asdict(item) for item in BACKGROUNDS],
        },
        "limitations": [
            "All donors and backgrounds were previously inspected discovery material.",
            "The reviewed event detector was negative-peak selected; positive-dominant coverage is absent unless observed post hoc.",
            "Manual labels are descriptive and not biological ground truth.",
            "Qualification uses a distinct reviewed event from the same discovery unit/window, not an independently curated unit identity.",
            "Short-array phase correction has finite-window boundary effects; injections are guarded from edges.",
            "No sorter, unit recovery, duplicate rate, residual fit, or motion endpoint is measured.",
        ],
        "files": [
            "donor_manifest.csv",
            "donor_templates.npz",
            "injection_schedule.csv",
            "background_manifest.csv",
            "retention_metrics.csv",
            "retention_summary.csv",
        ],
    }
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def plan(args: argparse.Namespace) -> dict:
    pairs = build_discovery_pairs(
        pd.read_csv(args.review_key), pd.read_csv(args.review_labels), args.n_donors
    )
    return {
        "status": STATUS,
        "donor_count": len(pairs),
        "backgrounds": [asdict(item) for item in BACKGROUNDS],
        "stages": list(REPORTED_STAGES),
        "raw_access": False,
        "sorter": False,
        "motion_estimation": False,
        "donor_pairs": pairs.to_dict(orient="records"),
    }


def main() -> None:
    args = parse_args()
    if not args.run and not args.plan_only:
        raise SystemExit("Choose --plan-only or --run")
    if args.plan_only:
        print(json.dumps(plan(args), indent=2))
    if args.run:
        print(json.dumps(run_pilot(args), indent=2))


if __name__ == "__main__":
    main()
