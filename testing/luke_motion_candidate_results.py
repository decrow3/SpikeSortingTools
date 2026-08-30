"""Build spatial-scale and fixed-event recovery evidence for Luke imec1."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from testing.luke_claimmask_window_sweep import (
    event_local_samples,
    events_in_window,
    load_reference_settings,
    local_match_mask,
)
from testing.luke_motion_visualizations import DEPTHS, RELATIVE_TIMES, sample_field
from testing.luke_motion_scale_characterization import decompose_spatial_field, recording_t_start
from testing.luke_motion_candidate_sort import motion_run_path
from testing.luke_upstream_sorter_ablation import DEFAULT_REVIEW, OUTPUT_ROOT, WINDOW


OUTPUT = Path("testing/outputs/luke_motion_candidate_results")
SWEEP_ROOT = Path(
    "/mnt/NPX/Luke/20250804/"
    "dredge_pipeline_results_Luke0804_V2V1_g0_imec1/motion_scale_sweep"
)
CONDITIONS = {
    "No external correction": OUTPUT_ROOT / "sorts/current_no_motion/sorter_output",
    "Zero-displacement identity": OUTPUT_ROOT / "sorts/zero_displacement_identity/sorter_output",
    "Kilosort internal rigid": OUTPUT_ROOT / "sorts/kilosort_internal_rigid/sorter_output",
    "Current DREDGE 150/100": OUTPUT_ROOT.parent / "claimmask_window_sweep/sorts/registration_outlier/claim_off/sorter_output",
    "Rigid from 300/200": OUTPUT_ROOT / "sorts/dredge_rigid_from_300_200/sorter_output",
    "Selected DREDGE 300/200": OUTPUT_ROOT / "sorts/dredge_nr_200_300_split/sorter_output",
    "Selected DREDGE p2 sigma 20": OUTPUT_ROOT / "sorts/dredge_nr_200_300_split_p2_extrapolate/sorter_output",
    "Selected DREDGE p2 sigma 28": OUTPUT_ROOT / "sorts/dredge_nr_200_300_split_p2_sigma28_extrapolate/sorter_output",
    "MEDiCINe default, sigma 10": OUTPUT_ROOT / "sorts/medicine_default_sigma10/sorter_output",
}
GAIN_CONDITIONS = {
    "Rigid gain 0.25": OUTPUT_ROOT / "sorts/dredge_rigid_gain_025/sorter_output",
    "Rigid gain 0.25 p2": OUTPUT_ROOT / "sorts/dredge_rigid_gain_025_p2_extrapolate/sorter_output",
    "Rigid gain 0.50": OUTPUT_ROOT / "sorts/dredge_rigid_gain_050/sorter_output",
    "Rigid gain 0.75": OUTPUT_ROOT / "sorts/dredge_rigid_gain_075/sorter_output",
}
PALETTE = {
    "No external correction": "#71864a",
    "Zero-displacement identity": "#8aa3a8",
    "Kilosort internal rigid": "#8b6fa6",
    "Current DREDGE 150/100": "#777777",
    "Rigid from 300/200": "#d08b27",
    "Selected DREDGE 300/200": "#3569a8",
    "Selected DREDGE p2 sigma 20": "#2a9d8f",
    "Selected DREDGE p2 sigma 28": "#64b5a8",
    "MEDiCINe default, sigma 10": "#b85c7a",
}


def finite_correlation(first: np.ndarray, second: np.ndarray) -> float:
    x, y = np.asarray(first).ravel(), np.asarray(second).ravel()
    valid = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(x[valid], y[valid])[0, 1])


def spatial_evidence() -> pd.DataFrame:
    agreement = pd.read_csv(SWEEP_ROOT / "motion_scale_sweep_agreement.csv")

    def lookup(scope: str, probe: str, left: str, right: str) -> float:
        rows = agreement[
            agreement.scope.eq(scope)
            & agreement.probe.eq(probe)
            & agreement.left_candidate.eq(left)
            & agreement.right_candidate.eq(right)
        ]
        if len(rows) != 1:
            raise ValueError((scope, probe, left, right, len(rows)))
        return float(rows.iloc[0].nonrigid_correlation)

    rows = []
    candidates = {
        "300 µm window / 200 µm step": {
            "split": "dredge_nr_200_300_split",
            "adjacent": ("dredge_nr_100_300", "dredge_nr_200_300_split"),
            "cross_method": ("dredge_nr_200_300_split", "decentralized_nr_200_300"),
            "cross_probe": "dredge_nr_200_300_split",
        },
        "600 µm window / 200 µm step": {
            "split": "dredge_nr_200_600",
            "adjacent": ("dredge_nr_200_600", "dredge_nr_400_600"),
            "cross_method": None,
            "cross_probe": "dredge_nr_200_600",
        },
    }
    for label, spec in candidates.items():
        split_values = [lookup("split_half", probe, spec["split"], spec["split"]) for probe in ("imec0", "imec1")]
        adjacent_values = [lookup("cross_candidate", probe, *spec["adjacent"]) for probe in ("imec0", "imec1")]
        rows.extend(
            [
                {"candidate": label, "criterion": "Split-half reproducibility", "correlation": np.mean(split_values), "probe_basis": "mean of imec0 and imec1"},
                {"candidate": label, "criterion": "Adjacent sampling agreement", "correlation": np.mean(adjacent_values), "probe_basis": "mean of imec0 and imec1"},
            ]
        )
        if spec["cross_method"] is not None:
            method_values = [lookup("cross_candidate", probe, *spec["cross_method"]) for probe in ("imec0", "imec1")]
            rows.append({"candidate": label, "criterion": "Independent-method agreement", "correlation": np.mean(method_values), "probe_basis": "mean of imec0 and imec1"})
        cross = agreement[
            agreement.scope.eq("cross_probe")
            & agreement.left_candidate.eq(spec["cross_probe"])
        ]
        rows.append({"candidate": label, "criterion": "Cross-probe agreement", "correlation": float(cross.iloc[0].nonrigid_correlation), "probe_basis": "imec0 versus imec1"})
    return pd.DataFrame(rows)


def event_recovery() -> pd.DataFrame:
    _, fs = load_reference_settings()
    events = events_in_window(pd.read_csv(DEFAULT_REVIEW), WINDOW)
    events = events[(events.review_label == "neural") & (events.status == "unmatched")].copy()
    samples = event_local_samples(events.sample_index.to_numpy(), WINDOW, fs)
    event_depths = events.peak_depth_um.to_numpy(float)
    motion_path = motion_run_path()
    motion = np.load(motion_path / "motion.npy")
    motion_times = np.load(motion_path / "time_bins.npy")
    motion_depths = np.load(motion_path / "depth_bins.npy")
    gradient = np.gradient(motion, motion_depths, axis=1)
    points = np.column_stack(
        (
            events.sample_index.to_numpy() / fs + recording_t_start("imec1"),
            event_depths,
        )
    )
    event_displacement = RegularGridInterpolator(
        (motion_times, motion_depths), motion, bounds_error=False, fill_value=None
    )(points)
    event_gradient = np.abs(
        RegularGridInterpolator(
            (motion_times, motion_depths), gradient, bounds_error=False, fill_value=None
        )(points)
    )
    tolerance = int(round(0.5e-3 * fs))
    rows = []
    for condition, path in {**CONDITIONS, **GAIN_CONDITIONS}.items():
        times = np.load(path / "spike_times.npy").reshape(-1).astype(np.int64)
        positions = np.load(path / "spike_positions.npy")
        valid = (times >= 0) & (times < int(round(WINDOW.duration_s * fs)))
        present = local_match_mask(samples, event_depths, times[valid], positions[valid, 1], tolerance, 100.0)
        for event_index, ((_, event), recovered) in enumerate(zip(events.iterrows(), present)):
            rows.append(
                {
                    "review_id": event.review_id,
                    "condition": condition,
                    "recovered": bool(recovered),
                    "event_time_s": float(samples[event_index] / fs),
                    "event_depth_um": float(event.peak_depth_um),
                    "selected_displacement_um": float(event_displacement[event_index]),
                    "selected_abs_displacement_um": float(abs(event_displacement[event_index])),
                    "selected_abs_spatial_gradient_um_per_um": float(event_gradient[event_index]),
                }
            )
    return pd.DataFrame(rows)


def sorter_metrics() -> pd.DataFrame:
    baseline = pd.read_csv(OUTPUT_ROOT / "upstream_sorter_ablation_scores.csv")
    candidate = pd.read_csv(OUTPUT_ROOT / "dredge_nr_200_300_split_scores.csv")
    rigid = pd.read_csv(OUTPUT_ROOT / "dredge_rigid_from_300_200_scores.csv")
    identity = pd.read_csv(OUTPUT_ROOT / "zero_displacement_identity_scores.csv")
    internal = pd.read_csv(OUTPUT_ROOT / "kilosort_internal_rigid_scores.csv")
    medicine = pd.read_csv(OUTPUT_ROOT / "medicine_default_sigma10_scores.csv")
    p2_nonrigid = [
        pd.read_csv(OUTPUT_ROOT / f"{condition}_scores.csv")
        for condition in (
            "dredge_nr_200_300_split_p2_extrapolate",
            "dredge_nr_200_300_split_p2_sigma28_extrapolate",
        )
    ]
    gains = [
        pd.read_csv(OUTPUT_ROOT / f"dredge_rigid_gain_{gain}_scores.csv")
        for gain in ("025", "050", "075")
    ]
    gains.append(pd.read_csv(OUTPUT_ROOT / "dredge_rigid_gain_025_p2_extrapolate_scores.csv"))
    baseline = baseline[baseline.population.eq("visual_neural_unmatched")]
    candidate = candidate[candidate.population.eq("visual_neural_unmatched")]
    rigid = rigid[rigid.population.eq("visual_neural_unmatched")]
    identity = identity[identity.population.eq("visual_neural_unmatched")]
    internal = internal[internal.population.eq("visual_neural_unmatched")]
    medicine = medicine[medicine.population.eq("visual_neural_unmatched")]
    p2_nonrigid = [
        frame[frame.population.eq("visual_neural_unmatched")] for frame in p2_nonrigid
    ]
    gains = [frame[frame.population.eq("visual_neural_unmatched")] for frame in gains]
    combined = pd.concat(
        [baseline, identity, internal, rigid, candidate, *p2_nonrigid, medicine, *gains],
        ignore_index=True,
    )
    mapping = {
        "current_no_motion": "No external correction",
        "zero_displacement_identity": "Zero-displacement identity",
        "kilosort_internal_rigid": "Kilosort internal rigid",
        "current_motion": "Current DREDGE 150/100",
        "dredge_rigid_from_300_200": "Rigid from 300/200",
        "dredge_nr_200_300_split": "Selected DREDGE 300/200",
        "dredge_nr_200_300_split_p2_extrapolate": "Selected DREDGE p2 sigma 20",
        "dredge_nr_200_300_split_p2_sigma28_extrapolate": "Selected DREDGE p2 sigma 28",
        "medicine_default_sigma10": "MEDiCINe default, sigma 10",
        "dredge_rigid_gain_025": "Rigid gain 0.25",
        "dredge_rigid_gain_025_p2_extrapolate": "Rigid gain 0.25 p2",
        "dredge_rigid_gain_050": "Rigid gain 0.50",
        "dredge_rigid_gain_075": "Rigid gain 0.75",
    }
    return combined[combined.condition.isin(mapping)].assign(condition=lambda d: d.condition.map(mapping))


def rigid_trace_comparison() -> pd.DataFrame:
    ops_path = CONDITIONS["Kilosort internal rigid"] / "ops.npy"
    ops = np.load(ops_path, allow_pickle=True).item()
    ks_shift = np.asarray(ops["dshift"], dtype=float).reshape(-1)
    batch_duration_s = float(ops["batch_size"]) / float(ops["fs"])
    relative_time_s = (np.arange(len(ks_shift)) + 0.5) * batch_duration_s

    motion_path = motion_run_path()
    dredge_motion = np.load(motion_path / "motion.npy").mean(axis=1)
    dredge_times = np.load(motion_path / "time_bins.npy")
    absolute_times = recording_t_start("imec1") + WINDOW.start_s + relative_time_s
    dredge_rigid = np.interp(absolute_times, dredge_times, dredge_motion)

    # The two estimators use opposite displacement signs. Centering removes the
    # arbitrary registration gauge so their temporal structure can be compared.
    ks_sign_aligned_centered = -(ks_shift - np.median(ks_shift))
    dredge_centered = dredge_rigid - np.median(dredge_rigid)
    return pd.DataFrame(
        {
            "time_s": relative_time_s,
            "kilosort_dshift_um": ks_shift,
            "kilosort_sign_aligned_centered_um": ks_sign_aligned_centered,
            "dredge_rigid_centered_um": dredge_centered,
            "kilosort_abs_step_um": np.r_[np.nan, np.abs(np.diff(ks_shift))],
            "dredge_abs_step_um": np.r_[np.nan, np.abs(np.diff(dredge_rigid))],
        }
    )


def save_spatial_figure(evidence: pd.DataFrame, output: Path) -> None:
    candidates = ["dredge_nr_100_300", "dredge_nr_200_300_split", "dredge_nr_200_600", "dredge_nr_400_600"]
    labels = ["300/100", "300/200", "600/200", "600/400"]
    fig = plt.figure(figsize=(15, 8.5))
    grid = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 1.25], wspace=0.12, hspace=0.12)
    image = None
    for row, probe in enumerate(("imec0", "imec1")):
        for col, (candidate, label) in enumerate(zip(candidates, labels)):
            ax = fig.add_subplot(grid[row, col])
            field = sample_field(probe, candidate)
            residual = decompose_spatial_field(field, DEPTHS)["residual"]
            image = ax.imshow(
                residual.T,
                origin="lower",
                aspect="auto",
                extent=[RELATIVE_TIMES[0], RELATIVE_TIMES[-1], DEPTHS[0], DEPTHS[-1]],
                cmap="RdBu_r",
                vmin=-35,
                vmax=35,
                interpolation="nearest",
            )
            if row == 0:
                ax.set_title(label)
            if col == 0:
                ax.set_ylabel(f"{probe}\nDepth (µm)")
            else:
                ax.set_yticklabels([])
            if row == 1:
                ax.set_xlabel("Time (s)")
            else:
                ax.set_xticklabels([])
    ax = fig.add_subplot(grid[:, 4])
    criteria = ["Split-half reproducibility", "Adjacent sampling agreement", "Independent-method agreement", "Cross-probe agreement"]
    y = np.arange(len(criteria))
    for offset, (candidate, color, marker) in enumerate([
        ("300 µm window / 200 µm step", "#3569a8", "o"),
        ("600 µm window / 200 µm step", "#d08b27", "s"),
    ]):
        subset = evidence[evidence.candidate.eq(candidate)].set_index("criterion")
        values = [subset.correlation.get(criterion, np.nan) for criterion in criteria]
        ax.scatter(values, y + (offset - 0.5) * 0.12, s=55, color=color, marker=marker, label=candidate.split(" window")[0] + " window")
    ax.axvline(0, color="#777777", lw=0.8)
    ax.set_xlim(-0.25, 1.05)
    ax.set_yticks(y, criteria)
    ax.invert_yaxis()
    ax.set_xlabel("Residual-field correlation (r)")
    ax.grid(axis="x", color="#dddddd", lw=0.7)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_title("Validation scorecard")
    fig.suptitle("Non-rigid spatial-scale validation", fontsize=15, y=0.99)
    fig.text(0.5, 0.955, "Residual displacement after removing rigid and linear-depth components; common ±35 µm heatmap scale", ha="center", fontsize=10)
    cax = fig.add_axes([0.24, 0.055, 0.34, 0.016])
    cbar = fig.colorbar(image, cax=cax, orientation="horizontal")
    cbar.set_label("Residual displacement (µm)")
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.12, top=0.90)
    fig.savefig(output / "spatial_scale_validation.png", dpi=180)
    plt.close(fig)


def save_recovery_figure(metrics: pd.DataFrame, output: Path) -> None:
    order = list(CONDITIONS)
    values = metrics.set_index("condition").loc[order]
    specs = [
        ("observed_recovery", "Reviewed neural-event recovery", "Fraction recovered", lambda x: f"{x:.1%}"),
        ("learned_detection_count", "Learned-template detections", "Detections", lambda x: f"{x/1e6:.3f}M"),
        ("n_final_spikes", "Final sorted spikes", "Spikes", lambda x: f"{x/1e3:.0f}k"),
        ("cross_unit_near_coincident_fraction", "Cross-unit coincidence", "Fraction of spikes", lambda x: f"{x:.1%}"),
        ("median_contamination_pct", "Median contamination", "Percent", lambda x: f"{x:.1f}%"),
        ("n_ks_good", "KS-good units", "Units", lambda x: f"{int(x)}"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(20, 8.2))
    colors = [PALETTE[name] for name in order]
    for ax, (field, title, ylabel, formatter) in zip(axes.ravel(), specs):
        data = values[field].to_numpy(float)
        bars = ax.bar(np.arange(len(order)), data, color=colors, edgecolor="#333333", lw=0.7)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        short_labels = {
            "No external correction": "No\ncorrection",
            "Zero-displacement identity": "Identity",
            "Kilosort internal rigid": "KS\nrigid",
            "Current DREDGE 150/100": "DREDGE\n150/100 p1",
            "Rigid from 300/200": "Rigid\n300/200 p1",
            "Selected DREDGE 300/200": "DREDGE\n300/200 p1",
            "Selected DREDGE p2 sigma 20": "DREDGE p2\nσ20",
            "Selected DREDGE p2 sigma 28": "DREDGE p2\nσ28",
            "MEDiCINe default, sigma 10": "MEDiCINe\nσ10 p1",
        }
        ax.set_xticks(np.arange(len(order)), [short_labels[name] for name in order])
        ax.tick_params(axis="x", labelsize=8)
        ax.set_ylim(0, max(data) * 1.20 if max(data) else 1)
        ax.grid(axis="y", color="#e0e0e0", lw=0.7)
        for bar, value in zip(bars, data):
            ax.text(bar.get_x() + bar.get_width() / 2, value + max(data) * 0.035, formatter(value), ha="center", va="bottom", fontsize=9)
    fig.suptitle("Motion-correction downstream diagnostic", fontsize=15, y=0.99)
    fig.text(0.5, 0.955, "Same 120 s imec1 source, Kilosort settings, claim-off configuration, and 27 prespecified reviewed neural misses", ha="center", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output / "motion_candidate_sorter_outcomes.png", dpi=180)
    plt.close(fig)


def save_event_matrix(events: pd.DataFrame, output: Path) -> None:
    matrix = events.pivot(index="review_id", columns="condition", values="recovered").astype(int)
    matrix = matrix[list(CONDITIONS)]
    matrix["pattern"] = matrix.astype(str).agg("".join, axis=1)
    matrix = matrix.sort_values(["pattern", matrix.index.name], ascending=[False, True]).drop(columns="pattern")
    fig, ax = plt.subplots(figsize=(12.5, 8.5))
    image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    totals = matrix.sum(axis=0).astype(int)
    labels = [
        "No correction",
        "Zero-motion identity",
        "KS internal rigid",
        "Current 150/100",
        "Rigid 300/200",
        "Non-rigid 300/200",
        "Non-rigid p2 sigma 20",
        "Non-rigid p2 sigma 28",
        "MEDiCINe sigma 10",
    ]
    ax.set_xticks(
        range(len(matrix.columns)),
        [f"{label}\n{total}/27" for label, total in zip(labels, totals)],
        rotation=18,
        ha="right",
    )
    ax.set_yticks(range(len(matrix)), matrix.index, fontsize=7)
    ax.set_ylabel("Prespecified reviewed event")
    for i in range(len(matrix)):
        for j in range(len(matrix.columns)):
            ax.text(j, i, "✓" if matrix.iloc[i, j] else "—", ha="center", va="center", color="white" if matrix.iloc[i, j] else "#555555", fontsize=9)
    fig.suptitle("Per-event recovery across correction conditions", fontsize=14, y=0.99)
    fig.text(0.5, 0.955, "Recovered = sorted spike within 0.5 ms and 100 µm; rows grouped by recovery pattern", ha="center", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output / "reviewed_event_recovery_matrix.png", dpi=180)
    plt.close(fig)


def save_rigid_gain_figure(metrics: pd.DataFrame, events: pd.DataFrame, output: Path) -> None:
    gain_order = [
        "No external correction",
        "Rigid gain 0.25",
        "Rigid gain 0.50",
        "Rigid gain 0.75",
        "Rigid from 300/200",
    ]
    gains = np.array([0.0, 0.25, 0.50, 0.75, 1.0])
    values = metrics.set_index("condition").loc[gain_order]
    specs = [
        ("observed_recovery", "Reviewed neural-event recovery", "Fraction recovered", lambda x: f"{x:.1%}"),
        ("n_ks_good", "KS-good units", "Units", lambda x: f"{int(x)}"),
        ("median_contamination_pct", "Median contamination", "Percent", lambda x: f"{x:.1f}%"),
        ("cross_unit_near_coincident_fraction", "Cross-unit coincidence", "Fraction of spikes", lambda x: f"{x:.1%}"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    for ax, (field, title, ylabel, formatter) in zip(axes.ravel(), specs):
        data = values[field].to_numpy(float)
        ax.plot(gains, data, color="#d08b27", marker="o", ms=7, lw=2)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Rigid displacement gain")
        ax.set_xticks(gains)
        ax.set_xlim(-0.04, 1.04)
        ax.grid(color="#dddddd", lw=0.7)
        pad = max(np.ptp(data) * 0.08, max(abs(data)) * 0.015, 0.01)
        for x, value in zip(gains, data):
            ax.text(x, value + pad, formatter(value), ha="center", va="bottom", fontsize=9)
        ax.set_ylim(min(0, np.min(data) - 2 * pad), np.max(data) + 4 * pad)
    fig.suptitle("Response to scaled rigid DREDGE displacement", fontsize=14, y=0.99)
    fig.text(0.5, 0.955, "Same rigid temporal trace; gain 0 is no external correction and gain 1 is the full depth mean", ha="center", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output / "rigid_gain_response.png", dpi=180)
    plt.close(fig)

    matrix = events.pivot(index="review_id", columns="condition", values="recovered").astype(int)
    matrix = matrix[gain_order]
    matrix["pattern"] = matrix.astype(str).agg("".join, axis=1)
    matrix = matrix.sort_values(["pattern", matrix.index.name], ascending=[False, True]).drop(columns="pattern")
    fig, ax = plt.subplots(figsize=(9.5, 8.5))
    ax.imshow(matrix.to_numpy(), aspect="auto", cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    totals = matrix.sum(axis=0).astype(int)
    ax.set_xticks(range(len(gains)), [f"{gain:g}×\n{total}/27" for gain, total in zip(gains, totals)])
    ax.set_yticks(range(len(matrix)), matrix.index, fontsize=7)
    ax.set_xlabel("Rigid displacement gain")
    ax.set_ylabel("Prespecified reviewed event")
    for i in range(len(matrix)):
        for j in range(len(matrix.columns)):
            ax.text(j, i, "✓" if matrix.iloc[i, j] else "—", ha="center", va="center", color="white" if matrix.iloc[i, j] else "#555555", fontsize=9)
    fig.suptitle("Paired event recovery across rigid gain", fontsize=14, y=0.99)
    fig.text(0.5, 0.955, "Rows grouped by recovery pattern; fixed 0.5 ms and 100 µm match rule", ha="center", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output / "rigid_gain_event_matrix.png", dpi=180)
    plt.close(fig)


def save_event_motion_association(events: pd.DataFrame, output: Path) -> None:
    selected = events[events.condition.eq("Selected DREDGE 300/200")].copy()
    selected["outcome"] = np.where(selected.recovered, "Recovered", "Missed")
    specs = [
        ("selected_abs_displacement_um", "Absolute displacement at event", "Displacement (µm)"),
        ("selected_abs_spatial_gradient_um_per_um", "Absolute spatial gradient at event", "|∂ displacement / ∂ depth|"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    rng = np.random.default_rng(20250804)
    for ax, (field, title, ylabel) in zip(axes, specs):
        groups = [selected[selected.outcome.eq(outcome)][field].to_numpy(float) for outcome in ("Recovered", "Missed")]
        boxes = ax.boxplot(groups, positions=[0, 1], widths=0.45, patch_artist=True, showfliers=False)
        for patch, color in zip(boxes["boxes"], ("#9fb27a", "#6f91bd")):
            patch.set_facecolor(color)
            patch.set_edgecolor("#333333")
        for index, values in enumerate(groups):
            ax.scatter(index + rng.uniform(-0.10, 0.10, len(values)), values, color="#333333", s=23, alpha=0.75, zorder=3)
            ax.text(index, max(values) * 1.06, f"n={len(values)}\nmedian={np.median(values):.3g}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks([0, 1], ["Recovered", "Missed"])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(0, max(map(np.max, groups)) * 1.25)
        ax.grid(axis="y", color="#dddddd", lw=0.7)
    fig.suptitle("Selected-field motion at reviewed event locations", fontsize=14, y=0.99)
    fig.text(0.5, 0.94, "Descriptive association for the 27 prespecified neural misses; no threshold was chosen from these outcomes", ha="center", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.89])
    fig.savefig(output / "event_motion_association.png", dpi=180)
    plt.close(fig)


def save_rigid_trace_figure(traces: pd.DataFrame, events: pd.DataFrame, output: Path) -> None:
    matrix = events.pivot(index="review_id", columns="condition", values="recovered").astype(bool)
    baseline = matrix["No external correction"]
    internal = matrix["Kilosort internal rigid"]
    lost_ids = matrix.index[baseline & ~internal]
    gained_ids = matrix.index[~baseline & internal]
    event_lookup = (
        events[events.condition.eq("Kilosort internal rigid")]
        .drop_duplicates("review_id")
        .set_index("review_id")
    )

    fig, axes = plt.subplots(2, 1, figsize=(12.5, 7.2), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(
        traces.time_s,
        traces.kilosort_sign_aligned_centered_um,
        color="#8b6fa6",
        lw=1.8,
        label="Kilosort internal rigid (sign-aligned, median-centered)",
    )
    axes[0].plot(
        traces.time_s,
        traces.dredge_rigid_centered_um,
        color="#d08b27",
        lw=1.8,
        label="DREDGE 300/200 depth mean (median-centered)",
    )
    for ids, color, label in ((lost_ids, "#b24c3f", "Lost vs no correction"), (gained_ids, "#3569a8", "Gained vs no correction")):
        for index, review_id in enumerate(ids):
            time_s = float(event_lookup.loc[review_id, "event_time_s"])
            axes[0].axvline(time_s, color=color, lw=1.1, alpha=0.75, label=label if index == 0 else None)
            axes[0].text(time_s + 0.8, axes[0].get_ylim()[1] * 0.82, review_id, color=color, rotation=90, va="top", fontsize=8)
    axes[0].axhline(0, color="#777777", lw=0.8)
    axes[0].set_ylabel("Centered displacement (µm)")
    axes[0].legend(frameon=False, ncol=2, fontsize=8, loc="upper left")
    axes[0].grid(axis="y", color="#dddddd", lw=0.7)

    axes[1].plot(traces.time_s, traces.kilosort_abs_step_um, color="#8b6fa6", lw=1.5, label="Kilosort")
    axes[1].plot(traces.time_s, traces.dredge_abs_step_um, color="#d08b27", lw=1.5, label="DREDGE")
    axes[1].axhline(20, color="#777777", lw=0.8, ls="--", label="20 µm per batch")
    axes[1].set_xlabel("Time in 120 s diagnostic window (s)")
    axes[1].set_ylabel("Absolute 2 s step (µm)")
    axes[1].grid(axis="y", color="#dddddd", lw=0.7)
    axes[1].legend(frameon=False, ncol=3, fontsize=8, loc="upper right")
    fig.suptitle("Rigid motion estimates on the untouched imec1 recording", fontsize=14, y=0.99)
    fig.text(
        0.5,
        0.955,
        "Kilosort follows the broad DREDGE trajectory but contains repeated 70–103 µm batch-to-batch jumps",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output / "internal_rigid_trace_comparison.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    evidence = spatial_evidence()
    events = event_recovery()
    metrics = sorter_metrics()
    rigid_traces = rigid_trace_comparison()
    evidence.to_csv(OUTPUT / "spatial_scale_evidence.csv", index=False)
    events.to_csv(OUTPUT / "reviewed_event_recovery.csv", index=False)
    metrics.to_csv(OUTPUT / "motion_candidate_sorter_metrics.csv", index=False)
    rigid_traces.to_csv(OUTPUT / "internal_rigid_trace_comparison.csv", index=False)
    save_spatial_figure(evidence, OUTPUT)
    save_recovery_figure(metrics, OUTPUT)
    save_event_matrix(events, OUTPUT)
    save_event_motion_association(events, OUTPUT)
    save_rigid_trace_figure(rigid_traces, events, OUTPUT)
    save_rigid_gain_figure(metrics, events, OUTPUT)
    chart_map = {
        "spatial_scale_validation.png": "Residual-field heatmaps and spatial-scale validation scorecard.",
        "motion_candidate_sorter_outcomes.png": "Fixed-event recovery and sorter-quality guardrails.",
        "reviewed_event_recovery_matrix.png": "Paired recovery of each prespecified visually neural missed event.",
        "event_motion_association.png": "Candidate displacement and spatial gradient for recovered versus missed reviewed events.",
        "internal_rigid_trace_comparison.png": "Kilosort internal rigid and DREDGE rigid traces plus batch-to-batch jump magnitudes.",
        "rigid_gain_response.png": "Sorter recovery and quality response across rigid displacement gains from zero to one.",
        "rigid_gain_event_matrix.png": "Paired reviewed-event recovery across rigid displacement gains.",
    }
    (OUTPUT / "chart_map.json").write_text(json.dumps(chart_map, indent=2) + "\n")


if __name__ == "__main__":
    main()
