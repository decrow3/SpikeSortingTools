#!/usr/bin/env python
"""Cheap cached lighthouse premise check for Luke0804 imec1 dots-RF.

Candidate selection uses only the unregistered static DARTsort arm.  The native
motion field is opened only after the candidate cohort and its observations are
frozen.  Static final labels are a deliberately cheap fixed-label control, not
depth-aware biological identity proof.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT = Path(
    "/media/huklab/Data/NPX/Ryansorting/Luke/DARTsort_motion_experiments/"
    "luke0804-imec1-local-spikeglx-dots-v4"
)
DEFAULT_OUTPUT = ROOT / "testing/outputs/luke_imec1_dots_lighthouse_direct_check_v1"
STATIC_ARM = "sorts/static-v1"
STATIC_QC = "qc/static-v2/unit_quality_metrics.csv"
NATIVE_ARM = "sorts/native-v1"
SEED_INTERVAL_S = (36.457968473, 56.457968473)
DOTS_INTERVAL_S = (36.457968473, 1134.774534925)
MOTION_WINDOWS = (
    (935.0, 941.0),
    (974.0, 986.0),
    (998.0, 1002.0),
    (1014.0, 1025.0),
    (1071.0, 1075.0),
    (1097.0, 1112.0),
)
WAVEFORM_OFFSETS = np.arange(-24, 25)
N_CANDIDATES = 25


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def exact_np1_patch(geom: np.ndarray, center_um: float) -> tuple[float, np.ndarray] | None:
    """Return the exact 16-channel, 40-um-lattice patch containing center_um."""
    base = 40.0 * math.floor(center_um / 40.0)
    channels = np.flatnonzero((geom[:, 1] >= base - 60.0) & (geom[:, 1] <= base + 80.0))
    if channels.size != 16:
        return None
    return base, channels


def normalized_seed_waveforms(
    templates: np.ndarray,
    geom: np.ndarray,
    trough_offset: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Recenter template footprints in exact relative geometry without depth gates."""
    rows: list[dict] = []
    waveforms: list[np.ndarray] = []
    for template_index, template in enumerate(templates):
        energy = np.square(template.astype(np.float64)).sum(axis=0)
        if not np.isfinite(energy).all() or energy.sum() <= 0:
            continue
        centroid = float(energy @ geom[:, 1] / energy.sum())
        patch = exact_np1_patch(geom, centroid)
        if patch is None:
            continue
        base, channels = patch
        start = trough_offset + int(WAVEFORM_OFFSETS[0])
        stop = trough_offset + int(WAVEFORM_OFFSETS[-1]) + 1
        if start < 0 or stop > template.shape[0]:
            continue
        waveform = template[start:stop, channels].astype(np.float32, copy=False)
        norm = float(np.linalg.norm(waveform))
        if not np.isfinite(norm) or norm == 0:
            continue
        local_fraction = float(np.square(waveform.astype(np.float64)).sum() / energy.sum())
        waveforms.append((waveform / norm).ravel())
        rows.append(
            {
                "template_index": template_index,
                "template_depth_um": centroid,
                "patch_base_um": base,
                "local_energy_fraction": local_fraction,
            }
        )
    return np.asarray(waveforms, dtype=np.float32), pd.DataFrame(rows)


def closest_rival_cosines(waveforms: np.ndarray, block: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Depth-blind nearest-rival cosine over all complete-support templates."""
    n = len(waveforms)
    best = np.full(n, -np.inf, dtype=np.float32)
    rival = np.full(n, -1, dtype=np.int64)
    for start in range(0, n, block):
        stop = min(start + block, n)
        similarities = waveforms[start:stop] @ waveforms.T
        local = np.arange(stop - start)
        similarities[local, np.arange(start, stop)] = -np.inf
        indices = similarities.argmax(axis=1)
        best[start:stop] = similarities[local, indices]
        rival[start:stop] = indices
    return best, rival


def unit_interval_counts(
    labels: np.ndarray,
    absolute_times_s: np.ndarray,
    interval: tuple[float, float],
) -> pd.Series:
    inside = (
        (labels >= 0)
        & (absolute_times_s >= interval[0])
        & (absolute_times_s < interval[1])
    )
    units, counts = np.unique(labels[inside], return_counts=True)
    return pd.Series(counts, index=units, dtype=np.int64)


def select_candidates(
    template_path: Path,
    sorting_path: Path,
    qc_path: Path,
    window_start_frame: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, float]:
    with np.load(sorting_path, allow_pickle=False) as sorting:
        labels = np.asarray(sorting["labels"], dtype=np.int64)
        times = np.asarray(sorting["times_samples"], dtype=np.int64)
        fs = float(sorting["sampling_frequency"])
        geom = np.asarray(sorting["geom"], dtype=np.float64)
    absolute_times = (times + window_start_frame) / fs
    seed_counts = unit_interval_counts(labels, absolute_times, SEED_INTERVAL_S)
    dots_counts = unit_interval_counts(labels, absolute_times, DOTS_INTERVAL_S)

    with np.load(template_path, allow_pickle=False) as saved:
        templates = np.asarray(saved["templates"], dtype=np.float32)
        unit_ids = np.asarray(saved["unit_ids"], dtype=np.int64)
        spike_counts = np.asarray(saved["spike_counts"], dtype=np.int64)
        trough_offset = int(saved["trough_offset_samples"])
        template_geom = np.asarray(saved["registered_geom"], dtype=np.float64)
    if not np.array_equal(geom, template_geom):
        raise RuntimeError("Static template and sorting geometries differ")
    waveforms, inventory = normalized_seed_waveforms(templates, geom, trough_offset)
    inventory["unit_id"] = unit_ids[inventory.template_index]
    inventory["template_spike_count"] = spike_counts[inventory.template_index]
    nearest, rival_index = closest_rival_cosines(waveforms)
    inventory["nearest_rival_cosine"] = nearest
    inventory["nearest_rival_unit"] = inventory.unit_id.to_numpy()[rival_index]
    inventory["seed_spikes"] = inventory.unit_id.map(seed_counts).fillna(0).astype(int)
    inventory["dots_spikes"] = inventory.unit_id.map(dots_counts).fillna(0).astype(int)

    qc = pd.read_csv(qc_path)
    inventory = inventory.merge(qc, on="unit_id", how="left", validate="one_to_one")
    inventory["eligible"] = (
        (inventory.seed_spikes >= 20)
        & (inventory.dots_spikes >= 500)
        & (inventory.local_energy_fraction >= 0.35)
        & (inventory.nearest_rival_cosine < 0.95)
        & (inventory.presence_ratio_60s >= 0.8)
        & inventory.endpoint_continuity_30s.fillna(False)
        & (inventory.probe_edge_burden <= 0.2)
    )
    inventory["rank_score"] = (
        (1.0 - inventory.nearest_rival_cosine.clip(upper=1.0))
        * inventory.local_energy_fraction
        * np.log1p(inventory.seed_spikes)
        * np.log1p(inventory.dots_spikes)
    )
    inventory = inventory.sort_values(
        ["eligible", "rank_score", "unit_id"], ascending=[False, False, True]
    ).reset_index(drop=True)
    inventory["selected"] = False
    selected_index = inventory.index[inventory.eligible][:N_CANDIDATES]
    inventory.loc[selected_index, "selected"] = True
    return inventory, labels, times, fs


def extract_candidate_events(
    h5_path: Path,
    inventory: pd.DataFrame,
    labels: np.ndarray,
    times: np.ndarray,
    fs: float,
    window_start_frame: int,
) -> pd.DataFrame:
    selected = inventory.loc[inventory.selected, "unit_id"].to_numpy(dtype=np.int64)
    baseline = inventory.set_index("unit_id")["template_depth_um"]
    chunks: list[pd.DataFrame] = []
    with h5py.File(h5_path, "r") as h5:
        if h5["point_source_localizations"].shape[0] != labels.size:
            raise RuntimeError("Static labels and matching localizations are not row-aligned")
        for start in range(0, labels.size, 250_000):
            stop = min(start + 250_000, labels.size)
            keep = np.isin(labels[start:stop], selected)
            if not keep.any():
                continue
            rows = np.flatnonzero(keep)
            unit = labels[start:stop][rows]
            depth = np.asarray(h5["point_source_localizations"][start:stop, 2])[rows]
            absolute_time = (times[start:stop][rows] + window_start_frame) / fs
            chunks.append(
                pd.DataFrame(
                    {
                        "unit_id": unit,
                        "time_s": absolute_time,
                        "observed_depth_um": depth,
                        "template_depth_um": baseline.loc[unit].to_numpy(),
                    }
                )
            )
    events = pd.concat(chunks, ignore_index=True)
    events = events[
        np.isfinite(events.observed_depth_um)
        & (events.time_s >= DOTS_INTERVAL_S[0])
        & (events.time_s < DOTS_INTERVAL_S[1])
    ].copy()
    seed = events[
        (events.time_s >= SEED_INTERVAL_S[0]) & (events.time_s < SEED_INTERVAL_S[1])
    ]
    seed_depth = seed.groupby("unit_id").observed_depth_um.median()
    events["seed_depth_um"] = events.unit_id.map(seed_depth)
    events["relative_depth_um"] = events.observed_depth_um - events.seed_depth_um
    events["time_bin_s"] = np.floor(events.time_s * 2.0) / 2.0 + 0.25
    return events.dropna(subset=["seed_depth_um"])


def bin_events(events: pd.DataFrame) -> pd.DataFrame:
    return (
        events.groupby(["unit_id", "time_bin_s"], as_index=False)
        .agg(
            observed_relative_um=("relative_depth_um", "median"),
            event_count=("relative_depth_um", "size"),
            seed_depth_um=("seed_depth_um", "first"),
        )
        .sort_values(["unit_id", "time_bin_s"])
    )


def load_motion(motion_path: Path):
    with motion_path.open("rb") as handle:
        payload = pickle.load(handle)
    motion = payload.get("dredge_motion_est") if isinstance(payload, dict) else payload
    if motion is None:
        raise RuntimeError("Native motion.pkl lacks dredge_motion_est")
    return motion


def add_field_comparison(
    binned: pd.DataFrame,
    motion_path: Path,
    window_start_s: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    motion = load_motion(motion_path)
    chunks = []
    summaries = []
    for unit_id, group in binned.groupby("unit_id", sort=False):
        group = group.copy()
        time = group.time_bin_s.to_numpy()
        depth = np.full(len(group), float(group.seed_depth_um.iloc[0]))
        predicted = np.asarray(motion.disp_at_s(time - window_start_s, depth)).ravel()
        seed = (time >= SEED_INTERVAL_S[0]) & (time < SEED_INTERVAL_S[1])
        if not seed.any():
            continue
        predicted -= np.median(predicted[seed])
        group["native_relative_um"] = predicted
        group["residual_um"] = group.observed_relative_um - predicted
        chunks.append(group)
        enough = group.event_count >= 3
        observed = group.loc[enough, "observed_relative_um"].to_numpy()
        field = group.loc[enough, "native_relative_um"].to_numpy()
        if len(observed) >= 3 and np.std(observed) and np.std(field):
            correlation = float(np.corrcoef(observed, field)[0, 1])
        else:
            correlation = np.nan
        common = group.loc[enough, ["time_bin_s", "observed_relative_um", "native_relative_um"]]
        increment = common.set_index("time_bin_s").diff().dropna()
        adjacent = np.diff(common.time_bin_s.to_numpy()) <= 0.51
        increment = increment.iloc[np.flatnonzero(adjacent)] if len(increment) else increment
        if len(increment) >= 3 and increment.observed_relative_um.std() and increment.native_relative_um.std():
            increment_correlation = float(
                np.corrcoef(increment.observed_relative_um, increment.native_relative_um)[0, 1]
            )
        else:
            increment_correlation = np.nan
        summaries.append(
            {
                "unit_id": int(unit_id),
                "bins_ge_3_events": int(enough.sum()),
                "field_level_correlation": correlation,
                "field_level_mae_um": float(np.mean(np.abs(observed - field))) if len(observed) else np.nan,
                "field_increment_correlation": increment_correlation,
                "field_increment_mae_um": float(
                    np.mean(np.abs(increment.observed_relative_um - increment.native_relative_um))
                ) if len(increment) else np.nan,
            }
        )
    return pd.concat(chunks, ignore_index=True), pd.DataFrame(summaries)


def pairwise_window_summary(binned: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for window_start, window_end in MOTION_WINDOWS:
        local = binned[
            (binned.time_bin_s >= window_start) & (binned.time_bin_s < window_end)
            & (binned.event_count >= 3)
        ]
        wide = local.pivot(index="time_bin_s", columns="unit_id", values="observed_relative_um")
        for i, unit_a in enumerate(wide.columns):
            for unit_b in wide.columns[i + 1 :]:
                pair = wide[[unit_a, unit_b]].dropna()
                if len(pair) < 5 or pair[unit_a].std() == 0 or pair[unit_b].std() == 0:
                    continue
                rows.append(
                    {
                        "window_start_s": window_start,
                        "window_end_s": window_end,
                        "unit_a": int(unit_a),
                        "unit_b": int(unit_b),
                        "common_bins": len(pair),
                        "correlation": float(pair.corr().iloc[0, 1]),
                        "median_abs_difference_um": float(np.median(np.abs(pair[unit_a] - pair[unit_b]))),
                    }
                )
    return pd.DataFrame(rows)


def plot_overview(binned: pd.DataFrame, inventory: pd.DataFrame, output: Path) -> None:
    selected = inventory[inventory.selected].sort_values("rank_score", ascending=False)
    fig, axes = plt.subplots(5, 5, figsize=(18, 14), sharex=True, sharey=True, constrained_layout=True)
    for ax, row in zip(axes.flat, selected.itertuples()):
        unit = binned[(binned.unit_id == row.unit_id) & (binned.event_count >= 3)]
        ax.plot(unit.time_bin_s, unit.observed_relative_um, color="#276FBF", lw=0.65, alpha=0.8)
        for start, end in MOTION_WINDOWS:
            ax.axvspan(start, end, color="#D87939", alpha=0.08)
        ax.axhline(0, color="#555555", lw=0.45)
        ax.set_title(
            f"static {row.unit_id} · {row.template_depth_um:.0f} µm\n"
            f"n={row.dots_spikes} · rival={row.nearest_rival_cosine:.2f}",
            fontsize=8,
        )
        ax.grid(color="#dddddd", lw=0.35)
    for ax in axes[-1]:
        ax.set_xlabel("absolute time (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("seed-relative depth (µm)")
    fig.suptitle(
        "Luke0804 imec1 dots-RF · static-arm fixed-label depth tracks\n"
        "Candidates selected without native motion, RF scores, or depth quotas",
        fontsize=14,
    )
    fig.savefig(output / "01_static_candidate_overview.png", dpi=180)
    fig.savefig(output / "01_static_candidate_overview.pdf")
    plt.close(fig)


def plot_windows(binned: pd.DataFrame, inventory: pd.DataFrame, output: Path) -> None:
    top = inventory[inventory.selected].sort_values("rank_score", ascending=False).head(8)
    colors = plt.cm.tab10(np.linspace(0, 1, len(top)))
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), constrained_layout=True)
    for ax, (start, end) in zip(axes.flat, MOTION_WINDOWS):
        for color, row in zip(colors, top.itertuples()):
            unit = binned[
                (binned.unit_id == row.unit_id)
                & (binned.time_bin_s >= start)
                & (binned.time_bin_s < end)
                & (binned.event_count >= 3)
            ]
            ax.plot(
                unit.time_bin_s,
                unit.observed_relative_um,
                color=color,
                lw=1.0,
                marker="o",
                ms=2.0,
                label=f"{row.unit_id} ({row.template_depth_um:.0f} µm)",
            )
        ax.axhline(0, color="#555555", lw=0.55)
        ax.set(xlim=(start, end), title=f"{start:g}–{end:g} s", xlabel="absolute time (s)", ylabel="seed-relative depth (µm)")
        ax.grid(color="#dddddd", lw=0.4)
    axes[0, 0].legend(ncol=2, fontsize=7, loc="best")
    fig.suptitle(
        "Direct replication check in prespecified motion-rich periods\n"
        "Eight highest-ranked static-only candidates; lines are 0.5 s event-depth medians",
        fontsize=14,
    )
    fig.savefig(output / "02_motion_window_replication.png", dpi=180)
    fig.savefig(output / "02_motion_window_replication.pdf")
    plt.close(fig)


def plot_field_comparison(binned: pd.DataFrame, inventory: pd.DataFrame, output: Path) -> None:
    selected = inventory[inventory.selected].sort_values("rank_score", ascending=False).head(12)
    fig, axes = plt.subplots(4, 3, figsize=(18, 12), sharex=True, sharey=True, constrained_layout=True)
    for ax, row in zip(axes.flat, selected.itertuples()):
        unit = binned[(binned.unit_id == row.unit_id) & (binned.event_count >= 3)]
        ax.plot(unit.time_bin_s, unit.observed_relative_um, color="#276FBF", lw=0.75, label="static observed")
        ax.plot(unit.time_bin_s, unit.native_relative_um, color="#D87939", lw=0.8, ls="--", label="native field")
        ax.axhline(0, color="#555555", lw=0.45)
        ax.set_title(f"static {row.unit_id} · seed depth {unit.seed_depth_um.median():.0f} µm", fontsize=9)
        ax.grid(color="#dddddd", lw=0.35)
    axes[0, 0].legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("absolute time (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("seed-relative depth (µm)")
    fig.suptitle(
        "Post-selection comparison to native DREDGE\n"
        "No fitted sign, gain, lag, or offset; both traces centered on the frozen seed interval",
        fontsize=14,
    )
    fig.savefig(output / "03_provisional_native_field_comparison.png", dpi=180)
    fig.savefig(output / "03_provisional_native_field_comparison.pdf")
    plt.close(fig)


def run(experiment: Path, output: Path) -> None:
    if output.exists() or output.with_name(output.name + ".partial").exists():
        raise FileExistsError(f"Refusing existing output: {output}")
    partial = output.with_name(output.name + ".partial")
    partial.mkdir(parents=True)
    static = experiment / STATIC_ARM
    native = experiment / NATIVE_ARM
    static_result = json.loads((static / "result.json").read_text())
    static_receipt = json.loads((static / "receipt.json").read_text())
    native_receipt = json.loads((native / "receipt.json").read_text())
    if static_receipt.get("status") != "complete" or native_receipt.get("status") != "complete":
        raise RuntimeError("Native/static dots arms are not complete")
    window_start_frame = int(static_result["window_start_frame"])
    window_start_s = window_start_frame / float(static_result["validation"]["sampling_frequency_hz"])
    sources = {
        "static_sorting": static / "dartsort_sorting.npz",
        "static_matching": static / "matching1.h5",
        "static_templates": static / "matching1_models/template_data.npz",
        "static_qc": experiment / STATIC_QC,
        "static_result": static / "result.json",
        "static_receipt": static / "receipt.json",
        "native_motion": native / "motion.pkl",
        "native_receipt": native / "receipt.json",
        "implementation": Path(__file__).resolve(),
    }
    try:
        inventory, labels, times, fs = select_candidates(
            sources["static_templates"],
            sources["static_sorting"],
            sources["static_qc"],
            window_start_frame,
        )
        if inventory.selected.sum() < 3:
            raise RuntimeError("Fewer than three static-only candidates survived frozen gates")
        inventory.to_csv(partial / "candidate_inventory.csv", index=False)
        events = extract_candidate_events(
            sources["static_matching"], inventory, labels, times, fs, window_start_frame
        )
        events.to_csv(partial / "candidate_events.csv", index=False)
        binned = bin_events(events)
        binned, field = add_field_comparison(binned, sources["native_motion"], window_start_s)
        binned.to_csv(partial / "candidate_halfsecond_tracks.csv", index=False)
        field.to_csv(partial / "native_field_candidate_metrics.csv", index=False)
        pairs = pairwise_window_summary(binned)
        pairs.to_csv(partial / "motion_window_pairwise_replication.csv", index=False)
        plot_overview(binned, inventory, partial)
        plot_windows(binned, inventory, partial)
        plot_field_comparison(binned, inventory, partial)

        selected = inventory[inventory.selected]
        summary = {
            "schema": "luke0804-imec1-dots-lighthouse-direct-check-v1",
            "status": "complete",
            "interpretation": {
                "candidate_status": "static fixed-label controls; not qualified depth-aware lighthouse identities",
                "automatic_motion_verdict": False,
                "next_gate": "Inspect direct replication and identity rivals before a managed raw-waveform whole-probe extraction.",
            },
            "selection": {
                "selected_candidates": int(selected.shape[0]),
                "selected_unit_ids": selected.unit_id.astype(int).tolist(),
                "eligible_before_top_n": int(inventory.eligible.sum()),
                "seed_interval_s": list(SEED_INTERVAL_S),
                "dots_interval_s": list(DOTS_INTERVAL_S),
                "absolute_depth_used_for_ranking": False,
                "native_motion_loaded_before_selection": False,
                "rf_metrics_loaded": False,
                "rules": {
                    "minimum_seed_spikes": 20,
                    "minimum_dots_spikes": 500,
                    "minimum_local_energy_fraction": 0.35,
                    "maximum_global_rival_cosine": 0.95,
                    "minimum_presence_ratio_60s": 0.8,
                    "endpoint_continuity_required": True,
                    "maximum_probe_edge_burden": 0.2,
                    "top_n": N_CANDIDATES,
                },
            },
            "direct_evidence": {
                "candidate_events": int(len(events)),
                "candidate_halfsecond_bins": int(len(binned)),
                "motion_window_pairwise_rows": int(len(pairs)),
                "median_pairwise_correlation_by_window": {
                    f"{start:g}-{end:g}": (
                        float(group.correlation.median()) if len(group) else None
                    )
                    for (start, end), group in pairs.groupby(
                        ["window_start_s", "window_end_s"], sort=True
                    )
                },
            },
            "provisional_native_field_comparison": {
                "equal_candidate_median_level_correlation": float(field.field_level_correlation.median()),
                "equal_candidate_median_level_mae_um": float(field.field_level_mae_um.median()),
                "equal_candidate_median_increment_correlation": float(field.field_increment_correlation.median()),
                "equal_candidate_median_increment_mae_um": float(field.field_increment_mae_um.median()),
                "no_fitted_sign_gain_lag_or_offset": True,
                "seed_centering_only": True,
            },
            "known_limits": [
                "Static final labels can fragment or merge identities as cells move.",
                "Candidate templates come from a sorter and are not independent biological identities.",
                "Exact 40-um patch support is lattice-phase biased; dropout cannot establish stationarity.",
                "The post-selection native comparison is descriptive until whole-probe waveform identities are frozen.",
            ],
            "sources": {name: {"path": str(path), "sha256": sha256(path)} for name, path in sources.items()},
        }
        atomic_json(partial / "summary.json", summary)
        (partial / "README.md").write_text(
            "# Luke0804 imec1 dots-RF lighthouse premise check\n\n"
            "This is the cheap cached direct check. Candidate selection uses only the unregistered "
            "static DARTsort arm, a common 36.458--56.458 s seed interval, relative multichannel "
            "template geometry, global waveform rivals, and static QC. Absolute depth, native "
            "motion, RF scores, and event agreement with another arm do not enter selection.\n\n"
            "Static labels are fixed-label controls, not depth-aware lighthouse identity proof. "
            "Inspect `01_static_candidate_overview`, then `02_motion_window_replication`. The native "
            "field is opened only for `03_provisional_native_field_comparison`, after candidates and "
            "observations are frozen. No sign, gain, lag, or offset is fit.\n\n"
            "A positive shared-movement lead justifies adapting the waveform-only whole-probe tracker "
            "to this imec1 interval. A negative result does not establish no motion because fixed "
            "labels and the exact 40-um translation lattice can drop moving identities.\n"
        )
        os.replace(partial, output)
        print(json.dumps(summary, indent=2))
    except BaseException:
        atomic_json(partial / "failure.json", {"status": "failed"})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.experiment.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
