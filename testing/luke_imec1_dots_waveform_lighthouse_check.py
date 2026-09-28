#!/usr/bin/env python
"""Depth-blind cached waveform tracking for Luke0804 imec1 dots-RF.

This is the second, still-cheap premise check.  It freezes the candidate list
from ``luke_imec1_dots_lighthouse_direct_check_v1``, builds relative-geometry
TPCA templates only in the frozen seed interval, and searches all detections in
the six prespecified motion-rich periods.  Native motion is opened only after
matching and measurement are complete.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LogNorm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from testing.luke_imec1_dots_lighthouse_direct_check import (
    DEFAULT_EXPERIMENT,
    MOTION_WINDOWS,
    ROOT,
    SEED_INTERVAL_S,
    add_field_comparison,
    atomic_json,
    load_motion,
    sha256,
)


FROZEN_DIRECT = ROOT / "testing/outputs/luke_imec1_dots_lighthouse_direct_check_v1"
DEFAULT_OUTPUT = ROOT / "testing/outputs/luke_imec1_dots_waveform_lighthouse_check_v1"
STRICT_COSINE = 0.86
DISPLAY_COSINE = 0.80
IDENTITY_MARGIN = 0.025
MAX_SEED_EVENTS_PER_TEMPLATE = 500
FEATURE_DATASET = "collisioncleaned_tpca_features"
COMPONENTS_KEY = "transformers.1.components"
MEAN_KEY = "transformers.1.mean"


def channel_signature_ids(geom: np.ndarray, channel_index: np.ndarray) -> np.ndarray:
    signatures = []
    for channel in range(len(geom)):
        neighbors = channel_index[channel]
        neighbors = neighbors[neighbors < len(geom)]
        relative = np.rint(geom[neighbors] - geom[channel]).astype(np.int64)
        signatures.append(tuple(map(tuple, relative)))
    lookup = {signature: index for index, signature in enumerate(dict.fromkeys(signatures))}
    return np.asarray([lookup[signature] for signature in signatures], dtype=np.int16)


def tpca_gram(model_path: Path) -> np.ndarray:
    state = torch.load(model_path, weights_only=False, map_location="cpu")
    components = np.asarray(state[COMPONENTS_KEY].cpu(), dtype=np.float32)
    mean = np.asarray(state[MEAN_KEY].cpu(), dtype=np.float32)
    if np.max(np.abs(mean), initial=0.0) > 1e-7:
        raise RuntimeError("Nonzero TPCA centering is not supported")
    return components @ components.T


def waveform_centroids(
    features: np.ndarray,
    main_channels: np.ndarray,
    channel_index: np.ndarray,
    geom: np.ndarray,
    gram: np.ndarray,
) -> np.ndarray:
    energy = np.einsum("ern,rs,esn->en", features, gram, features, optimize=True)
    energy = np.maximum(energy, 0.0)
    physical = channel_index[main_channels]
    valid = physical < len(geom)
    safe = np.where(valid, physical, 0)
    energy[~valid] = 0.0
    denominator = energy.sum(axis=1)
    return np.divide(
        (energy * geom[safe, 1]).sum(axis=1),
        denominator,
        out=np.full(len(features), np.nan),
        where=denominator > 0,
    )


def build_seed_templates(
    h5: h5py.File,
    labels: np.ndarray,
    absolute_times: np.ndarray,
    eligible: pd.DataFrame,
    event_signatures: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame]:
    seed = (absolute_times >= SEED_INTERVAL_S[0]) & (absolute_times < SEED_INTERVAL_S[1])
    templates = []
    rows = []
    for row in eligible.itertuples():
        indices = np.flatnonzero(seed & (labels == row.unit_id))
        signatures, counts = np.unique(event_signatures[indices], return_counts=True)
        signature = int(signatures[counts.argmax()])
        compatible = indices[event_signatures[indices] == signature]
        if len(compatible) < 5:
            continue
        if len(compatible) > MAX_SEED_EVENTS_PER_TEMPLATE:
            compatible = compatible[
                np.linspace(0, len(compatible) - 1, MAX_SEED_EVENTS_PER_TEMPLATE, dtype=int)
            ]
        feature = np.asarray(h5[FEATURE_DATASET][compatible], dtype=np.float32)
        template = np.median(feature, axis=0).ravel()
        norm = np.linalg.norm(template)
        if not np.isfinite(norm) or norm == 0:
            continue
        templates.append(template / norm)
        rows.append(
            {
                "unit_id": int(row.unit_id),
                "signature_id": signature,
                "template_seed_events": len(compatible),
                "selected": bool(row.selected),
                "static_rank_score": float(row.rank_score),
                "static_template_depth_um": float(row.template_depth_um),
            }
        )
    return np.asarray(templates, dtype=np.float32), pd.DataFrame(rows)


def match_cached_events(
    h5_path: Path,
    labels: np.ndarray,
    absolute_times: np.ndarray,
    channel_signatures: np.ndarray,
    templates: np.ndarray,
    template_table: pd.DataFrame,
    gram: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    analysis = (absolute_times >= SEED_INTERVAL_S[0]) & (absolute_times < SEED_INTERVAL_S[1])
    for start, end in MOTION_WINDOWS:
        analysis |= (absolute_times >= start) & (absolute_times < end)
    selected_units = set(template_table.loc[template_table.selected, "unit_id"].astype(int))
    template_units = template_table.unit_id.to_numpy(dtype=np.int64)
    template_signatures = template_table.signature_id.to_numpy(dtype=np.int16)
    events = []
    recovery = {
        unit: {"compatible_seed_events": 0, "strict_correct_seed_events": 0}
        for unit in selected_units
    }
    counters = {"analysis_events": 0, "below_display_gate": 0, "winner_not_selected": 0}
    with h5py.File(h5_path, "r") as h5:
        channels = np.asarray(h5["channels"], dtype=np.int64)
        event_signatures = channel_signatures[channels]
        channel_index = np.asarray(h5["channel_index"], dtype=np.int64)
        geom = np.asarray(h5["geom"], dtype=np.float64)
        for chunk_start in range(0, len(labels), 250_000):
            chunk_stop = min(chunk_start + 250_000, len(labels))
            chunk_rows = np.flatnonzero(analysis[chunk_start:chunk_stop]) + chunk_start
            if not len(chunk_rows):
                continue
            counters["analysis_events"] += len(chunk_rows)
            features = np.asarray(h5[FEATURE_DATASET][chunk_start:chunk_stop], dtype=np.float32)[
                chunk_rows - chunk_start
            ]
            main = channels[chunk_rows]
            signatures = event_signatures[chunk_rows]
            centroids = waveform_centroids(features, main, channel_index, geom, gram)
            for signature in np.unique(signatures):
                local = np.flatnonzero(signatures == signature)
                candidate = np.flatnonzero(template_signatures == signature)
                if len(candidate) < 2:
                    continue
                flattened = features[local].reshape(len(local), -1)
                flattened /= np.maximum(np.linalg.norm(flattened, axis=1, keepdims=True), 1e-12)
                similarity = flattened @ templates[candidate].T
                order = np.argsort(similarity, axis=1, kind="stable")
                row_index = np.arange(len(local))
                best = order[:, -1]
                second = order[:, -2]
                score = similarity[row_index, best]
                margin = score - similarity[row_index, second]
                winner = template_units[candidate[best]]
                global_rows = chunk_rows[local]
                in_seed = (
                    (absolute_times[global_rows] >= SEED_INTERVAL_S[0])
                    & (absolute_times[global_rows] < SEED_INTERVAL_S[1])
                )
                strict = (score >= STRICT_COSINE) & (margin >= IDENTITY_MARGIN)
                for unit in selected_units:
                    source = in_seed & (labels[global_rows] == unit)
                    if source.any():
                        recovery[unit]["compatible_seed_events"] += int(source.sum())
                        recovery[unit]["strict_correct_seed_events"] += int(
                            (source & strict & (winner == unit)).sum()
                        )
                display = score >= DISPLAY_COSINE
                counters["below_display_gate"] += int((~display).sum())
                counters["winner_not_selected"] += int((display & ~np.isin(winner, list(selected_units))).sum())
                keep = display & np.isin(winner, list(selected_units))
                for k in np.flatnonzero(keep):
                    status = (
                        "strict"
                        if strict[k]
                        else ("lower_score" if margin[k] >= IDENTITY_MARGIN else "identity_ambiguous")
                    )
                    source_row = int(global_rows[k])
                    events.append(
                        {
                            "row": source_row,
                            "unit_id": int(winner[k]),
                            "source_static_label": int(labels[source_row]),
                            "time_s": float(absolute_times[source_row]),
                            "main_channel": int(main[local[k]]),
                            "signature_id": int(signature),
                            "waveform_centroid_um": float(centroids[local[k]]),
                            "point_source_depth_um": float(h5["point_source_localizations"][source_row, 2]),
                            "score": float(score[k]),
                            "margin": float(margin[k]),
                            "status": status,
                            "seed": bool(in_seed[k]),
                        }
                    )
    recovery_rows = []
    for unit, values in sorted(recovery.items()):
        denominator = values["compatible_seed_events"]
        numerator = values["strict_correct_seed_events"]
        recovery_rows.append(
            {
                "unit_id": unit,
                **values,
                "strict_seed_recovery_fraction": numerator / denominator if denominator else 0.0,
                "seed_qualified": numerator >= 5 and numerator >= 0.2 * denominator,
            }
        )
    return pd.DataFrame(events), pd.DataFrame(recovery_rows), counters


def measured_tracks(events: pd.DataFrame, recovery: pd.DataFrame) -> pd.DataFrame:
    qualified = set(recovery.loc[recovery.seed_qualified, "unit_id"].astype(int))
    strict = events[(events.status == "strict") & events.unit_id.isin(qualified)].copy()
    seed_depth = strict[strict.seed].groupby("unit_id").waveform_centroid_um.median()
    strict["seed_depth_um"] = strict.unit_id.map(seed_depth)
    strict["relative_depth_um"] = strict.waveform_centroid_um - strict.seed_depth_um
    strict["time_bin_s"] = np.floor(strict.time_s * 2.0) / 2.0 + 0.25
    return (
        strict.dropna(subset=["seed_depth_um"])
        .groupby(["unit_id", "time_bin_s"], as_index=False)
        .agg(
            observed_relative_um=("relative_depth_um", "median"),
            event_count=("relative_depth_um", "size"),
            seed_depth_um=("seed_depth_um", "first"),
        )
        .sort_values(["unit_id", "time_bin_s"])
    )


def pairwise_replication(tracks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for start, end in MOTION_WINDOWS:
        local = tracks[
            (tracks.time_bin_s >= start)
            & (tracks.time_bin_s < end)
            & (tracks.event_count >= 2)
        ]
        wide = local.pivot(index="time_bin_s", columns="unit_id", values="observed_relative_um")
        for i, unit_a in enumerate(wide.columns):
            for unit_b in wide.columns[i + 1 :]:
                pair = wide[[unit_a, unit_b]].dropna()
                if len(pair) < 5 or not pair[unit_a].std() or not pair[unit_b].std():
                    continue
                rows.append(
                    {
                        "window_start_s": start,
                        "window_end_s": end,
                        "unit_a": int(unit_a),
                        "unit_b": int(unit_b),
                        "common_bins": len(pair),
                        "correlation": float(pair.corr().iloc[0, 1]),
                        "median_abs_difference_um": float(np.median(np.abs(pair[unit_a] - pair[unit_b]))),
                    }
                )
    return pd.DataFrame(rows)


def track_plausibility(events: pd.DataFrame, recovery: pd.DataFrame) -> pd.DataFrame:
    """Post-match diagnostic only; depth never feeds identity scoring or ranking."""
    rows = []
    for row in recovery[recovery.seed_qualified].itertuples():
        target = events[
            (events.unit_id == row.unit_id) & (events.status == "strict") & (~events.seed)
        ]
        if target.empty:
            depth_min = depth_max = depth_span = np.nan
        else:
            depth_min = float(target.waveform_centroid_um.min())
            depth_max = float(target.waveform_centroid_um.max())
            depth_span = depth_max - depth_min
        rows.append(
            {
                "unit_id": int(row.unit_id),
                "strict_target_matches": int(len(target)),
                "target_depth_min_um": depth_min,
                "target_depth_max_um": depth_max,
                "target_depth_span_um": depth_span,
                "implausible_gt500um_span": bool(np.isfinite(depth_span) and depth_span > 500.0),
                "provisional_track_plausible": bool(
                    len(target) >= 20 and np.isfinite(depth_span) and depth_span <= 500.0
                ),
                "diagnostic_role": "post_match_only_not_identity_gate",
            }
        )
    return pd.DataFrame(rows)


def plot_depth_atlas(
    events: pd.DataFrame,
    recovery: pd.DataFrame,
    all_times: np.ndarray,
    all_depths: np.ndarray,
    output: Path,
) -> None:
    qualified = set(recovery.loc[recovery.seed_qualified, "unit_id"].astype(int))
    colors = {unit: plt.cm.tab20(index % 20) for index, unit in enumerate(sorted(qualified))}
    fig, axes = plt.subplots(3, 2, figsize=(17, 13), constrained_layout=True)
    for ax, (start, end) in zip(axes.flat, MOTION_WINDOWS):
        time_edges = np.arange(start, end + 0.1001, 0.1)
        depth_edges = np.arange(0.0, 3840.1, 10.0)
        inside = (all_times >= start) & (all_times < end)
        density = np.histogram2d(all_times[inside], all_depths[inside], bins=(time_edges, depth_edges))[0].T
        positive = density[density > 0]
        vmax = np.percentile(positive, 99.5) if positive.size else 2
        ax.pcolormesh(time_edges, depth_edges, np.maximum(density, 1), cmap="Greys", norm=LogNorm(1, max(2, vmax)), shading="auto", rasterized=True)
        local = events[(events.time_s >= start) & (events.time_s < end) & events.unit_id.isin(qualified)]
        for unit, group in local.groupby("unit_id"):
            strict = group[group.status == "strict"]
            lower = group[group.status == "lower_score"]
            ambiguous = group[group.status == "identity_ambiguous"]
            ax.scatter(strict.time_s, strict.waveform_centroid_um, s=8, color=colors[unit], alpha=0.75, label=str(unit), rasterized=True)
            ax.scatter(lower.time_s, lower.waveform_centroid_um, s=13, facecolors="none", edgecolors=[colors[unit]], alpha=0.7, rasterized=True)
            ax.scatter(ambiguous.time_s, ambiguous.waveform_centroid_um, s=14, marker="x", color=colors[unit], alpha=0.5, rasterized=True)
        ax.set(xlim=(start, end), ylim=(0, 3840), title=f"{start:g}–{end:g} s", xlabel="absolute time (s)", ylabel="absolute waveform centroid (µm)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels, handles))
        axes[0, 0].legend(by_label.values(), by_label.keys(), ncol=3, fontsize=7, title="candidate")
    fig.suptitle(
        "Depth-blind waveform matches over unregistered detected-peak density\n"
        "Filled=strict, open=lower score, x=identity ambiguous; absolute depth revealed after matching",
        fontsize=14,
    )
    fig.savefig(output / "01_waveform_match_depth_atlas.png", dpi=180)
    fig.savefig(output / "01_waveform_match_depth_atlas.pdf")
    plt.close(fig)


def plot_candidate_field_comparison(tracks: pd.DataFrame, recovery: pd.DataFrame, output: Path) -> None:
    order = recovery[recovery.seed_qualified].sort_values(
        ["strict_seed_recovery_fraction", "strict_correct_seed_events"], ascending=False
    ).head(12)
    fig, axes = plt.subplots(4, 3, figsize=(17, 12), sharey=True, constrained_layout=True)
    for ax, row in zip(axes.flat, order.itertuples()):
        unit = tracks[(tracks.unit_id == row.unit_id) & (tracks.event_count >= 2)]
        ax.plot(unit.time_bin_s, unit.observed_relative_um, color="#276FBF", marker="o", ms=2.0, lw=0.8, label="waveform observed")
        ax.plot(unit.time_bin_s, unit.native_relative_um, color="#D87939", ls="--", lw=0.9, label="native field")
        ax.axhline(0, color="#555555", lw=0.45)
        ax.set_title(f"candidate {row.unit_id} · seed recovery {row.strict_seed_recovery_fraction:.0%}", fontsize=9)
        ax.set_xlabel("absolute time (s)")
        ax.set_ylabel("seed-relative depth (µm)")
        ax.grid(color="#dddddd", lw=0.4)
    for ax in axes.flat[len(order):]:
        ax.remove()
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "Provisional lighthouse observations versus native DREDGE in prespecified windows\n"
        "No fitted sign, gain, lag, or offset; candidate order fixed by seed recovery",
        fontsize=14,
    )
    fig.savefig(output / "02_waveform_candidate_native_comparison.png", dpi=180)
    fig.savefig(output / "02_waveform_candidate_native_comparison.pdf")
    plt.close(fig)


def run(experiment: Path, frozen_direct: Path, output: Path) -> None:
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError(f"Refusing existing output: {output}")
    partial.mkdir(parents=True)
    static = experiment / "sorts/static-v1"
    native = experiment / "sorts/native-v1"
    sources = {
        "frozen_candidates": frozen_direct / "candidate_inventory.csv",
        "frozen_direct_summary": frozen_direct / "summary.json",
        "static_sorting": static / "dartsort_sorting.npz",
        "static_matching": static / "matching1.h5",
        "static_tpca_model": static / "matching1_models/featurization_pipeline.pt",
        "static_result": static / "result.json",
        "native_motion": native / "motion.pkl",
        "implementation": Path(__file__).resolve(),
    }
    try:
        inventory = pd.read_csv(sources["frozen_candidates"])
        eligible = inventory[inventory.eligible].copy()
        direct_summary = json.loads(sources["frozen_direct_summary"].read_text())
        expected_selected = direct_summary["selection"]["selected_unit_ids"]
        if inventory.loc[inventory.selected, "unit_id"].astype(int).tolist() != expected_selected:
            raise RuntimeError("Frozen candidate order differs from direct-check receipt")
        result = json.loads(sources["static_result"].read_text())
        window_start_frame = int(result["window_start_frame"])
        with np.load(sources["static_sorting"], allow_pickle=False) as sorting:
            labels = np.asarray(sorting["labels"], dtype=np.int64)
            times = np.asarray(sorting["times_samples"], dtype=np.int64)
            fs = float(sorting["sampling_frequency"])
        absolute_times = (times + window_start_frame) / fs
        gram = tpca_gram(sources["static_tpca_model"])
        with h5py.File(sources["static_matching"], "r") as h5:
            geom = np.asarray(h5["geom"], dtype=np.float64)
            channel_index = np.asarray(h5["channel_index"], dtype=np.int64)
            channels = np.asarray(h5["channels"], dtype=np.int64)
            signature_by_channel = channel_signature_ids(geom, channel_index)
            event_signatures = signature_by_channel[channels]
            templates, template_table = build_seed_templates(
                h5, labels, absolute_times, eligible, event_signatures
            )
            all_depths = np.asarray(h5["point_source_localizations"][:, 2], dtype=np.float32)
        template_table.to_csv(partial / "seed_template_inventory.csv", index=False)
        events, recovery, counters = match_cached_events(
            sources["static_matching"], labels, absolute_times, signature_by_channel,
            templates, template_table, gram
        )
        events.to_csv(partial / "waveform_matches.csv", index=False)
        recovery.to_csv(partial / "seed_recovery.csv", index=False)
        tracks = measured_tracks(events, recovery)
        window_start_s = window_start_frame / fs
        tracks, field = add_field_comparison(tracks, sources["native_motion"], window_start_s)
        tracks.to_csv(partial / "strict_halfsecond_tracks.csv", index=False)
        field.to_csv(partial / "native_field_candidate_metrics.csv", index=False)
        pairs = pairwise_replication(tracks)
        pairs.to_csv(partial / "pairwise_replication.csv", index=False)
        plausibility = track_plausibility(events, recovery)
        plausibility.to_csv(partial / "track_plausibility.csv", index=False)
        analysis_time = np.zeros(len(absolute_times), dtype=bool)
        for start, end in MOTION_WINDOWS:
            analysis_time |= (absolute_times >= start) & (absolute_times < end)
        plot_depth_atlas(events, recovery, absolute_times[analysis_time], all_depths[analysis_time], partial)
        plot_candidate_field_comparison(tracks, recovery, partial)

        qualified = recovery[recovery.seed_qualified]
        plausible = plausibility[plausibility.provisional_track_plausible]
        strict_target = events[(events.status == "strict") & (~events.seed) & events.unit_id.isin(qualified.unit_id)]
        summary = {
            "schema": "luke0804-imec1-dots-waveform-lighthouse-check-v1",
            "status": "complete",
            "selection_and_matching": {
                "frozen_selected_candidates": len(expected_selected),
                "all_static_only_rival_templates": int(len(template_table)),
                "seed_qualified_candidates": int(len(qualified)),
                "seed_qualified_unit_ids": qualified.unit_id.astype(int).tolist(),
                "post_match_plausible_candidates": int(len(plausible)),
                "post_match_plausible_unit_ids": plausible.unit_id.astype(int).tolist(),
                "implausible_gt500um_span_candidates": plausibility.loc[
                    plausibility.implausible_gt500um_span, "unit_id"
                ].astype(int).tolist(),
                "strict_target_matches": int(len(strict_target)),
                "lower_score_target_matches": int(len(events[(events.status == "lower_score") & (~events.seed)])),
                "ambiguous_target_matches": int(len(events[(events.status == "identity_ambiguous") & (~events.seed)])),
                "motion_loaded_before_matching": False,
                "absolute_depth_used_for_matching": False,
                "continuity_used_for_matching": False,
                "thresholds": {
                    "strict_cosine": STRICT_COSINE,
                    "display_cosine": DISPLAY_COSINE,
                    "identity_margin": IDENTITY_MARGIN,
                    "seed_qualification": "at least 5 strict correct and 20% compatible seed-event recovery",
                },
            },
            "direct_replication": {
                "pairwise_rows": int(len(pairs)),
                "median_pairwise_correlation_by_window": {
                    f"{start:g}-{end:g}": float(group.correlation.median())
                    for (start, end), group in pairs.groupby(["window_start_s", "window_end_s"])
                },
            },
            "provisional_native_field_comparison": {
                "candidates_with_metrics": int(len(field)),
                "median_level_correlation": float(field.field_level_correlation.median()),
                "median_level_mae_um": float(field.field_level_mae_um.median()),
                "median_increment_correlation": float(field.field_increment_correlation.median()),
                "median_increment_mae_um": float(field.field_increment_mae_um.median()),
                "no_fitted_sign_gain_lag_or_offset": True,
                "interpretation_valid": bool(len(plausible)),
                "invalid_reason_if_false": (
                    None if len(plausible) else
                    "Every seed-qualified cached track has an implausible >500-um target depth span."
                ),
            },
            "counters": counters,
            "interpretation": {
                "candidate_status": "provisional cached TPCA waveform tracks; not yet raw-voltage-qualified lighthouse cells",
                "lighthouse_units_established": bool(len(plausible)),
                "automatic_motion_verdict": False,
                "inspection_required": "Inspect waveforms, rival identities, depth atlas, lattice phase, and artifact coincidences.",
                "selection_lookahead_note": "The six target windows were already viewed in the native peak raster; this is an exploratory method check, not untouched confirmation.",
            },
            "known_limits": [
                "The cached TPCA representation was learned by DARTsort and is not a raw-voltage identity measurement.",
                "Exact neighborhood signatures preserve relative geometry but inherit the 40-um lattice-phase dropout risk.",
                "Seed templates inherit static sorter labels and may contain mixtures.",
                "Only the six prespecified motion-rich windows were searched in this bounded check.",
            ],
            "sources": {name: {"path": str(path), "sha256": sha256(path)} for name, path in sources.items()},
        }
        atomic_json(partial / "summary.json", summary)
        (partial / "README.md").write_text(
            "# Luke0804 imec1 dots-RF cached waveform lighthouse check\n\n"
            "Candidate identities and their order are inherited from the frozen static-only direct check. "
            "All 56 eligible static templates remain rivals. TPCA footprints are compared only among exact "
            "relative-neighborhood geometries; absolute depth and temporal continuity are absent from scoring.\n\n"
            "Start with `01_waveform_match_depth_atlas`. Absolute waveform-centroid depth is revealed only "
            "after matching and shown over the unregistered peak density. `02_waveform_candidate_native_comparison` "
            "adds native DREDGE only after identity matching; no sign, gain, lag, or offset is fit.\n\n"
            "These are provisional cached-feature tracks. Raw-waveform examples, global rival review, decoy "
            "controls, and a calibrated treatment of the 40-um lattice remain required before calling any "
            "candidate a qualified lighthouse cell or issuing a motion-field verdict. `track_plausibility.csv` "
            "is a post-match sanity check only; absolute depth never enters matching or candidate ranking.\n"
        )
        os.replace(partial, output)
        print(json.dumps(summary, indent=2))
    except BaseException:
        atomic_json(partial / "failure.json", {"status": "failed"})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--frozen-direct", type=Path, default=FROZEN_DIRECT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.experiment.resolve(), args.frozen_direct.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
