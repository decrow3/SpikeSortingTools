#!/usr/bin/env python
"""Apply correction M to the completed Luke0804 imec1 MEDiCINe stage-1 sweep.

This is a label-free, no-sort post-processing pass. It forms candidate episodes
from all completed fields plus population rate, measures each episode using the
cached pilot-frontend depth x x peak map, validates matched quiet nulls, and
only then scores/selects configurations under M.2.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from testing.luke_imec1_medicine_reference_sweep_v1 import (
    BIN, CONFIGS, WINDOWS, atomic_json, pareto_front, population_source,
    sample_field, sha256,
)


SHIFTS = np.arange(-320.0, 120.0 + 1e-6, 10.0)
DEPTH_BIN_UM = 10.0
X_BIN_UM = 8.0
MIN_PEAKS = 150
MIN_CORR_GAIN = 0.05
MAX_JOIN_GAP_S = 1.0
SEED = 20260923


def join_short_gaps(mask: np.ndarray, max_bins: int) -> np.ndarray:
    result = np.asarray(mask, bool).copy()
    false = ~result
    starts = np.flatnonzero(false & np.r_[True, result[:-1]])
    stops = np.flatnonzero(false & np.r_[result[1:], True]) + 1
    for start, stop in zip(starts, stops):
        if start and stop < len(result) and stop - start <= max_bins:
            result[start:stop] = True
    return result


def intervals(mask: np.ndarray, centers: np.ndarray) -> list[tuple[int, int, float, float]]:
    starts = np.flatnonzero(mask & np.r_[True, ~mask[:-1]])
    stops = np.flatnonzero(mask & np.r_[~mask[1:], True]) + 1
    return [(int(a), int(b), float(centers[a] - BIN / 2),
             float(centers[b - 1] + BIN / 2)) for a, b in zip(starts, stops)]


def field_rigid(path: Path, centers: np.ndarray) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        times = np.asarray(data["session_time_s"], float)
        rigid = np.nanmedian(np.asarray(data["displacement_um"], float), axis=1)
    values = np.full(len(centers), np.nan)
    valid = (centers >= times[0]) & (centers <= times[-1])
    values[valid] = np.interp(centers[valid], times, rigid)
    return values


def load_peaks(output: Path, window: str) -> dict[str, np.ndarray]:
    path, start = population_source(output, window, 240)
    with np.load(path, allow_pickle=False) as data:
        if "x_um" not in data.files:
            raise RuntimeError(f"M.1 requires x_um in {path}")
        return {
            "time": np.asarray(data["time_s"], float) + start,
            "depth": np.asarray(data["depth_um"], float),
            "x": np.asarray(data["x_um"], float),
        }


def candidate_table(output: Path):
    import pandas as pd

    rows, bin_rows = [], []
    for window, (start, stop) in WINDOWS.items():
        centers = np.arange(start + BIN / 2, stop, BIN)
        masks = []
        for config in CONFIGS:
            values = field_rigid(output / f"fields/{config}/{window}/field.npz", centers)
            finite = np.isfinite(values)
            threshold = float(np.percentile(values[finite], 5))
            mask = finite & (values <= threshold)
            masks.append(mask)
            for index in np.flatnonzero(mask):
                bin_rows.append({"window": window, "bin_center_s": centers[index],
                                 "source": f"field_p5:{config}", "value": values[index],
                                 "threshold": threshold})
        field_union = np.any(masks, axis=0)
        peaks = load_peaks(output, window)
        counts = np.histogram(peaks["time"], bins=np.r_[centers - BIN / 2, stop])[0]
        median_rate = float(np.median(counts))
        rate_mask = counts < 0.5 * median_rate
        masks.append(rate_mask)
        for index in np.flatnonzero(rate_mask):
            bin_rows.append({"window": window, "bin_center_s": centers[index],
                             "source": "rate_below_half_median", "value": counts[index],
                             "threshold": 0.5 * median_rate})
        union = join_short_gaps(field_union | rate_mask, round(MAX_JOIN_GAP_S / BIN))
        for episode_id, (i0, i1, a, b) in enumerate(intervals(union, centers)):
            rows.append({"window": window, "episode_id": episode_id,
                         "start_s": a, "stop_s": b, "duration_s": b - a,
                         "n_bins": i1 - i0, "field_p5_source": bool(field_union[i0:i1].any()),
                         "low_rate_source": bool(rate_mask[i0:i1].any()),
                         "invalid_eye_source": False})
    return pd.DataFrame(rows), pd.DataFrame(bin_rows)


def histogram(peaks: dict[str, np.ndarray], select: np.ndarray,
              depth_edges: np.ndarray, x_edges: np.ndarray) -> np.ndarray:
    values, _, _ = np.histogram2d(peaks["depth"][select], peaks["x"][select],
                                  bins=(depth_edges, x_edges))
    return np.log1p(values)


def shifted_correlation(episode: np.ndarray, rest: np.ndarray) -> tuple[float, float, list[float]]:
    correlations = []
    for shift in SHIFTS:
        # The reported shift is the observed displacement convention used by
        # MEDiCINe/DREDGE. Applying it to register the observed episode map
        # therefore uses the opposite array offset.
        offset = -int(round(shift / DEPTH_BIN_UM))
        if offset < 0:
            left, right = episode[-offset:], rest[:offset]
        elif offset > 0:
            left, right = episode[:-offset], rest[offset:]
        else:
            left, right = episode, rest
        a, b = left.ravel(), right.ravel()
        correlations.append(float(np.corrcoef(a, b)[0, 1]) if a.std() and b.std() else np.nan)
    correlations = np.asarray(correlations)
    if not np.isfinite(correlations).any():
        return np.nan, np.nan, correlations.tolist()
    best_index = int(np.nanargmax(correlations))
    zero_index = int(np.flatnonzero(SHIFTS == 0)[0])
    return float(SHIFTS[best_index]), float(correlations[zero_index]), correlations.tolist()


def map_shift(peaks: dict[str, np.ndarray], episode_select: np.ndarray,
              rest_select: np.ndarray, rng: np.random.Generator | None = None,
              episode_target: int | None = None, rest_target: int | None = None) -> dict:
    episode_indices = np.flatnonzero(episode_select)
    rest_indices = np.flatnonzero(rest_select)
    if rng is not None:
        if episode_target is not None and len(episode_indices) >= episode_target:
            episode_indices = rng.choice(episode_indices, episode_target, replace=False)
        if rest_target is not None and len(rest_indices) >= rest_target:
            rest_indices = rng.choice(rest_indices, rest_target, replace=False)
    n_episode, n_rest = len(episode_indices), len(rest_indices)
    result = {"episode_peaks": n_episode, "rest_peaks": n_rest, "best_shift_um": np.nan,
              "corr_zero": np.nan, "corr_best": np.nan, "corr_gain": np.nan,
              "correlations": []}
    if n_episode < MIN_PEAKS or n_rest < MIN_PEAKS:
        return result
    all_indices = np.r_[episode_indices, rest_indices]
    z = peaks["depth"][all_indices]
    x = peaks["x"][all_indices]
    depth_edges = np.arange(np.floor(z.min() / DEPTH_BIN_UM) * DEPTH_BIN_UM,
                            np.ceil(z.max() / DEPTH_BIN_UM) * DEPTH_BIN_UM + DEPTH_BIN_UM,
                            DEPTH_BIN_UM)
    x_edges = np.arange(np.floor(x.min() / X_BIN_UM) * X_BIN_UM,
                        np.ceil(x.max() / X_BIN_UM) * X_BIN_UM + X_BIN_UM, X_BIN_UM)
    episode_map = histogram(peaks, np.isin(np.arange(len(peaks["time"])), episode_indices),
                            depth_edges, x_edges)
    rest_map = histogram(peaks, np.isin(np.arange(len(peaks["time"])), rest_indices),
                         depth_edges, x_edges)
    best_shift, corr_zero, correlations = shifted_correlation(episode_map, rest_map)
    if np.isfinite(best_shift):
        best_index = int(np.flatnonzero(SHIFTS == best_shift)[0])
        corr_best = float(correlations[best_index])
        result.update(best_shift_um=best_shift, corr_zero=corr_zero, corr_best=corr_best,
                      corr_gain=corr_best - corr_zero, correlations=correlations)
    return result


def measure_references(output: Path, candidates):
    import pandas as pd

    measured, nulls = [], []
    rng = np.random.default_rng(SEED)
    for window, group in candidates.groupby("window", sort=False):
        start, stop = WINDOWS[window]
        centers = np.arange(start + BIN / 2, stop, BIN)
        active = np.zeros(len(centers), bool)
        for row in group.itertuples():
            active |= (centers >= row.start_s) & (centers < row.stop_s)
        peaks = load_peaks(output, window)
        for row in group.itertuples():
            episode = (peaks["time"] >= row.start_s) & (peaks["time"] < row.stop_s)
            rest = (peaks["time"] >= row.start_s - 4) & (peaks["time"] < row.start_s - 1)
            # Exclude peaks falling in any other candidate interval.
            peak_bins = np.floor((peaks["time"] - start) / BIN).astype(int)
            in_window = (peak_bins >= 0) & (peak_bins < len(active))
            candidate_peak = np.zeros(len(peaks["time"]), bool)
            candidate_peak[in_window] = active[peak_bins[in_window]]
            rest &= ~candidate_peak
            result = map_shift(peaks, episode, rest)
            rest_in_window = row.start_s - 4 >= start
            accepted = (rest_in_window and result["episode_peaks"] >= MIN_PEAKS and result["rest_peaks"] >= MIN_PEAKS
                        and np.isfinite(result["corr_gain"]) and result["corr_gain"] >= MIN_CORR_GAIN)
            measured.append({**row._asdict(), **{k: v for k, v in result.items() if k != "correlations"},
                             "accepted": bool(accepted), "rest_fully_in_window": bool(rest_in_window),
                             "resolution_note": "10um shift grid; <0.5s episodes often unresolved"})

            duration_bins = int(row.n_bins)
            valid_starts = []
            for i0 in range(round(4 / BIN), len(active) - duration_bins + 1):
                i1 = i0 + duration_bins
                rest0, rest1 = i0 - round(4 / BIN), i0 - round(1 / BIN)
                if active[i0:i1].any() or active[rest0:rest1].any():
                    continue
                pa, pb = centers[i0] - BIN / 2, centers[i1 - 1] + BIN / 2
                pr0, pr1 = pa - 4, pa - 1
                n_ep = int(((peaks["time"] >= pa) & (peaks["time"] < pb)).sum())
                n_rest = int(((peaks["time"] >= pr0) & (peaks["time"] < pr1)).sum())
                if n_ep >= result["episode_peaks"] and n_rest >= result["rest_peaks"]:
                    valid_starts.append((pa, pb, pr0, pr1))
            if not valid_starts:
                nulls.append({"window": window, "matched_episode_id": row.episode_id,
                              "resolved": False, "reason": "no matched quiet placement"})
                continue
            pa, pb, pr0, pr1 = valid_starts[int(rng.integers(len(valid_starts)))]
            pseudo_episode = (peaks["time"] >= pa) & (peaks["time"] < pb)
            pseudo_rest = (peaks["time"] >= pr0) & (peaks["time"] < pr1)
            null = map_shift(peaks, pseudo_episode, pseudo_rest, rng,
                             result["episode_peaks"], result["rest_peaks"])
            nulls.append({"window": window, "matched_episode_id": row.episode_id,
                          "start_s": pa, "stop_s": pb, "rest_start_s": pr0,
                          "rest_stop_s": pr1, **{k: v for k, v in null.items() if k != "correlations"},
                          "resolved": bool(np.isfinite(null["best_shift_um"]))})
    return pd.DataFrame(measured), pd.DataFrame(nulls)


def null_validation(nulls) -> dict:
    result = {"status": "pass", "criterion": "per-window mode=0um and median=0um"}
    windows = {}
    for window in WINDOWS:
        values = nulls.loc[(nulls.window == window) & nulls.resolved, "best_shift_um"].to_numpy(float)
        if not len(values):
            windows[window] = {"n": 0, "status": "fail_no_resolved_nulls"}
            result["status"] = "fail"
            continue
        unique, counts = np.unique(values, return_counts=True)
        mode = float(unique[np.argmax(counts)])
        median = float(np.median(values))
        exact_zero = float(np.mean(values == 0))
        status = "pass" if mode == 0 and median == 0 else "fail"
        if status == "fail":
            result["status"] = "fail"
        windows[window] = {"n": int(len(values)), "mode_um": mode, "median_um": median,
                           "exact_zero_fraction": exact_zero, "status": status,
                           "counts": {str(int(k)): int(v) for k, v in zip(unique, counts)}}
    result["windows"] = windows
    return result


def episode_prediction(field: dict, start: float, stop: float) -> float:
    depth = np.asarray(field["depth_um"], float)
    times = np.arange(start + BIN / 2, stop, BIN)
    rest = np.arange(start - 4 + BIN / 2, start - 1, BIN)
    if not len(times) or not len(rest):
        return np.nan
    episode_values = sample_field(field, *np.meshgrid(times, depth, indexing="ij"))
    rest_values = sample_field(field, *np.meshgrid(rest, depth, indexing="ij"))
    return float(np.nanmedian(episode_values) - np.nanmedian(rest_values))


def quiet_increment(field: dict) -> float:
    start, stop = WINDOWS["p50"]
    times = np.arange(start, stop + 1e-6, 5.0)
    depth = np.asarray(field["depth_um"], float)
    values = sample_field(field, *np.meshgrid(times, depth, indexing="ij"))
    differences = np.diff(values, axis=0).ravel()
    differences = differences[np.isfinite(differences)]
    return float(np.sqrt(np.mean(differences ** 2))) if len(differences) else np.nan


def score(output: Path, measured, nulls):
    import pandas as pd

    accepted = measured.loc[measured.accepted].copy()
    detail, rows = [], []
    pooled_quiet: dict[str, list[float]] = {config: [] for config in CONFIGS}
    for config in CONFIGS:
        for window in WINDOWS:
            with np.load(output / f"fields/{config}/{window}/field.npz", allow_pickle=False) as data:
                field = {key: np.asarray(data[key]) for key in data.files}
            observed, predicted = [], []
            for episode in accepted.loc[accepted.window == window].itertuples():
                value = episode_prediction(field, episode.start_s, episode.stop_s)
                observed.append(float(episode.best_shift_um))
                predicted.append(value)
                detail.append({"config": config, "window": window,
                               "episode_id": int(episode.episode_id),
                               "measured_shift_um": float(episode.best_shift_um),
                               "field_displacement_um": value,
                               "abs_error_um": abs(value - episode.best_shift_um)})
            null_values = []
            for pseudo in nulls.loc[(nulls.window == window) & nulls.resolved].itertuples():
                null_values.append(abs(episode_prediction(field, pseudo.start_s, pseudo.stop_s)))
            pooled_quiet[config].extend(value for value in null_values if np.isfinite(value))
            observed, predicted = np.asarray(observed), np.asarray(predicted)
            finite = np.isfinite(observed) & np.isfinite(predicted)
            ratio_ok = finite & (np.abs(observed) >= DEPTH_BIN_UM)
            receipt = json.loads((output / f"fields/{config}/{window}/receipt.json").read_text())
            rows.append({"config": config, "window": window,
                         "episode_err": float(np.median(np.abs(predicted[finite] - observed[finite]))) if finite.any() else np.nan,
                         "episode_ratio": float(np.median(predicted[ratio_ok] / observed[ratio_ok])) if ratio_ok.any() else np.nan,
                         "accepted_episodes": int(finite.sum()),
                         "quiet_abs": float(np.median(null_values)) if null_values else np.nan,
                         "null_episodes": len(null_values),
                         "quiet_inc": quiet_increment(field) if window == "p50" else np.nan,
                         "runtime_s": float(receipt["runtime_s"])})
    table, detail = pd.DataFrame(rows), pd.DataFrame(detail)
    aggregate = []
    for config, group in table.groupby("config", sort=False):
        d = detail.loc[detail.config == config].dropna(
            subset=["field_displacement_um", "measured_shift_um", "abs_error_um"])
        ratios = d.field_displacement_um.to_numpy(float) / d.measured_shift_um.to_numpy(float)
        quiet = np.asarray(pooled_quiet[config], float)
        aggregate.append({"config": config,
                          "episode_err": float(np.median(d.abs_error_um)) if len(d) else np.nan,
                          "episode_ratio": float(np.median(ratios)) if len(ratios) else np.nan,
                          "accepted_episodes": len(d),
                          "quiet_abs": float(np.median(quiet)) if len(quiet) else np.nan,
                          "quiet_inc": float(group.loc[group.window == "p50", "quiet_inc"].iloc[0]),
                          "runtime_s": float(group.runtime_s.sum())})
    return table, detail, pd.DataFrame(aggregate)


def select_config(aggregate) -> dict:
    eligible = aggregate.loc[(aggregate.quiet_abs <= 10) & (aggregate.quiet_inc <= 8) &
                             aggregate.episode_ratio.between(0.8, 1.2)]
    if len(eligible):
        row = eligible.sort_values(["episode_err", "config"]).iloc[0]
        return {"outcome": "selected", "config": str(row.config),
                "rule": "quiet_abs<=10, quiet_inc<=8, ratio 0.8..1.2; lowest pooled episode_err"}
    finite = aggregate.dropna(subset=["episode_err", "quiet_abs", "quiet_inc"])
    front = pareto_front(finite, ["episode_err", "quiet_abs", "quiet_inc"])
    return {"outcome": "no_selection", "pareto_front": front.config.tolist(),
            "rule": "no configuration passed all frozen M.2 gates"}


def plot_episode_scatter(output: Path, detail) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = dict(zip(WINDOWS, ["#0077BB", "#33BBEE", "#009988", "#EE7733", "#CC3311", "#AA3377"]))
    fig, axes = plt.subplots(4, 4, figsize=(15, 15), sharex=True, sharey=True, layout="constrained")
    for axis, config in zip(axes.ravel(), CONFIGS):
        group = detail.loc[detail.config == config]
        for window, points in group.groupby("window"):
            axis.scatter(points.measured_shift_um, points.field_displacement_um, s=13,
                         alpha=.7, color=colors[window], label=window)
        axis.plot([-330, 130], [-330, 130], color="0.2", lw=.7, linestyle="--")
        axis.axhline(0, color="0.75", lw=.5); axis.axvline(0, color="0.75", lw=.5)
        axis.set(title=config, xlim=(-330, 130), ylim=(-330, 130))
    handles = [plt.Line2D([], [], marker="o", linestyle="none", color=color, label=window)
               for window, color in colors.items()]
    fig.legend(handles=handles, loc="outside upper center", ncol=6)
    fig.supxlabel("Measured depth×x map shift (µm)")
    fig.supylabel("MEDiCINe episode displacement, rest-centred (µm)")
    fig.savefig(output / "figure_m_episode_shift_vs_field.png", dpi=180)
    plt.close(fig)


def write_readme(output: Path, candidates, measured, nulls, null_check, aggregate,
                 selection, old_selection) -> None:
    counts = measured.groupby("window").accepted.agg(["sum", "count"])
    null_lines = []
    for window, row in null_check["windows"].items():
        null_lines.append(f"- {window}: n={row['n']}, mode={row.get('mode_um', 'NA')} µm, "
                          f"median={row.get('median_um', 'NA')} µm, exact-zero={row.get('exact_zero_fraction', float('nan')):.3f}; "
                          f"distribution={row.get('counts', {})}")
    old_repr = json.dumps(old_selection, sort_keys=True)
    changed = (old_selection.get("outcome") != selection.get("outcome") or
               old_selection.get("config") != selection.get("config"))
    text = [
        "# Luke0804 imec1 MEDiCINe stage-1 — correction M", "",
        "No spike sort or voltage modification was run. The stage-1 fields are unchanged; this README replaces the earlier pause reference and selection rule.", "",
        "## Corrected label-free reference", "",
        "The earlier static-unit ‘stay home’ result is not evidence of zero motion: fragmentation/reassignment lets a static cluster retain whichever spikes match its home template. Gabor suppression/eye-loss episodes are therefore treated as candidate displacement episodes, not quiet pauses.",
        "Candidate episodes are the union of runs at or below every field's own P5 and raw-peak rate below 0.5× the window median, after joining gaps ≤1 s. Independent eye validity was unavailable for Luke0804 (`testing/luke_within_rigid_motion_dose_response.py:88`), so it contributed no candidate runs.",
        f"Each episode was compared with 4–1 s before onset using log1p depth (10 µm) × x (8 µm) pilot-frontend peak maps over shifts −320..+120 µm. Acceptance required ≥{MIN_PEAKS} peaks in both maps and correlation gain ≥{MIN_CORR_GAIN:.2f}. Shifts are quantised to 10 µm; episodes shorter than about 0.5 s are often unresolved.", "",
        "Accepted/candidate episode counts:", "",
    ]
    text.extend([f"- {window}: {int(counts.loc[window, 'sum'])}/{int(counts.loc[window, 'count'])}"
                 for window in WINDOWS])
    text.extend(["", "## Matched quiet null", "", f"Null validation: **{null_check['status']}**. The frozen pass condition was modal and median best shift of 0 µm in every window.", ""])
    text.extend(null_lines)
    text.extend(["", "Null pseudo-episodes are the quiet reference (0 µm) and are matched in duration and thinned peak count. Full distributions are in `references_m/null_pseudo_episodes.csv`.", "",
                 "## M.2 scores and selection", "",
                 f"Outcome: `{selection['outcome']}`. " + (f"Selected `{selection['config']}`." if selection["outcome"] == "selected" else f"No configuration passed; Pareto front: {', '.join(selection['pareto_front'])}."),
                 "The frozen rule is quiet_abs ≤10 µm and quiet_inc ≤8 µm, then lowest pooled episode_err only when episode_ratio is 0.8–1.2.",
                 f"Legacy pause-based result: `{old_repr}`. It **{'would' if changed else 'would not'}** have produced a different selection outcome/configuration.",
                 "See `scores_m.csv`, `config_summary_m.csv`, `episode_field_comparison_m.csv`, and `figure_m_episode_shift_vs_field.png`.", "",
                 "## DARTsort deployment audit (K.3)", "",
                 "The shared-recovery and merge-bias runs passed MEDiCINe as external `dredge_motion_est`. DARTsort therefore did **not** apply its internal 500 µm/s or 250 µm/51-bin median post-filters and did **not** resample the external field to `temporal_bin_length_s`; both external fields already had a 0.25 s grid. The counterfactual saved-p10 replay replaces 22/480 time bins (22 depth×time cells), with largest raw excursion 259.7 µm and largest 4 s-centred excursion 271.7 µm. Exact file:line evidence is in `dartsort_postfilter_audit.json`.", "",
                 "## Proposed stage-2 grid (not started)", "",
                 "At most eight configurations: the M-selected/Pareto leader, base, useful neighboring kernels chosen from 0.5/1/2/**5/10/30 s**, one neighboring depth-bin count, seed replication of the leader, native amplitude top-50%, and either rate equalization or crop/full-probe. Freeze the exact list after reviewing M scores.", "",
                 "## Provenance", "",
                 f"Candidate episodes: {len(candidates)}. Measured table SHA-256: `{sha256(output/'references_m/measured_episodes.csv')}`.",
                 "The 16-arm fit grid includes native `amplitude_threshold_quantile=0.5` (`amp50`) and `0.75` (`amp25`), plus `kern5`, `kern10`, and `kern30`. No hand-filtered amplitude arms were run.",
                 ])
    (output / "README.md").write_text("\n".join(text) + "\n")


def main() -> None:
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not (output / "sweep_complete.json").exists():
        raise RuntimeError("Stage-1 fits are not complete")
    reference_dir = output / "references_m"
    reference_dir.mkdir(exist_ok=False)
    atomic_json(reference_dir / "preregistration.json", {
        "correction": "M", "shifts_um": SHIFTS.tolist(), "depth_bin_um": DEPTH_BIN_UM,
        "x_bin_um": X_BIN_UM, "minimum_peaks_each_map": MIN_PEAKS,
        "minimum_correlation_gain": MIN_CORR_GAIN, "join_gaps_s": MAX_JOIN_GAP_S,
        "eye_validity": "unavailable", "null_seed": SEED,
        "null_pass": "per-window mode=0um and median=0um",
        "selection": "quiet_abs<=10 and quiet_inc<=8; lowest pooled episode_err if ratio in [0.8,1.2]",
        "no_sort": True, "voltage_modified": False,
    })
    candidates, bins = candidate_table(output)
    candidates.to_csv(reference_dir / "candidate_episodes.csv", index=False)
    bins.to_csv(reference_dir / "candidate_source_bins.csv", index=False)
    measured, nulls = measure_references(output, candidates)
    measured.to_csv(reference_dir / "measured_episodes.csv", index=False)
    nulls.to_csv(reference_dir / "null_pseudo_episodes.csv", index=False)
    null_check = null_validation(nulls)
    atomic_json(reference_dir / "null_validation.json", null_check)
    if null_check["status"] != "pass":
        atomic_json(output / "m_stopped.json", {"reason": "null_not_centered_at_zero", "detail": null_check})
        raise RuntimeError("M null did not sit at zero; stopped before scoring")

    for old, legacy in (("scores.csv", "scores_jk_legacy.csv"),
                        ("config_summary.csv", "config_summary_jk_legacy.csv"),
                        ("selection.json", "selection_jk_legacy.json")):
        if (output / old).exists() and not (output / legacy).exists():
            shutil.copy2(output / old, output / legacy)
    old_selection = json.loads((output / "selection_jk_legacy.json").read_text())
    scores, detail, aggregate = score(output, measured, nulls)
    scores.to_csv(output / "scores_m.csv", index=False)
    detail.to_csv(output / "episode_field_comparison_m.csv", index=False)
    selection = select_config(aggregate)
    aggregate["selected"] = aggregate.config.eq(selection.get("config"))
    aggregate.to_csv(output / "config_summary_m.csv", index=False)
    atomic_json(output / "selection_m.json", selection)
    plot_episode_scatter(output, detail)
    write_readme(output, candidates, measured, nulls, null_check, aggregate,
                 selection, old_selection)
    atomic_json(output / "m_validation.json", {
        "status": "complete", "candidate_episodes": len(candidates),
        "accepted_episodes": int(measured.accepted.sum()), "nulls": len(nulls),
        "scores": len(scores), "expected_scores": len(CONFIGS) * len(WINDOWS),
        "configs": len(aggregate), "selection": selection,
        "artifacts": {str(path.relative_to(output)): sha256(path) for path in [
            reference_dir / "preregistration.json", reference_dir / "candidate_episodes.csv",
            reference_dir / "measured_episodes.csv", reference_dir / "null_pseudo_episodes.csv",
            output / "scores_m.csv", output / "config_summary_m.csv",
            output / "episode_field_comparison_m.csv", output / "selection_m.json",
            output / "figure_m_episode_shift_vs_field.png", output / "README.md"]},
    })
    print(json.dumps({"status": "complete", "selection": selection,
                      "accepted": int(measured.accepted.sum()), "null": null_check}, indent=2))


if __name__ == "__main__":
    main()
