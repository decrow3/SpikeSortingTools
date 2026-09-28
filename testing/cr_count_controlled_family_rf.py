"""Frozen W2 count-controlled RF comparison for exact force/no-force families."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cp_w2_rf_evaluator import explicit_imec1_clock


DARTSORT_ROOT = Path("/home/huklab/Documents/DARTsort")
sys.path.insert(0, str(DARTSORT_ROOT))
from experiments.luke0804_imec1.dots_rf_common import (  # noqa: E402
    crossvalidated_sta_snr,
    dog_standardize,
    frozen_kfold_trial_folds,
)
from experiments.luke0804_imec1.evaluate_dots_rf import (  # noqa: E402
    lag_safe_mask,
    lagged_stimulus_gaze_mask,
    load_gaze,
    validate_gaze_calibration_receipt,
)


SEEDS = (1729, 1730, 1731)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def map_events_to_frames(
    times_s: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    trials: np.ndarray,
) -> np.ndarray:
    """Map each event to the same half-open trial frame used by RF binning."""
    times_s = np.asarray(times_s, dtype=np.float64)
    result = np.full(times_s.size, -1, dtype=np.int64)
    for trial in np.unique(trials):
        frame_ix = np.flatnonzero(trials == trial)
        candidate = np.flatnonzero(
            (times_s >= left[frame_ix[0]]) & (times_s < right[frame_ix[-1]])
        )
        local = np.searchsorted(right[frame_ix], times_s[candidate], side="right")
        inside = local < frame_ix.size
        candidate = candidate[inside]
        local = local[inside]
        inside = times_s[candidate] >= left[frame_ix[local]]
        result[candidate[inside]] = frame_ix[local[inside]]
    return result


def load_context(args: argparse.Namespace, *, build_stimulus: bool) -> dict:
    config = json.loads(args.config.read_text())
    metadata = json.loads(args.interval_metadata.read_text())
    fs = float(metadata["sampling_frequency_hz"])
    source_start, source_end = map(
        int, metadata["exact_source_frame_interval_half_open"]
    )
    if source_end - source_start != int(metadata["local_frame_count"]):
        raise ValueError("Explicit source interval is inconsistent")
    imec_to_nidaq, clock = explicit_imec1_clock(
        Path(config["imec_to_nidaq_timing_mat"])
    )
    nidaq_fs = float(config["nidaq_sampling_frequency_hz"])
    cache = args.stimulus_cache.resolve()
    manifest = json.loads((cache / "manifest.json").read_text())
    for filename, key in (
        ("stimulus_frames.npz", "stimulus_frames_sha256"),
        ("trial_table.csv", "trial_table_sha256"),
        ("trial_split.json", "trial_split_sha256"),
    ):
        if sha256(cache / filename) != manifest[key]:
            raise ValueError(f"Stimulus hash mismatch: {filename}")
    with np.load(cache / "stimulus_frames.npz", allow_pickle=False) as data:
        positive = np.asarray(data["valid_positive_time"], dtype=bool)
        centers = np.asarray(data["t_bin_centers_s"], dtype=np.float64)[positive]
        left = np.asarray(data["t_bin_left_s"], dtype=np.float64)[positive]
        right = np.asarray(data["t_bin_right_s"], dtype=np.float64)[positive]
        dots = np.asarray(data["dots_pix"])[positive]
        trials = np.asarray(data["trial_ordinals"], dtype=np.int64)[positive]
        ppd = float(data["pix_per_deg"])
    unique_trials = np.unique(trials)
    selection = config["parameter_selection"]
    outer = frozen_kfold_trial_folds(
        unique_trials,
        seed=int(selection["outer_split_seed"]),
        n_folds=int(selection["outer_folds"]),
        n_temporal_blocks=int(selection["outer_temporal_blocks"]),
    )
    holdout_fold = int(selection["final_holdout_fold"])
    development = {int(t) for t in unique_trials if outer[int(t)] != holdout_fold}
    holdout = {int(t) for t in unique_trials if outer[int(t)] == holdout_fold}
    gaze_validation = validate_gaze_calibration_receipt(
        config, args.gaze_csv.resolve(), development, holdout
    )
    mapped_bounds = imec_to_nidaq(np.asarray([source_start, source_end])) / nidaq_fs
    complete = {
        int(t)
        for t in unique_trials
        if np.all(left[trials == t] >= mapped_bounds[0])
        and np.all(right[trials == t] <= mapped_bounds[1])
    }
    selected_development = complete & development
    selected_holdout = complete & holdout
    inner = {
        int(key): int(value)
        for key, value in json.loads((cache / "trial_split.json").read_text())[
            "fold_by_trial_ordinal"
        ].items()
    }
    gaze_args = argparse.Namespace(
        gaze_csv=args.gaze_csv,
        gaze_time_column="t_ephys",
        gaze_x_column="i",
        gaze_y_column="j",
        gaze_valid_column="valid",
    )
    frame_gaze, gaze_valid = load_gaze(gaze_args, centers)
    lags = np.asarray(config["lags"], dtype=np.int64)
    selected_frame = np.asarray(
        [int(t) in selected_development for t in trials], dtype=bool
    )
    base_valid = (
        lag_safe_mask(trials, int(lags.max()))
        & lagged_stimulus_gaze_mask(gaze_valid, trials, lags)
        & selected_frame
    )
    frame_fold = np.asarray([inner[int(t)] for t in trials], dtype=np.int8)
    stimulus = None
    modules = None
    if build_stimulus:
        rowley_repo = Path(config["data_rowley_repo"]).resolve()
        sys.path.insert(0, str(rowley_repo))
        import DataRowleyV1V2.dots_calibration.training as training_module
        import DataRowleyV1V2.utils.rf as rf_module

        half = float(config["roi_half_width_deg"])
        roi_pix = np.flipud(np.asarray([[-half, half], [-half, half]]) * ppd)
        dxy_pix = float(config["dxy_deg"]) * ppd
        edges = [np.arange(lo, hi + dxy_pix, dxy_pix) for lo, hi in roi_pix]
        stimulus = training_module.bin_dots_to_stimulus(
            dots, frame_gaze, *edges
        )
        modules = (training_module, rf_module)
    with np.load(args.all_force, allow_pickle=False) as data:
        af_times = np.asarray(data["times_samples"], dtype=np.int64)
        af_labels = np.asarray(data["labels"], dtype=np.int64)
    with np.load(args.no_force, allow_pickle=False) as data:
        nf_times = np.asarray(data["times_samples"], dtype=np.int64)
        nf_labels = np.asarray(data["labels"], dtype=np.int64)
    if not (af_times.shape == af_labels.shape == nf_times.shape == nf_labels.shape):
        raise ValueError("All-force/no-force row domains differ")
    af_seconds = imec_to_nidaq(af_times + source_start) / nidaq_fs
    nf_seconds = imec_to_nidaq(nf_times + source_start) / nidaq_fs
    af_frame = map_events_to_frames(af_seconds, left, right, trials)
    nf_frame = map_events_to_frames(nf_seconds, left, right, trials)
    af_valid = (af_frame >= 0) & base_valid[np.maximum(af_frame, 0)]
    nf_valid = (nf_frame >= 0) & base_valid[np.maximum(nf_frame, 0)]
    return {
        "config": config,
        "fs": fs,
        "source_start": source_start,
        "source_end": source_end,
        "clock": clock,
        "left": left,
        "right": right,
        "trials": trials,
        "base_valid": base_valid,
        "frame_fold": frame_fold,
        "lags": lags,
        "stimulus": stimulus,
        "modules": modules,
        "af_times": af_times,
        "af_labels": af_labels,
        "nf_times": nf_times,
        "nf_labels": nf_labels,
        "af_frame": af_frame,
        "nf_frame": nf_frame,
        "af_valid": af_valid,
        "nf_valid": nf_valid,
        "selected_development": sorted(selected_development),
        "selected_holdout": sorted(selected_holdout),
        "gaze_validation": gaze_validation,
    }


def family_support(context: dict) -> pd.DataFrame:
    af = context["af_labels"]
    nf = context["nf_labels"]
    af_frame = context["af_frame"]
    nf_frame = context["nf_frame"]
    fold = context["frame_fold"]
    rows = []
    for parent in np.unique(af[af >= 0]):
        own = af == parent
        children, counts = np.unique(nf[own & (nf >= 0)], return_counts=True)
        order = sorted(
            zip(children.astype(int), counts.astype(int)),
            key=lambda item: (-item[1], item[0]),
        )
        if len(order) < 2 or order[1][1] < 20 or order[1][1] / sum(counts) < 0.1:
            continue
        parent_valid = own & context["af_valid"]
        parent_fold_counts = [
            int((parent_valid & (fold[np.maximum(af_frame, 0)] == f)).sum())
            for f in (0, 1)
        ]
        for rank, (child, exact_rows) in enumerate(order[:2], start=1):
            child_all = nf == child
            child_valid = child_all & context["nf_valid"]
            child_fold_counts = [
                int((child_valid & (fold[np.maximum(nf_frame, 0)] == f)).sum())
                for f in (0, 1)
            ]
            thinned = [
                min(parent_fold_counts[f], child_fold_counts[f]) for f in (0, 1)
            ]
            qualifies = sum(thinned) >= 500 and min(thinned) >= 200
            failures = []
            if sum(thinned) < 500:
                failures.append("total<500")
            if thinned[0] < 200:
                failures.append("fold0<200")
            if thinned[1] < 200:
                failures.append("fold1<200")
            rows.append(
                {
                    "all_force_parent": int(parent),
                    "child_rank": rank,
                    "no_force_child": child,
                    "parent_all_rows": int(own.sum()),
                    "exact_parent_child_rows": exact_rows,
                    "child_all_rows": int(child_all.sum()),
                    "actual_final_child_parent_purity": float(
                        exact_rows / child_all.sum()
                    ),
                    "parent_fold0": parent_fold_counts[0],
                    "parent_fold1": parent_fold_counts[1],
                    "child_fold0": child_fold_counts[0],
                    "child_fold1": child_fold_counts[1],
                    "thinned_fold0": thinned[0],
                    "thinned_fold1": thinned[1],
                    "thinned_total": sum(thinned),
                    "qualifies": qualifies,
                    "failure": "eligible" if qualifies else ";".join(failures),
                }
            )
    result = pd.DataFrame(rows).sort_values(
        ["all_force_parent", "child_rank"]
    ).reset_index(drop=True)
    if result.all_force_parent.nunique() != 88 or len(result) != 176:
        raise RuntimeError(
            f"Expected 88 families/176 comparisons, got "
            f"{result.all_force_parent.nunique()}/{len(result)}"
        )
    return result


def freeze(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    selection_path = args.output / "FROZEN_FAMILY_SUPPORT.csv"
    receipt_path = args.output / "FROZEN_SELECTION.json"
    if selection_path.exists() or receipt_path.exists():
        raise FileExistsError("Frozen selection already exists")
    context = load_context(args, build_stimulus=False)
    support = family_support(context)
    both = support.groupby("all_force_parent").qualifies.all()
    child_min = support.groupby("all_force_parent").thinned_total.min()
    display = sorted(
        [int(parent) for parent in both.index[both]],
        key=lambda parent: (-int(child_min.loc[parent]), parent),
    )[:6]
    support["display_family_frozen"] = support.all_force_parent.isin(display)
    support.to_csv(selection_path, index=False)
    receipt = {
        "status": "frozen_before_cr_rf_outcomes",
        "selection_rule": (
            "88 substantial W2 families; top two children by descending exact-row "
            "count then child ID; uniform per-fold thinning eligibility computed "
            "without RF values"
        ),
        "families": int(support.all_force_parent.nunique()),
        "comparisons": int(len(support)),
        "qualifying_comparisons": int(support.qualifies.sum()),
        "families_both_children_qualify": int(both.sum()),
        "display_parents": display,
        "display_rule": (
            "both children qualify; descending minimum child thinned total, then parent ID"
        ),
        "seeds": list(SEEDS),
        "thresholds": {"minimum_total": 500, "minimum_each_fold": 200},
        "development_trials": context["selected_development"],
        "outer_holdout_trials_not_opened": context["selected_holdout"],
        "outer_holdout_opened": False,
        "gaze_calibration": context["gaze_validation"],
        "hashes": {
            "all_force": sha256(args.all_force),
            "no_force": sha256(args.no_force),
            "config": sha256(args.config),
            "stimulus_manifest": sha256(args.stimulus_cache / "manifest.json"),
            "gaze_csv": sha256(args.gaze_csv),
            "interval_metadata": sha256(args.interval_metadata),
        },
        "elapsed_s": time.perf_counter() - started,
    }
    receipt["frozen_family_support_sha256"] = sha256(selection_path)
    atomic_json(receipt_path, receipt)


def sampled_rows(
    context: dict, parent: int, child: int, seed: int
) -> tuple[dict, dict]:
    af_frame, nf_frame = context["af_frame"], context["nf_frame"]
    fold = context["frame_fold"]
    parent_mask = (context["af_labels"] == parent) & context["af_valid"]
    child_mask = (context["nf_labels"] == child) & context["nf_valid"]
    rng = np.random.default_rng(np.random.SeedSequence([seed, parent, child]))
    selected_parent, selected_child = {}, {}
    for f in (0, 1):
        parent_rows = np.flatnonzero(
            parent_mask & (fold[np.maximum(af_frame, 0)] == f)
        )
        child_rows = np.flatnonzero(
            child_mask & (fold[np.maximum(nf_frame, 0)] == f)
        )
        n = min(len(parent_rows), len(child_rows))
        selected_parent[f] = np.sort(rng.choice(parent_rows, n, replace=False))
        selected_child[f] = np.sort(rng.choice(child_rows, n, replace=False))
    return selected_parent, selected_child


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).ravel()
    right = np.asarray(right, dtype=np.float64).ravel()
    denom = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denom) if denom > 0 else np.nan


def execute(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    result = args.output / "results"
    partial = args.output / "results.partial"
    if result.exists() or partial.exists():
        raise FileExistsError("Results already exist")
    partial.mkdir(parents=True)
    frozen_path = args.output / "FROZEN_FAMILY_SUPPORT.csv"
    frozen_receipt = json.loads((args.output / "FROZEN_SELECTION.json").read_text())
    if sha256(frozen_path) != frozen_receipt["frozen_family_support_sha256"]:
        raise ValueError("Frozen family support changed")
    support = pd.read_csv(frozen_path)
    context = load_context(args, build_stimulus=True)
    fresh = family_support(context)
    comparable = support.drop(columns=["display_family_frozen"])
    pd.testing.assert_frame_equal(
        comparable.reset_index(drop=True), fresh.reset_index(drop=True),
        check_dtype=False,
    )
    config = context["config"]
    training_module, rf_module = context["modules"]
    selections = {}
    metric_rows = []
    family_rows = []
    sta_products = {}
    display_cache = {}
    qualifying = support[support.qualifies].reset_index(drop=True)
    for seed in SEEDS:
        series = []
        responses = np.zeros(
            (len(context["left"]), 2 * len(qualifying)), dtype=np.int32
        )
        for comparison, row in qualifying.iterrows():
            parent = int(row.all_force_parent)
            child = int(row.no_force_child)
            selected_parent, selected_child = sampled_rows(
                context, parent, child, seed
            )
            for f in (0, 1):
                pkey = f"s{seed}_p{parent}_c{child}_parent_fold{f}"
                ckey = f"s{seed}_p{parent}_c{child}_child_fold{f}"
                selections[pkey] = selected_parent[f]
                selections[ckey] = selected_child[f]
                np.add.at(
                    responses[:, 2 * comparison],
                    context["af_frame"][selected_parent[f]],
                    1,
                )
                np.add.at(
                    responses[:, 2 * comparison + 1],
                    context["nf_frame"][selected_child[f]],
                    1,
                )
            series.extend(
                [
                    (parent, child, "parent", int(row.child_rank)),
                    (parent, child, "child", int(row.child_rank)),
                ]
            )
        stas = []
        for f in (0, 1):
            dfs = context["base_valid"] & (context["frame_fold"] == f)
            sta = rf_module.calc_sta(
                context["stimulus"],
                responses,
                context["lags"],
                dfs=dfs.astype(np.float32),
                reverse_correlate=bool(config.get("sta_reverse_correlate", True)),
                batch_size=args.batch_size,
                device="cpu",
                progress=True,
            ).cpu().numpy()
            if sta.ndim == 5 and sta.shape[2] == 1:
                sta = sta[:, :, 0]
            stas.append(sta)
        scores = crossvalidated_sta_snr(
            stas[0], stas[1], dxy_deg=float(config["dxy_deg"])
        )
        for index, (parent, child, role, rank) in enumerate(series):
            metric_rows.append(
                {
                    "seed": seed,
                    "all_force_parent": parent,
                    "no_force_child": child,
                    "child_rank": rank,
                    "role": role,
                    "cv_snr": float(scores["cv_snr"][index]),
                    "fold0_to_fold1_snr": float(
                        scores["fold0_to_fold1_snr"][index]
                    ),
                    "fold1_to_fold0_snr": float(
                        scores["fold1_to_fold0_snr"][index]
                    ),
                    "self_fold_reliability": float(
                        scores["fold_map_cosine"][index]
                    ),
                }
            )
        index_by = {
            (parent, child, role): index
            for index, (parent, child, role, _rank) in enumerate(series)
        }
        both = support.groupby("all_force_parent").qualifies.all()
        for parent in sorted(map(int, both.index[both])):
            family = support[support.all_force_parent == parent].sort_values(
                "child_rank"
            )
            child1, child2 = map(int, family.no_force_child)
            i1 = index_by[(parent, child1, "child")]
            i2 = index_by[(parent, child2, "child")]
            _, hp10 = dog_standardize(
                stas[0][[i1]], dxy_deg=float(config["dxy_deg"])
            )
            _, hp11 = dog_standardize(
                stas[1][[i1]], dxy_deg=float(config["dxy_deg"])
            )
            _, hp20 = dog_standardize(
                stas[0][[i2]], dxy_deg=float(config["dxy_deg"])
            )
            _, hp21 = dog_standardize(
                stas[1][[i2]], dxy_deg=float(config["dxy_deg"])
            )
            cross01 = cosine(hp10[0], hp21[0])
            cross10 = cosine(hp11[0], hp20[0])
            family_rows.append(
                {
                    "seed": seed,
                    "all_force_parent": parent,
                    "child1": child1,
                    "child2": child2,
                    "child1_self_fold_reliability": float(
                        scores["fold_map_cosine"][i1]
                    ),
                    "child2_self_fold_reliability": float(
                        scores["fold_map_cosine"][i2]
                    ),
                    "child_cross_fold_0to1_cosine": cross01,
                    "child_cross_fold_1to0_cosine": cross10,
                    "child_cross_fold_symmetric_cosine": float(
                        np.nanmean([cross01, cross10])
                    ),
                }
            )
            for child, index in ((child1, i1), (child2, i2)):
                sta_products[f"s{seed}_p{parent}_c{child}_fold0_sta"] = stas[0][index]
                sta_products[f"s{seed}_p{parent}_c{child}_fold1_sta"] = stas[1][index]
                parent_index = index_by[(parent, child, "parent")]
                sta_products[
                    f"s{seed}_p{parent}_matched_to_c{child}_fold0_sta"
                ] = stas[0][parent_index]
                sta_products[
                    f"s{seed}_p{parent}_matched_to_c{child}_fold1_sta"
                ] = stas[1][parent_index]
            if parent in frozen_receipt["display_parents"] and seed == SEEDS[0]:
                parent_index = index_by[(parent, child1, "parent")]
                display_cache[parent] = {
                    "parent": parent,
                    "children": (child1, child2),
                    "fold0": stas[0][[parent_index, i1, i2]],
                    "fold1": stas[1][[parent_index, i1, i2]],
                }
    np.savez_compressed(partial / "SELECTED_EVENT_ROW_IDS.npz", **selections)
    np.savez_compressed(partial / "BOTH_CHILD_FOLD_STAS.npz", **sta_products)
    metrics = pd.DataFrame(metric_rows)
    wide = metrics.pivot(
        index=["seed", "all_force_parent", "no_force_child", "child_rank"],
        columns="role",
        values=[
            "cv_snr",
            "fold0_to_fold1_snr",
            "fold1_to_fold0_snr",
            "self_fold_reliability",
        ],
    )
    wide.columns = [f"{role}_{metric}" for metric, role in wide.columns]
    wide = wide.reset_index()
    wide["child_minus_parent_cv_snr"] = wide.child_cv_snr - wide.parent_cv_snr
    wide = wide.merge(
        support,
        on=["all_force_parent", "no_force_child", "child_rank"],
        how="left",
        validate="many_to_one",
    )
    wide.to_csv(partial / "COUNT_CONTROLLED_PARENT_CHILD_RF.csv", index=False)
    family_metrics = pd.DataFrame(family_rows)
    family_metrics.to_csv(
        partial / "BOTH_CHILD_RF_AGREEMENT.csv", index=False
    )
    if display_cache:
        figure, axes = plt.subplots(
            len(display_cache), 3,
            figsize=(8.4, 2.3 * len(display_cache)),
            squeeze=False,
        )
        for row_index, parent in enumerate(frozen_receipt["display_parents"]):
            item = display_cache[parent]
            z0, _ = dog_standardize(
                item["fold0"][[0]], dxy_deg=float(config["dxy_deg"])
            )
            selected = int(np.argmax(np.abs(z0[0])))
            lag_index = np.unravel_index(selected, z0[0].shape)[0]
            maps = item["fold1"][:, lag_index]
            limit = float(np.max(np.abs(maps))) or 1.0
            labels = [
                f"parent {parent}",
                f"child {item['children'][0]}",
                f"child {item['children'][1]}",
            ]
            for column, (axis, rf_map, label) in enumerate(
                zip(axes[row_index], maps, labels)
            ):
                image = axis.imshow(
                    rf_map, cmap="BrBG", vmin=-limit, vmax=limit,
                    origin="lower", interpolation="nearest"
                )
                axis.set_title(f"{label}; lag {context['lags'][lag_index]}")
                axis.set_xticks([]); axis.set_yticks([])
                if column == 2:
                    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        figure.suptitle(
            "Frozen count-support families: fold-0 parent selects lag; fold-1 maps shown"
        )
        figure.tight_layout()
        figure.savefig(partial / "FROZEN_FAMILY_RF_MAPS.png", dpi=180)
        plt.close(figure)
    seed_summaries = []
    for seed, rows in wide.groupby("seed"):
        delta = rows.child_minus_parent_cv_snr.to_numpy()
        seed_summaries.append(
            {
                "seed": int(seed),
                "comparisons": int(len(rows)),
                "median_child_minus_parent_cv_snr": float(np.median(delta)),
                "mean_child_minus_parent_cv_snr": float(np.mean(delta)),
                "child_wins": int((delta > 0).sum()),
                "child_losses": int((delta < 0).sum()),
                "exact_ties": int((delta == 0).sum()),
            }
        )
    signs = wide.pivot_table(
        index=["all_force_parent", "no_force_child", "child_rank"],
        columns="seed", values="child_minus_parent_cv_snr"
    )
    consistent_positive = int((signs.gt(0).all(axis=1)).sum())
    consistent_negative = int((signs.lt(0).all(axis=1)).sum())
    summary = {
        "status": "complete_development_only",
        "families_total": int(support.all_force_parent.nunique()),
        "comparisons_total": int(len(support)),
        "qualifying_comparisons": int(support.qualifies.sum()),
        "failed_comparisons": int((~support.qualifies).sum()),
        "families_both_children_qualify": int(
            support.groupby("all_force_parent").qualifies.all().sum()
        ),
        "seed_summaries": seed_summaries,
        "comparisons_positive_all_three_seeds": consistent_positive,
        "comparisons_negative_all_three_seeds": consistent_negative,
        "dependent_scores": True,
        "interpretation": (
            "Parent and child scores overlap in event lineage. Seed spread is a "
            "thinning sensitivity, not an independent-trial confidence interval."
        ),
        "outer_holdout_opened": False,
        "gaze_calibration": context["gaze_validation"],
        "elapsed_s": time.perf_counter() - started,
    }
    atomic_json(partial / "SUMMARY.json", summary)
    products = []
    for path in sorted(partial.iterdir()):
        products.append(
            {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
        )
    atomic_json(
        partial / "COMPLETE.json",
        {"status": "complete", "written_last": True, "products": products},
    )
    os.replace(partial, result)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("mode", choices=("freeze", "execute"))
    result.add_argument("--all-force", type=Path, required=True)
    result.add_argument("--no-force", type=Path, required=True)
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--stimulus-cache", type=Path, required=True)
    result.add_argument("--gaze-csv", type=Path, required=True)
    result.add_argument("--interval-metadata", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--batch-size", type=int, default=5000)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    if arguments.mode == "freeze":
        freeze(arguments)
    else:
        execute(arguments)
