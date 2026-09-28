#!/usr/bin/env python
"""Score frozen imec1 compact-core observations against independent fields."""
from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from testing.luke_imec1_lighthouse_motion_adapters_v1 import (
    MotionField, load_dartsort_pickle, load_lfp, load_medicine, rigid_projection,
    sha256, zero_field,
)


ROOT = Path(__file__).resolve().parents[1]
ASSESSMENT = ROOT / "testing/outputs/luke_imec1_twenty_candidate_assessment_v1/twenty_candidate_assessment.csv"
EVENTS = ROOT / "testing/outputs/luke_imec1_compact_alignment_pilot_v2/qualification_events_compact.csv"
TEMPLATES = ROOT / "testing/outputs/luke_imec1_compact_alignment_pilot_v2/frozen_training_templates.npz"
FAMILIES = ROOT / "testing/outputs/luke_imec1_dots_sorterfree_waveform_discovery_v3_s300/families_after_depth_reveal.csv"
NATIVE = Path("/media/huklab/Data/NPX/Ryansorting/Luke/DARTsort_motion_experiments/luke0804-imec1-local-spikeglx-dots-v4/sorts/native-v1/motion.pkl")
MEDICINE = Path("/media/huklab/Data/NPX/Ryansorting/Luke/DARTsort_motion_experiments/luke0804-imec1-local-spikeglx-dots-v4/fine_medicine_v6_estimate_v1/medicine_v6_documented")
DEFAULT_LFP = ROOT / "testing/inputs/luke_imec1_dredge_lfp_full_crop_v1"
DEFAULT_OUT = ROOT / "testing/outputs/luke_imec1_lighthouse_motion_comparison_v1"
FS = 29999.759166666667
CROP_START_FRAME = 193737
BIN_S = 5.0
QUIET_UM = 20.0
PRIMARY_IDS = ("p08_f025", "p08_f033", "p06_f010", "p08_f029", "p06_f045", "p09_f020", "p09_f024", "p08_f044")


def _write_table(table: pd.DataFrame, path: Path) -> None:
    try:
        table.to_parquet(path, index=False)
    except ImportError:
        table.to_csv(path.with_suffix(".csv"), index=False)


def _template_hashes() -> dict[str, str]:
    with np.load(TEMPLATES, allow_pickle=True) as data:
        ids = data["family_id"].astype(str)
        templates = data["compact"]
        scales = data["compact_scale"]
    result = {}
    for family_id, template, scale in zip(ids, templates, scales):
        digest = hashlib.sha256()
        digest.update(np.ascontiguousarray(template).view(np.uint8))
        digest.update(np.asarray(scale, dtype=np.float64).tobytes())
        result[family_id] = digest.hexdigest()
    return result


def _decorate_events(events: pd.DataFrame, hashes: dict[str, str], families: pd.DataFrame) -> pd.DataFrame:
    events = events.copy()
    events["template_sha256"] = events.winner_family.map(hashes)
    events["family_uid"] = events.apply(
        lambda r: f"compact-v2:s300:{r.winner_family}:{r.template_sha256[:16]}", axis=1
    )
    events["reference_depth_um"] = events.winner_family.map(families.seed_depth_median_um)
    events["raw_ap_frame"] = np.rint(events.time_s.to_numpy(float) * FS).astype(np.int64)
    events["crop_time_s"] = (events.raw_ap_frame - CROP_START_FRAME) / FS
    events["event_key"] = events.apply(
        lambda r: f"{r.family_uid}:{int(r.raw_ap_frame)}:{r.source_event_id}", axis=1
    )
    events["observed_relative_um"] = events.waveform_centroid_um - events.reference_depth_um
    return events


def load_frozen_events(extra_events: Path | None = None) -> pd.DataFrame:
    assessment = pd.read_csv(ASSESSMENT)
    if tuple(assessment.loc[assessment.assessment_tier.eq("retained_450um"), "family_id"]) != PRIMARY_IDS:
        raise RuntimeError("primary proposal cohort changed")
    qualification = pd.read_csv(EVENTS)
    qualification = qualification.loc[
        qualification.winner_bank.eq("s300")
        & qualification.winner_family.isin(PRIMARY_IDS)
        & qualification.winner_kind.eq(0)
        & qualification.status.eq("strict")
    ].copy()
    qualification["evidence_role"] = "qualification"
    qualification["window_id"] = "qualification_310_320"
    qualification["window_start_s"] = 310.0
    qualification["window_stop_s"] = 320.0
    qualification["source_event_id"] = qualification.event_id.astype(str)
    tables = [qualification]
    if extra_events is not None:
        extra = pd.read_csv(extra_events)
        if not extra.empty:
            extra["evidence_role"] = "extra_validation"
            extra["window_id"] = extra.window_start_s.map(lambda x: f"validation_{x:g}")
            extra["source_event_id"] = extra.apply(
                lambda r: f"{r.window_start_s:g}:{int(r.event_index_window)}", axis=1
            )
            tables.append(extra)
    events = pd.concat(tables, ignore_index=True, sort=False)
    families = pd.read_csv(FAMILIES).set_index("family_id")
    hashes = _template_hashes()
    events = _decorate_events(events, hashes, families)
    if events.event_key.duplicated().any() or len(qualification) != 167:
        raise RuntimeError("frozen strict-event grain changed")
    return events


def load_fields(lfp_root: Path) -> tuple[dict[str, MotionField], dict]:
    fields: dict[str, MotionField] = {
        "zero": zero_field(),
        "native_ap": load_dartsort_pickle(NATIVE, name="native_ap"),
        "medicine": load_medicine(MEDICINE, name="medicine"),
    }
    receipt: dict[str, object] = {
        "native_pickle_sha256": sha256(NATIVE),
        "medicine": {p.name: sha256(p) for p in sorted(MEDICINE.glob("*.npy"))},
        "lfp_available": False,
    }
    spec_path = lfp_root / "spec.json"
    crop_source = 0.0
    if spec_path.exists():
        spec = json.loads(spec_path.read_text())
        # Producer arrays use absolute acquisition time; ap_origin_s maps that
        # to source-recording seconds, then crop_start_source_s maps to crop.
        crop_source = float(spec.get("ap_origin_s", 0.0)) + float(spec.get("crop_start_source_s", 0.0))
    lfp_files = {
        "lfp80": lfp_root / "unsharpened_80um_motion.npz",
        "lfp260": lfp_root / "unsharpened_260um_motion.npz",
    }
    if all(path.exists() for path in lfp_files.values()):
        for name, path in lfp_files.items():
            fields[name] = load_lfp(path, name=name, crop_start_source_s=crop_source)
        receipt["lfp_available"] = True
        receipt["lfp"] = {path.name: sha256(path) for path in lfp_root.iterdir() if path.is_file()}
    medicine = fields["medicine"]
    fields["medicine_rigid"] = rigid_projection(medicine, name="medicine_rigid")
    return fields, receipt


def add_predictions(events: pd.DataFrame, fields: dict[str, MotionField]) -> pd.DataFrame:
    out = events.copy()
    for name, field in fields.items():
        values, supported = field.sample(out.crop_time_s.to_numpy(), out.reference_depth_um.to_numpy())
        out[f"{name}_um"] = values
        out[f"{name}_supported"] = supported
    primary = [name for name in ("zero", "native_ap", "medicine", "lfp80", "lfp260") if name in fields]
    out["all_primary_common_support"] = out[[f"{x}_supported" for x in primary]].all(axis=1)
    return out


def make_increments(events: pd.DataFrame, fields: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    # Reduce every primary field on exactly the same events. This deliberately
    # drops an entire event from all medians when any primary field lacks native
    # support; pairwise support is reported separately in support_summary.csv.
    strict = events.loc[events.all_primary_common_support].copy()
    strict["bin"] = np.floor((strict.time_s - strict.window_start_s) / BIN_S).astype(int)
    support_cols = [f"{name}_supported" for name in fields]
    for (family_uid, window_id), family in strict.groupby(["family_uid", "window_id"]):
        levels = family.groupby(["window_id", "bin"], as_index=False).agg(
            observed_um=("observed_relative_um", "median"),
            first_event_s=("time_s", "min"), last_event_s=("time_s", "max"),
            event_count=("event_key", "size"), reference_depth_um=("reference_depth_um", "first"),
            **{f"{name}_um": (f"{name}_um", "median") for name in fields},
            **{col: (col, "all") for col in support_cols},
        ).sort_values("bin")
        for previous, current in zip(levels.iloc[:-1].itertuples(), levels.iloc[1:].itertuples()):
            if current.bin != previous.bin + 1:
                continue
            observed_delta = current.observed_um - previous.observed_um
            regime = "quiet" if abs(observed_delta) < QUIET_UM else "movement"
            for name in fields:
                supported = bool(getattr(previous, f"{name}_supported") and getattr(current, f"{name}_supported"))
                predicted = getattr(current, f"{name}_um") - getattr(previous, f"{name}_um") if supported else np.nan
                rows.append({
                    "family_uid": family_uid, "candidate": name, "window_id": current.window_id,
                    "bin_from": previous.bin, "bin_to": current.bin,
                    "elapsed_s": BIN_S, "reference_depth_um": current.reference_depth_um,
                    "events_from": previous.event_count, "events_to": current.event_count,
                    "first_event_s": previous.first_event_s, "last_event_s": current.last_event_s,
                    "lighthouse_delta_um": observed_delta, "candidate_delta_um": predicted,
                    "residual_um": observed_delta - predicted if supported else np.nan,
                    "supported": supported, "regime": regime,
                })
    return pd.DataFrame(rows)


def family_balanced_metrics(increments: pd.DataFrame) -> dict[str, float | int | bool]:
    good = increments.loc[np.isfinite(increments.residual_um)].copy()
    if good.empty:
        return {"families": 0, "scored_families": 0, "increments": 0, "scored_increments": 0,
                "rmse_um": np.nan, "mae_um": np.nan, "eligible": False}
    per_family = good.groupby("family_uid").residual_um.agg(
        mse=lambda x: float(np.mean(np.square(x))), mae=lambda x: float(np.mean(np.abs(x))), n="size"
    )
    eligible_ids = per_family.index[per_family.n >= 3]
    scored = good.loc[good.family_uid.isin(eligible_ids)]
    scored_family = per_family.loc[eligible_ids]
    error = scored.residual_um.to_numpy(float)
    return {
        "families": int(len(per_family)), "increments": int(len(good)),
        "scored_families": int(len(scored_family)), "scored_increments": int(len(scored)),
        "rmse_um": float(np.sqrt(scored_family.mse.mean())) if len(scored_family) else np.nan,
        "mae_um": float(scored_family.mae.mean()) if len(scored_family) else np.nan,
        "p95_abs_error_um": float(np.quantile(np.abs(error), .95)) if len(error) else np.nan,
        "largest_abs_error_um": float(np.max(np.abs(error))) if len(error) else np.nan,
        "exceed_40_fraction": float(np.mean(np.abs(error) > 40)) if len(error) else np.nan,
        "exceed_80_fraction": float(np.mean(np.abs(error) > 80)) if len(error) else np.nan,
        "exceed_120_fraction": float(np.mean(np.abs(error) > 120)) if len(error) else np.nan,
        "eligible": bool(len(scored_family) >= 3),
    }


def shared_movement_summary(increments: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Compare independently measured lighthouse deltas on identical time-bin edges."""
    zero = increments.loc[increments.candidate.eq("zero")].copy()
    pivot = zero.pivot_table(
        index=["window_id", "bin_from", "bin_to"],
        columns="family_uid",
        values="lighthouse_delta_um",
        aggfunc="first",
    )
    rows = []
    for left, right in combinations(pivot.columns, 2):
        pair = pivot[[left, right]].dropna()
        rows.append({
            "family_uid_left": left,
            "family_uid_right": right,
            "shared_increments": int(len(pair)),
            "pearson_r": float(pair[left].corr(pair[right])) if len(pair) >= 3 else np.nan,
            "sign_agreement_fraction": float(
                np.mean(np.sign(pair[left].to_numpy()) == np.sign(pair[right].to_numpy()))
            ) if len(pair) else np.nan,
            "both_movement_increments": int(
                ((pair[left].abs() >= QUIET_UM) & (pair[right].abs() >= QUIET_UM)).sum()
            ),
        })
    occupancy = pivot.notna().sum(axis=1)
    return pd.DataFrame(rows), {
        "time_bin_edges": int(len(pivot)),
        "time_bin_edges_with_two_or_more_families": int((occupancy >= 2).sum()),
        "time_bin_edges_with_three_or_more_families": int((occupancy >= 3).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--lfp-root", type=Path, default=DEFAULT_LFP)
    parser.add_argument("--extra-events", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    events = load_frozen_events(args.extra_events)
    fields, source_receipt = load_fields(args.lfp_root)
    predictions = add_predictions(events, fields)
    _write_table(predictions, args.output / "event_predictions.parquet")
    depth_span_rows = []
    for (family_uid, role), group in predictions.groupby(["family_uid", "evidence_role"]):
        depth = group.waveform_centroid_um.to_numpy(float)
        depth_span_rows.append({
            "family_uid": family_uid,
            "evidence_role": role,
            "events": int(len(depth)),
            "minimum_depth_um": float(np.min(depth)),
            "maximum_depth_um": float(np.max(depth)),
            "full_span_um": float(np.ptp(depth)),
            "p05_p95_span_um": float(np.quantile(depth, .95) - np.quantile(depth, .05)),
        })
    pd.DataFrame(depth_span_rows).to_csv(args.output / "candidate_depth_spans.csv", index=False)

    primary = [x for x in ("zero", "native_ap", "medicine", "lfp80", "lfp260") if x in fields]
    support_rows = []
    for name in fields:
        support_rows.append({"grain": "all", "family_uid": "all", "candidate": name, "events": len(predictions),
                             "supported_events": int(predictions[f"{name}_supported"].sum()),
                             "supported_fraction": float(predictions[f"{name}_supported"].mean())})
        for family_uid, group in predictions.groupby("family_uid"):
            support_rows.append({"grain": "family", "family_uid": family_uid, "candidate": name,
                                 "events": len(group), "supported_events": int(group[f"{name}_supported"].sum()),
                                 "supported_fraction": float(group[f"{name}_supported"].mean())})
    support_rows.append({"grain": "all", "family_uid": "all", "candidate": "all_primary_common", "events": len(predictions),
                         "supported_events": int(predictions.all_primary_common_support.sum()),
                         "supported_fraction": float(predictions.all_primary_common_support.mean())})
    for challenger in [x for x in ("medicine", "lfp80", "lfp260") if x in fields]:
        paired = predictions.native_ap_supported & predictions[f"{challenger}_supported"]
        support_rows.append({"grain": "paired_native", "family_uid": "all", "candidate": challenger,
                             "events": len(predictions), "supported_events": int(paired.sum()),
                             "supported_fraction": float(paired.mean())})
    pd.DataFrame(support_rows).to_csv(args.output / "support_summary.csv", index=False)

    increments = make_increments(predictions, primary + (["medicine_rigid"] if "medicine_rigid" in fields else []))
    increments.to_csv(args.output / "family_increments.csv", index=False)
    shared_movement, shared_movement_counts = shared_movement_summary(increments)
    shared_movement.to_csv(args.output / "shared_movement_pairwise.csv", index=False)

    paired_tables = []
    for challenger in [x for x in ("native_ap", "medicine", "lfp80", "lfp260") if x in fields]:
        pair_events = predictions.copy()
        pair_events["all_primary_common_support"] = pair_events[f"{challenger}_supported"]
        table = make_increments(pair_events, ["zero", challenger])
        table["support_set"] = f"zero_vs_{challenger}"
        paired_tables.append(table)
    for challenger in [x for x in ("medicine", "lfp80", "lfp260") if x in fields]:
        pair_events = predictions.copy()
        pair_events["all_primary_common_support"] = pair_events.native_ap_supported & pair_events[f"{challenger}_supported"]
        table = make_increments(pair_events, ["native_ap", challenger])
        table["support_set"] = f"native_ap_vs_{challenger}"
        paired_tables.append(table)
    paired_increments = pd.concat(paired_tables, ignore_index=True)
    paired_increments.to_csv(args.output / "paired_family_increments.csv", index=False)
    paired_metrics = []
    for (support_set, candidate, regime), group in pd.concat([
        paired_increments.assign(metric_regime=paired_increments.regime),
        paired_increments.assign(metric_regime="all"),
    ]).groupby(["support_set", "candidate", "metric_regime"]):
        paired_metrics.append({"support_set": support_set, "candidate": candidate, "regime": regime,
                               **family_balanced_metrics(group)})
    pd.DataFrame(paired_metrics).to_csv(args.output / "paired_metrics.csv", index=False)
    metrics = []
    for (candidate, regime), group in increments.groupby(["candidate", "regime"]):
        metrics.append({"candidate": candidate, "regime": regime, **family_balanced_metrics(group)})
    for candidate, group in increments.groupby("candidate"):
        metrics.append({"candidate": candidate, "regime": "all", **family_balanced_metrics(group)})
    metric_table = pd.DataFrame(metrics)
    metric_table.to_csv(args.output / "family_metrics.csv", index=False)
    metric_table.to_csv(args.output / "regime_metrics.csv", index=False)

    per_family_rows = []
    for (candidate, regime, family_uid), group in pd.concat([
        increments.assign(metric_regime=increments.regime),
        increments.assign(metric_regime="all"),
    ]).groupby(["candidate", "metric_regime", "family_uid"]):
        error = group.residual_um.dropna().to_numpy(float)
        per_family_rows.append({
            "candidate": candidate, "regime": regime, "family_uid": family_uid,
            "increments": len(error), "rmse_um": float(np.sqrt(np.mean(error ** 2))) if len(error) else np.nan,
            "mae_um": float(np.mean(np.abs(error))) if len(error) else np.nan,
            "eligible_three_increments": bool(len(error) >= 3),
        })
    pd.DataFrame(per_family_rows).to_csv(args.output / "per_family_metrics.csv", index=False)

    window_rows = []
    for (candidate, regime, window_id), group in pd.concat([
        increments.assign(metric_regime=increments.regime),
        increments.assign(metric_regime="all"),
    ]).groupby(["candidate", "metric_regime", "window_id"]):
        window_rows.append({"candidate": candidate, "regime": regime, "window_id": window_id,
                            **family_balanced_metrics(group)})
    pd.DataFrame(window_rows).to_csv(args.output / "window_metrics.csv", index=False)

    loo_rows = []
    for (candidate, regime), group in pd.concat([
        increments.assign(metric_regime=increments.regime),
        increments.assign(metric_regime="all"),
    ]).groupby(["candidate", "metric_regime"]):
        for omitted in sorted(group.family_uid.unique()):
            loo_rows.append({"candidate": candidate, "regime": regime, "omitted_family_uid": omitted,
                             **family_balanced_metrics(group.loc[group.family_uid.ne(omitted)])})
    pd.DataFrame(loo_rows).to_csv(args.output / "leave_one_family_out.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 6), layout="constrained")
    for family_uid, group in predictions.groupby("family_uid"):
        ax.scatter(group.time_s, group.observed_relative_um, s=20, alpha=.75, label=family_uid.split(":")[2])
    ax.set(xlabel="AP recording-relative time (s)", ylabel="training-reference-relative waveform depth (µm)",
           title="Frozen imec1 compact-core strict events across qualification and validation windows")
    ax.legend(ncol=4, fontsize=7)
    fig.savefig(args.output / "candidate_labels_over_time.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(4, 2, figsize=(15, 13), layout="constrained", sharex=True)
    colors = {"native_ap": "#0072B2", "medicine": "#CC79A7", "lfp80": "#D55E00", "lfp260": "#009E73"}
    for ax, (family_uid, group) in zip(axes.flat, predictions.groupby("family_uid")):
        family_id = family_uid.split(":")[2]
        for window_id, window in group.groupby("window_id"):
            observed = window.observed_relative_um - window.observed_relative_um.median()
            ax.scatter(window.time_s, observed, s=12, color="black", alpha=.65)
            for name in [x for x in colors if x in fields]:
                supported = window[f"{name}_supported"]
                if supported.any():
                    prediction = window.loc[supported, f"{name}_um"]
                    prediction = prediction - prediction.median()
                    ax.plot(window.loc[supported, "time_s"], prediction, ".", ms=2.5,
                            color=colors[name], alpha=.7, label=name if window_id == group.window_id.iloc[0] else None)
        ax.set(title=family_id, ylabel="within-window centered depth (µm)")
    axes[-1, 0].set_xlabel("AP recording-relative time (s)"); axes[-1, 1].set_xlabel("AP recording-relative time (s)")
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle("Frozen waveform-only events and field samples; gaps are not bridged")
    fig.savefig(args.output / "exact_event_tracks_and_fields.png", dpi=170)
    plt.close(fig)

    all_metrics = metric_table.loc[metric_table.regime.eq("all")].set_index("candidate")
    quiet_metrics = metric_table.loc[metric_table.regime.eq("quiet")].set_index("candidate")
    movement_metrics = metric_table.loc[metric_table.regime.eq("movement")].set_index("candidate")
    common_candidates = sorted(set(all_metrics.index) & set(quiet_metrics.index) & set(movement_metrics.index))
    fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
    for name in common_candidates:
        ax.scatter(quiet_metrics.loc[name, "rmse_um"], movement_metrics.loc[name, "p95_abs_error_um"], s=70)
        ax.annotate(name, (quiet_metrics.loc[name, "rmse_um"], movement_metrics.loc[name, "p95_abs_error_um"]), xytext=(4, 4), textcoords="offset points")
    ax.set(xlabel="Quiet-increment RMSE (µm; lower is better)", ylabel="Movement P95 absolute residual (µm; lower is better)",
           title="Quiet penalty versus movement-tail failure")
    fig.savefig(args.output / "movement_tail_vs_quiet_penalty.png", dpi=170)
    plt.close(fig)

    summary = {
        "schema": "luke0804-imec1-lighthouse-motion-comparison-v1",
        "status": (
            "validation_windows_complete"
            if events.evidence_role.eq("extra_validation").any()
            else "qualification_smoke_complete"
        ),
        "strict_primary_events": len(events), "families": int(events.family_uid.nunique()),
        "qualification_events": int(events.evidence_role.eq("qualification").sum()),
        "extra_validation_events": int(events.evidence_role.eq("extra_validation").sum()),
        "windows": sorted(events.window_id.unique()), "crop_start_frame": CROP_START_FRAME,
        "crop_start_source_s": CROP_START_FRAME / FS, "primary_fields_present": primary,
        "eligible_aggregate_rows": int(metric_table.eligible.sum()) if len(metric_table) else 0,
        "ranking_permitted": bool(len(metric_table) and metric_table.eligible.any()),
        "coverage_limit": "Ranking requires at least three adjacent increments in at least three families; inspect eligible flags.",
        "shared_movement": shared_movement_counts,
        "source_receipt": source_receipt,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
