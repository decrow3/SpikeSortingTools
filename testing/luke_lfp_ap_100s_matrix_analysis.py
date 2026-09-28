"""Evaluate the matched LFP/AP sort matrix with frozen lighthouse events."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from testing.luke_medicine_5min_matrix_analysis import load_sort, match_events, unit_metrics


START_S = 930.0
TRAIN_STOP_S = 940.0
STOP_S = 1030.0
TOLERANCE_MS = 0.5
DEPTH_TOLERANCE_UM = 100.0
MIN_ALL_EVENTS = 20
MIN_REGIME_EVENTS = 10
ARMS = ["unwarped", "lfp_rigid", "ap_rigid", "ap_nonrigid"]


def expected_depth(events: pd.DataFrame, fields: np.lib.npyio.NpzFile, arm: str) -> np.ndarray:
    observed = events.centroid_um.to_numpy(float)
    if arm == "unwarped":
        return observed
    if arm == "lfp_rigid":
        displacement = np.interp(events.time_s, fields["lfp_time_s"], fields["lfp_displacement_um"])
    elif arm == "lfp_savgol":
        displacement = np.interp(
            events.time_s,
            fields["lfp_savgol_time_s"],
            fields["lfp_savgol_displacement_um"],
        )
    elif arm == "lfp_native":
        displacement = np.interp(
            events.time_s,
            fields["lfp_native_time_s"],
            fields["lfp_native_displacement_um"],
        )
    elif arm == "ap_rigid":
        displacement = np.interp(events.time_s, fields["ap_time_s"], fields["ap_rigid_displacement_um"])
    elif arm == "ap_nonrigid":
        sampler = RegularGridInterpolator(
            (fields["ap_time_s"], fields["ap_depth_um"]),
            fields["ap_nonrigid_displacement_um"], bounds_error=False, fill_value=None,
        )
        displacement = sampler(np.column_stack([events.time_s, observed]))
    elif arm == "dartsort_native":
        sampler = RegularGridInterpolator(
            (fields["dartsort_time_s"], fields["dartsort_depth_um"]),
            fields["dartsort_displacement_um"], bounds_error=False, fill_value=None,
        )
        displacement = sampler(np.column_stack([events.time_s, observed]))
    else:
        raise ValueError(arm)
    return observed - displacement


def analyze_arm(arm: str, arm_dir: Path, events: pd.DataFrame, fields: np.lib.npyio.NpzFile):
    sort = load_sort(arm_dir)
    fs = sort["fs"]
    tolerance = int(round(TOLERANCE_MS * 1e-3 * fs))
    depths = expected_depth(events, fields, arm)
    samples = events.frame.to_numpy(np.int64) - int(round(START_S * fs))
    matched_rows = []
    for unit_id, indices in events.groupby("unit_id", sort=False).groups.items():
        index = np.asarray(list(indices), dtype=int)
        matched = match_events(
            samples[index], depths[index], sort["times"], sort["depths"], sort["clusters"],
            tolerance, DEPTH_TOLERANCE_UM,
        )
        matched["event_row"] = index
        matched["unit_id"] = int(unit_id)
        matched["arm"] = arm
        matched_rows.append(matched)
    matched = pd.concat(matched_rows, ignore_index=True)
    metadata = events[["unit_id", "family_id", "time_s", "bin", "regime"]].reset_index().rename(columns={"index": "event_row"})
    matched = matched.merge(metadata, on=["event_row", "unit_id"], how="left", validate="one_to_one")

    unit_rows = []
    for unit_id, group in matched.groupby("unit_id"):
        unit_rows.append({"arm": arm, "regime": "all", "unit_id": int(unit_id), **unit_metrics(group)})
    for (regime, unit_id), group in matched.loc[matched.regime.notna()].groupby(["regime", "unit_id"]):
        unit_rows.append({"arm": arm, "regime": regime, "unit_id": int(unit_id), **unit_metrics(group)})
    unit_table = pd.DataFrame(unit_rows)

    labels = sort["labels"]
    label_col = next(column for column in labels.columns if column != "cluster_id")
    contamination = sort["contam"]
    contam_col = next(column for column in contamination.columns if column != "cluster_id")
    good_ids = set(labels.loc[labels[label_col].astype(str).str.lower().eq("good"), "cluster_id"].astype(int))
    isi = []
    for cluster in np.unique(sort["clusters"]):
        train = np.sort(sort["times"][sort["clusters"] == cluster])
        if train.size > 1:
            isi.append(float(np.mean(np.diff(train) < int(round(1.5e-3 * fs)))))
    all_units = unit_table.loc[unit_table.regime.eq("all")]
    aggregate = {
        "arm": arm,
        "spikes": int(len(sort["times"])),
        "units": int(labels.cluster_id.nunique()),
        "good_units": int(len(good_ids)),
        "median_contamination_pct": float(contamination[contam_col].median()),
        "median_refractory_fraction": float(np.median(isi)),
        "macro_lighthouse_recovery": float(all_units.recovery_fraction.mean()),
        "macro_single_cluster_all_events": float(all_units.single_cluster_fraction_all_events.mean()),
        "macro_duplicate_event_fraction": float(all_units.duplicate_event_fraction.mean()),
        "lighthouses": int(len(all_units)),
    }
    return aggregate, unit_table, matched


def regime_metrics(unit_table: pd.DataFrame, family_map: pd.DataFrame) -> pd.DataFrame:
    table = unit_table.merge(family_map[["unit_id", "family_id"]], on="unit_id", how="left", validate="many_to_one")
    rows = []
    for (arm, regime), group in table.loc[table.regime.isin(["all", "quiet", "movement"])].groupby(["arm", "regime"]):
        threshold = MIN_ALL_EVENTS if regime == "all" else MIN_REGIME_EVENTS
        eligible = group.loc[group.events.ge(threshold)].copy()
        families = eligible.groupby("family_id", as_index=False).agg(
            events=("events", "sum"), units=("unit_id", "nunique"),
            recovery_fraction=("recovery_fraction", "mean"),
            single_cluster_fraction=("single_cluster_fraction_all_events", "mean"),
            duplicate_event_fraction=("duplicate_event_fraction", "mean"),
            fragment_clusters=("fragment_clusters", "mean"),
        )
        rows.append({
            "arm": arm,
            "regime": regime,
            "eligible_families": int(len(families)),
            "eligible_units": int(eligible.unit_id.nunique()),
            "events": int(families.events.sum()) if len(families) else 0,
            "family_macro_recovery": float(families.recovery_fraction.mean()) if len(families) else np.nan,
            "family_macro_single_cluster": float(families.single_cluster_fraction.mean()) if len(families) else np.nan,
            "family_macro_duplicate_fraction": float(families.duplicate_event_fraction.mean()) if len(families) else np.nan,
            "median_family_fragment_clusters": float(families.fragment_clusters.median()) if len(families) else np.nan,
            "minimum_events_per_unit": threshold,
        })
    return pd.DataFrame(rows)


def plot_results(global_summary: pd.DataFrame, regimes: pd.DataFrame, output: Path) -> None:
    colors = {"unwarped": "#6B7280", "lfp_rigid": "#84A80B", "ap_rigid": "#D97706", "ap_nonrigid": "#2563EB"}
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    metrics = [
        ("family_macro_single_cluster", "All held-out lighthouse events", regimes.loc[regimes.regime.eq("all")]),
        ("family_macro_single_cluster", "Lighthouse-quiet events", regimes.loc[regimes.regime.eq("quiet")]),
        ("family_macro_single_cluster", "Lighthouse-movement events", regimes.loc[regimes.regime.eq("movement")]),
        ("good_units", "KS-good units", global_summary),
    ]
    for ax, (column, title, data) in zip(axes.flat, metrics):
        data = data.set_index("arm").reindex(ARMS).reset_index()
        ax.bar(np.arange(len(data)), data[column], color=[colors[name] for name in data.arm], edgecolor="#374151")
        ax.set_xticks(np.arange(len(data)), [name.replace("_", "\n") for name in data.arm])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.15)
    fig.suptitle("Matched 930–1030 s correction-sort matrix\nFamily-balanced single-cluster concentration is the primary identity endpoint")
    fig.savefig(output / "01_sort_matrix_summary.png", dpi=180)
    fig.savefig(output / "01_sort_matrix_summary.pdf")
    plt.close(fig)


def run(output_root: Path, events_path: Path, family_path: Path, regime_path: Path) -> dict:
    analysis = output_root / "analysis"
    analysis.mkdir(exist_ok=False)
    events = pd.read_csv(events_path)
    family = pd.read_csv(family_path).rename(columns={"family": "family_id"})
    events = events.merge(family[["unit_id", "family_id"]], on="unit_id", how="left", validate="many_to_one")
    events = events.loc[
        events.evidence.eq("strict_accepted") & ~events.training.astype(bool)
        & events.time_s.ge(TRAIN_STOP_S) & events.time_s.lt(STOP_S)
    ].copy()
    events = events.sort_values(["unit_id", "frame", "score"], ascending=[True, True, False]).drop_duplicates(["unit_id", "frame"]).reset_index(drop=True)
    events["bin"] = np.floor((events.time_s - START_S) / 5.0).astype(int)
    regime = pd.read_csv(regime_path)
    regime = regime.loc[
        regime.window.eq("930_1030") & regime.sensitivity.eq("plausibility_lattice_node_5um")
        & regime.candidate.eq("zero"), ["family_id", "bin_to", "regime"],
    ].drop_duplicates().rename(columns={"bin_to": "bin"})
    events = events.merge(regime, on=["family_id", "bin"], how="left", validate="many_to_one")
    if events.empty:
        raise RuntimeError("no frozen held-out lighthouse events")

    fields = np.load(output_root / "resolved_candidate_fields.npz", allow_pickle=False)
    aggregates, units, matched = [], [], []
    for arm in ARMS:
        aggregate, unit_table, event_table = analyze_arm(arm, output_root / "arms" / arm, events, fields)
        aggregates.append(aggregate)
        units.append(unit_table)
        matched.append(event_table)
    global_summary = pd.DataFrame(aggregates)
    baseline = global_summary.set_index("arm").loc["unwarped"]
    for column in ["macro_lighthouse_recovery", "macro_single_cluster_all_events", "macro_duplicate_event_fraction", "median_contamination_pct", "median_refractory_fraction", "good_units"]:
        global_summary[f"delta_vs_unwarped_{column}"] = global_summary[column] - baseline[column]
    unit_table = pd.concat(units, ignore_index=True)
    matched_table = pd.concat(matched, ignore_index=True)
    regimes = regime_metrics(unit_table, family)
    for column in ["family_macro_recovery", "family_macro_single_cluster", "family_macro_duplicate_fraction", "median_family_fragment_clusters"]:
        base = regimes.loc[regimes.arm.eq("unwarped"), ["regime", column]].set_index("regime")[column]
        regimes[f"delta_vs_unwarped_{column}"] = regimes[column] - regimes.regime.map(base)

    global_summary.to_csv(analysis / "arm_summary.csv", index=False)
    regimes.to_csv(analysis / "regime_summary.csv", index=False)
    unit_table.to_csv(analysis / "lighthouse_unit_metrics.csv", index=False)
    matched_table.to_csv(analysis / "matched_events.csv", index=False)
    plot_results(global_summary, regimes, analysis)
    result = {
        "status": "complete",
        "primary_endpoint": "family-macro single-cluster fraction among frozen held-out strict lighthouse events",
        "interval_s": [START_S, STOP_S],
        "training_excluded_s": [START_S, TRAIN_STOP_S],
        "time_tolerance_ms": TOLERANCE_MS,
        "depth_tolerance_um": DEPTH_TOLERANCE_UM,
        "all_events_minimum_per_unit": MIN_ALL_EVENTS,
        "regime_events_minimum_per_unit": MIN_REGIME_EVENTS,
        "regime_policy": "frozen depth-aware lighthouse 5 s increment: quiet <20 µm, movement >=20 µm",
        "global_arms": json.loads(global_summary.to_json(orient="records")),
        "regime_arms": json.loads(regimes.to_json(orient="records")),
        "interpretation": "development comparison; raw unit count and event recovery are not sufficient efficacy endpoints",
    }
    (analysis / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    raise SystemExit("Run through the managed matrix worker so analysis follows completed sorts.")
