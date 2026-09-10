"""Cheap post-hoc audit of depth-blind Kilosort template families.

Family membership is frozen from v2 before this script sees depth or time.  This
script (1) labels depth/time-coherent families using explicit diagnostics,
(2) asks which *pre-reveal* properties separate those families, and (3) compares
the coherent families with the previously frozen waveform-only lighthouse
measurements.  It reads cached tables/arrays only; it does not read voltage or
run a sorter.
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[1]
FAMILY_OUT = ROOT / "testing/outputs/luke_kilosort_depthblind_families_v2"
EVENTS = ROOT / "testing/outputs/luke_kilosort_cid_depth_scatter_v1/spikes_930_1030.npz"
EVENT_SUMMARY = ROOT / "testing/outputs/luke_kilosort_cid_depth_scatter_v1/summary.json"
LIGHTHOUSE = ROOT / "testing/outputs/luke_waveform_only_expansion_v1/overlay_events.csv"
OUT = ROOT / "testing/outputs/luke_kilosort_family_plausibility_v3"

INTERVAL_S = (930.0, 1030.0)
COACTIVE_BIN_S = 1.0
COACTIVE_MIN_SPIKES_PER_CID = 2
SEPARATION_UM = 80.0
RAPID_JUMP_UM = 100.0
MIN_CROSS_CID_JOINS = 5
MAX_RAPID_JUMP_FRACTION = 0.10
MIN_COACTIVE_BINS = 3
MAX_SEPARATED_COACTIVE_FRACTION = 0.10
MAX_REFRACTORY_RATIO = 3.0
TRACK_BIN_S = 5.0
MIN_MATCHED_TRACK_BINS = 5
MIN_CONSENSUS_UNITS = 3
PERMUTATIONS = 20_000
RNG_SEED = 20260909


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def binned_median(time_s: np.ndarray, value: np.ndarray, width_s: float) -> pd.Series:
    count = int(round((INTERVAL_S[1] - INTERVAL_S[0]) / width_s))
    index = np.floor((np.asarray(time_s) - INTERVAL_S[0]) / width_s).astype(int)
    table = pd.DataFrame({"bin": index, "value": np.asarray(value, float)})
    table = table[(table["bin"] >= 0) & (table["bin"] < count)]
    return table.groupby("bin")["value"].median()


def seed_center(series: pd.Series, width_s: float) -> pd.Series:
    seed_bins = int(round(10.0 / width_s))
    baseline = series[series.index < seed_bins].median()
    return series - baseline


def coactive_depth_separation(
    time_s: np.ndarray,
    depth_um: np.ndarray,
    event_cids: np.ndarray,
    member_cids: np.ndarray,
    width_s: float = COACTIVE_BIN_S,
) -> tuple[int, float, float]:
    """Count bins where >=2 CIDs have support and measure their median-depth span."""
    index = np.floor((time_s - INTERVAL_S[0]) / width_s).astype(int)
    spans = []
    for bin_index in np.unique(index):
        medians = []
        for cid in member_cids:
            take = (index == bin_index) & (event_cids == cid)
            if np.count_nonzero(take) >= COACTIVE_MIN_SPIKES_PER_CID:
                medians.append(float(np.median(depth_um[take])))
        if len(medians) >= 2:
            spans.append(float(np.ptp(medians)))
    if not spans:
        return 0, np.nan, np.nan
    spans = np.asarray(spans)
    return len(spans), float(np.mean(spans >= SEPARATION_UM)), float(np.median(spans))


def classify_family(row: pd.Series) -> str:
    observed_fail = (
        (row["cross_cid_joined_intervals"] >= MIN_CROSS_CID_JOINS
         and row["rapid_jump_fraction_ge_100um"] > MAX_RAPID_JUMP_FRACTION)
        or (row["coactive_one_second_bins"] >= MIN_COACTIVE_BINS
            and row["separated_coactive_fraction_ge_80um"] > MAX_SEPARATED_COACTIVE_FRACTION)
        or row["merged_to_poisson_short_isi_ratio"] > MAX_REFRACTORY_RATIO
    )
    fully_supported_pass = (
        row["cross_cid_joined_intervals"] >= MIN_CROSS_CID_JOINS
        and row["rapid_jump_fraction_ge_100um"] <= MAX_RAPID_JUMP_FRACTION
        and row["coactive_one_second_bins"] >= MIN_COACTIVE_BINS
        and row["separated_coactive_fraction_ge_80um"] <= MAX_SEPARATED_COACTIVE_FRACTION
        and row["merged_to_poisson_short_isi_ratio"] <= MAX_REFRACTORY_RATIO
    )
    if observed_fail:
        return "implausible"
    if fully_supported_pass:
        return "depth_time_coherent"
    return "underpowered"


def auc_as_listed(values: np.ndarray, labels: np.ndarray) -> float:
    """ROC AUC where larger feature values predict label==1, with tie handling."""
    values = np.asarray(values, float)
    labels = np.asarray(labels, bool)
    ranks = rankdata(values)
    n1 = int(labels.sum())
    n0 = int((~labels).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    return float((ranks[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def permutation_auc_p(values: np.ndarray, labels: np.ndarray, rng: np.random.Generator) -> float:
    observed = abs(auc_as_listed(values, labels) - 0.5)
    exceed = 0
    for _ in range(PERMUTATIONS):
        permuted = rng.permutation(labels)
        exceed += abs(auc_as_listed(values, permuted) - 0.5) >= observed - 1e-12
    return float((exceed + 1) / (PERMUTATIONS + 1))


def waveform_predictors(families: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    aggregate = members.groupby("family_id").agg(
        median_local_energy_fraction=("local_energy_fraction", "median"),
        minimum_local_energy_fraction=("local_energy_fraction", "min"),
        minimum_member_spikes=("snippet_spikes", "min"),
        maximum_member_spikes=("snippet_spikes", "max"),
        mean_template_peak_abs=("template_peak_abs_ks_units", "mean"),
        std_template_peak_abs=("template_peak_abs_ks_units", "std"),
    ).reset_index()
    result = families.merge(aggregate, on="family_id", validate="one_to_one")
    result["kilosort_good_fraction"] = result["good_count"] / result["member_count"]
    result["member_spike_balance"] = result["minimum_member_spikes"] / result["maximum_member_spikes"]
    result["template_amplitude_cv"] = result["std_template_peak_abs"] / result["mean_template_peak_abs"]
    result["log10_total_snippet_spikes"] = np.log10(result["total_snippet_spikes"])
    result["minimum_internal_minus_external_cosine"] = (
        result["all_pairs_min_cosine"] - result["max_cosine_to_outside_template"]
    )
    return result


def predictor_associations(audit: pd.DataFrame) -> pd.DataFrame:
    features = [
        ("member_count", "smaller"),
        ("edge_density", "larger"),
        ("all_pairs_min_cosine", "larger"),
        ("all_pairs_median_cosine", "larger"),
        ("minimum_internal_minus_external_cosine", "larger"),
        ("complete_link_at_0p95", "true"),
        ("kilosort_good_fraction", "larger"),
        ("log10_total_snippet_spikes", "unknown"),
        ("member_spike_balance", "larger"),
        ("median_local_energy_fraction", "larger"),
        ("minimum_local_energy_fraction", "larger"),
        ("template_amplitude_cv", "smaller"),
    ]
    decided = audit[audit["depth_time_class"] != "underpowered"].copy()
    labels = decided["depth_time_class"].eq("depth_time_coherent").to_numpy()
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for feature, prior_expectation in features:
        values = decided[feature].astype(float).to_numpy()
        auc = auc_as_listed(values, labels)
        rows.append({
            "feature": feature,
            "n_coherent": int(labels.sum()),
            "n_implausible": int((~labels).sum()),
            "coherent_median": float(np.median(values[labels])),
            "implausible_median": float(np.median(values[~labels])),
            "auc_larger_predicts_coherent": auc,
            "separation_auc": max(auc, 1 - auc),
            "observed_favorable_direction": "larger" if auc >= 0.5 else "smaller",
            "prior_expected_direction": prior_expectation,
            "two_sided_label_permutation_p": permutation_auc_p(values, labels, rng),
        })
    result = pd.DataFrame(rows)
    order = np.argsort(result["two_sided_label_permutation_p"].to_numpy())
    ordered_p = result.loc[order, "two_sided_label_permutation_p"].to_numpy()
    adjusted = np.minimum.accumulate((ordered_p * len(ordered_p) / np.arange(1, len(ordered_p) + 1))[::-1])[::-1]
    result["benjamini_hochberg_q"] = np.nan
    result.loc[order, "benjamini_hochberg_q"] = np.minimum(adjusted, 1.0)
    return result.sort_values(
        ["separation_auc", "two_sided_label_permutation_p"], ascending=[False, True]
    )


def pearson_and_spearman(a: pd.Series, b: pd.Series) -> tuple[int, float, float]:
    both = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
    if len(both) < MIN_MATCHED_TRACK_BINS:
        return len(both), np.nan, np.nan
    pearson = correlation(both["a"].to_numpy(), both["b"].to_numpy())
    spearman = correlation(rankdata(both["a"]), rankdata(both["b"]))
    return len(both), pearson, spearman


def circular_shift_best_p(family: pd.Series, prior: dict[int, pd.Series], observed_max: float) -> float:
    """Library-wise p: shift every prior cell together and recompute the best match."""
    null_max = []
    bin_count = int(round((INTERVAL_S[1] - INTERVAL_S[0]) / TRACK_BIN_S))
    for shift in range(1, bin_count):
        shifted_library = {}
        for unit_id, series in prior.items():
            shifted = series.copy()
            shifted.index = (shifted.index + shift) % bin_count
            shifted_library[unit_id] = shifted
        correlations = [pearson_and_spearman(family, series)[1] for series in shifted_library.values()]
        finite = [value for value in correlations if np.isfinite(value)]
        if finite:
            null_max.append(max(finite))
    if not null_max or not np.isfinite(observed_max):
        return np.nan
    return float((1 + np.count_nonzero(np.asarray(null_max) >= observed_max)) / (1 + len(null_max)))


def circular_shift_series_p(family: pd.Series, reference: pd.Series, observed: float) -> float:
    """One-sided exact circular-shift p on the 20-bin grid."""
    bin_count = int(round((INTERVAL_S[1] - INTERVAL_S[0]) / TRACK_BIN_S))
    null = []
    for shift in range(1, bin_count):
        shifted = reference.copy()
        shifted.index = (shifted.index + shift) % bin_count
        value = pearson_and_spearman(family, shifted)[1]
        if np.isfinite(value):
            null.append(value)
    if not null or not np.isfinite(observed):
        return np.nan
    return float((1 + np.count_nonzero(np.asarray(null) >= observed)) / (1 + len(null)))


def classification_sensitivity(audit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for jump_max in (0.05, 0.10, 0.20):
        for coactive_max in (0.05, 0.10, 0.25):
            for refractory_max in (2.0, 3.0, 5.0):
                supported = (
                    (audit["cross_cid_joined_intervals"] >= MIN_CROSS_CID_JOINS)
                    & (audit["coactive_one_second_bins"] >= MIN_COACTIVE_BINS)
                )
                coherent = (
                    supported
                    & (audit["rapid_jump_fraction_ge_100um"] <= jump_max)
                    & (audit["separated_coactive_fraction_ge_80um"] <= coactive_max)
                    & (audit["merged_to_poisson_short_isi_ratio"] <= refractory_max)
                )
                ids = audit.loc[coherent, "family_id"].sort_values().tolist()
                rows.append({
                    "rapid_jump_fraction_max": jump_max,
                    "separated_coactive_fraction_max": coactive_max,
                    "merged_to_poisson_short_isi_ratio_max": refractory_max,
                    "coherent_family_count": len(ids),
                    "coherent_family_ids": " ".join(ids),
                })
    return pd.DataFrame(rows)


def make_tracks(
    coherent_ids: list[str], members: pd.DataFrame, time_s: np.ndarray,
    event_cids: np.ndarray, depth_um: np.ndarray, lighthouse: pd.DataFrame,
) -> tuple[dict[str, pd.Series], dict[int, pd.Series], pd.Series, pd.Series]:
    family_tracks = {}
    for family_id in coherent_ids:
        cids = members.loc[members["family_id"] == family_id, "cid"].to_numpy(int)
        take = np.isin(event_cids, cids)
        family_tracks[family_id] = seed_center(binned_median(time_s[take], depth_um[take], TRACK_BIN_S), TRACK_BIN_S)
    strict = lighthouse[lighthouse["evidence"] == "strict_accepted"]
    prior_tracks = {
        int(unit_id): seed_center(binned_median(group["time_s"], group["relative_um"], TRACK_BIN_S), TRACK_BIN_S)
        for unit_id, group in strict.groupby("unit_id")
    }
    wide = pd.concat(prior_tracks, axis=1)
    support = wide.notna().sum(axis=1)
    consensus = wide.median(axis=1, skipna=True)[support >= MIN_CONSENSUS_UNITS].sort_index()
    return family_tracks, prior_tracks, consensus, support.sort_index()


def compare_tracks(
    family_tracks: dict[str, pd.Series], prior_tracks: dict[int, pd.Series], consensus: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    best_rows = []
    for family_id, family in family_tracks.items():
        family_rows = []
        for unit_id, prior in prior_tracks.items():
            n, pearson, spearman = pearson_and_spearman(family, prior)
            family_rows.append({
                "family_id": family_id, "lighthouse_unit_id": unit_id,
                "matched_5s_bins": n, "pearson_r": pearson, "spearman_rho": spearman,
            })
        rows.extend(family_rows)
        valid = pd.DataFrame(family_rows).dropna(subset=["pearson_r"])
        n_consensus, r_consensus, rho_consensus = pearson_and_spearman(family, consensus)
        p_consensus = circular_shift_series_p(family, consensus, r_consensus)
        if len(valid):
            best = valid.sort_values("pearson_r", ascending=False).iloc[0]
            p_best = circular_shift_best_p(family, prior_tracks, float(best["pearson_r"]))
            best_rows.append({
                "family_id": family_id,
                "family_5s_bins": int(len(family)),
                "family_displacement_min_um": float(family.min()),
                "family_displacement_max_um": float(family.max()),
                "best_lighthouse_unit_id": int(best["lighthouse_unit_id"]),
                "best_matched_5s_bins": int(best["matched_5s_bins"]),
                "best_pearson_r": float(best["pearson_r"]),
                "best_spearman_rho": float(best["spearman_rho"]),
                "best_library_match_circular_shift_p": p_best,
                "consensus_matched_5s_bins": n_consensus,
                "consensus_pearson_r": r_consensus,
                "consensus_spearman_rho": rho_consensus,
                "consensus_circular_shift_p": p_consensus,
            })
    return pd.DataFrame(rows), pd.DataFrame(best_rows)


def save_figures(
    audit: pd.DataFrame, associations: pd.DataFrame, comparisons: pd.DataFrame,
    best: pd.DataFrame, family_tracks: dict[str, pd.Series], prior_tracks: dict[int, pd.Series],
    consensus: pd.Series,
) -> None:
    colors = {"depth_time_coherent": "#2E8B57", "implausible": "#C94C4C", "underpowered": "#8A8A8A"}
    fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
    for status, group in audit.groupby("depth_time_class"):
        ax.scatter(group["rapid_jump_fraction_ge_100um"], group["separated_coactive_fraction_ge_80um"],
                   s=35 + 12 * group["member_count"], color=colors[status], alpha=0.8, label=status.replace("_", " "))
        for row in group.itertuples(index=False):
            ax.annotate(row.family_id, (row.rapid_jump_fraction_ge_100um, row.separated_coactive_fraction_ge_80um),
                        xytext=(3, 2), textcoords="offset points", fontsize=7)
    ax.axvline(MAX_RAPID_JUMP_FRACTION, color="0.3", ls="--", lw=1)
    ax.axhline(MAX_SEPARATED_COACTIVE_FRACTION, color="0.3", ls="--", lw=1)
    ax.set(xlabel=f"Fraction of cross-CID joins with ≥{RAPID_JUMP_UM:g} µm jump",
           ylabel=f"Fraction of coactive 1-s bins separated by ≥{SEPARATION_UM:g} µm",
           title="Depth/time plausibility revealed after waveform-only family construction")
    ax.grid(alpha=0.2); ax.legend(frameon=False)
    fig.savefig(OUT / "01_depth_time_family_assessment.png", dpi=180)
    fig.savefig(OUT / "01_depth_time_family_assessment.pdf")
    plt.close(fig)

    shown = associations.sort_values("separation_auc").copy()
    labels = [value.replace("_", " ") for value in shown["feature"]]
    bar_colors = ["#2E8B57" if direction == "larger" else "#6B6ECF" for direction in shown["observed_favorable_direction"]]
    fig, ax = plt.subplots(figsize=(10, 6.5), layout="constrained")
    ax.barh(labels, shown["separation_auc"], left=0, color=bar_colors, alpha=0.85)
    ax.axvline(0.5, color="0.25", lw=1)
    ax.set(xlim=(0.48, 1.02), xlabel="Univariate separation AUC (direction chosen only for display)",
           title="What pre-reveal properties distinguish depth/time-coherent families?\n5 coherent versus decided implausible families; exploratory, not a trained classifier")
    for y, row in enumerate(shown.itertuples(index=False)):
        ax.text(row.separation_auc + 0.008, y, f"{row.observed_favorable_direction}; p={row.two_sided_label_permutation_p:.3g}", va="center", fontsize=8)
    ax.grid(axis="x", alpha=0.2)
    fig.savefig(OUT / "02_waveform_only_predictor_associations.png", dpi=180)
    fig.savefig(OUT / "02_waveform_only_predictor_associations.pdf")
    plt.close(fig)

    coherent = list(family_tracks)
    matrix = comparisons.pivot(index="family_id", columns="lighthouse_unit_id", values="pearson_r").reindex(coherent)
    fig, ax = plt.subplots(figsize=(12, max(3.5, 0.65 * len(coherent))), layout="constrained")
    image = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=90)
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    ax.set(xlabel="Previous lighthouse unit", ylabel="Depth/time-coherent Kilosort family",
           title=f"5-s displacement correlation on matched support (strict lighthouse evidence; ≥{MIN_MATCHED_TRACK_BINS} bins)")
    fig.colorbar(image, ax=ax, label="Pearson r")
    fig.savefig(OUT / "03_family_by_lighthouse_correlation.png", dpi=180)
    fig.savefig(OUT / "03_family_by_lighthouse_correlation.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(len(coherent), 1, figsize=(13, 3.0 * len(coherent)), sharex=True, layout="constrained", squeeze=False)
    centers = INTERVAL_S[0] + (np.arange(20) + 0.5) * TRACK_BIN_S
    full_bins = np.arange(20)
    for ax, family_id in zip(axes[:, 0], coherent):
        family = family_tracks[family_id]
        row = best.set_index("family_id").loc[family_id]
        best_unit = int(row["best_lighthouse_unit_id"])
        prior = prior_tracks[best_unit]
        # Reindex to the complete grid so unsupported intervals break lines.
        family_plot = family.reindex(full_bins)
        consensus_plot = consensus.reindex(full_bins)
        prior_plot = prior.reindex(full_bins)
        ax.plot(centers, family_plot, "o-", color="#222222", lw=1.7, ms=4, label=f"{family_id} cached Kilosort depth")
        ax.plot(centers, consensus_plot, "o-", color="#2E78B7", lw=1.2, ms=3, alpha=0.75, label="median of prior lighthouse cells")
        ax.plot(centers, prior_plot, "o--", color="#E07A2D", lw=1.0, ms=3, alpha=0.8, label=f"best prior cell {best_unit}")
        ax.axhline(0, color="0.5", lw=0.7)
        ax.set(ylabel="seed-relative depth (µm)", title=(
            f"{family_id}: best cell r={row.best_pearson_r:.2f} ({int(row.best_matched_5s_bins)} bins, library-shift p={row.best_library_match_circular_shift_p:.3g}); "
            f"prior-cell median r={row.consensus_pearson_r:.2f} (shift p={row.consensus_circular_shift_p:.3g})"
        ))
        ax.grid(alpha=0.2); ax.legend(frameon=False, ncol=3, fontsize=8)
    axes[-1, 0].set_xlabel("recording time (s)")
    fig.suptitle("Independent post-selection check against previous lighthouse displacement\nNo lag, sign, gain, or cell identity fitted", fontsize=13)
    fig.savefig(OUT / "04_coherent_family_motion_comparison.png", dpi=180)
    fig.savefig(OUT / "04_coherent_family_motion_comparison.pdf")
    plt.close(fig)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    family = pd.read_csv(FAMILY_OUT / "family_summary_depth_time_revealed.csv")
    joined = pd.read_csv(FAMILY_OUT / "family_joined_index.csv")
    waveform_members = pd.read_csv(FAMILY_OUT / "family_members_waveform_only.csv")
    members = pd.read_csv(FAMILY_OUT / "family_members_depth_time_revealed.csv")
    lighthouse = pd.read_csv(LIGHTHOUSE)
    event_summary = json.loads(EVENT_SUMMARY.read_text())
    event_file = np.load(EVENTS)
    fs = float(event_summary["sampling_frequency_hz"])
    time_s = event_file["frames"].astype(float) / fs
    event_cids = event_file["cid"].astype(int)
    depth_um = event_file["xy_um"][:, 1].astype(float)

    pre = waveform_predictors(family, waveform_members)
    audit = pre.merge(joined[[
        "family_id", "cross_cid_joined_intervals", "cross_cid_depth_jump_median_um",
        "cross_cid_depth_jump_p95_um", "cross_cid_depth_jumps_ge_40_um", "cross_cid_depth_jumps_ge_100_um",
    ]], on="family_id", validate="one_to_one")
    audit["rapid_jump_fraction_ge_100um"] = (
        audit["cross_cid_depth_jumps_ge_100_um"] / audit["cross_cid_joined_intervals"].replace(0, np.nan)
    )
    coactive_rows = []
    for family_id, group in members.groupby("family_id"):
        cids = group["cid"].to_numpy(int)
        take = np.isin(event_cids, cids)
        count, fraction, median_span = coactive_depth_separation(
            time_s[take], depth_um[take], event_cids[take], cids
        )
        coactive_rows.append({
            "family_id": family_id, "coactive_one_second_bins": count,
            "separated_coactive_fraction_ge_80um": fraction,
            "coactive_median_between_cid_span_um": median_span,
        })
    audit = audit.merge(pd.DataFrame(coactive_rows), on="family_id", validate="one_to_one")
    audit["depth_time_class"] = audit.apply(classify_family, axis=1)
    audit.to_csv(OUT / "family_plausibility_audit.csv", index=False)
    sensitivity = classification_sensitivity(audit)
    sensitivity.to_csv(OUT / "classification_threshold_sensitivity.csv", index=False)

    associations = predictor_associations(audit)
    associations.to_csv(OUT / "waveform_only_predictor_associations.csv", index=False)

    coherent_ids = audit.loc[audit["depth_time_class"] == "depth_time_coherent", "family_id"].sort_values().tolist()
    family_tracks, prior_tracks, consensus, consensus_support = make_tracks(
        coherent_ids, members, time_s, event_cids, depth_um, lighthouse
    )
    comparisons, best = compare_tracks(family_tracks, prior_tracks, consensus)
    comparisons.to_csv(OUT / "coherent_family_by_lighthouse_comparisons.csv", index=False)
    best.to_csv(OUT / "coherent_family_motion_summary.csv", index=False)
    track_rows = []
    for family_id, series in family_tracks.items():
        for bin_index, value in series.items():
            track_rows.append({"track_type": "kilosort_family", "track_id": family_id, "bin": bin_index,
                               "time_s": INTERVAL_S[0] + (bin_index + 0.5) * TRACK_BIN_S, "relative_depth_um": value})
    for unit_id, series in prior_tracks.items():
        for bin_index, value in series.items():
            track_rows.append({"track_type": "previous_lighthouse_strict", "track_id": str(unit_id), "bin": bin_index,
                               "time_s": INTERVAL_S[0] + (bin_index + 0.5) * TRACK_BIN_S, "relative_depth_um": value})
    for bin_index, value in consensus.items():
        track_rows.append({"track_type": "previous_lighthouse_median", "track_id": "median", "bin": bin_index,
                           "time_s": INTERVAL_S[0] + (bin_index + 0.5) * TRACK_BIN_S, "relative_depth_um": value,
                           "contributing_lighthouse_units": int(consensus_support.loc[bin_index])})
    pd.DataFrame(track_rows).sort_values(["track_type", "track_id", "bin"]).to_csv(
        OUT / "binned_displacement_tracks.csv", index=False
    )

    save_figures(audit, associations, comparisons, best, family_tracks, prior_tracks, consensus)

    summary = {
        "status": "complete",
        "purpose": "post-hoc plausibility predictors and comparison with previous lighthouse displacement",
        "raw_voltage_read": False,
        "new_spike_sort_run": False,
        "family_construction_refit": False,
        "family_count": int(len(audit)),
        "depth_time_class_counts": audit["depth_time_class"].value_counts().to_dict(),
        "depth_time_coherent_family_ids": coherent_ids,
        "threshold_sensitivity_unique_family_sets": int(sensitivity["coherent_family_ids"].nunique()),
        "classification_rule": {
            "cross_cid_joins_min": MIN_CROSS_CID_JOINS,
            "rapid_jump_um": RAPID_JUMP_UM,
            "rapid_jump_fraction_max": MAX_RAPID_JUMP_FRACTION,
            "coactive_bin_s": COACTIVE_BIN_S,
            "coactive_bins_min": MIN_COACTIVE_BINS,
            "coactive_separation_um": SEPARATION_UM,
            "separated_coactive_fraction_max": MAX_SEPARATED_COACTIVE_FRACTION,
            "merged_to_poisson_short_isi_ratio_max": MAX_REFRACTORY_RATIO,
        },
        "motion_comparison": {
            "bin_s": TRACK_BIN_S,
            "lighthouse_evidence": "strict_accepted only",
            "seed_center_interval_s": [930.0, 940.0],
            "minimum_matched_bins": MIN_MATCHED_TRACK_BINS,
            "no_fitted_transform": "no lag, sign, gain, or cell selection used to form families or label coherence",
            "best_match_multiple_comparison_control": "19 nonzero circular shifts of the entire prior-cell library",
        },
        "limitations": [
            "Depth/time coherence is a screen, not biological identity validation or a motion estimate.",
            "The cutoffs are explicit but chosen for this pilot; family labels require threshold sensitivity and visual review.",
            "Only 24 families are available, so predictor associations are univariate and exploratory.",
            "The prior 17 lighthouse candidates are not fully independent and strict observations are sparse for many cells.",
            "Kilosort spike_positions are sorter-derived PC-weighted depths, not independent raw-voltage localizations.",
        ],
        "source_sha256": {str(path): sha256(path) for path in [
            FAMILY_OUT / "family_summary_depth_time_revealed.csv", FAMILY_OUT / "family_joined_index.csv",
            FAMILY_OUT / "family_members_waveform_only.csv", FAMILY_OUT / "family_members_depth_time_revealed.csv",
            EVENTS, LIGHTHOUSE,
        ]},
    }
    summary["script_sha256"] = sha256(Path(__file__))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
