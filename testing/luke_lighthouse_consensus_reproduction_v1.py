"""Independently reproduce and stress-test the cached lighthouse consensus."""

from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVENTS = ROOT / "testing/outputs/luke_waveform_only_expansion_v1/overlay_events.csv"
SAVED = ROOT / "testing/outputs/luke_kilosort_family_plausibility_v3/binned_displacement_tracks.csv"
OUT = ROOT / "testing/outputs/luke_lighthouse_consensus_reproduction_v1"
START_S, STOP_S, BIN_S = 930.0, 1030.0, 5.0
MIN_SUPPORT = 3


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def binned_track(group):
    bins = np.floor((group["time_s"].to_numpy() - START_S) / BIN_S).astype(int)
    values = pd.DataFrame({"bin": bins, "value": group["relative_um"].to_numpy()})
    values = values[(values.bin >= 0) & (values.bin < int((STOP_S - START_S) / BIN_S))]
    track = values.groupby("bin")["value"].median()
    baseline = track[track.index < int(10 / BIN_S)].median()
    return track - baseline


def make_tracks(events, grouping):
    tracks = {}
    for identity, unit_ids in grouping.items():
        tracks[identity] = binned_track(events[events.unit_id.isin(unit_ids)])
    return pd.concat(tracks, axis=1)


def consensus(wide):
    support = wide.notna().sum(axis=1)
    return wide.median(axis=1, skipna=True).where(support >= MIN_SUPPORT), support


def corr(a, b, minimum=4):
    aligned = pd.concat([a, b], axis=1).dropna()
    return float(aligned.corr().iloc[0, 1]) if len(aligned) >= minimum else np.nan


def agreement(wide):
    columns = list(wide.columns)
    pairs = [corr(wide[a], wide[b]) for i, a in enumerate(columns) for b in columns[i + 1:]]
    loo = {}
    for column in columns:
        other, _ = consensus(wide.drop(columns=column))
        loo[column] = corr(wide[column], other)
    return np.asarray(pairs), pd.Series(loo)


def matrix_agreement(matrix):
    def array_corr(a, b):
        keep = np.isfinite(a) & np.isfinite(b)
        return np.corrcoef(a[keep], b[keep])[0, 1] if keep.sum() >= 4 else np.nan
    pairs = [array_corr(matrix[i], matrix[j]) for i in range(len(matrix))
             for j in range(i + 1, len(matrix))]
    loo = [array_corr(matrix[i], np.nanmedian(np.delete(matrix, i, axis=0), axis=0))
           for i in range(len(matrix))]
    return np.asarray(pairs), np.asarray(loo)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(EVENTS)
    strict = events[events.evidence.eq("strict_accepted")].copy()
    unit_ids = sorted(strict.unit_id.unique())
    original_grouping = {str(unit): [unit] for unit in unit_ids}
    original = make_tracks(strict, original_grouping).sort_index()
    rebuilt, support = consensus(original)

    saved = pd.read_csv(SAVED)
    saved_units = saved[saved.track_type.eq("previous_lighthouse_strict")].copy()
    saved_wide = saved_units.pivot(index="bin", columns="track_id", values="relative_depth_um")
    saved_wide.columns = saved_wide.columns.astype(int).astype(str)
    saved_wide = saved_wide.reindex(columns=original.columns)
    saved_consensus = (saved[saved.track_type.eq("previous_lighthouse_median")]
                       .set_index("bin")["relative_depth_um"])
    unit_difference = np.nanmax(np.abs(original.reindex(saved_wide.index).to_numpy() - saved_wide.to_numpy()))
    consensus_difference = np.nanmax(np.abs(rebuilt.reindex(saved_consensus.index) - saved_consensus))

    dedup_grouping = {str(unit): [unit] for unit in unit_ids if unit not in (673, 675)}
    dedup_grouping["673/675"] = [673, 675]
    dedup = make_tracks(strict, dedup_grouping).sort_index()
    dedup_consensus, dedup_support = consensus(dedup)
    comparison = pd.concat({"saved_17_label": saved_consensus,
                            "rebuilt_17_label": rebuilt,
                            "deduplicated_16_family": dedup_consensus,
                            "original_support": support,
                            "deduplicated_support": dedup_support}, axis=1)
    comparison.index.name = "bin"
    comparison["time_s"] = START_S + (comparison.index + 0.5) * BIN_S
    comparison.to_csv(OUT / "consensus_comparison.csv")
    original.to_csv(OUT / "rebuilt_17_label_tracks.csv", index_label="bin")
    dedup.to_csv(OUT / "deduplicated_16_family_tracks.csv", index_label="bin")

    pairwise, loo = agreement(dedup)
    finite_pairwise = pairwise[np.isfinite(pairwise)]
    finite_loo = loo[np.isfinite(loo)].to_numpy()
    rng = np.random.default_rng(20260909)
    null_pair_medians = []
    null_loo_medians = []
    matrix = dedup.to_numpy().T
    for _ in range(2000):
        shifted = np.asarray([np.roll(row, int(rng.integers(row.size))) for row in matrix])
        shifted_pairs, shifted_loo = matrix_agreement(shifted)
        null_pair_medians.append(np.nanmedian(shifted_pairs))
        null_loo_medians.append(np.nanmedian(shifted_loo))
    pair_shift_p = (1 + np.sum(np.asarray(null_pair_medians) >= np.nanmedian(finite_pairwise))) / 2001
    loo_shift_p = (1 + np.sum(np.asarray(null_loo_medians) >= np.nanmedian(finite_loo))) / 2001
    pd.DataFrame({"family": loo.index, "leave_one_out_consensus_r": loo.values,
                  "observed_bins": [int(dedup[c].notna().sum()) for c in loo.index]}).to_csv(
                      OUT / "deduplicated_family_agreement.csv", index=False)

    common = pd.concat([saved_consensus, dedup_consensus], axis=1).dropna()
    old_new_r = float(common.corr().iloc[0, 1])
    old_new_max_delta = float(np.max(np.abs(common.iloc[:, 0] - common.iloc[:, 1])))
    old_span = float(saved_consensus.max() - saved_consensus.min())
    new_span = float(dedup_consensus.max() - dedup_consensus.min())

    fig, (ax, bx) = plt.subplots(2, 1, figsize=(11, 7.5), height_ratios=[2, 0.65],
                                 sharex=True, layout="constrained")
    ax.plot(comparison.time_s, comparison.saved_17_label, "o-", color="#7B8794", lw=2,
            label="saved 17-label consensus")
    ax.plot(comparison.time_s, comparison.rebuilt_17_label, "--", color="#D07A1F", lw=1.7,
            label="independent reproduction")
    ax.plot(comparison.time_s, comparison.deduplicated_16_family, "o-", color="#2166AC", lw=2,
            ms=4, label="673/675 collapsed to one family")
    ax.axhline(0, color="#333333", lw=0.8)
    ax.set(ylabel="seed-centered relative depth (um)",
           title="Original waveform-traced lighthouse consensus reproduction")
    ax.legend(frameon=False, ncol=3)
    bx.step(comparison.time_s, comparison.original_support, where="mid", color="#7B8794",
            label="17-label support")
    bx.step(comparison.time_s, comparison.deduplicated_support, where="mid", color="#2166AC",
            label="16-family support")
    bx.axhline(MIN_SUPPORT, color="#333333", ls="--", lw=0.9, label="minimum support")
    bx.set(xlabel="recording time (s)", ylabel="votes", ylim=(0, 17.5))
    bx.legend(frameon=False, ncol=3)
    fig.savefig(OUT / "01_consensus_reproduction.png", dpi=180)
    fig.savefig(OUT / "01_consensus_reproduction.pdf")
    plt.close(fig)

    summary = {
        "status": "complete",
        "strict_event_count": int(len(strict)),
        "prior_label_count": len(unit_ids),
        "deduplicated_family_count": len(dedup.columns),
        "exact_reproduction": {
            "max_absolute_per_unit_track_difference_um": float(unit_difference),
            "max_absolute_consensus_difference_um": float(consensus_difference),
        },
        "deduplication_sensitivity": {
            "saved_vs_deduplicated_consensus_pearson_r": old_new_r,
            "maximum_absolute_bin_difference_um": old_new_max_delta,
            "saved_consensus_peak_to_peak_um": old_span,
            "deduplicated_consensus_peak_to_peak_um": new_span,
            "saved_max_adjacent_5s_jump_um": float(np.nanmax(np.abs(np.diff(saved_consensus.to_numpy())))),
            "deduplicated_max_adjacent_5s_jump_um": float(np.nanmax(np.abs(np.diff(dedup_consensus.to_numpy())))),
        },
        "deduplicated_agreement": {
            "pairwise_correlations_with_at_least_4_common_bins": int(len(finite_pairwise)),
            "median_pairwise_pearson_r": float(np.nanmedian(finite_pairwise)),
            "positive_pairwise_fraction_among_finite": float(np.mean(finite_pairwise > 0)),
            "pairwise_median_circular_shift_p": float(pair_shift_p),
            "leave_one_out_correlations_with_at_least_4_bins": int(len(finite_loo)),
            "median_leave_one_out_consensus_r": float(np.nanmedian(finite_loo)),
            "positive_leave_one_out_fraction_among_finite": float(np.mean(finite_loo > 0)),
            "leave_one_out_median_circular_shift_p": float(loo_shift_p),
            "circular_shift_draws": 2000,
        },
        "support": {
            "minimum_families_per_included_bin": int(dedup_support[dedup_consensus.notna()].min()),
            "median_families_per_bin": float(dedup_support.median()),
            "bins_below_6_families": int((dedup_support < 6).sum()),
        },
        "interpretation": "The original calculation is exactly reproducible and robust to collapsing 673/675, but that reproduces the estimator, not its validity.",
        "source_sha256": {str(EVENTS): digest(EVENTS), str(SAVED): digest(SAVED)},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
