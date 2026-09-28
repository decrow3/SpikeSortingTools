"""Compare frozen LFP and AP motion candidates with lighthouse observations.

This is an independent validation pass, not a motion estimator. Lighthouse
identities, family assignments, plausibility exclusions, and lattice rules are
frozen before any candidate field is sampled. Candidate values are evaluated
at accepted events' actual times and at each family's frozen reference depth.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "testing/outputs/luke_motion_candidate_lighthouse_comparison_v1"
DOC = ROOT / "docs/luke_motion_candidate_lighthouse_comparison_20260910.md"
EARLY = ROOT / "testing/outputs/luke_waveform_only_expansion_v1/overlay_events.csv"
EXTENDED = ROOT / "testing/outputs/luke_lighthouse_extension_300s_v1/new_event_predictions.csv"
FAMILY_MAP = ROOT / "testing/outputs/luke_lighthouse_method_audit_v1/d_candidate_families.csv"
LFP_PACKAGE = ROOT / "testing/outputs/luke_lfp_lighthouse_validation_v1/lfp_package"
AP_PACKAGE = Path(
    "/mnt/NPX/Luke/20250804/shared_analysis/"
    "luke_improved_motion_two_machine_20260909_v1/estimation_huklaban1_v1"
)
AP_FIELD = AP_PACKAGE / "candidate_fields.npz"
AP_MANIFEST = AP_PACKAGE / "manifest.json"

WINDOWS = {"930_1030": (930.0, 1030.0), "1150_1200": (1150.0, 1200.0)}
LFP_ARM = "band_0p5_8"
BIN_S = 5.0
QUIET_THRESHOLD_UM = 20.0
LATTICE_UM = 40.0
NODE_TOL_UM = 5.0
MIN_ALL_INCREMENTS = 3
PREDECLARED_CONCERN_UNITS = {125, 161, 353, 557, 698, 705}
SENSITIVITIES = ["all_strict", "plausibility_restricted", "plausibility_lattice_node_5um"]
PRIMARY_SENSITIVITY = "plausibility_lattice_node_5um"
CANDIDATES = ["zero", "lfp_rigid", "ap_rigid", "ap_nonrigid"]
LABELS = {
    "zero": "No correction",
    "lfp_rigid": "LFP rigid",
    "ap_rigid": "AP rigid",
    "ap_nonrigid": "AP nonrigid",
}
COLORS = {
    "zero": "#6B7280",
    "lfp_rigid": "#84A80B",
    "ap_rigid": "#D97706",
    "ap_nonrigid": "#2563EB",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() < 3 or np.std(x[good]) == 0 or np.std(y[good]) == 0:
        return np.nan
    return float(np.corrcoef(x[good], y[good])[0, 1])


def slope_through_origin(x: np.ndarray, y: np.ndarray) -> float:
    good = np.isfinite(x) & np.isfinite(y)
    denominator = float(np.dot(x[good], x[good]))
    return float(np.dot(x[good], y[good]) / denominator) if good.any() and denominator > 0 else np.nan


def load_events() -> pd.DataFrame:
    early = pd.read_csv(EARLY)
    extended = pd.read_csv(EXTENDED).drop(columns=["prediction_um", "difference_um", "variant"])
    events = pd.concat([early, extended], ignore_index=True)
    family = pd.read_csv(FAMILY_MAP).rename(columns={"family": "family_id"})
    events = events.merge(
        family[["unit_id", "family_id", "seed_centroid_um"]],
        on="unit_id",
        how="left",
        validate="many_to_one",
    )
    if events[["family_id", "seed_centroid_um"]].isna().any().any():
        raise RuntimeError("Frozen family assignment or reference depth is missing")
    events["family_id"] = events.family_id.astype(int)
    events["lattice_phase_um"] = np.abs(
        ((events.relative_um + LATTICE_UM / 2) % LATTICE_UM) - LATTICE_UM / 2
    )
    return events


def apply_sensitivity(events: pd.DataFrame, sensitivity: str) -> pd.DataFrame:
    if sensitivity == "all_strict":
        return events.copy()
    if sensitivity == "plausibility_restricted":
        return events.loc[~events.unit_id.isin(PREDECLARED_CONCERN_UNITS)].copy()
    if sensitivity == "plausibility_lattice_node_5um":
        return events.loc[
            ~events.unit_id.isin(PREDECLARED_CONCERN_UNITS)
            & events.lattice_phase_um.le(NODE_TOL_UM)
        ].copy()
    raise ValueError(sensitivity)


def nearest_supported_lfp(events: pd.DataFrame, window: str) -> pd.DataFrame:
    z = np.load(LFP_PACKAGE / f"{window}_{LFP_ARM}_motion.npz")
    time_s = z["time_s"].astype(float)
    values = z["displacement_um"].astype(float)
    supported = z["supported"].astype(bool) & ~z["invalid"].astype(bool) & np.isfinite(values)
    query = events.time_s.to_numpy(float)
    right = np.clip(np.searchsorted(time_s, query, side="left"), 0, len(time_s) - 1)
    left = np.clip(right - 1, 0, len(time_s) - 1)
    choose_right = np.abs(time_s[right] - query) < np.abs(time_s[left] - query)
    index = np.where(choose_right, right, left)
    time_error = np.abs(time_s[index] - query)
    valid = supported[index] & (time_error <= 0.01)
    out = events.copy()
    out["lfp_rigid_um"] = np.where(valid, values[index], np.nan)
    out["lfp_supported"] = valid
    out["lfp_time_error_s"] = time_error
    return out


def sample_ap(events: pd.DataFrame, ap: np.lib.npyio.NpzFile) -> pd.DataFrame:
    time_s = ap["time_s"].astype(float)
    depth_um = ap["depth_um"].astype(float)
    nonrigid = ap["nonrigid_displacement_um"].astype(float)
    rigid = ap["rigid_displacement_um"].astype(float)
    query_t = events.time_s.to_numpy(float)
    query_d = events.seed_centroid_um.to_numpy(float)
    if query_d.min() < depth_um.min() or query_d.max() > depth_um.max():
        raise RuntimeError("A lighthouse reference depth falls outside the AP field")

    right_t = np.clip(np.searchsorted(time_s, query_t, side="left"), 1, len(time_s) - 1)
    left_t = right_t - 1
    alpha_t = (query_t - time_s[left_t]) / (time_s[right_t] - time_s[left_t])
    at_time = nonrigid[left_t] * (1 - alpha_t[:, None]) + nonrigid[right_t] * alpha_t[:, None]

    right_d = np.clip(np.searchsorted(depth_um, query_d, side="left"), 1, len(depth_um) - 1)
    left_d = right_d - 1
    alpha_d = (query_d - depth_um[left_d]) / (depth_um[right_d] - depth_um[left_d])
    row = np.arange(len(events))

    out = events.copy()
    out["ap_rigid_um"] = np.interp(query_t, time_s, rigid, left=np.nan, right=np.nan)
    out["ap_nonrigid_um"] = (
        at_time[row, left_d] * (1 - alpha_d) + at_time[row, right_d] * alpha_d
    )
    out["zero_um"] = 0.0
    return out


def bin_events(events: pd.DataFrame, start: float, stop: float) -> pd.DataFrame:
    strict = events.loc[
        events.evidence.eq("strict_accepted") & events.time_s.ge(start) & events.time_s.lt(stop)
    ].copy()
    strict["bin"] = np.floor((strict.time_s - start) / BIN_S).astype(int)
    binned = (
        strict.groupby(["family_id", "bin"], as_index=False)
        .agg(
            time_s=("time_s", "median"),
            lighthouse_um=("relative_um", "median"),
            reference_depth_um=("seed_centroid_um", "median"),
            event_count=("relative_um", "size"),
            unit_count=("unit_id", "nunique"),
            temporal_span_s=("time_s", lambda x: float(x.max() - x.min())),
            lattice_phase_um=("lattice_phase_um", "median"),
            lfp_supported_fraction=("lfp_supported", "mean"),
            zero_um=("zero_um", "median"),
            lfp_rigid_um=("lfp_rigid_um", "median"),
            ap_rigid_um=("ap_rigid_um", "median"),
            ap_nonrigid_um=("ap_nonrigid_um", "median"),
        )
    )
    binned["bin_time_s"] = start + (binned.bin + 0.5) * BIN_S
    binned["common_support"] = binned[[f"{name}_um" for name in CANDIDATES]].notna().all(axis=1)
    return binned


def make_increments(levels: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for family_id, group in levels.groupby("family_id"):
        group = group.sort_values("bin").reset_index(drop=True)
        previous = group.shift(1)
        valid = (
            group.common_support
            & previous.common_support.eq(True)
            & group.bin.sub(previous.bin).eq(1)
        )
        for index in np.flatnonzero(valid.to_numpy()):
            lighthouse_delta = float(group.loc[index, "lighthouse_um"] - previous.loc[index, "lighthouse_um"])
            regime = "quiet" if abs(lighthouse_delta) < QUIET_THRESHOLD_UM else "movement"
            base = {
                "family_id": int(family_id),
                "bin_from": int(previous.loc[index, "bin"]),
                "bin_to": int(group.loc[index, "bin"]),
                "time_s": float(group.loc[index, "bin_time_s"]),
                "reference_depth_um": float(group.loc[index, "reference_depth_um"]),
                "lighthouse_delta_um": lighthouse_delta,
                "regime": regime,
                "event_count_from": int(previous.loc[index, "event_count"]),
                "event_count_to": int(group.loc[index, "event_count"]),
            }
            for candidate in CANDIDATES:
                predicted = float(group.loc[index, f"{candidate}_um"] - previous.loc[index, f"{candidate}_um"])
                rows.append(
                    {
                        **base,
                        "candidate": candidate,
                        "candidate_delta_um": predicted,
                        "residual_um": lighthouse_delta - predicted,
                    }
                )
    return pd.DataFrame(rows)


def per_family_metrics(increments: pd.DataFrame) -> pd.DataFrame:
    rows = []
    regimes = {
        "all": increments,
        "quiet": increments.loc[increments.regime.eq("quiet")],
        "movement": increments.loc[increments.regime.eq("movement")],
    }
    for regime, subset in regimes.items():
        for (candidate, family_id), group in subset.groupby(["candidate", "family_id"]):
            x = group.lighthouse_delta_um.to_numpy(float)
            y = group.candidate_delta_um.to_numpy(float)
            residual = x - y
            rows.append(
                {
                    "regime": regime,
                    "candidate": candidate,
                    "family_id": int(family_id),
                    "n_increments": int(len(group)),
                    "rmse_um": float(np.sqrt(np.mean(residual**2))),
                    "mae_um": float(np.mean(np.abs(residual))),
                    "signed_residual_um": float(np.mean(residual)),
                    "correlation": correlation(x, y),
                    "slope": slope_through_origin(x, y),
                }
            )
    return pd.DataFrame(rows)


def summarize(family: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for (regime, candidate), group in family.groupby(["regime", "candidate"]):
        minimum = MIN_ALL_INCREMENTS if regime == "all" else 1
        eligible = group.loc[group.n_increments.ge(minimum)].copy()
        zero = family.loc[
            family.regime.eq(regime)
            & family.candidate.eq("zero")
            & family.family_id.isin(eligible.family_id)
            & family.n_increments.ge(minimum)
        ].set_index("family_id")
        eligible = eligible.loc[eligible.family_id.isin(zero.index)].copy()
        zero = zero.loc[eligible.family_id]
        if eligible.empty:
            continue
        rmse = float(np.sqrt(np.mean(eligible.rmse_um.to_numpy() ** 2)))
        zero_rmse = float(np.sqrt(np.mean(zero.rmse_um.to_numpy() ** 2)))
        skill = float(1 - rmse / zero_rmse) if zero_rmse > 0 else np.nan
        draws = []
        for _ in range(5000):
            index = rng.integers(0, len(eligible), len(eligible))
            candidate_draw = float(np.sqrt(np.mean(eligible.rmse_um.to_numpy()[index] ** 2)))
            zero_draw = float(np.sqrt(np.mean(zero.rmse_um.to_numpy()[index] ** 2)))
            draws.append(1 - candidate_draw / zero_draw if zero_draw > 0 else np.nan)
        rows.append(
            {
                "regime": regime,
                "candidate": candidate,
                "families": int(len(eligible)),
                "increments": int(eligible.n_increments.sum()),
                "family_balanced_rmse_um": rmse,
                "family_balanced_mae_um": float(eligible.mae_um.mean()),
                "mean_family_signed_residual_um": float(eligible.signed_residual_um.mean()),
                "median_family_correlation": float(eligible.correlation.median()),
                "median_family_slope": float(eligible.slope.median()),
                "zero_rmse_um": zero_rmse,
                "skill_vs_zero": skill,
                "skill_ci_low": float(np.nanquantile(draws, 0.025)),
                "skill_ci_high": float(np.nanquantile(draws, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def support_summary(events: pd.DataFrame, levels: pd.DataFrame, window: str, sensitivity: str) -> dict:
    within = events.loc[events.time_s.ge(WINDOWS[window][0]) & events.time_s.lt(WINDOWS[window][1])]
    strict = within.loc[within.evidence.eq("strict_accepted")]
    return {
        "window": window,
        "sensitivity": sensitivity,
        "events_all_evidence": int(len(within)),
        "strict_events": int(len(strict)),
        "lower_score_events": int(within.evidence.eq("lower_score").sum()),
        "identity_ambiguous_events": int(within.evidence.eq("identity_ambiguous").sum()),
        "strict_families": int(strict.family_id.nunique()),
        "occupied_family_bins": int(len(levels)),
        "common_support_family_bins": int(levels.common_support.sum()),
        "lfp_event_supported_fraction": float(strict.lfp_supported.mean()) if len(strict) else np.nan,
        "median_lattice_phase_um": float(strict.lattice_phase_um.median()) if len(strict) else np.nan,
    }


def make_figures(summary: pd.DataFrame, levels: pd.DataFrame) -> None:
    primary = summary.loc[summary.sensitivity.eq(PRIMARY_SENSITIVITY)].copy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), constrained_layout=True, sharey=True)
    order = ["zero", "lfp_rigid", "ap_rigid", "ap_nonrigid"]
    hatches = {"zero": "//", "lfp_rigid": "..", "ap_rigid": "xx", "ap_nonrigid": ""}
    for ax, regime in zip(axes, ["all", "quiet", "movement"]):
        subset = primary.loc[primary.regime.eq(regime)]
        x = np.arange(len(WINDOWS))
        width = 0.19
        offsets = (np.arange(len(order)) - (len(order) - 1) / 2) * width
        any_finite = False
        for offset, candidate in zip(offsets, order):
            values = []
            for window in WINDOWS:
                row = subset.loc[subset.window.eq(window) & subset.candidate.eq(candidate)]
                values.append(float(row.family_balanced_rmse_um.iloc[0]) if len(row) else np.nan)
            any_finite |= bool(np.isfinite(values).any())
            ax.bar(
                x + offset,
                values,
                width=width,
                color=COLORS[candidate],
                edgecolor="#374151",
                linewidth=0.6,
                hatch=hatches[candidate],
                label=LABELS[candidate],
            )
        if regime == "movement" and subset.loc[subset.window.eq("1150_1200")].empty:
            ax.text(1, 8, "No ≥20 µm\nlighthouse increments", ha="center", va="bottom", color="#4B5563", fontsize=9)
        ax.set_xticks(x, ["930–1030", "1150–1200"])
        ax.set_title({"all": "All increments", "quiet": "Lighthouse-quiet (<20 µm)", "movement": "Lighthouse movement (≥20 µm)"}[regime])
        ax.set_xlabel("Recording window (s)")
        ax.grid(axis="y", alpha=0.18)
    axes[0].set_ylabel("Family-balanced 5 s increment RMSE (µm)")
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "Frozen motion candidates versus depth-aware lighthouse observations\n"
        "Strict evidence · plausibility restricted · within 5 µm of template-lattice nodes"
    )
    fig.savefig(OUT / "01_candidate_rmse.png", dpi=180)
    fig.savefig(OUT / "01_candidate_rmse.pdf")
    plt.close(fig)

    primary_levels = levels.loc[levels.sensitivity.eq(PRIMARY_SENSITIVITY)].copy()
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    for ax, (window, (start, stop)) in zip(axes, WINDOWS.items()):
        subset = primary_levels.loc[primary_levels.window.eq(window)]
        centered = []
        for _, family_group in subset.groupby("family_id"):
            family_group = family_group.sort_values("bin").copy()
            reference = family_group.loc[family_group.bin.lt(2), "lighthouse_um"].median()
            family_group["lighthouse_centered_um"] = family_group.lighthouse_um - reference
            centered.append(family_group)
            ax.plot(
                family_group.bin_time_s,
                family_group.lighthouse_centered_um,
                color="#9CA3AF",
                alpha=0.45,
                lw=0.8,
            )
        if not centered:
            continue
        combined = pd.concat(centered, ignore_index=True)
        consensus = combined.groupby("bin", as_index=False).agg(
            time_s=("bin_time_s", "first"),
            lighthouse=("lighthouse_centered_um", "median"),
            families=("family_id", "nunique"),
            lfp_rigid=("lfp_rigid_um", "median"),
            ap_rigid=("ap_rigid_um", "median"),
            ap_nonrigid=("ap_nonrigid_um", "median"),
        )
        ax.plot(consensus.time_s, consensus.lighthouse, color="#111827", lw=2.3, marker="o", ms=3.5, label="Lighthouse family median")
        for candidate in ["lfp_rigid", "ap_rigid", "ap_nonrigid"]:
            values = consensus[candidate]
            first = values.loc[values.notna()].iloc[0] if values.notna().any() else np.nan
            ax.plot(consensus.time_s, values - first, color=COLORS[candidate], lw=1.7, label=LABELS[candidate])
        ax.axhline(0, color=COLORS["zero"], ls="--", lw=1.2, label="No correction")
        ax.set(xlim=(start, stop), ylabel="Relative displacement (µm)", title=f"{start:.0f}–{stop:.0f} s")
        ax.grid(alpha=0.15)
        ax.legend(ncol=5, fontsize=8)
    axes[-1].set_xlabel("Recording time (s)")
    fig.suptitle("Candidate trajectories on primary lighthouse support (display references only)")
    fig.savefig(OUT / "02_candidate_tracks.png", dpi=180)
    fig.savefig(OUT / "02_candidate_tracks.pdf")
    plt.close(fig)


def write_report(summary: pd.DataFrame, support: pd.DataFrame) -> None:
    primary = summary.loc[
        summary.sensitivity.eq(PRIMARY_SENSITIVITY) & summary.regime.eq("all")
    ].copy()
    lines = [
        "# Frozen motion candidates against depth-aware lighthouse observations",
        "",
        "## Result",
        "",
        "This cached comparison treats every field as a candidate and the lighthouse panel as independent validation. "
        "It does not authorize correction. The AP candidates are the completed full-session MEDiCINe nonrigid field "
        "and its frozen equal-depth rigid projection; the LFP candidate is the frozen `band_0p5_8` rigid trace.",
        "",
        "The result is regime-dependent. In 930–1030 s, LFP rigid has the lowest all-increment RMSE (42.35 µm), "
        "versus 83.76 µm for AP rigid, 84.62 µm for AP nonrigid, and 109.17 µm for no motion. But on the 24 "
        "lighthouse-quiet increments in that same window, no motion is best (1.44 µm), while LFP rigid rises to "
        "48.86 µm. In the entirely quiet 1150–1200 s window, no motion is again best (1.43 µm); AP rigid, AP "
        "nonrigid, and LFP rigid produce 4.92, 9.03, and 21.86 µm RMSE, respectively. Thus LFP rigid captures the "
        "large shared excursions best but also introduces the most quiet-period false motion.",
        "",
        "## All-increment comparison",
        "",
        "| Window | Candidate | Families | Increments | RMSE (µm) | Skill vs zero | Median family r | Median slope |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for window in WINDOWS:
        for candidate in CANDIDATES:
            row = primary.loc[primary.window.eq(window) & primary.candidate.eq(candidate)]
            if row.empty:
                continue
            value = row.iloc[0]
            lines.append(
                f"| {window.replace('_', '–')} | {LABELS[candidate]} | {int(value.families)} | "
                f"{int(value.increments)} | {value.family_balanced_rmse_um:.2f} | "
                f"{value.skill_vs_zero:.3f} | {value.median_family_correlation:.3f} | {value.median_family_slope:.3f} |"
            )
    lines += [
        "",
        "## Quiet and movement checks",
        "",
        "| Window | Regime | Candidate | Families | Increments | RMSE (µm) | Skill vs zero |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    stratified = summary.loc[
        summary.sensitivity.eq(PRIMARY_SENSITIVITY) & summary.regime.isin(["quiet", "movement"])
    ]
    for window in WINDOWS:
        for regime in ["quiet", "movement"]:
            for candidate in CANDIDATES:
                row = stratified.loc[
                    stratified.window.eq(window)
                    & stratified.regime.eq(regime)
                    & stratified.candidate.eq(candidate)
                ]
                if row.empty:
                    continue
                value = row.iloc[0]
                lines.append(
                    f"| {window.replace('_', '–')} | {regime} | {LABELS[candidate]} | {int(value.families)} | "
                    f"{int(value.increments)} | {value.family_balanced_rmse_um:.2f} | {value.skill_vs_zero:.3f} |"
                )
    lines += [
        "",
        "There were no ≥20 µm lighthouse increments in 1150–1200 s, so that window contributes only to the quiet stratum.",
        "",
        "## Fair-comparison contract",
        "",
        "- Lighthouse identities, family mapping, plausibility exclusions, and the 5 µm lattice-node sensitivity were frozen before sampling candidates.",
        "- Each candidate is sampled at accepted events' actual times. AP nonrigid is also interpolated at each event's frozen template reference depth.",
        "- Strict events are reduced to consecutive 5 s family increments. Every family receives one vote; labels 557/673/675 remain one family.",
        "- The primary table uses common support across all candidates and requires at least three increments per family.",
        "- Quiet and movement strata are defined independently from the candidates using absolute lighthouse increments below or above 20 µm.",
        "- No sign, gain, lag, offset, smoothing, interpolation across LFP support gaps, or candidate-informed lighthouse selection is fitted.",
        "",
        "## Evidence retained",
        "",
        "Strict, lower-score, identity-ambiguous, unsupported, lattice-phase, per-family residual, correlation, slope, and support evidence are saved separately. "
        "The AP package remains marked `requires_review` and `correction_ready: false`.",
        "",
        "## Interpretation boundary",
        "",
        "Agreement can reject obvious candidate failures but cannot certify biological identity, physical displacement calibration, or correction safety. "
        "The quiet/movement split is descriptive because it uses the observed lighthouse increment magnitude. Only the two windows covered by the frozen "
        "lighthouse panel are evaluated here.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "MPLCONFIGDIR=/tmp/luke-motion-candidate-lighthouse-mpl python -m testing.luke_motion_candidate_lighthouse_comparison_v1",
        "```",
        "",
    ]
    DOC.write_text("\n".join(lines))
    (OUT / "README.md").write_text(
        "# Motion-candidate lighthouse comparison\n\n"
        "Cached, plotting-and-scoring-only comparison of frozen LFP rigid, AP rigid, AP nonrigid, and no-motion candidates. "
        "See `docs/luke_motion_candidate_lighthouse_comparison_20260910.md` for the decision-facing summary.\n"
    )


def build_artifact(summary: pd.DataFrame) -> None:
    generated = pd.Timestamp.now(tz="America/Los_Angeles").isoformat()
    primary = summary.loc[summary.sensitivity.eq(PRIMARY_SENSITIVITY)].copy()
    def finite_or_none(value: float) -> float | None:
        return float(value) if np.isfinite(value) else None

    chart_rows = []
    for row in primary.itertuples():
        if row.regime == "movement" and row.window == "1150_1200":
            continue
        chart_rows.append(
            {
                "window_regime": f"{row.window.replace('_', '–')} s · {row.regime}",
                "window": row.window.replace("_", "–") + " s",
                "regime": row.regime,
                "candidate": LABELS[row.candidate],
                "rmse_um": float(row.family_balanced_rmse_um),
                "families": int(row.families),
                "increments": int(row.increments),
                "skill_vs_zero": float(row.skill_vs_zero),
                "median_family_correlation": finite_or_none(row.median_family_correlation),
                "median_family_slope": finite_or_none(row.median_family_slope),
            }
        )
    table_rows = [
        {
            "window": row.window.replace("_", "–") + " s",
            "candidate": LABELS[row.candidate],
            "families": int(row.families),
            "increments": int(row.increments),
            "rmse_um": round(float(row.family_balanced_rmse_um), 2),
            "skill_vs_zero": float(row.skill_vs_zero),
            "skill_ci_low": float(row.skill_ci_low),
            "skill_ci_high": float(row.skill_ci_high),
            "median_family_correlation": finite_or_none(row.median_family_correlation),
            "median_family_slope": finite_or_none(row.median_family_slope),
        }
        for row in primary.loc[primary.regime.eq("all")].itertuples()
    ]
    source = {
        "id": "candidate_summary",
        "label": "Frozen candidate lighthouse comparison",
        "path": str(OUT / "candidate_summary.csv"),
        "query": {
            "engine": "DuckDB",
            "language": "sql",
            "sql": (
                f"SELECT * FROM read_csv_auto('{OUT / 'candidate_summary.csv'}') "
                f"WHERE sensitivity = '{PRIMARY_SENSITIVITY}' ORDER BY window, regime, candidate"
            ),
            "description": "Family-balanced comparison of frozen candidate fields on common lighthouse support.",
            "executed_at": generated,
            "tables_used": [str(OUT / "candidate_summary.csv")],
            "filters": [
                "strict_accepted lighthouse evidence",
                "predeclared plausibility exclusions",
                "lattice phase at most 5 µm",
                "common support across all four candidates",
            ],
            "metric_definitions": [
                "Family-balanced RMSE is the square root of the unweighted mean of per-family mean squared 5 s increment residuals.",
                "Residual is lighthouse increment minus candidate increment.",
                "Skill versus zero is 1 - candidate RMSE / no-motion RMSE; positive favors the candidate.",
                "Quiet is absolute lighthouse increment below 20 µm; movement is at least 20 µm.",
            ],
        },
    }
    title = "Frozen motion candidates against depth-aware lighthouse observations"
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": "Independent comparison of LFP rigid, AP rigid, AP nonrigid, and no-motion candidates.",
            "generatedAt": generated,
            "blocks": [
                {"id": "title", "type": "markdown", "body": f"# {title}", "layout": "full"},
                {"id": "technical-summary", "type": "markdown", "body": (
                    "## Technical summary\n\n"
                    "**No single candidate wins across movement regimes.** In 930–1030 s, LFP rigid best captures the large shared excursions and has the lowest all-increment RMSE (42.35 µm versus 83.76 AP rigid, 84.62 AP nonrigid, and 109.17 no motion). During lighthouse-quiet increments, however, no motion is best in both windows. The LFP trace has the largest quiet-period false-motion error. This supports retaining LFP rigid as a diagnostic candidate, not authorizing correction."
                ), "layout": "full"},
                {"id": "key-findings", "type": "markdown", "body": (
                    "## Performance reverses between movement and quiet periods\n\n"
                    "On 33 early-window movement increments, LFP rigid reaches 39.75 µm family-balanced RMSE versus 103.32 AP rigid, 105.38 AP nonrigid, and 133.19 no motion. On 24 early-window quiet increments, no motion is 1.44 µm; AP rigid is 8.28, AP nonrigid 10.38, and LFP rigid 48.86 µm. All 28 increments in 1150–1200 s are quiet, where no motion is 1.43 µm and LFP rigid is 21.86 µm."
                ), "layout": "full", "sourceId": "candidate_summary"},
                {"id": "rmse-chart", "type": "chart", "chartId": "candidate-rmse", "layout": "full"},
                {"id": "exact-results", "type": "markdown", "body": (
                    "## All-increment results and uncertainty\n\n"
                    "The table reports exact primary-sensitivity values and family-bootstrap 95% intervals for skill versus no motion. Positive skill means lower RMSE than no motion."
                ), "layout": "full"},
                {"id": "summary-table", "type": "table", "tableId": "candidate-table", "layout": "full"},
                {"id": "scope", "type": "markdown", "body": (
                    "## Scope, data, and metric definitions\n\n"
                    "The comparison covers only 930–1030 s and 1150–1200 s, the two LFP-audit windows overlapping the frozen lighthouse panel. Strict waveform observations are grouped into independent identity families and consecutive 5 s increments. Each eligible family receives one vote. The primary sensitivity excludes six predeclared concern units and retains observations within 5 µm of 40 µm template-lattice nodes."
                ), "layout": "full"},
                {"id": "method", "type": "markdown", "body": (
                    "## Candidate sampling preserves independence\n\n"
                    "Identities, gates, family mapping, plausibility exclusions, and lattice rules were frozen before candidate evaluation. Every candidate is sampled at actual event times. AP nonrigid is bilinearly interpolated at the event's frozen template reference depth; AP rigid and LFP rigid are depth-independent. Candidate increments use common support. No sign, gain, lag, offset, smoothing, or candidate-informed event selection is fitted."
                ), "layout": "full"},
                {"id": "limitations", "type": "markdown", "body": (
                    "## Limitations and robustness\n\n"
                    "The primary result has six eligible families and 53 increments early, and four families with 28 increments in the quiet window. The all-strict sensitivity is dominated by known implausible/lookalike-prone tracks and is preserved only as a control. Quiet/movement strata are descriptive because they are defined by observed lighthouse magnitude. Lighthouse observations remain provisional biological evidence, and the AP package itself is marked `requires_review` and `correction_ready: false`."
                ), "layout": "full"},
                {"id": "next", "type": "markdown", "body": (
                    "## Recommended next step\n\n"
                    "Do not select one field globally from this result. The cheapest next test is a simple quiet-state gate or regime-aware diagnostic evaluated on a later, independently chosen high-motion lighthouse window. Only after that cross-modal check should raw LFP be resampled under candidate corrections, using held-out LFP channels or features to avoid circular validation."
                ), "layout": "full"},
                {"id": "questions", "type": "markdown", "body": (
                    "## Further questions\n\n"
                    "Does LFP rigid retain its movement advantage in a late independently selected high-motion interval? Can an LFP-only confidence measure identify the quiet false-motion periods without AP or lighthouse input? Does depth-specific AP nonrigidity improve for individual well-supported families even though its family-balanced aggregate does not?"
                ), "layout": "full"},
            ],
            "charts": [
                {
                    "id": "candidate-rmse",
                    "title": "Candidate error by window and lighthouse-motion regime",
                    "subtitle": "Family-balanced 5 s increment RMSE on strict common support",
                    "type": "bar",
                    "dataset": "candidate_comparison",
                    "sourceId": "candidate_summary",
                    "encodings": {
                        "x": {"field": "window_regime", "type": "nominal", "label": "Window and regime"},
                        "y": {"field": "rmse_um", "type": "quantitative", "label": "RMSE", "unit": "µm"},
                        "color": {"field": "candidate", "type": "nominal", "label": "Candidate"},
                        "tooltip": [
                            {"field": "families", "type": "quantitative", "label": "Families"},
                            {"field": "increments", "type": "quantitative", "label": "Increments"},
                            {"field": "skill_vs_zero", "type": "quantitative", "label": "Skill vs zero"},
                        ],
                    },
                    "layout": "full",
                }
            ],
            "tables": [
                {
                    "id": "candidate-table",
                    "title": "All-increment candidate results",
                    "subtitle": "Primary plausibility and lattice sensitivity",
                    "dataset": "all_increment_results",
                    "sourceId": "candidate_summary",
                    "defaultSort": {"field": "window", "direction": "asc"},
                    "layout": "full",
                    "columns": [
                        {"field": "window", "label": "Window", "type": "text"},
                        {"field": "candidate", "label": "Candidate", "type": "text"},
                        {"field": "families", "label": "Families", "type": "number"},
                        {"field": "increments", "label": "Increments", "type": "number"},
                        {"field": "rmse_um", "label": "RMSE (µm)", "type": "number"},
                        {"field": "skill_vs_zero", "label": "Skill vs zero", "type": "number", "movement": True},
                        {"field": "skill_ci_low", "label": "Skill CI low", "type": "number"},
                        {"field": "skill_ci_high", "label": "Skill CI high", "type": "number"},
                        {"field": "median_family_correlation", "label": "Median r", "type": "number"},
                        {"field": "median_family_slope", "label": "Median slope", "type": "number"},
                    ],
                }
            ],
            "sources": [
                source,
                {"id": "policy", "label": "Depth-aware lighthouse policy", "path": str(ROOT / "docs/luke_depth_aware_lighthouse_policy_20260908.md")},
                {"id": "lfp-audit", "label": "LFP conditioning audit", "path": str(LFP_PACKAGE / "README.md")},
                {"id": "ap-field", "label": "Frozen AP candidate field manifest", "path": str(AP_MANIFEST)},
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated,
            "status": "ready",
            "datasets": {
                "candidate_comparison": chart_rows,
                "all_increment_results": table_rows,
            },
        },
        "sources": [source],
    }
    (OUT / "artifact.json").write_text(json.dumps(artifact, indent=2) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    required = [EARLY, EXTENDED, FAMILY_MAP, AP_FIELD, AP_MANIFEST]
    required += [LFP_PACKAGE / f"{window}_{LFP_ARM}_motion.npz" for window in WINDOWS]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    manifest = json.loads(AP_MANIFEST.read_text())
    expected_ap_hash = manifest["files"]["candidate_fields.npz"]
    actual_ap_hash = sha256(AP_FIELD)
    if actual_ap_hash != expected_ap_hash:
        raise RuntimeError("AP candidate field hash does not match its manifest")
    if manifest.get("correction_ready") is not False:
        raise RuntimeError("Expected a review-only AP candidate package")

    base_events = load_events()
    ap = np.load(AP_FIELD)
    rng = np.random.default_rng(20260910)
    level_outputs = []
    increment_outputs = []
    family_outputs = []
    summary_outputs = []
    support_outputs = []

    for window, (start, stop) in WINDOWS.items():
        window_events = nearest_supported_lfp(base_events, window)
        window_events = sample_ap(window_events, ap)
        for sensitivity in SENSITIVITIES:
            events = apply_sensitivity(window_events, sensitivity)
            levels = bin_events(events, start, stop)
            increments = make_increments(levels)
            family = per_family_metrics(increments)
            summary = summarize(family, rng)
            for frame in [levels, increments, family, summary]:
                frame.insert(0, "sensitivity", sensitivity)
                frame.insert(0, "window", window)
            level_outputs.append(levels)
            increment_outputs.append(increments)
            family_outputs.append(family)
            summary_outputs.append(summary)
            support_outputs.append(support_summary(events, levels, window, sensitivity))

    levels = pd.concat(level_outputs, ignore_index=True)
    increments = pd.concat(increment_outputs, ignore_index=True)
    family = pd.concat(family_outputs, ignore_index=True)
    summary = pd.concat(summary_outputs, ignore_index=True)
    support = pd.DataFrame(support_outputs)

    levels.to_csv(OUT / "binned_family_tracks.csv", index=False)
    increments.to_csv(OUT / "candidate_increments.csv", index=False)
    family.to_csv(OUT / "per_family_metrics.csv", index=False)
    summary.to_csv(OUT / "candidate_summary.csv", index=False)
    support.to_csv(OUT / "support_summary.csv", index=False)
    make_figures(summary, levels)
    write_report(summary, support)
    build_artifact(summary)

    primary = summary.loc[
        summary.sensitivity.eq(PRIMARY_SENSITIVITY) & summary.regime.eq("all")
    ]
    result = {
        "status": "complete",
        "scientific_status": "independent_comparison_not_correction_authorization",
        "primary_sensitivity": PRIMARY_SENSITIVITY,
        "windows": WINDOWS,
        "quiet_threshold_um": QUIET_THRESHOLD_UM,
        "candidate_fields": {
            "lfp_rigid": f"{LFP_ARM} frozen rigid LFP trace",
            "ap_rigid": "frozen full-session MEDiCINe equal-depth rigid projection",
            "ap_nonrigid": "frozen full-session MEDiCINe field at family reference depth",
            "zero": "no-motion baseline",
        },
        "primary_results": primary[
            ["window", "candidate", "families", "increments", "family_balanced_rmse_um", "skill_vs_zero"]
        ].to_dict(orient="records"),
        "source_sha256": {str(path): sha256(path) for path in required},
        "ap_manifest_scientific_status": manifest.get("scientific_status"),
        "ap_manifest_correction_ready": manifest.get("correction_ready"),
        "correction_applied": False,
        "estimator_fitted": False,
        "sorting_launched": False,
    }
    (OUT / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
