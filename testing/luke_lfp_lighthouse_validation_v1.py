"""Independent lighthouse-unit assessment of the imec0 LFP motion audit.

This is a comparison/validation script, not a motion estimator.  It evaluates
the frozen LFP fields only where the pre-existing waveform-only lighthouse
panel has temporal support.  Identity decisions are never changed using LFP.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "testing/outputs/luke_lfp_lighthouse_validation_v1"
LFP = OUT / "lfp_package"
EARLY = ROOT / "testing/outputs/luke_waveform_only_expansion_v1/overlay_events.csv"
EXTENDED = ROOT / "testing/outputs/luke_lighthouse_extension_300s_v1/new_event_predictions.csv"
FAMILY_MAP = ROOT / "testing/outputs/luke_lighthouse_method_audit_v1/d_candidate_families.csv"

WINDOWS = {"930_1030": (930.0, 1030.0), "1150_1200": (1150.0, 1200.0)}
ARMS = [
    "baseline",
    "screen_none_raw",
    "screen_median_raw",
    "screen_none_d1",
    "screen_median_d1",
    "screen_none_d2",
    "screen_median_d2",
    "band_0p5_8",
    "band_8_30",
    "band_30_100",
    "cap8",
]
PRIMARY_ARM = "band_0p5_8"
BIN_S = 5.0
MIN_FAMILIES = 3
MIN_COMMON_INCREMENTS = 3
LATTICE_UM = 40.0
NODE_TOL_UM = 5.0
# Defined by the pre-LFP tracker audit: five >500 um tracks plus the one seed
# whose recovery did not beat the circular-shift null.  Unit 557 is removed in
# this sensitivity, but its family remains represented by pooled 673/675.
PREDECLARED_CONCERN_UNITS = {125, 161, 353, 557, 698, 705}


def corr(x: np.ndarray, y: np.ndarray) -> float:
    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() < 3 or np.std(x[good]) == 0 or np.std(y[good]) == 0:
        return np.nan
    return float(np.corrcoef(x[good], y[good])[0, 1])


def slope_through_origin(x: np.ndarray, y: np.ndarray) -> float:
    """Slope for y~x after differencing; x=lighthouse, y=LFP."""
    good = np.isfinite(x) & np.isfinite(y)
    denom = float(np.dot(x[good], x[good]))
    return float(np.dot(x[good], y[good]) / denom) if good.sum() and denom > 0 else np.nan


def load_events() -> pd.DataFrame:
    early = pd.read_csv(EARLY)
    extended = pd.read_csv(EXTENDED).drop(columns=["prediction_um", "difference_um", "variant"])
    events = pd.concat([early, extended], ignore_index=True)
    fam = pd.read_csv(FAMILY_MAP).rename(columns={"family": "family_id"})
    events = events.merge(fam[["unit_id", "family_id"]], on="unit_id", how="left", validate="many_to_one")
    if events.family_id.isna().any():
        raise RuntimeError("Missing frozen family assignment")
    events["family_id"] = events.family_id.astype(int)
    events["lattice_phase_um"] = np.abs(((events.relative_um + LATTICE_UM / 2) % LATTICE_UM) - LATTICE_UM / 2)
    return events


def load_lfp(window: str, arm: str) -> pd.DataFrame:
    z = np.load(LFP / f"{window}_{arm}_motion.npz")
    return pd.DataFrame(
        {
            "time_s": z["time_s"].astype(float),
            "lfp_um": z["displacement_um"].astype(float),
            "supported": z["supported"].astype(bool),
            "invalid": z["invalid"].astype(bool),
        }
    )


def sample_lfp_at_events(events: pd.DataFrame, trace: pd.DataFrame) -> pd.DataFrame:
    """Attach the nearest full-rate LFP output without bridging unsupported gaps."""
    d = events.copy()
    t = trace.time_s.to_numpy(float)
    q = d.time_s.to_numpy(float)
    right = np.searchsorted(t, q, side="left")
    right = np.clip(right, 0, len(t) - 1)
    left = np.clip(right - 1, 0, len(t) - 1)
    choose_right = np.abs(t[right] - q) < np.abs(t[left] - q)
    ix = np.where(choose_right, right, left)
    dt = np.abs(t[ix] - q)
    valid = (
        (dt <= 0.01)
        & trace.supported.to_numpy(bool)[ix]
        & ~trace.invalid.to_numpy(bool)[ix]
        & np.isfinite(trace.lfp_um.to_numpy(float)[ix])
    )
    d["lfp_sample_time_s"] = t[ix]
    d["lfp_time_error_s"] = dt
    d["lfp_event_supported"] = valid
    d["lfp_event_um"] = np.where(valid, trace.lfp_um.to_numpy(float)[ix], np.nan)
    return d


def bin_lighthouse(events: pd.DataFrame, start: float, stop: float, min_events: int = 1) -> pd.DataFrame:
    strict = events.loc[events.evidence.eq("strict_accepted") & events.time_s.ge(start) & events.time_s.lt(stop)].copy()
    strict["bin"] = np.floor((strict.time_s - start) / BIN_S).astype(int)
    out = (
        strict.groupby(["family_id", "bin"], as_index=False)
        .agg(
            time_s=("time_s", "median"),
            lighthouse_um=("relative_um", "median"),
            event_count=("relative_um", "size"),
            temporal_span_s=("time_s", lambda x: float(x.max() - x.min()) if len(x) else np.nan),
            unit_count=("unit_id", "nunique"),
            lattice_phase_um=("lattice_phase_um", "median"),
            lfp_um=("lfp_event_um", "median"),
            lfp_samples=("lfp_event_supported", "sum"),
            lfp_supported_fraction=("lfp_event_supported", "mean"),
        )
    )
    out = out.loc[out.event_count.ge(min_events)].copy()
    out["bin_time_s"] = start + (out.bin + 0.5) * BIN_S
    return out


def bin_lfp(trace: pd.DataFrame, start: float, stop: float) -> pd.DataFrame:
    d = trace.loc[trace.time_s.ge(start) & trace.time_s.lt(stop)].copy()
    d["bin"] = np.floor((d.time_s - start) / BIN_S).astype(int)
    rows = []
    for b, g in d.groupby("bin"):
        valid = g.supported & ~g.invalid & np.isfinite(g.lfp_um)
        rows.append(
            {
                "bin": int(b),
                "lfp_um": float(np.median(g.loc[valid, "lfp_um"])) if valid.any() else np.nan,
                "lfp_supported_fraction": float(valid.mean()),
                "lfp_samples": int(valid.sum()),
            }
        )
    return pd.DataFrame(rows)


def build_level_tracks(lh: pd.DataFrame, lfp_ref: float) -> pd.DataFrame:
    d = lh.copy()
    d["lfp_centered_um"] = d.lfp_um - lfp_ref
    centered = []
    for _, g in d.groupby("family_id", sort=False):
        g = g.sort_values("bin").copy()
        ref = g.loc[g.bin.lt(2), "lighthouse_um"].median()
        g["lighthouse_centered_um"] = g.lighthouse_um - ref
        g["has_reference_support"] = np.isfinite(ref) and np.isfinite(lfp_ref)
        centered.append(g)
    return pd.concat(centered, ignore_index=True) if centered else d


def increments(level: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family_id, g in level.groupby("family_id"):
        g = g.sort_values("bin").copy()
        prev = g.shift(1)
        adjacent = g.bin.to_numpy() - prev.bin.to_numpy() == 1
        valid = adjacent & g.lighthouse_um.notna() & prev.lighthouse_um.notna() & g.lfp_um.notna() & prev.lfp_um.notna()
        for ix in np.flatnonzero(valid):
            dl = float(g.iloc[ix].lighthouse_um - prev.iloc[ix].lighthouse_um)
            dp = float(g.iloc[ix].lfp_um - prev.iloc[ix].lfp_um)
            rows.append(
                {
                    "family_id": int(family_id),
                    "bin_from": int(prev.iloc[ix].bin),
                    "bin_to": int(g.iloc[ix].bin),
                    "time_s": float(g.iloc[ix].bin_time_s),
                    "lighthouse_delta_um": dl,
                    "lfp_delta_um": dp,
                    "residual_um": dl - dp,
                    "zero_residual_um": dl,
                    "event_count_from": int(prev.iloc[ix].event_count),
                    "event_count_to": int(g.iloc[ix].event_count),
                }
            )
    columns = [
        "family_id", "bin_from", "bin_to", "time_s", "lighthouse_delta_um",
        "lfp_delta_um", "residual_um", "zero_residual_um",
        "event_count_from", "event_count_to",
    ]
    return pd.DataFrame(rows, columns=columns)


def family_metrics(inc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family_id, g in inc.groupby("family_id"):
        x = g.lighthouse_delta_um.to_numpy(float)
        y = g.lfp_delta_um.to_numpy(float)
        r = x - y
        rows.append(
            {
                "family_id": int(family_id),
                "n_increments": len(g),
                "lfp_rmse_um": float(np.sqrt(np.mean(r**2))),
                "zero_rmse_um": float(np.sqrt(np.mean(x**2))),
                "lfp_mae_um": float(np.mean(np.abs(r))),
                "zero_mae_um": float(np.mean(np.abs(x))),
                "signed_residual_um": float(np.mean(r)),
                "correlation": corr(x, y),
                "lfp_vs_lighthouse_slope": slope_through_origin(x, y),
            }
        )
    columns = [
        "family_id", "n_increments", "lfp_rmse_um", "zero_rmse_um",
        "lfp_mae_um", "zero_mae_um", "signed_residual_um", "correlation",
        "lfp_vs_lighthouse_slope",
    ]
    return pd.DataFrame(rows, columns=columns)


def equal_family_summary(fm: pd.DataFrame, rng: np.random.Generator) -> dict[str, float]:
    eligible = fm.loc[fm.n_increments.ge(MIN_COMMON_INCREMENTS)].copy()
    if eligible.empty:
        return {
            "families": 0,
            "increments": 0,
            "family_balanced_lfp_rmse_um": np.nan,
            "family_balanced_zero_rmse_um": np.nan,
            "skill_vs_zero": np.nan,
            "median_family_correlation": np.nan,
            "median_family_signed_residual_um": np.nan,
            "median_lfp_vs_lighthouse_slope": np.nan,
            "skill_ci_low": np.nan,
            "skill_ci_high": np.nan,
        }
    mse_lfp = eligible.lfp_rmse_um.to_numpy() ** 2
    mse_zero = eligible.zero_rmse_um.to_numpy() ** 2
    rmse_lfp = float(np.sqrt(np.mean(mse_lfp)))
    rmse_zero = float(np.sqrt(np.mean(mse_zero)))
    skill = float(1 - rmse_lfp / rmse_zero) if rmse_zero > 0 else np.nan
    draws = []
    for _ in range(5000):
        idx = rng.integers(0, len(eligible), len(eligible))
        a = float(np.sqrt(np.mean(mse_lfp[idx])))
        z = float(np.sqrt(np.mean(mse_zero[idx])))
        draws.append(1 - a / z if z > 0 else np.nan)
    return {
        "families": int(len(eligible)),
        "increments": int(eligible.n_increments.sum()),
        "family_balanced_lfp_rmse_um": rmse_lfp,
        "family_balanced_zero_rmse_um": rmse_zero,
        "skill_vs_zero": skill,
        "median_family_correlation": float(eligible.correlation.median()),
        "median_family_signed_residual_um": float(eligible.signed_residual_um.median()),
        "median_lfp_vs_lighthouse_slope": float(eligible.lfp_vs_lighthouse_slope.median()),
        "skill_ci_low": float(np.nanquantile(draws, 0.025)),
        "skill_ci_high": float(np.nanquantile(draws, 0.975)),
    }


def population_metrics(level: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    cons = (
        level.groupby("bin", as_index=False)
        .agg(
            time_s=("bin_time_s", "first"),
            lighthouse_um=("lighthouse_centered_um", "median"),
            family_support=("family_id", "nunique"),
            strict_events=("event_count", "sum"),
            lfp_um=("lfp_centered_um", "median"),
            lfp_supported_fraction=("lfp_supported_fraction", "median"),
        )
    )
    cons = cons.loc[cons.family_support.ge(MIN_FAMILIES) & cons.lighthouse_um.notna() & cons.lfp_um.notna()].copy()
    cons = cons.sort_values("bin")
    prev = cons.shift(1)
    adjacent = cons.bin.to_numpy() - prev.bin.to_numpy() == 1
    dx = (cons.lighthouse_um - prev.lighthouse_um).to_numpy(float)[adjacent]
    dy = (cons.lfp_um - prev.lfp_um).to_numpy(float)[adjacent]
    if len(dx):
        rmse = float(np.sqrt(np.mean((dx - dy) ** 2)))
        zero = float(np.sqrt(np.mean(dx**2)))
    else:
        rmse = zero = np.nan
    metrics = {
        "population_bins": int(len(cons)),
        "population_increments": int(len(dx)),
        "population_correlation": corr(dx, dy),
        "population_lfp_rmse_um": rmse,
        "population_zero_rmse_um": zero,
        "population_skill_vs_zero": float(1 - rmse / zero) if np.isfinite(zero) and zero > 0 else np.nan,
        "median_family_support_per_bin": float(cons.family_support.median()) if len(cons) else np.nan,
    }
    return cons, metrics


def sensitivity_events(events: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, int]:
    d = events.copy()
    min_events = 1
    if mode == "plausibility_restricted":
        d = d.loc[~d.unit_id.isin(PREDECLARED_CONCERN_UNITS)].copy()
    elif mode == "plausibility_lattice_node_5um":
        d = d.loc[
            ~d.unit_id.isin(PREDECLARED_CONCERN_UNITS)
            & d.lattice_phase_um.le(NODE_TOL_UM)
        ].copy()
    elif mode == "lattice_node_5um":
        d = d.loc[d.lattice_phase_um.le(NODE_TOL_UM)].copy()
    elif mode == "min3_events_per_bin":
        min_events = 3
    elif mode != "all_strict":
        raise ValueError(mode)
    return d, min_events


def make_figures(all_levels: pd.DataFrame, summaries: pd.DataFrame, family: pd.DataFrame) -> None:
    # Chart contract: temporal relationship; two-panel highlighted line/points;
    # blue lighthouse consensus, orange LFP, gray independent family tracks.
    primary = all_levels.loc[(all_levels.arm == PRIMARY_ARM) & (all_levels.sensitivity == "all_strict")]
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    for ax, (window, (start, stop)) in zip(axes, WINDOWS.items()):
        d = primary.loc[primary.window == window]
        for _, g in d.groupby("family_id"):
            ax.plot(g.bin_time_s, g.lighthouse_centered_um, color="#9CA3AF", alpha=0.45, lw=0.8)
            ax.scatter(g.bin_time_s, g.lighthouse_centered_um, color="#6B7280", alpha=0.5, s=10)
        cons = d.groupby("bin", as_index=False).agg(
            time_s=("bin_time_s", "first"), lighthouse_um=("lighthouse_centered_um", "median"),
            support=("family_id", "nunique"), lfp_um=("lfp_centered_um", "first")
        )
        cons = cons.loc[cons.support.ge(MIN_FAMILIES)]
        ax.plot(cons.time_s, cons.lighthouse_um, color="#1D4ED8", lw=2.2, marker="o", ms=4, label="Lighthouse family median")
        ax.plot(cons.time_s, cons.lfp_um, color="#EA580C", lw=2.2, label="LFP 0.5–8 Hz d1")
        ax.axhline(0, color="#111827", lw=0.8, alpha=0.5)
        ax.set(xlim=(start, stop), ylabel="Within-window displacement (µm)", title=f"{start:.0f}–{stop:.0f} s")
        ax.grid(axis="y", color="#E5E7EB", lw=0.7)
    axes[0].legend(frameon=False, ncol=2, loc="upper left")
    axes[-1].set_xlabel("Recording time (s)")
    fig.suptitle("LFP motion and depth-aware lighthouse observations", fontsize=14)
    fig.savefig(OUT / "01_lfp_lighthouse_tracks.png", dpi=180)
    fig.savefig(OUT / "01_lfp_lighthouse_tracks.pdf")
    plt.close(fig)

    # Arm comparison: family-balanced increment RMSE, with zero-motion reference.
    d = summaries.loc[summaries.sensitivity.eq("plausibility_restricted")].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True, constrained_layout=True)
    labels = ARMS
    for ax, (window, _) in zip(axes, WINDOWS.items()):
        w = d.loc[d.window.eq(window)].set_index("arm").reindex(labels)
        y = np.arange(len(labels))
        enough = w.families.ge(3)
        ax.scatter(w.family_balanced_zero_rmse_um.where(enough), y, color="#1D4ED8", marker="|", s=220, linewidths=2.5, label="Zero motion")
        ax.scatter(w.family_balanced_lfp_rmse_um.where(enough), y, color="#EA580C", s=40, label="LFP arm")
        for yi, (_, row) in enumerate(w.iterrows()):
            if row.families >= 3 and np.isfinite(row.family_balanced_lfp_rmse_um) and np.isfinite(row.family_balanced_zero_rmse_um):
                ax.plot([row.family_balanced_zero_rmse_um, row.family_balanced_lfp_rmse_um], [yi, yi], color="#9CA3AF", lw=0.8, zorder=0)
        ax.set(title=window.replace("_", "–") + " s", xlabel="Family-balanced 5 s increment RMSE (µm)")
        ax.grid(axis="x", color="#E5E7EB", lw=0.7)
        ax.set_yticks(y, labels)
    axes[0].legend(frameon=False, loc="lower right")
    fig.suptitle("LFP arms versus zero motion after predeclared plausibility exclusions")
    fig.savefig(OUT / "02_arm_rmse_vs_zero.png", dpi=180)
    fig.savefig(OUT / "02_arm_rmse_vs_zero.pdf")
    plt.close(fig)

    # Family heterogeneity for the primary arm.
    d = family.loc[(family.arm == PRIMARY_ARM) & (family.sensitivity == "all_strict")].copy()
    d["skill_vs_zero"] = 1 - d.lfp_rmse_um / d.zero_rmse_um
    pivot = d.pivot(index="family_id", columns="window", values="skill_vs_zero").sort_index()
    pivot = pivot.reindex(columns=list(WINDOWS))
    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    vals = pivot.to_numpy(float)
    im = ax.imshow(vals, aspect="auto", cmap="coolwarm", vmin=-2, vmax=1)
    ax.set_xticks(np.arange(len(pivot.columns)), [x.replace("_", "–") for x in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)), [f"Family {x}" for x in pivot.index])
    ax.set_title("0.5–8 Hz d1 skill versus zero motion by identity family")
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            if np.isfinite(vals[i, j]):
                ax.text(j, i, f"{vals[i,j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(vals[i, j]) > 1.1 else "#111827")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Skill = 1 − LFP RMSE / zero-motion RMSE")
    fig.savefig(OUT / "03_primary_arm_family_skill.png", dpi=180)
    fig.savefig(OUT / "03_primary_arm_family_skill.pdf")
    plt.close(fig)


def build_artifact(summaries: pd.DataFrame) -> None:
    generated = pd.Timestamp.now(tz="America/Los_Angeles").isoformat()
    primary = summaries.loc[
        summaries.arm.eq(PRIMARY_ARM)
        & summaries.sensitivity.eq("plausibility_lattice_node_5um")
    ].copy()
    chart_rows = []
    for _, row in primary.iterrows():
        base = {
            "window": row.window.replace("_", "–") + " s",
            "families": int(row.families),
            "increments": int(row.increments),
            "skill_vs_zero": float(row.skill_vs_zero),
            "median_family_correlation": float(row.median_family_correlation),
            "median_slope": float(row.median_lfp_vs_lighthouse_slope),
            "event_support": float(row.lfp_median_supported_fraction),
            "sensitivity": "Plausibility-restricted and ≤5 µm from lattice nodes",
        }
        chart_rows.append({**base, "estimator": "LFP 0.5–8 Hz d1", "rmse_um": float(row.family_balanced_lfp_rmse_um)})
        chart_rows.append({**base, "estimator": "Zero motion", "rmse_um": float(row.family_balanced_zero_rmse_um)})

    table = summaries.loc[
        summaries.arm.eq(PRIMARY_ARM)
        & summaries.sensitivity.isin(
            ["all_strict", "plausibility_restricted", "lattice_node_5um", "plausibility_lattice_node_5um"]
        )
    ].copy()
    labels = {
        "all_strict": "All strict tracks",
        "plausibility_restricted": "Predeclared plausibility exclusions",
        "lattice_node_5um": "Within 5 µm of lattice nodes",
        "plausibility_lattice_node_5um": "Both restrictions",
    }
    table_rows = []
    for _, row in table.iterrows():
        table_rows.append(
            {
                "window": row.window.replace("_", "–") + " s",
                "sensitivity": labels[row.sensitivity],
                "families": int(row.families),
                "increments": int(row.increments),
                "lfp_rmse_um": round(float(row.family_balanced_lfp_rmse_um), 2),
                "zero_rmse_um": round(float(row.family_balanced_zero_rmse_um), 2),
                "skill_vs_zero": float(row.skill_vs_zero),
                "median_correlation": float(row.median_family_correlation),
                "median_slope": float(row.median_lfp_vs_lighthouse_slope),
                "event_support": float(row.lfp_median_supported_fraction),
            }
        )

    source = {
        "id": "analysis_summary",
        "label": "LFP–lighthouse validation summary",
        "path": str(OUT / "arm_summary.csv"),
        "query": {
            "engine": "DuckDB",
            "language": "sql",
            "sql": (
                f"SELECT * FROM read_csv_auto('{OUT / 'arm_summary.csv'}') "
                "WHERE arm = 'band_0p5_8' AND sensitivity IN "
                "('all_strict', 'plausibility_restricted', 'lattice_node_5um', "
                "'plausibility_lattice_node_5um') ORDER BY window, sensitivity"
            ),
            "description": "Full-rate LFP values sampled at frozen lighthouse event times, summarized as consecutive 5-second family increments.",
            "executed_at": generated,
            "tables_used": [str(EARLY), str(EXTENDED), str(FAMILY_MAP), str(LFP / "*_motion.npz")],
            "filters": [
                "imec0 only",
                "strict_accepted lighthouse observations for scored comparisons",
                "LFP supported=True and invalid=False",
                "identity families receive equal weight",
            ],
            "metric_definitions": [
                "Family-balanced RMSE: square root of the unweighted mean of each eligible family's mean squared 5-second increment residual.",
                "Residual sign: lighthouse increment minus LFP increment.",
                "Skill versus zero: 1 - LFP RMSE / zero-motion RMSE; positive favors LFP.",
                "Eligible family: at least three adjacent supported 5-second increments.",
            ],
        },
    }
    sources = [
        source,
        {"id": "policy", "label": "Depth-aware lighthouse policy", "path": str(ROOT / "docs/luke_depth_aware_lighthouse_policy_20260908.md")},
        {"id": "lfp_audit", "label": "LFP conditioning audit", "path": str(LFP / "README.md")},
        {"id": "tracker_audit", "label": "Lighthouse tracker method audit", "path": str(ROOT / "testing/outputs/luke_lighthouse_method_audit_v1/README.md")},
    ]
    title = "Lighthouse assessment of the imec0 LFP motion traces"
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": "Independent, depth-aware lighthouse validation of the 2026-09-10 LFP conditioning audit.",
            "generatedAt": generated,
            "blocks": [
                {"id": "title", "type": "markdown", "body": f"# {title}", "layout": "full"},
                {"id": "technical-summary", "type": "markdown", "body": (
                    "## Technical summary\n\n"
                    "**The highlighted 0.5–8 Hz first-derivative trace is not validated for correction.** "
                    "It tracks shared lighthouse movement in 930–1030 s, including after the strongest predeclared identity and lattice controls, but it produces substantial false motion in the quieter 1150–1200 s window. The failure to generalize outweighs the early-window agreement.\n\n"
                    "Only two audit windows overlap the frozen depth-aware lighthouse panel. The 5000–5100 s and 6000–6100 s traces remain untested by this evidence."
                ), "layout": "full"},
                {"id": "key-findings", "type": "markdown", "body": (
                    "## Agreement in the early window does not generalize\n\n"
                    "After removing units already flagged as implausible/lookalike-prone and restricting observations to within 5 µm of the 40 µm template lattice nodes, the highlighted arm reduced family-balanced increment RMSE from **109.2 µm for zero motion to 42.4 µm** in 930–1030 s. Six families and 53 adjacent increments contributed; median family correlation was **0.955** and median LFP-versus-lighthouse slope was **0.814**.\n\n"
                    "In 1150–1200 s, the same control set shows the opposite result: lighthouse increments were nearly stationary (**1.43 µm zero-motion RMSE**) while the LFP trace had **21.9 µm RMSE**, more than 15 times worse. Four families and 28 increments contributed, with median correlation **−0.112**."
                ), "layout": "full"},
                {"id": "rmse-chart", "type": "chart", "chartId": "primary-rmse", "layout": "full"},
                {"id": "chart-interpretation", "type": "markdown", "body": (
                    "## The quiet-window false positives are decision-limiting\n\n"
                    "The chart compares the candidate with the zero-motion baseline on identical lighthouse support. A reliable displacement estimator must improve on zero motion in both regimes; good alignment during a large-motion interval is insufficient when a quiet interval produces tens of microns of spurious movement."
                ), "layout": "full"},
                {"id": "sensitivity-table", "type": "table", "tableId": "primary-sensitivity", "layout": "full"},
                {"id": "scope-definitions", "type": "markdown", "body": (
                    "## Scope and metric definitions\n\n"
                    "The frozen 17 template labels map to 15 independent identity families; labels 557/673/675 share one family vote. LFP outputs were sampled at each lighthouse event's actual time, using only supported, non-invalid solver samples. Within each family, strict observations were reduced to 5-second medians and scored on consecutive-bin increments, avoiding arbitrary position offsets. Families with at least three increments received equal weight. Lower-score and identity-ambiguous events remain saved as separate support evidence and do not enter the primary score."
                ), "layout": "full"},
                {"id": "methodology", "type": "markdown", "body": (
                    "## Validation design\n\n"
                    "Candidate identities, acceptance gates, family assignments, plausibility flags, and lattice thresholds were all defined before this LFP comparison. No LFP, AP, DREDGE, depth prior, smoothing, gain fit, sign fit, lag optimization, or continuity rule entered lighthouse matching. The comparison reports per-family residuals before equal-family aggregation and uses zero displacement as the baseline."
                ), "layout": "full"},
                {"id": "limitations", "type": "markdown", "body": (
                    "## Limitations and robustness\n\n"
                    "The all-strict panel contains known waveform lookalikes with biologically implausible hundreds-of-microns jumps. Those rows are preserved, but they can make large erroneous LFP excursions appear useful. The positive 930–1030 s result survives both the predeclared plausibility exclusion and a lattice-node restriction; the 1150–1200 s failure becomes stronger under the same controls. Support is sparse: only 6 and 4 families, respectively, meet the increment threshold in the combined sensitivity. This evidence is descriptive validation, not certified biological ground truth."
                ), "layout": "full"},
                {"id": "next-steps", "type": "markdown", "body": (
                    "## Recommended next steps\n\n"
                    "1. Keep the 0.5–8 Hz derivative output as a diagnostic candidate only; do not use it for correction.\n"
                    "2. Before changing the estimator, use the cheapest direct follow-up: obtain frozen waveform-only lighthouse coverage in one of the late LFP windows and repeat this exact comparison.\n"
                    "3. Treat quiet-window false-motion rejection as a required gate for any future LFP estimator.\n"
                    "4. If more precision is needed, expand independent family support before adding consensus, smoothing, or calibration machinery."
                ), "layout": "full"},
                {"id": "further-questions", "type": "markdown", "body": (
                    "## Further questions\n\n"
                    "Would the 930–1030 s agreement replicate in a late, independently selected high-motion window? Are the quiet-window LFP excursions tied to chunk boundaries, low accepted-pair counts, or temporally coherent non-translational LF structure? Those checks could change which conditioning failure mode to address, but they do not change the current no-correction decision."
                ), "layout": "full"},
            ],
            "charts": [
                {
                    "id": "primary-rmse",
                    "title": "LFP and zero-motion increment error",
                    "subtitle": "Plausibility-restricted lighthouse observations within 5 µm of lattice nodes",
                    "showDescription": True,
                    "intent": "comparison",
                    "question": "Does the highlighted LFP trace beat zero motion in both lighthouse-covered windows?",
                    "rationale": "Grouped bars directly compare same-unit RMSE values against the required zero-motion baseline in each window.",
                    "comparisonContext": {"baseline": "Zero motion", "grain": "Identity family", "unit": "µm RMSE"},
                    "type": "bar",
                    "dataset": "primary_comparison",
                    "sourceId": "analysis_summary",
                    "encodings": {
                        "x": {"field": "window", "type": "nominal", "label": "Window"},
                        "y": {"field": "rmse_um", "type": "quantitative", "label": "5 s increment RMSE", "unit": "µm"},
                        "color": {"field": "estimator", "type": "nominal", "label": "Estimator"},
                        "tooltip": [
                            {"field": "families", "type": "quantitative", "label": "Families"},
                            {"field": "increments", "type": "quantitative", "label": "Increments"},
                            {"field": "median_family_correlation", "type": "quantitative", "label": "Median family r"},
                        ],
                    },
                    "combinationRationale": "Color distinguishes the candidate from its zero-motion baseline within each time window.",
                    "valueFormat": "number",
                    "unit": "µm",
                    "layout": "full",
                }
            ],
            "tables": [
                {
                    "id": "primary-sensitivity",
                    "title": "Highlighted-arm sensitivity results",
                    "subtitle": "Strict waveform evidence; family-balanced consecutive 5-second increments",
                    "showDescription": True,
                    "dataset": "sensitivity_results",
                    "defaultSort": {"field": "window", "direction": "asc"},
                    "density": "spacious",
                    "sourceId": "analysis_summary",
                    "layout": "full",
                    "columns": [
                        {"field": "window", "label": "Window", "type": "text"},
                        {"field": "sensitivity", "label": "Evidence cut", "type": "text"},
                        {"field": "families", "label": "Families", "type": "number"},
                        {"field": "increments", "label": "Increments", "type": "number"},
                        {"field": "lfp_rmse_um", "label": "LFP RMSE (µm)", "type": "number"},
                        {"field": "zero_rmse_um", "label": "Zero RMSE (µm)", "type": "number"},
                        {"field": "skill_vs_zero", "label": "Skill vs zero", "type": "number", "movement": True},
                        {"field": "median_correlation", "label": "Median r", "type": "number"},
                        {"field": "median_slope", "label": "Median slope", "type": "number"},
                        {"field": "event_support", "label": "LFP event support", "type": "percent"},
                    ],
                }
            ],
            "sources": sources,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated,
            "status": "ready",
            "datasets": {"primary_comparison": chart_rows, "sensitivity_results": table_rows},
        },
        "sources": sources,
    }
    (OUT / "artifact.json").write_text(json.dumps(artifact, indent=2) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    events = load_events()
    rng = np.random.default_rng(20260910)
    sensitivities = [
        "all_strict",
        "plausibility_restricted",
        "lattice_node_5um",
        "plausibility_lattice_node_5um",
        "min3_events_per_bin",
    ]
    summary_rows, family_rows, level_rows, pop_rows, primary_event_rows = [], [], [], [], []

    evidence = []
    for window, (start, stop) in WINDOWS.items():
        w = events.loc[events.time_s.ge(start) & events.time_s.lt(stop)]
        for (family_id, unit_id, ev), g in w.groupby(["family_id", "unit_id", "evidence"]):
            evidence.append(
                {
                    "window": window,
                    "family_id": int(family_id),
                    "unit_id": int(unit_id),
                    "evidence": ev,
                    "events": int(len(g)),
                    "occupied_5s_bins": int(np.floor((g.time_s - start) / BIN_S).nunique()),
                    "median_lattice_phase_um": float(g.lattice_phase_um.median()),
                }
            )

        for arm in ARMS:
            trace = load_lfp(window, arm)
            trace_in_window = trace.loc[trace.time_s.ge(start) & trace.time_s.lt(stop)]
            trace_ref_mask = (
                trace_in_window.time_s.lt(start + 10)
                & trace_in_window.supported
                & ~trace_in_window.invalid
                & trace_in_window.lfp_um.notna()
            )
            lfp_ref = float(trace_in_window.loc[trace_ref_mask, "lfp_um"].median())
            sampled = sample_lfp_at_events(w, trace)
            if arm == PRIMARY_ARM:
                keep = [
                    "window", "arm", "family_id", "unit_id", "time_s", "evidence",
                    "relative_um", "lattice_phase_um", "lfp_sample_time_s",
                    "lfp_time_error_s", "lfp_event_supported", "lfp_event_um",
                ]
                e = sampled.copy()
                e["window"] = window
                e["arm"] = arm
                primary_event_rows.append(e[keep])
            for sensitivity in sensitivities:
                ev, min_events = sensitivity_events(sampled, sensitivity)
                lh = bin_lighthouse(ev, start, stop, min_events=min_events)
                level = build_level_tracks(lh, lfp_ref)
                level["window"] = window
                level["arm"] = arm
                level["sensitivity"] = sensitivity
                level_rows.append(level)
                inc = increments(level)
                fm = family_metrics(inc)
                if not fm.empty:
                    fm["window"] = window
                    fm["arm"] = arm
                    fm["sensitivity"] = sensitivity
                    family_rows.append(fm)
                summary = equal_family_summary(fm, rng)
                cons, pop = population_metrics(level)
                cons["window"] = window
                cons["arm"] = arm
                cons["sensitivity"] = sensitivity
                pop_rows.append(cons)
                summary_rows.append(
                    {
                        "window": window,
                        "arm": arm,
                        "sensitivity": sensitivity,
                        "strict_events": int((ev.evidence == "strict_accepted").sum()),
                        "families_with_any_strict": int(lh.family_id.nunique()),
                        "occupied_family_bins": int(len(lh)),
                        "median_events_per_occupied_family_bin": float(lh.event_count.median()) if len(lh) else np.nan,
                        "lfp_median_supported_fraction": float(
                            ev.loc[ev.evidence.eq("strict_accepted"), "lfp_event_supported"].mean()
                        ),
                        **summary,
                        **pop,
                    }
                )

    summaries = pd.DataFrame(summary_rows)
    families = pd.concat(family_rows, ignore_index=True) if family_rows else pd.DataFrame()
    levels = pd.concat(level_rows, ignore_index=True)
    populations = pd.concat(pop_rows, ignore_index=True)
    pd.DataFrame(evidence).to_csv(OUT / "evidence_support.csv", index=False)
    summaries.to_csv(OUT / "arm_summary.csv", index=False)
    families.to_csv(OUT / "family_metrics.csv", index=False)
    levels.to_csv(OUT / "binned_family_tracks.csv", index=False)
    populations.to_csv(OUT / "population_tracks.csv", index=False)
    pd.concat(primary_event_rows, ignore_index=True).to_csv(OUT / "primary_arm_event_evaluations.csv.gz", index=False)

    make_figures(levels, summaries, families)
    build_artifact(summaries)

    primary = summaries.loc[(summaries.arm == PRIMARY_ARM) & (summaries.sensitivity == "all_strict")]
    result = {
        "status": "complete",
        "scope": {
            "tested_windows": list(WINDOWS),
            "untested_lfp_windows": ["5000_5100", "6000_6100"],
            "arms": ARMS,
            "identity_families": 15,
            "family_11_members": [557, 673, 675],
        },
        "comparison": {
            "bin_s": BIN_S,
            "quantity": "consecutive-bin displacement increments",
            "residual_sign": "lighthouse minus LFP",
            "family_weighting": "equal family vote after within-family scoring",
            "baseline": "zero motion",
            "lfp_support": "supported and non-invalid samples only",
        },
        "primary_arm": primary.to_dict(orient="records"),
        "predeclared_concern_units": sorted(PREDECLARED_CONCERN_UNITS),
        "random_seed": 20260910,
    }
    (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
