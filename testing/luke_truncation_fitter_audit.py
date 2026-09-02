"""Audit the amplitude-truncation fitter and the cohort comparison it feeds.

Follow-up items 1-3 of docs/decisions/0008-amplitude-completeness-gates-promotion.md
require validating the truncation estimator before any numeric gate is set on it.
This script does that against the three stored Luke0804 imec0 analyses, and tests
whether the reported rescue completeness deficit survives controlling for which
units each configuration admits.

Outputs go to testing/outputs/luke_truncation_fitter_audit/ (gitignored, local).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/mnt/NPX/Luke/20250804")
OUTPUT = REPO_ROOT / "testing/outputs/luke_truncation_fitter_audit"

METHODS = {
    "rescue": (
        DATA_ROOT / "rescue_pipeline_results_Luke0804_V2V1_g0_imec0",
        "cur/cur_output",
    ),
    "legacy": (
        DATA_ROOT / "pipeline_results_Luke0804_V2V1_g0_imec0",
        "cur/cur_sorter_output",
    ),
    "claim_mask": (
        DATA_ROOT / "patched_pipeline_results_Luke0804_V2V1_g0_imec0",
        "cur/cur_sorter_output",
    ),
}

DURATION_S = 10473.55
CEILING = 50.0


def load_units(base: Path, cur_rel: str) -> pd.DataFrame:
    """Per-unit label, spike count, rate, and amplitude scale from the curated sort."""
    cur = base / cur_rel
    clusters = np.load(cur / "spike_clusters.npy").reshape(-1)
    amps = np.load(cur / "amplitudes.npy").reshape(-1)
    labels = pd.read_csv(cur / "cluster_KSLabel.tsv", sep="\t")
    label_col = next(c for c in labels.columns if c != "cluster_id")

    uniq, counts = np.unique(clusters, return_counts=True)
    med_amp = np.array([np.median(amps[clusters == u]) for u in uniq])
    frame = pd.DataFrame(
        {
            "cluster_id": uniq,
            "n_spikes": counts,
            "rate_hz": counts / DURATION_S,
            "median_amplitude": med_amp,
        }
    ).merge(
        labels.rename(columns={label_col: "ks_label"})[["cluster_id", "ks_label"]],
        on="cluster_id",
        how="left",
    )
    frame["is_good"] = frame["ks_label"].astype(str).str.strip().str.lower() == "good"
    return frame


def load_truncation(base: Path) -> pd.DataFrame:
    """Per-window fit results, with x_min recovered by inverting the reported mpct."""
    d = np.load(base / "qc/amp_truncation/truncation_qc.npz")
    frame = pd.DataFrame(
        {
            "cluster_id": d["cid"].astype(int),
            "mpct": d["mpcts"],
            "x0": d["popts"][:, 0],
            "k": d["popts"][:, 1],
            "A": d["popts"][:, 2],
        }
    )
    frame["on_ceiling"] = np.isclose(frame["mpct"], CEILING)
    # mpct = 100 * sigmoid(x_min; x0, k)  ->  x_min = x0 + log(m/(100-m))/k
    with np.errstate(divide="ignore", invalid="ignore"):
        frame["x_min"] = frame["x0"] + np.log(
            frame["mpct"] / (100 - frame["mpct"])
        ) / frame["k"]
    frame.loc[frame["on_ceiling"], "x_min"] = frame.loc[frame["on_ceiling"], "x0"]
    # Independent estimate implied by the renormalisation parameter A:
    # for a truncated sigmoid CDF, A = 1/(1-F(x_min)) so F(x_min) = 1 - 1/A.
    frame["mpct_from_A"] = 100 * (1 - 1 / frame["A"].clip(lower=1e-12))
    return frame


def per_unit(trunc: pd.DataFrame) -> pd.DataFrame:
    grouped = trunc.groupby("cluster_id")
    return pd.DataFrame(
        {
            "median_mpct": grouped["mpct"].median(),
            "median_mpct_from_A": grouped["mpct_from_A"].median(),
            "n_windows": grouped.size(),
            "frac_ceiling": grouped["on_ceiling"].mean(),
            "median_mpct_off_ceiling": grouped.apply(
                lambda g: g.loc[~g["on_ceiling"], "mpct"].median(), include_groups=False
            ),
        }
    ).reset_index()


def build() -> dict[str, pd.DataFrame]:
    tables = {}
    for name, (base, cur_rel) in METHODS.items():
        units = load_units(base, cur_rel)
        trunc = load_truncation(base)
        merged = units.merge(per_unit(trunc), on="cluster_id", how="left")
        merged["eligible"] = merged["n_windows"].notna()
        merged["method"] = name
        tables[name] = merged
    return tables


def cohort_summary(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    cohorts = {
        "all eligible good": lambda f: f["is_good"] & f["eligible"],
        "good >1 Hz": lambda f: f["is_good"] & f["eligible"] & (f["rate_hz"] > 1),
        "good 1-5 Hz": lambda f: f["is_good"] & f["eligible"] & (f["rate_hz"] > 1) & (f["rate_hz"] <= 5),
        "good >5 Hz": lambda f: f["is_good"] & f["eligible"] & (f["rate_hz"] > 5),
        "all eligible units": lambda f: f["eligible"],
    }
    rows = []
    for cohort, mask_fn in cohorts.items():
        for method, frame in tables.items():
            sub = frame[mask_fn(frame)]
            if sub.empty:
                continue
            rows.append(
                {
                    "cohort": cohort,
                    "method": method,
                    "n_units": len(sub),
                    "median_mpct": sub["median_mpct"].median(),
                    "pct_below_10": 100 * (sub["median_mpct"] < 10).mean(),
                    "median_frac_ceiling": sub["frac_ceiling"].median(),
                    "median_mpct_off_ceiling": sub["median_mpct_off_ceiling"].median(),
                    "median_rate_hz": sub["rate_hz"].median(),
                    "median_amplitude": sub["median_amplitude"].median(),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    tables = build()
    units = pd.concat(tables.values(), ignore_index=True)
    units.to_csv(OUTPUT / "unit_truncation.csv", index=False)
    summary = cohort_summary(tables)
    summary.to_csv(OUTPUT / "cohort_summary.csv", index=False)

    pd.set_option("display.width", 200, "display.max_columns", 50)
    for cohort in summary["cohort"].unique():
        print(f"\n### {cohort}")
        print(
            summary[summary["cohort"] == cohort]
            .drop(columns="cohort")
            .to_string(index=False, float_format=lambda v: f"{v:.3g}")
        )
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary.to_dict(orient="records"), indent=2) + "\n"
    )
    print(f"\nwrote {OUTPUT}")


if __name__ == "__main__":
    main()
