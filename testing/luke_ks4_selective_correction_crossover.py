"""Summarize displacement-specific KS4 correction crossover evidence.

This is a post hoc stratification of the completed discovery operator audit. It
does not alter the preregistered all-displacement decision and cannot authorize
a supplied-trajectory sort.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_ROOT = Path("testing/outputs/luke_ks4_native_operator_audit")
PAIR_KEYS = [
    "template_id",
    "background",
    "motion_class",
    "generator",
    "displacement_um",
    "displacement_sign",
    "edge_status",
]
THRESHOLDS = {
    "median_delta_residual_max": -0.005,
    "median_delta_absolute_amplitude_error_max": 0.005,
    "median_delta_cosine_min": 0.0,
}


def paired_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    """Pair native and uncorrected moved cases and compute signed deltas."""
    native = metrics.loc[metrics.arm.eq("moved_ks4_native_inverse")].copy()
    control = metrics.loc[metrics.arm.eq("moved_no_correction")].copy()
    if native.duplicated(PAIR_KEYS).any() or control.duplicated(PAIR_KEYS).any():
        raise ValueError("operator case keys are not unique")

    paired = native.merge(
        control,
        on=PAIR_KEYS,
        suffixes=("_native", "_uncorrected"),
        validate="one_to_one",
    )
    if len(paired) != len(native) or len(paired) != len(control):
        raise ValueError("native and uncorrected cases do not match exactly")

    paired["abs_displacement_um"] = paired.displacement_um.abs()
    paired["delta_residual"] = (
        paired.residual_fraction_native - paired.residual_fraction_uncorrected
    )
    paired["delta_absolute_amplitude_error"] = (
        (paired.amplitude_retention_native - 1.0).abs()
        - (paired.amplitude_retention_uncorrected - 1.0).abs()
    )
    paired["delta_cosine"] = (
        paired.template_cosine_native - paired.template_cosine_uncorrected
    )
    return paired


def _add_pass_columns(summary: pd.DataFrame) -> pd.DataFrame:
    result = summary.copy()
    result["residual_pass"] = (
        result.median_delta_residual <= THRESHOLDS["median_delta_residual_max"]
    )
    result["amplitude_pass"] = (
        result.median_delta_absolute_amplitude_error
        <= THRESHOLDS["median_delta_absolute_amplitude_error_max"]
    )
    result["cosine_pass"] = (
        result.median_delta_cosine >= THRESHOLDS["median_delta_cosine_min"]
    )
    result["complete_pass"] = result[
        ["residual_pass", "amplitude_pass", "cosine_pass"]
    ].all(axis=1)
    return result


def summarize(paired: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return signed strata and worst-generator/sign summaries by magnitude."""
    signed = (
        paired.groupby(
            ["generator", "abs_displacement_um", "displacement_sign"],
            as_index=False,
        )
        .agg(
            cases=("delta_residual", "size"),
            median_delta_residual=("delta_residual", "median"),
            median_delta_absolute_amplitude_error=(
                "delta_absolute_amplitude_error",
                "median",
            ),
            median_delta_cosine=("delta_cosine", "median"),
        )
        .sort_values(["abs_displacement_um", "generator", "displacement_sign"])
    )
    signed = _add_pass_columns(signed)

    worst = (
        signed.groupby("abs_displacement_um", as_index=False)
        .agg(
            generator_sign_strata=("complete_pass", "size"),
            worst_delta_residual=("median_delta_residual", "max"),
            worst_delta_absolute_amplitude_error=(
                "median_delta_absolute_amplitude_error",
                "max",
            ),
            worst_delta_cosine=("median_delta_cosine", "min"),
            residual_all_strata_pass=("residual_pass", "all"),
            amplitude_all_strata_pass=("amplitude_pass", "all"),
            cosine_all_strata_pass=("cosine_pass", "all"),
            complete_all_strata_pass=("complete_pass", "all"),
        )
        .sort_values("abs_displacement_um")
    )
    return signed, worst


def crossover_result(worst: pd.DataFrame) -> dict:
    complete = worst.loc[worst.complete_all_strata_pass, "abs_displacement_um"]
    residual_cosine = worst.loc[
        worst.residual_all_strata_pass & worst.cosine_all_strata_pass,
        "abs_displacement_um",
    ]
    return {
        "schema_version": "luke-ks4-selective-correction-crossover-v1",
        "analysis_role": "post_hoc_discovery_stratification",
        "changes_original_operator_decision": False,
        "thresholds": THRESHOLDS,
        "tested_displacement_magnitudes_um": worst.abs_displacement_um.tolist(),
        "complete_crossover_um": None if complete.empty else float(complete.min()),
        "residual_and_cosine_crossover_um": (
            None if residual_cosine.empty else float(residual_cosine.min())
        ),
        "selective_policy_authorized": False,
        "interpretation": (
            "No tested magnitude passed residual, amplitude, and cosine in every "
            "forward-generator/sign stratum. Residual and cosine first passed "
            "together at 20 um, but amplitude preservation still failed. A refined "
            "confirmatory grid and exact-identity switching audit are required."
        ),
        "prospective_holdout_accessed": False,
        "sorter_run": False,
    }


def run(root: Path = DEFAULT_ROOT) -> dict:
    metrics = pd.read_csv(root / "case_metrics.csv")
    paired = paired_deltas(metrics)
    signed, worst = summarize(paired)
    result = crossover_result(worst)

    output = root / "selective_correction"
    output.mkdir(parents=True, exist_ok=True)
    paired.to_csv(output / "paired_case_deltas.csv", index=False)
    signed.to_csv(output / "signed_crossover_summary.csv", index=False)
    worst.to_csv(output / "worst_stratum_crossover_summary.csv", index=False)
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args().root), indent=2))


if __name__ == "__main__":
    main()
