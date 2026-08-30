"""Consolidate Luke conditioning audits into a provisional pipeline decision."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


TWO_BY_TWO = Path(
    "/media/huklab/Data/NPX/Ryansorting/Luke/"
    "Luke0804_conditioning_2x2_good_imec1/conditioning_2x2_scores.csv"
)
HARDER = Path(
    "/media/huklab/Data/NPX/Ryansorting/Luke/"
    "Luke0804_conditioning_harder_gate_imec1/harder_window_scores.csv"
)
ARTIFACT = Path(
    "testing/outputs/luke_kilosort_artifact_threshold_audit/threshold_summary.csv"
)
DEFAULT_OUTPUT = Path("testing/outputs/luke_conditioning_final_decision/decision.json")


def build_decision(
    factorial: pd.DataFrame, harder: pd.DataFrame, artifact: pd.DataFrame
) -> dict:
    factor = factorial.set_index("cell")
    hard = harder.set_index(["window", "condition"])
    recovery = {
        window: {
            condition: float(hard.loc[(window, condition), "neural_unmatched_recovery"])
            for condition in ("legacy_blank_interpolate", "unchanged_interpolate")
        }
        for window in ("neutral_template", "pathological")
    }
    artifact_300 = artifact[artifact.artifact_threshold_counts == 300].set_index(
        "window"
    )
    return {
        "decision": "retain_legacy_conditioning_provisionally_with_artifact_sidecar",
        "selected_upstream_sequence": [
            "Neuropixels phase correction",
            "samplewise bilateral blanking at 500 uV",
            "interpolate physical channel 191 and include it in Kilosort",
            "single internal Kilosort CAR/high-pass/whitening pass",
        ],
        "sorter_settings": {
            "external_motion_correction": False,
            "kilosort_internal_motion_correction": False,
            "cross_template_claim_mask": "off",
            "kilosort_artifact_threshold": "infinite/disabled",
            "bad_channels": None,
        },
        "required_sidecar": (
            "Preserve raw over-500-uV sample/channel intervals and exclude nearby sorter "
            "detections from artifact-sensitive claims; do not treat blank-induced peaks as neural."
        ),
        "reviewed_neural_recovery": recovery,
        "good_window_channel_191_evidence": {
            "interpolate_include_ks_good": int(
                factor.loc["blank_interpolate_include", "n_ks_good"]
            ),
            "exclude_ks_good": int(factor.loc["blank_exclude191", "n_ks_good"]),
            "interpolate_include_median_contamination_pct": float(
                factor.loc["blank_interpolate_include", "median_contamination_pct"]
            ),
            "exclude_median_contamination_pct": float(
                factor.loc["blank_exclude191", "median_contamination_pct"]
            ),
        },
        "native_artifact_threshold_rejected": {
            window: {
                "threshold_counts": 300,
                "rejected_batch_fraction": float(
                    artifact_300.loc[window, "rejected_batch_fraction"]
                ),
                "reviewed_neural_erased_fraction": float(
                    artifact_300.loc[window, "reviewed_neural_erased_fraction"]
                ),
            }
            for window in ("neutral_template", "pathological")
        },
        "known_cost": (
            "The 500-uV samplewise blanker can create local ringing/false peaks and worsens "
            "event density in saturation-enriched seconds. It is retained because removing it "
            "causes much larger sorter-level reviewed-neural recovery losses."
        ),
        "positive_polarity_excess": {
            "status": "deferred_but_likely_coupled_to_required_blanking",
            "return_to": (
                "Measure the same raw-voltage polarity and over-500-uV rates in multiple Luke "
                "sessions, both imec streams, Yates controls, and another acquisition rig."
            ),
            "decision_use_now": (
                "Do not use positive-polarity event density for biological claims and retain "
                "the artifact sidecar exclusion."
            ),
        },
        "advancement": (
            "Conditioning is fixed provisionally. Continue only with motion disabled until a "
            "matched no-motion baseline using this exact sequence is established."
        ),
    }


def main() -> None:
    decision = build_decision(
        pd.read_csv(TWO_BY_TWO), pd.read_csv(HARDER), pd.read_csv(ARTIFACT)
    )
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT.write_text(json.dumps(decision, indent=2) + "\n")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
