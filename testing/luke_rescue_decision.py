"""Consolidate the current evidence-based Luke rescue policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_OUTPUT = Path("testing/outputs/luke_rescue_decision")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_json(path: str) -> dict:
    return json.loads(Path(path).read_text())


def run(output_dir: Path) -> dict:
    claim = read_json("testing/outputs/luke_claimmask_rescue_audit/decision.json")
    preprocessing = read_json(
        "testing/outputs/luke_preprocessing_rescue_decision/decision.json"
    )
    channel = read_json("testing/outputs/luke_channel_artifact_audit/decision.json")
    interpolation = read_json(
        "testing/outputs/luke_bad_channel_interpolation_audit/decision.json"
    )
    motion = pd.read_csv(
        "testing/outputs/luke_motion_candidate_results/motion_candidate_sorter_metrics.csv"
    )
    motion = motion.loc[motion["population"] == "visual_neural_unmatched"]
    motion_index = motion.set_index("condition")
    interpolation_impl = pd.read_csv(
        "testing/outputs/luke_interpolation_implementation_audit/paired_event_metrics.csv"
    )
    interpolation_impl = interpolation_impl.loc[
        interpolation_impl["review_label"] == "neural"
    ]
    implementation_index = interpolation_impl.groupby("variant").median(numeric_only=True)
    replication = pd.read_csv(
        "/mnt/NPX/Luke/20250804/"
        "dredge_pipeline_results_Luke0804_V2V1_g0_imec1/"
        "motion_candidate_replication/shared_template/replication_scores.csv"
    )
    replication = replication.loc[
        replication["population"] == "visual_neural_unmatched"
    ].set_index("condition")
    required_motion = {
        "No external correction",
        "Current DREDGE 150/100",
        "Selected DREDGE 300/200",
        "MEDiCINe default, sigma 10",
    }
    if not required_motion.issubset(set(motion_index.index)):
        raise ValueError("Motion evidence is incomplete")
    if not claim["decision"].startswith("reject_current_claim_mask"):
        raise ValueError("Claim-mask decision changed")
    if not preprocessing["decision"].endswith("not_production"):
        raise ValueError("Preprocessing decision changed")

    evidence_rows = [
        {
            "domain": "Raw voltage",
            "status": "Resolved diagnostic",
            "decision": "No global threshold reduction",
            "key_evidence": "Luke has excess shared high-frequency voltage; imec1 retains a 3.51–5.31 positive/negative ratio after three reference controls.",
        },
        {
            "domain": "External motion",
            "status": "Implementation diagnosed; no incremental Luke benefit",
            "decision": (
                "Disable external voltage resampling in the rescue candidate"
            ),
            "key_evidence": (
                f"No external correction recovered {motion_index.loc['No external correction', 'observed_recovery']:.1%}; "
                f"current DREDGE {motion_index.loc['Current DREDGE 150/100', 'observed_recovery']:.1%}, "
                f"selected DREDGE {motion_index.loc['Selected DREDGE 300/200', 'observed_recovery']:.1%}, and "
                f"MEDiCINe {motion_index.loc['MEDiCINe default, sigma 10', 'observed_recovery']:.1%}. "
                f"However, all external sorts shared an uncalibrated p=1 interpolation path. "
                f"On 128 matched neural events, calibrated p=2 retained median amplitude "
                f"{implementation_index.loc['official_p2_extrapolate_int16', 'ratio_anchor_peak_amplitude_counts']:.1%} "
                f"versus {implementation_index.loc['pipeline_p1_zero_int16', 'ratio_anchor_peak_amplitude_counts']:.1%} "
                f"and local RMS {implementation_index.loc['official_p2_extrapolate_int16', 'ratio_local_snippet_rms_counts']:.1%} "
                f"versus {implementation_index.loc['pipeline_p1_zero_int16', 'ratio_local_snippet_rms_counts']:.1%}. "
                f"Sorter validation showed conservative p=2 was safe but neutral: in the "
                f"240-s replication it recovered "
                f"{replication.loc['rigid_gain_025_p2', 'observed_recovery']:.1%}, identical to "
                f"no-motion {replication.loc['no_external_correction', 'observed_recovery']:.1%}. "
                f"Adding it to single-pass preprocessing recovered "
                f"{replication.loc['single_ks_preprocessing_rigid_gain_025_p2', 'observed_recovery']:.1%}, "
                f"identical to single-pass alone "
                f"{replication.loc['single_ks_preprocessing', 'observed_recovery']:.1%}; "
                f"the 400/400 nonrigid p=2 control fell to "
                f"{replication.loc['single_ks_preprocessing_dredge_400_400_p2', 'observed_recovery']:.1%}."
            ),
        },
        {
            "domain": "Preprocessing",
            "status": "Advance to validation",
            "decision": "Run a full-session single-pass candidate",
            "key_evidence": "Effective KS-good families increased 123→163 in 240 s and 87→108 in 120 s, robust across agreement thresholds; event recovery was unchanged.",
        },
        {
            "domain": "Bad channel 191",
            "status": "Retain with guardrail",
            "decision": "Interpolate, but exclude the synthetic row from claim metrics",
            "key_evidence": "Interpolation preserved exact peak amplitude for 96% of 200 reviewed events and redirected all six false channel-191 peak localizations.",
        },
        {
            "domain": "Claim mask",
            "status": "Reject",
            "decision": "Keep claim mask disabled",
            "key_evidence": "Every nonzero tested mask reduced reviewed-neural recovery; the production mask lost 25.9–48.6 percentage points in matched windows.",
        },
        {
            "domain": "Biological Luke–Yates claim",
            "status": "Not supportable",
            "decision": "Do not claim lower Luke neural density",
            "key_evidence": "Event density is reference-, polarity-, probe-, and layer-dependent; normalized depth is not matched cortical layer.",
        },
    ]
    evidence = pd.DataFrame(evidence_rows)
    decision = {
        "overall_decision": "luke_is_plausibly_partially_rescuable_full_session_validation_required",
        "best_supported_candidate": {
            "conditioning": "single Kilosort preprocessing",
            "external_voltage_motion_correction": "disabled",
            "motion_use": (
                "calibrated p=2 is technically viable, but no Luke field passed the "
                "incremental-benefit gate; retain motion for diagnostics or downstream tracking"
            ),
            "bad_channel_191": "interpolate but exclude synthetic row from claim metrics",
            "claim_mask": "disabled",
        },
        "production_status": "not_ready",
        "required_gate": (
            "Run a matched full-session single-pass validation with external voltage motion "
            "correction and the claim mask disabled. It must preserve reviewed-event recovery "
            "and improve stable unit-family yield without increasing duplicates, contamination, "
            "or refractory violations."
        ),
        "operational_constraint": (
            "CUDA validation is available outside the sandbox. The remaining full-session "
            "candidate will require substantially more storage and runtime than the completed "
            "120-s and 240-s gates."
        ),
        "evidence_domains": evidence_rows,
        "claim_mask_decision": claim["decision"],
        "preprocessing_decision": preprocessing["decision"],
        "channel_decision": channel["decision"],
        "interpolation_decision": interpolation["decision"],
        "motion_interpolation_implementation_audit": {
            "n_reviewed_neural_events": int(
                interpolation_impl["review_id"].nunique()
            ),
            "current_p1_median_amplitude_retention": float(
                implementation_index.loc[
                    "pipeline_p1_zero_int16", "ratio_anchor_peak_amplitude_counts"
                ]
            ),
            "official_p2_median_amplitude_retention": float(
                implementation_index.loc[
                    "official_p2_extrapolate_int16",
                    "ratio_anchor_peak_amplitude_counts",
                ]
            ),
            "sorter_conclusion": (
                "p2 fixes the p1 attenuation/explosion phenotype, but conservative motion is "
                "neural-recovery neutral and nonrigid motion is harmful across tested scales"
            ),
            "replication_single_pass_recovery": float(
                replication.loc["single_ks_preprocessing", "observed_recovery"]
            ),
            "replication_single_pass_plus_rigid_p2_recovery": float(
                replication.loc[
                    "single_ks_preprocessing_rigid_gain_025_p2", "observed_recovery"
                ]
            ),
            "replication_single_pass_plus_400_400_p2_recovery": float(
                replication.loc[
                    "single_ks_preprocessing_dredge_400_400_p2", "observed_recovery"
                ]
            ),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence.to_csv(output_dir / "evidence_matrix.csv", index=False)
    (output_dir / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    return decision


def main() -> None:
    args = parse_args()
    print(json.dumps(run(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
