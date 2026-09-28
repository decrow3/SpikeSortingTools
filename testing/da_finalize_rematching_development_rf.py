"""Finalize the DA saved-output development RF comparison packet."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    args = parser.parse_args()
    packet = args.packet.resolve()
    run = packet / "run"
    summary = json.loads((run / "SUMMARY.json").read_text())
    expected_hashes = {
        "arm:D2L": "03c339b4af8081e7e09bcbf23ebb2ae439b4cc8d468538d352cb8e79b61fc0b0",
        "arm:REMATCH0": "82210d21996cee55aeef6c1bf73d23f0e3861cac867f404a2af96bbfb4024882",
        "arm:CD1_FULL": "ba69b03c4c62d57acd5b51407ffcc63c1f078ba2ab4bc6b96932fe56e4f3958d",
        "config": "e23297fb79ec4aca9150a42c4611c39f8fd33ff23647f4170514581d11e19d15",
        "gaze_csv": "3a72672721ad83ad4810e764ea365d268794bafd2ceb65c955bc51f8c3e2e2a2",
        "stimulus_frames": "92b460154266ba0b8aaf14eecc22babc8a9b432a1b5fd4286dcd177b5342bfbe",
        "stimulus_manifest": "d5241af5de121eba5b5354b1890a83b57f156f52dc9664195a7cee56df9730f2",
        "trial_split": "399858bbd6dbd689d43e5b333c24f6e850910f5f5ff3b18709477747a9e1a18c",
        "trial_table": "6a950b31bff1031456b394271d88af7796fbf7d38d526168ba504a7d147b62f0",
    }
    hash_checks = {key: summary["hashes"].get(key) == value for key, value in expected_hashes.items()}
    support_checks = {
        "source_frames_exact": summary["support"]["source_frame_interval_half_open"] == [26999783, 37199701],
        "development_trials_12": len(summary["support"]["development_trial_ordinals"]) == 12,
        "inner_folds_7_5": summary["support"]["development_inner_fold_counts"] == {"0": 7, "1": 5},
        "valid_frames_6052": summary["support"]["development_valid_frames"] == 6052,
        "outer_holdout_unopened": summary["support"]["outer_holdout_opened"] is False,
        "complete_holdout_5": len(summary["support"]["outer_holdout_trial_ordinals_not_evaluated"]) == 5,
    }
    if not all(hash_checks.values()) or not all(support_checks.values()):
        raise ValueError({"hash_checks": hash_checks, "support_checks": support_checks})

    score_rows = []
    failure_rows = []
    for arm in ("D2L", "REMATCH0", "CD1_FULL"):
        eligibility = pd.read_csv(run / f"{arm}_rf_eligibility.csv")
        metrics = pd.read_csv(run / f"{arm}_unit_rf_metrics.csv")
        values = metrics.cv_snr.to_numpy(float)
        eligible = eligibility.eligible.astype(bool).to_numpy()
        total_fail = eligibility.n_spikes.to_numpy() < 500
        fold0_fail = eligibility.n_spikes_fold0.to_numpy() < 200
        fold1_fail = eligibility.n_spikes_fold1.to_numpy() < 200
        reason = np.full(len(eligibility), "eligible", dtype=object)
        reason[~eligible] = "multiple_or_fold_threshold_failure"
        reason[~eligible & total_fail] = "total_spikes_lt_500"
        reason[~eligible & ~total_fail & fold0_fail & ~fold1_fail] = "fold0_spikes_lt_200_only"
        reason[~eligible & ~total_fail & ~fold0_fail & fold1_fail] = "fold1_spikes_lt_200_only"
        reason[~eligible & ~total_fail & fold0_fail & fold1_fail] = "both_folds_lt_200"
        for name in sorted(set(reason)):
            failure_rows.append({"arm": arm, "exclusive_status": name, "n_units": int(np.sum(reason == name))})
        score_rows.append(
            {
                "arm": arm,
                "n_units_total": int(len(eligibility)),
                "n_units_eligible": int(eligible.sum()),
                "eligible_fraction": float(eligible.mean()),
                "n_total_spikes_lt_500": int(total_fail.sum()),
                "n_fold0_spikes_lt_200": int(fold0_fail.sum()),
                "n_fold1_spikes_lt_200": int(fold1_fail.sum()),
                "cv_snr_median": float(np.median(values)),
                "cv_snr_mean": float(np.mean(values)),
                "cv_snr_sum_all_eligible": float(values.sum()),
                "n_cv_snr_positive": int(np.sum(values > 0)),
                "cv_snr_q25": float(np.quantile(values, 0.25)),
                "cv_snr_q75": float(np.quantile(values, 0.75)),
            }
        )
    scores = pd.DataFrame(score_rows)
    scores.to_csv(packet / "DA_SCORES.csv", index=False)
    pd.DataFrame(failure_rows).to_csv(packet / "ELIGIBILITY_FAILURES.csv", index=False)

    # Compact decision figure: population coverage and the full eligible-score
    # distributions. Color is redundant with hatch/marker shape and labels.
    arms = ["D2L", "REMATCH0", "CD1_FULL"]
    colors = ["#0072B2", "#E69F00", "#D55E00"]
    markers = ["o", "s", "^"]
    hatches = ["//", "..", "xx"]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), constrained_layout=True)
    coverage = [row["n_units_eligible"] / row["n_units_total"] for row in score_rows]
    bars = axes[0].bar(arms, coverage, color=colors, edgecolor="#222222", linewidth=0.8)
    for bar, hatch, row in zip(bars, hatches, score_rows):
        bar.set_hatch(hatch)
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                     f"{row['n_units_eligible']}/{row['n_units_total']}", ha="center", va="bottom", fontsize=9)
    axes[0].set_ylim(0, max(coverage) * 1.25)
    axes[0].set_ylabel("Eligible fraction")
    axes[0].set_title("Frozen RF eligibility")
    axes[0].grid(axis="y", color="#D9D9D9", linewidth=0.7)
    rng = np.random.default_rng(20260928)
    for x, (arm, color, marker) in enumerate(zip(arms, colors, markers), start=1):
        values = pd.read_csv(run / f"{arm}_unit_rf_metrics.csv").cv_snr.to_numpy(float)
        jitter = rng.uniform(-0.12, 0.12, size=values.size)
        axes[1].scatter(x + jitter, values, s=14, marker=marker, facecolor="none",
                        edgecolor=color, linewidth=0.8, alpha=0.75)
        med = float(np.median(values))
        axes[1].plot([x - 0.22, x + 0.22], [med, med], color="#222222", linewidth=2.2)
        axes[1].text(x, 0.97, f"median {med:.2f}", transform=axes[1].get_xaxis_transform(),
                     ha="center", va="top", fontsize=8,
                     bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.5})
    axes[1].axhline(0, color="#666666", linestyle="--", linewidth=0.9)
    axes[1].set_xticks([1, 2, 3], arms)
    axes[1].set_ylabel("Cross-validated RF SNR")
    axes[1].set_title("All eligible units (unmatched populations)")
    axes[1].grid(axis="y", color="#D9D9D9", linewidth=0.7)
    fig.suptitle("W2 development-only RF comparison", fontsize=13)
    fig.savefig(packet / "DA_RF_COMPARISON.png", dpi=180, facecolor="white")
    plt.close(fig)

    existing_static = {
        "role": "optional contextual comparator; reused, not rerun",
        "source": "testing/outputs/cp_h1_independent_outcome_review_v1/rf_w2_development/SUMMARY.json",
        "same_frozen_W2_stimulus_gaze_and_trial_support": True,
        "arm_final_hash": "be6106ed0cb0f99759fd23629dd3bcb5f50657dbffa238ad3721c15d1b21fba7",
        "n_units_total": 538,
        "n_units_eligible": 67,
        "cv_snr_median": 0.38762410122014607,
        "cv_snr_mean": 1.2703160602818409,
    }
    write_json(packet / "EXISTING_STATIC_CONTEXT.json", existing_static)
    validation = {
        "status": "pass",
        "hash_checks": hash_checks,
        "support_checks": support_checks,
        "clock": summary["clock"],
        "gaze_calibration_checks": summary["gaze_calibration"]["checks"],
        "new_label_namespaces_not_joined": True,
        "paired_summary_attempted": False,
        "interpretation": "development-only total-pipeline functional comparison; adaptive localization differs across rematching arms",
    }
    write_json(packet / "VALIDATION.json", validation)

    by_arm = {row["arm"]: row for row in score_rows}
    report = f"""# DA W2 development RF result

## Verdict first

The original accepted **D2L** final has the strongest descriptive result of the
three saved W2 outputs on this frozen development-only RF screen. It retains
more eligible units (80) and has higher aggregate and mean cross-validated SNR
than either fresh rematching result. This is not a pipeline winner based on
eligibility alone: the arm populations are unmatched and selection changes the
denominator. **CD1_FULL does not show a functional improvement over REMATCH0**:
it has fewer eligible units (68 versus 75), a slightly lower median (0.221 versus 0.235),
and a lower all-eligible score sum (45.66 versus 52.96). This provides no
functional reason to add the coordinate-descent round.

| Arm | All units | Eligible | Coverage | Median cvSNR | Mean cvSNR | Sum, all eligible | Positive |
|---|---:|---:|---:|---:|---:|---:|---:|
| D2L | {by_arm['D2L']['n_units_total']} | {by_arm['D2L']['n_units_eligible']} | {by_arm['D2L']['eligible_fraction']:.1%} | {by_arm['D2L']['cv_snr_median']:.3f} | {by_arm['D2L']['cv_snr_mean']:.3f} | {by_arm['D2L']['cv_snr_sum_all_eligible']:.2f} | {by_arm['D2L']['n_cv_snr_positive']} |
| REMATCH0 | {by_arm['REMATCH0']['n_units_total']} | {by_arm['REMATCH0']['n_units_eligible']} | {by_arm['REMATCH0']['eligible_fraction']:.1%} | {by_arm['REMATCH0']['cv_snr_median']:.3f} | {by_arm['REMATCH0']['cv_snr_mean']:.3f} | {by_arm['REMATCH0']['cv_snr_sum_all_eligible']:.2f} | {by_arm['REMATCH0']['n_cv_snr_positive']} |
| CD1_FULL | {by_arm['CD1_FULL']['n_units_total']} | {by_arm['CD1_FULL']['n_units_eligible']} | {by_arm['CD1_FULL']['eligible_fraction']:.1%} | {by_arm['CD1_FULL']['cv_snr_median']:.3f} | {by_arm['CD1_FULL']['cv_snr_mean']:.3f} | {by_arm['CD1_FULL']['cv_snr_sum_all_eligible']:.2f} | {by_arm['CD1_FULL']['n_cv_snr_positive']} |

The existing static same-support context (not rerun) has 67 eligible units,
median cvSNR 0.388 and mean 1.270. It is contextual only and is not mixed into
the three-arm DA table.

## Validation and limits

All frozen config, stimulus, split, gaze and arm hashes pass. The analysis uses
the exact imec1 W2 AP-frame interval `[26999783, 37199701)`, the original
61-development/20-sealed-holdout split, the same 12 complete W2 development
trials, 7/5 inner folds, and 6,052 valid gaze/lag frames. The five complete W2
holdout trials were not evaluated. Eligibility remains 500 total and 200 per
fold; `ELIGIBILITY_FAILURES.csv` reports every unit, including overlapping raw
threshold-failure counts in `DA_SCORES.csv` and mutually exclusive outcomes.

No cross-arm label join was used. Therefore the population summaries compare
saved pipeline outputs, not matched biological neurons, and neither raw ISI nor
eligibility alone selects a winner. TPCA/common waveform bases are exact, but
adaptive localization differs between rematching arms, so the final-output
contrast is total-pipeline mediation. The matching-stage CZ counts remain
descriptive and do not override this functional result.

## Runtime and resources

The successful evaluator took {summary['elapsed_s']:.2f} seconds internally
(30.35 seconds service wall, 30.84 user + 2.73 system CPU seconds, 2.93 GB peak
RSS). The first infrastructure attempt failed before scoring because it used an
environment without `mat73`; its evidence is preserved and the unchanged run
was retried in the existing locked Rowley environment. No sort, raw voltage,
GPU, gaze/lag refit, threshold tuning, holdout evaluation, or count calibration
was performed. Directly metered evaluator CPU across the failed and successful
attempts is 50.38 seconds; setup, validation, and reporting are conservatively
charged 69.62 seconds. The 120-second total DA charge takes H1 cumulative usage
from 17,333.22 to 17,453.22 seconds, below the 20,500-second ceiling.
"""
    (packet / "REPORT.md").write_text(report)
    (packet / "README.md").write_text(
        "# DA rematching development RF packet\n\n"
        "Verdict: the original accepted D2L final has the strongest descriptive result on the frozen W2 "
        "development-only RF screen; CD1_FULL shows no functional improvement "
        "over REMATCH0. See `REPORT.md`, `DA_SCORES.csv`, "
        "`ELIGIBILITY_FAILURES.csv`, and `VALIDATION.json`.\n"
    )
    receipt = {
        "successful_evaluator_elapsed_s": summary["elapsed_s"],
        "successful_service_wall_s": 30.35,
        "successful_user_cpu_s": 30.84,
        "successful_system_cpu_s": 2.73,
        "failed_attempt_user_cpu_s": 15.09,
        "failed_attempt_system_cpu_s": 1.72,
        "peak_rss_gb": 2.927216,
        "raw_voltage_reads": 0,
        "gpu_seconds": 0,
        "sorts": 0,
        "directly_metered_evaluator_cpu_s": 50.38,
        "setup_validation_reporting_conservative_charge_s": 69.62,
        "total_da_charge_s": 120.0,
        "h1_cumulative_before_s": 17333.22,
        "h1_cumulative_after_s": 17453.22,
        "h1_ceiling_s": 20500.0,
    }
    write_json(packet / "RESOURCE_RECEIPT.json", receipt)
    excluded = {"MANIFEST.json", "COMPLETE.json"}
    products = []
    for path in sorted(p for p in packet.rglob("*") if p.is_file() and p.name not in excluded):
        products.append({"path": str(path.relative_to(packet)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    write_json(packet / "MANIFEST.json", {"products": products})
    complete = {
        "status": "complete_development_only",
        "verdict": "D2L strongest descriptive development-RF result; CD1_FULL shows no functional improvement over REMATCH0; eligibility alone does not select a pipeline",
        "holdout_opened": False,
        "hash_and_support_validation": "pass",
        "written_last_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": sha256(packet / "MANIFEST.json"),
    }
    write_json(packet / "COMPLETE.json", complete)


if __name__ == "__main__":
    main()
