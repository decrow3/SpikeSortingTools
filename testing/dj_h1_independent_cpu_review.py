#!/usr/bin/env python3
"""Finite DJ source/endpoint review without rerunning matching or RF fitting."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DA = Path("/mnt/NPX/Luke/DARTsort_motion_experiments/da_rematching_development_rf_20260928")
DG = Path("/mnt/NPX/Luke/DARTsort_motion_experiments/dg_lattice_exploratory_w2_20260928")
CX = ROOT / "testing/outputs/cx_masked_donor_qualification_v1/run"
OUT = ROOT / "testing/outputs/dj_h1_independent_cpu_review_20260928"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(partial, path)


def main() -> None:
    started = time.monotonic()
    cpu0 = time.process_time()
    if OUT.exists():
        raise FileExistsError(OUT)
    if os.statvfs(OUT.parent).f_bavail * os.statvfs(OUT.parent).f_frsize < 30_000_000_000:
        raise RuntimeError("30 GB free-space gate")
    OUT.mkdir(parents=True)
    write_json(OUT / "START_RECEIPT.json", {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "cpu_allwork_cap_s": 300,
        "threads_max": 2,
        "readers_max": 1,
        "memory_cap_gb": 20,
        "final_cap_bytes": 500_000_000,
        "raw_voltage_reads": 0,
        "gpu_s": 0,
        "sorts": 0,
    })

    complete = json.loads((DA / "COMPLETE.json").read_text())
    manifest = json.loads((DA / "MANIFEST.json").read_text())
    if complete["manifest_sha256"] != sha(DA / "MANIFEST.json"):
        raise RuntimeError("DA manifest hash mismatch")
    for row in manifest["products"]:
        path = DA / row["path"]
        if path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
            raise RuntimeError(f"DA product mismatch: {path}")

    summary = json.loads((DA / "run/SUMMARY.json").read_text())
    metrics = pd.read_csv(DA / "run/D2L_unit_rf_metrics.csv")
    eligibility = pd.read_csv(DA / "run/D2L_rf_eligibility.csv")
    published_scores = pd.read_csv(DA / "DA_SCORES.csv").set_index("arm").loc["D2L"]
    values = metrics.cv_snr.to_numpy(float)
    recomputed = {
        "arm": "D2L",
        "n_units_before_eligibility": int(len(eligibility)),
        "n_units_eligible": int(eligibility.eligible.astype(bool).sum()),
        "metric_rows": int(len(metrics)),
        "cv_snr_median": float(np.median(values)),
        "cv_snr_mean": float(np.mean(values)),
        "cv_snr_sum_all_eligible": float(values.sum()),
        "n_cv_snr_positive": int((values > 0).sum()),
    }
    expected = summary["arms"]["D2L"]
    checks = {
        "n_units_before_eligibility": recomputed["n_units_before_eligibility"] == expected["n_units_before_eligibility"],
        "n_units_eligible": recomputed["n_units_eligible"] == expected["n_units_eligible"] == len(metrics),
        "cv_snr_median": bool(np.isclose(recomputed["cv_snr_median"], expected["cv_snr_median"], rtol=0, atol=1e-14)),
        "cv_snr_mean": bool(np.isclose(recomputed["cv_snr_mean"], expected["cv_snr_mean"], rtol=0, atol=1e-14)),
        "cv_snr_sum": bool(np.isclose(recomputed["cv_snr_sum_all_eligible"], float(published_scores.cv_snr_sum_all_eligible), rtol=0, atol=1e-12)),
        "positive": recomputed["n_cv_snr_positive"] == int(published_scores.n_cv_snr_positive),
        "outer_holdout_unopened": summary["support"]["outer_holdout_opened"] is False,
        "development_trials_12": len(summary["support"]["development_trial_ordinals"]) == 12,
        "inner_folds_7_5": summary["support"]["development_inner_fold_counts"] == {"0": 7, "1": 5},
        "valid_frames_6052": summary["support"]["development_valid_frames"] == 6052,
    }
    if not all(checks.values()):
        raise RuntimeError(f"D2L endpoint check failed: {checks}")
    pd.DataFrame([recomputed]).to_csv(OUT / "D2L_RF_ENDPOINT_RECOMPUTE.csv", index=False)

    cx_summary = json.loads((CX / "SUMMARY.json").read_text())
    cx_rows = pd.read_csv(CX / "PER_DONOR_MODIFIED_QUALIFICATION.csv")
    cx_checks = {
        "denominator": int(len(cx_rows)),
        "old_qualified": int(cx_rows.old_source_qualified.sum()),
        "modified_qualified": int(cx_rows.modified_source_qualified.sum()),
        "swapped_qualified": int(cx_rows.swapped_half_qualified.sum()),
        "dual_qualified": int((cx_rows.modified_source_qualified & cx_rows.swapped_half_qualified).sum()),
        "agreement": int(cx_rows.qualification_stable_across_swap.sum()),
        "disagreements": int((~cx_rows.qualification_stable_across_swap).sum()),
        "median_mask_cosine": float(cx_rows.weight_cosine_train_swap.median()),
    }
    if cx_checks["dual_qualified"] != 25 or cx_checks["old_qualified"] != 5 or cx_checks["disagreements"] != 6:
        raise RuntimeError("CX engineering-only denominator changed")

    dg_files = sorted(path for path in DG.rglob("*") if path.is_file()) if DG.exists() else []
    review = {
        "status": "complete_source_and_one_endpoint; paired_DG_arrays_unavailable",
        "verdict": (
            "The published DA evaluator and D2L endpoint reproduce exactly. The DG shared directory "
            "contains no files, so h1 cannot independently execute or audit the frozen S0/SL reciprocal-best "
            "paired correspondence or verify their reported RF endpoint arrays."
        ),
        "da_packet": {
            "path": str(DA),
            "manifest_sha256": sha(DA / "MANIFEST.json"),
            "source_sha256": sha(DA / "source/cp_w2_rf_evaluator.py"),
            "checks": checks,
            "endpoint": recomputed,
            "scope": "development-only total-pipeline RF endpoint; unmatched populations; not identity preserving",
        },
        "frozen_correspondence_rule": {
            "rule": "reciprocal best Jaccard; both directional fractions >=0.5; best-minus-runner-up margin >=0.1; at least 100 matched events",
            "rivals": "retain all rivals and ambiguity",
            "interpretation": "uncalibrated output correspondence, not biological identity",
            "executed_on_h1": False,
            "reason": "DG final arrays and candidate-match source are absent from the empty shared DG directory",
        },
        "dg_dependency": {"path": str(DG), "exists": DG.exists(), "file_count": len(dg_files)},
        "hub_reported_not_independently_verified": {
            "S0": {"eligible": 76, "all_units": 528, "median_cv_snr": 0.306244, "mean_cv_snr": 0.995563},
            "SL": {"eligible": 80, "all_units": 504, "median_cv_snr": 0.390758, "mean_cv_snr": 0.823962},
            "interpretation": "mixed medians/means on unmatched populations cannot by themselves establish benefit, non-detriment, identity, or motion specificity",
        },
        "original_CX_qualification": {
            **cx_checks,
            "summary_sha256": sha(CX / "SUMMARY.json"),
            "interpretation": cx_summary["interpretation"],
            "engineering_only": True,
        },
        "prohibited_inferences": [
            "RF population summaries do not establish donor identity",
            "the 25-donor set is not pristine or held-out donor validation",
            "mixed median/mean RF changes are not a non-detriment proof",
            "no motion specificity follows without the paired correspondence and synthetic-truth scoring",
        ],
    }
    write_json(OUT / "DJ_REVIEW.json", review)
    (OUT / "REPORT.md").write_text(f"""# DJ independent CPU review

## Verdict

The published DA evaluator and one existing D2L RF endpoint are hash-valid and
reproduce exactly: {recomputed['n_units_eligible']}/{recomputed['n_units_before_eligibility']}
eligible, median cvSNR {recomputed['cv_snr_median']:.6f}, mean
{recomputed['cv_snr_mean']:.6f}, and all-eligible sum
{recomputed['cv_snr_sum_all_eligible']:.6f}. This is a development-only,
unmatched-population total-pipeline endpoint; the outer holdout remains unopened.

The DG shared directory exists but contains zero files. Therefore the h1 review
cannot independently inspect the S0/SL final arrays or candidate-match source,
execute the frozen reciprocal-best Jaccard correspondence, or validate the
hub-relayed S0/SL RF numbers. Those numbers are preserved in `DJ_REVIEW.json`
as reported context only.

The original donor limitation is unchanged: 5/90 unmodified qualifiers versus
25/90 dual-half modified-source qualifiers, with six swapped-half disagreements
and median taper-mask cosine {cx_checks['median_mask_cosine']:.3f}. The bank is
an exploratory engineering source set, not validated biological identities.
No matching, RF fit, sort, raw read, GPU work, or holdout evaluation ran here.
""")

    elapsed = time.monotonic() - started
    cpu = time.process_time() - cpu0
    receipt = {
        "status": "complete",
        "cpu_allwork_cap_s": 300,
        "measured_cpu_s": cpu,
        "measured_wall_s": elapsed,
        "conservative_active_charge_s": 60.0,
        "prior_h1_cumulative_active_s": 18383.22,
        "new_h1_cumulative_active_s": 18443.22,
        "overall_ceiling_active_s": 26000.0,
        "raw_voltage_reads": 0,
        "gpu_s": 0,
        "sorts": 0,
        "rf_fits": 0,
        "matching_runs": 0,
        "dh_failed_setup_attempts": "two sub-second import/class-name failures preserved; included inside DH's prior conservative 120 s, not charged again",
    }
    write_json(OUT / "RESOURCE_RECEIPT.json", receipt)
    products = []
    for path in sorted(OUT.iterdir()):
        if path.name not in {"MANIFEST.json", "COMPLETE.json"}:
            products.append({"path": path.name, "bytes": path.stat().st_size, "sha256": sha(path)})
    write_json(OUT / "MANIFEST.json", {"products": products})
    write_json(OUT / "COMPLETE.json", {"status": "complete_source_and_one_endpoint", "manifest_sha256": sha(OUT / "MANIFEST.json"), "written_last": True})
    print(json.dumps({"status": "complete", "endpoint": recomputed, "dg_files": len(dg_files), "cpu_s": cpu, "wall_s": elapsed}))


if __name__ == "__main__":
    main()
