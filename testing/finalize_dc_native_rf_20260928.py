"""Finalize DC's four-arm development RF table and independent native-bank audit."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


PACKET = Path("/mnt/NPX/Luke/DARTsort_motion_experiments/dc_motion_domain_and_native_bank_review_20260928")
DA = Path("/mnt/NPX/Luke/DARTsort_motion_experiments/da_rematching_development_rf_20260928")
CONTROL = Path("/mnt/NPX/Luke/DARTsort_motion_experiments/dc_native_bank_control_20260928")
ROOT = Path("/home/huklab/Documents/RyanSorting/SpikeSortingTools")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def main() -> None:
    # Preserve the prior domain-only terminal marker before extending the packet.
    if not (PACKET / "DOMAIN_COMPLETE.json").exists():
        shutil.copy2(PACKET / "COMPLETE.json", PACKET / "DOMAIN_COMPLETE.json")

    prior = json.loads((DA / "run/SUMMARY.json").read_text())
    native = json.loads((PACKET / "native_rf_run/SUMMARY.json").read_text())
    frozen_hashes = {
        "D2L": "03c339b4af8081e7e09bcbf23ebb2ae439b4cc8d468538d352cb8e79b61fc0b0",
        "REMATCH0": "82210d21996cee55aeef6c1bf73d23f0e3861cac867f404a2af96bbfb4024882",
        "CD1_FULL": "ba69b03c4c62d57acd5b51407ffcc63c1f078ba2ab4bc6b96932fe56e4f3958d",
        "NATIVE_REMATCH0": "6bbdc7442781d56c4991bb81344451b60e9bd9da6a434ce7bc49509206a8b15a",
    }
    for arm in ("D2L", "REMATCH0", "CD1_FULL"):
        if prior["hashes"][f"arm:{arm}"] != frozen_hashes[arm]:
            raise ValueError(f"prior DA arm hash mismatch: {arm}")
    if native["hashes"]["arm:NATIVE_REMATCH0"] != frozen_hashes["NATIVE_REMATCH0"]:
        raise ValueError("native arm hash mismatch")
    if native["support"] != prior["support"]:
        raise ValueError("native evaluator support differs from frozen DA support")
    if native["hashes"]["config"] != prior["hashes"]["config"]:
        raise ValueError("evaluator config mismatch")

    rows = []
    for arm in ("D2L", "REMATCH0", "CD1_FULL", "NATIVE_REMATCH0"):
        base = DA / "run" if arm != "NATIVE_REMATCH0" else PACKET / "native_rf_run"
        frame = pd.read_csv(base / f"{arm}_unit_rf_metrics.csv")
        elig = pd.read_csv(base / f"{arm}_rf_eligibility.csv")
        rows.append({
            "arm": arm,
            "final_sorting_sha256": frozen_hashes[arm],
            "n_units_total": len(elig),
            "n_units_eligible": len(frame),
            "eligible_fraction": len(frame) / len(elig),
            "cv_snr_median": frame.cv_snr.median(),
            "cv_snr_mean": frame.cv_snr.mean(),
            "cv_snr_sum": frame.cv_snr.sum(),
            "n_cv_snr_positive": int((frame.cv_snr > 0).sum()),
        })
    pd.DataFrame(rows).to_csv(PACKET / "FOUR_ARM_DEVELOPMENT_RF.csv", index=False)

    final_path = CONTROL / "NATIVE_REMATCH0/dartsort_sorting.npz"
    h5_path = CONTROL / "NATIVE_REMATCH0/matching2.h5"
    bank_path = Path(json.loads((CONTROL / "ARM_FREEZE.json").read_text())["bank"]["path"])
    with np.load(bank_path, allow_pickle=False) as bank:
        bank_checks = {
            "unit_ids_are_dense_0_to_727": bool(np.array_equal(bank["unit_ids"], np.arange(728))),
            "n_units": int(bank["unit_ids"].size),
            "whiten_strategy": str(bank["whiten_strategy"].item()),
        }
    with np.load(final_path, allow_pickle=False) as final, h5py.File(h5_path, "r") as h5:
        delta = np.asarray(final["times_samples"], dtype=np.int64) - np.asarray(h5["times_samples"], dtype=np.int64)
        source_review = {
            "status": "pass",
            "control_label": "NATIVE_REMATCH0 (authoritative receipt; generic COMPLETE arm label is not used)",
            "bank_path": str(bank_path),
            "bank_sha256": sha256(bank_path),
            "bank_checks": bank_checks,
            "worker_sha256": sha256(CONTROL / "source/cx_fixed_bank_worker.py"),
            "expected_worker_sha256": "73360f38deba3ce81b5c46e2e0e9f4196a6f92e93cb114d4ba936e7969316d8a",
            "matching_h5_sha256": sha256(h5_path),
            "final_sorting_sha256": sha256(final_path),
            "parent_h5_path": str(final["parent_h5_path"].item()),
            "matching_rows": int(h5["times_samples"].shape[0]),
            "final_rows": int(final["times_samples"].shape[0]),
            "channels_rowwise_equal": bool(np.array_equal(final["channels"], h5["channels"][:])),
            "time_delta_nonzero_fraction": float(np.count_nonzero(delta) / delta.size),
            "time_delta_min_samples": int(delta.min()),
            "time_delta_max_samples": int(delta.max()),
            "sampling_frequency_hz": float(final["sampling_frequency"]),
            "row_namespace": "within-arm matching2-to-final only; no cross-arm row-index join",
            "endpoint": "final labels after the one configured refinement iteration",
        }
    required = [
        source_review["bank_sha256"] == "43ccbd40f3868d6831b4b08607986308e675d235474218a37b8da1d5f59a54f0",
        source_review["worker_sha256"] == source_review["expected_worker_sha256"],
        source_review["matching_h5_sha256"] == "0b9c855005467a6ec66a7e6dbd99cbdc76619898bdddf5bd8a5f73fc036b1e63",
        source_review["final_sorting_sha256"] == frozen_hashes["NATIVE_REMATCH0"],
        bank_checks["unit_ids_are_dense_0_to_727"],
        bank_checks["whiten_strategy"] == "prewhiten_postapply",
        source_review["parent_h5_path"] == "matching2.h5",
        source_review["matching_rows"] == source_review["final_rows"],
        source_review["channels_rowwise_equal"],
    ]
    if not all(required):
        raise ValueError("native source/bank/lineage review failed")
    write_json(PACKET / "NATIVE_SOURCE_BANK_LINEAGE_REVIEW.json", source_review)

    shutil.copy2(ROOT / "testing/cp_w2_rf_evaluator.py", PACKET / "source/cp_w2_rf_evaluator.py")
    shutil.copy2(ROOT / "testing/run_dc_native_rf_20260928.py", PACKET / "source/run_dc_native_rf_20260928.py")
    shutil.copy2(ROOT / "testing/finalize_dc_native_rf_20260928.py", PACKET / "source/finalize_dc_native_rf_20260928.py")

    runtime = json.loads((PACKET / "NATIVE_RF_RUNTIME.json").read_text())
    write_json(PACKET / "RESOURCE_RECEIPT_FINAL.json", {
        "domain_charge_s": 90.0,
        "native_rf_charge_s": 30.0,
        "dc_total_charge_s": 120.0,
        "native_rf_actual_wall_s": runtime["elapsed_wall_s"],
        "native_rf_actual_cpu_s": runtime["user_cpu_s"] + runtime["system_cpu_s"],
        "h1_cumulative_before_dc_s": 17513.22,
        "h1_cumulative_after_dc_s": 17633.22,
        "h1_ceiling_s": 20500.0,
        "gpu_s": 0,
        "raw_voltage_reads": 0,
    })
    table = pd.DataFrame(rows).set_index("arm")
    report = f"""# DC motion-domain and native-bank development RF review

## Verdict first

The native-bank control completed and passed the independent source, bank, and
within-arm endpoint review. On the frozen development-only W2 RF screen, D2L
remains strongest descriptively. NATIVE_REMATCH0 has the lowest median cvSNR
({table.loc['NATIVE_REMATCH0','cv_snr_median']:.3f}) but a higher mean and sum
than regenerated-bank REMATCH0; this mixed result does not show a native-bank
functional uplift and does not identify corresponding neurons across arms.

| Arm | Eligible / total | Median cvSNR | Mean cvSNR | Sum cvSNR | Positive |
|---|---:|---:|---:|---:|---:|
| D2L | 80 / 566 | 0.262 | 1.046 | 83.684 | 48 |
| REMATCH0 | 75 / 586 | 0.235 | 0.706 | 52.962 | 40 |
| CD1_FULL | 68 / 613 | 0.221 | 0.671 | 45.660 | 36 |
| NATIVE_REMATCH0 | 76 / 573 | 0.116 | 0.757 | 57.521 | 41 |

The original 81-trial universe, 12 development trials, 7/5 inner folds, 6,052
valid frames, and 500-total/200-per-fold eligibility thresholds are unchanged.
The 20-trial outer holdout was not evaluated. Prior three-arm results were
reused by verified final-sort hashes; only NATIVE_REMATCH0 was newly scored.

`NATIVE_SOURCE_BANK_LINEAGE_REVIEW.json` verifies the accepted native bank
(728 dense units, `prewhiten_postapply`), executed worker, matching H5, final
NPZ, row count, channel alignment, and matching2-to-final timing refinement.
The generic upstream `COMPLETE.json` arm text is superseded by its authoritative
NATIVE_REMATCH0 receipts. No cross-arm row-index join was performed.

## Exact W2 motion domains

| Arm | Negative excursion assigned Hz | Outside-mask flat assigned Hz | Catalogue-outside remainder assigned Hz |
|---|---:|---:|---:|
| STATIC_S | 1115.33 | 1896.06 | 1201.18 |
| D2L | 1148.75 | 2026.98 | 1281.27 |
| REMATCH0 | 1132.28 | 2062.21 | 1299.74 |
| CD1_FULL | 1150.76 | 2123.02 | 1319.76 |

The exact piecewise-linear domains cover W2 without gaps or overlap.
`DOMAIN_EVENT_COUNTS.csv` reports exposure, assigned/noise counts and rates;
`SEGMENT_SAFE_ISI.csv` prevents intervals from crossing domain boundaries.
The catalogue-outside remainder is explicitly **not true rest**. Across-unit
rank correlations are descriptive only, with unequal arm-local cohorts; they
are not neuron identity matches. No point-source depth from T16 is used.
"""
    (PACKET / "REPORT.md").write_text(report)
    (PACKET / "README.md").write_text("# DC complete packet\n\nSee `REPORT.md`, `FOUR_ARM_DEVELOPMENT_RF.csv`, and the native lineage review.\n")

    validation = {
        "status": "pass",
        "prior_arm_hashes_verified": True,
        "native_arm_hash_verified": True,
        "frozen_support_identical": True,
        "outer_holdout_opened": False,
        "newly_scored_arms": ["NATIVE_REMATCH0"],
        "reused_arms": ["D2L", "REMATCH0", "CD1_FULL"],
        "native_source_bank_lineage_review": "pass",
    }
    write_json(PACKET / "NATIVE_RF_VALIDATION.json", validation)

    files = []
    for path in sorted(PACKET.rglob("*")):
        if not path.is_file() or path.name in {"COMPLETE.json", "MANIFEST.json"}:
            continue
        files.append({"path": str(path.relative_to(PACKET)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    write_json(PACKET / "MANIFEST.json", {"files": files})
    # COMPLETE is deliberately the last packet write.
    write_json(PACKET / "COMPLETE.json", {
        "status": "complete",
        "scope": "DC exact motion domains plus native-bank four-arm development RF",
        "manifest_sha256": sha256(PACKET / "MANIFEST.json"),
        "outer_holdout_opened": False,
        "written_last": True,
    })


if __name__ == "__main__":
    main()
