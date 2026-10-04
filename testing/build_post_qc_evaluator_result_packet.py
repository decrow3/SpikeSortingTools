#!/usr/bin/env python3
"""Build the immutable H5 result packet for the one-shot post-QC evaluator run."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path


DEST = Path("/mnt/NPX/Luke/DARTsort_motion_experiments/en_first_medium_evaluator_post_qc_execution_result_20261002_v1_h5")
OUTPUT = Path("/media/huklaban5/Data/en_common_support_crop_screen_20260930_v1/evaluator_post_qc_v1")
STAGE = Path("/media/huklaban5/Data/en_common_support_crop_screen_20260930_v1/evaluator_post_qc_v1_staging/source")
REQUEST = Path("/mnt/NPX/Luke/DARTsort_motion_experiments/en_first_medium_evaluator_post_qc_execution_request_20261002_v1_h1")
SOURCE_PACKET = Path("/mnt/NPX/Luke/DARTsort_motion_experiments/en_first_medium_four_arm_evaluator_repair_candidate_20261002_v3_h1")
TRAVERSAL_REVIEW = Path("/mnt/NPX/Luke/DARTsort_motion_experiments/en_first_medium_four_arm_evaluator_v3_h5_review_20261002_v1_h5")
QC_RESULT = Path("/mnt/NPX/Luke/DARTsort_motion_experiments/en_truncation_qc_regeneration_v6_real_result_20261002_v1_h5")
QC_REVIEW = Path("/mnt/NPX/Luke/DARTsort_motion_experiments/en_truncation_qc_regeneration_v6_postexecution_review_20261002_v1_h1")
SERVICE = Path("/home/huklaban5/Documents/SpikeSortingTools/SpikeSortingTools/testing/en_first_medium_evaluator_post_qc_v1.service")
LOGS = Path("/home/huklaban5/Documents/SpikeSortingTools/SpikeSortingTools/testing/outputs/en_first_medium_evaluator_post_qc_v1")

EXPECTED_MODULES = {
    "testing/first_medium_evaluator.py": "de977776c1aa48c2bba0c849d9940e37c8fd6b81522ec80cf947e943104ea291",
    "testing/sort_comparison.py": "b4065ba7e6c326e74f5f283de9a6ee76df0dfd3b70b65fd534b1d4559a80de0c",
    "testing/luke_amplitude_dropout_audit.py": "12e105dc797693957701e0b0a81130c2debced70d2588a8ea3c28a03e2a26431",
    "pipeline/config.py": "74424d444e05a7ba2d3968d22c1490475ebe8d7c0d213687316ace7cec5cf59c",
    "pipeline/truncation.py": "eec2476397c1856891a4b9f1856c1af4a5e2f81a13e934d1ade2f96a02a6e3e6",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def rows(path: Path) -> int:
    with path.open(newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def verified_binding(packet: Path, expected_manifest: str, expected_complete: str | None = None) -> dict:
    manifest = sha(packet / "MANIFEST.sha256")
    complete = sha(packet / "COMPLETE.json")
    assert manifest == expected_manifest, (packet, manifest)
    if expected_complete is not None:
        assert complete == expected_complete, (packet, complete)
    return {"path": str(packet), "manifest_sha256": manifest, "complete_sha256": complete}


def main() -> None:
    assert not DEST.exists(), f"refusing to overwrite {DEST}"
    assert OUTPUT.is_dir()
    report_path = OUTPUT / "FOUR_ARM_REPORT.json"
    report = json.loads(report_path.read_text())
    pair_id = "REF384__existing_corrected_B384"
    pair = OUTPUT / "pairs" / pair_id

    module_hashes = {name: sha(STAGE / name) for name in EXPECTED_MODULES}
    assert module_hashes == EXPECTED_MODULES
    report_modules = {v["path"].removeprefix(str(STAGE) + "/"): v["sha256"] for v in report["implementation_provenance"].values()}
    assert report_modules == EXPECTED_MODULES

    cfg_path = REQUEST / "config/en_first_medium_four_arm_evaluator.post_qc.execution_disabled.v1.json"
    assert sha(cfg_path) == "559f50a4c4a166c3a9a9fae3266abf8ad2d07a63626ee3c58f6792a47c2b3205"
    cfg = json.loads(cfg_path.read_text())
    input_receipts = {}
    expected_identities = {
        "REF384": "739f433cd65e369af83cad4fa3dbfc2763ebc79a087c902ae98bb0405b9d1cc4",
        "existing_corrected_B384": "55496fca68263842b83a5fdc5ae7f1a758a2dcb325e601d3981e13f83d953e20",
    }
    expected_qc = {
        "REF384": "ae07ecf970f0c6de93f5f1717254fd11a181ce81b8a693232171f03e4ed524af",
        "existing_corrected_B384": "66088862cc80de3f7d259844acdddf0ff72dba5a36c9cb84dce70f259688dd13",
    }
    for arm in cfg["arms"]:
        if arm["evidence_kind"] != "real_saved":
            continue
        arm_id = arm["arm_id"]
        curated = Path(arm["curated_output"])
        files = {
            name: sha(curated / name)
            for name in ("spike_times.npy", "spike_clusters.npy", "full_st.npy", "kept_spikes.npy", "cluster_KSLabel.tsv")
        }
        qc_file = Path(arm["qc_dir"]) / "amp_truncation/truncation_qc.npz"
        qc_sha = sha(qc_file)
        assert arm["expected_identity_digest"] == expected_identities[arm_id]
        assert report["arms"][arm_id]["provenance"]["curated_identity_digest"] == expected_identities[arm_id]
        assert qc_sha == expected_qc[arm_id] == arm["expected_qc_request_digest"]
        assert report["arms"][arm_id]["provenance"]["qc_request_digest"] == expected_qc[arm_id]
        input_receipts[arm_id] = {
            "curated_output": str(curated),
            "file_sha256": files,
            "identity_digest": expected_identities[arm_id],
            "identity_match": True,
            "qc_path": str(qc_file),
            "qc_sha256": qc_sha,
            "qc_match": True,
            "clock_normalization": report["arms"][arm_id]["provenance"]["clock_normalization"],
        }

    bindings = {
        "request": verified_binding(REQUEST, "933fa3f33956d137e70a3bcc91f9d05ff4b0394ba424713d757f1baaf6d987a3", "c9b0f3790fa7bba46e2bf3c4b6ae1e374e241bad25a3c90b07c0869f01671ae8"),
        "accepted_source": verified_binding(SOURCE_PACKET, "7e0d4d7e16c5ade2ceb376cd058b3ad3e6f05ffb9cd17b4830450f14277bfd1e", "959a1e139c3289851d3905ed815bd543b32fa821884937eb1519cdbbac777ec6"),
        "accepted_traversal_review": verified_binding(TRAVERSAL_REVIEW, "ea994640050807fd03d1ef6beda83acce664552fcea692c2944efb2b711e8c30"),
        "qc_result": verified_binding(QC_RESULT, "695d0fdb58fbb82c4d1213f665eb3a4a065b4708d3b41c418fce7d193a37db08", "4cd7a0a3bfd4e3c18c92f3a6316e7ef559ed0def63a8bcc345f6266cc18283de"),
        "qc_postexecution_review": verified_binding(QC_REVIEW, "c2ee013d27540a7facf39fc12373758330d219c1f30a7127503003b0a753c7aa", "a159632006fac3f97df1f8f62f1746990ed1c8e2f8e109e00c87860565b2e3f9"),
    }

    output_inventory = []
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file():
            output_inventory.append({"path": str(path.relative_to(OUTPUT)), "bytes": path.stat().st_size, "sha256": sha(path)})

    scorecard = report["scientific_scorecard"][pair_id]
    closure = {
        "pair_id": pair_id,
        "row_counts": {
            name: rows(pair / name)
            for name in (
                "baseline_eligibility.csv", "primary_matches.csv", "correspondence_edges.csv",
                "amplitude_completeness_pairs.csv", "amplitude_windows.csv", "guardrail_summary.csv",
                "unit_metrics_baseline.csv", "unit_metrics_candidate.csv",
            )
        },
        "summary": json.loads((pair / "summary.json").read_text()),
        "coverage": json.loads((pair / "coverage_summary.json").read_text()),
        "split_merge": json.loads((pair / "split_merge_summary.json").read_text()),
        "decision": json.loads((pair / "decision.json").read_text()),
    }

    receipt = {
        "schema": "first-medium-evaluator-post-qc-execution-receipt-v1",
        "host": "huklaban5",
        "execution_count": 1,
        "retry_count": 0,
        "service": {
            "name": "en_first_medium_evaluator_post_qc_v1.service",
            "type_at_execution": "exec",
            "start_utc": "2026-10-02T09:34:19Z",
            "completion_utc": "2026-10-02T09:34:31Z",
            "wall_time_observed_s": 12,
            "cpu_time_journal_s": 11.148,
            "result": "success",
            "exec_main_status": 0,
            "n_restarts": 0,
            "restart": "no",
            "allowed_cpus": "0-3",
            "cpu_quota": "400%",
            "memory_max_bytes": 17179869184,
            "runtime_max_s": 3600,
            "timeout_start_s": 3600,
            "memory_peak": "not retained after inactive Type=exec cgroup",
            "command": "/home/huklaban5/anaconda3/bin/python -m testing.first_medium_evaluator --request /mnt/NPX/Luke/DARTsort_motion_experiments/en_first_medium_evaluator_post_qc_execution_request_20261002_v1_h1/config/en_first_medium_four_arm_evaluator.post_qc.execution_disabled.v1.json --output /media/huklaban5/Data/en_common_support_crop_screen_20260930_v1/evaluator_post_qc_v1",
            "environment": {
                "PYTHONPATH": str(STAGE),
                "NUMBA_CACHE_DIR": "/media/huklaban5/Data/en_common_support_crop_screen_20260930_v1/evaluator_post_qc_v1_numba_cache",
                "PYTHONDONTWRITEBYTECODE": "1",
                "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
                "OPENBLAS_NUM_THREADS": "4", "NUMEXPR_NUM_THREADS": "4",
            },
        },
        "prelaunch": {
            "output_absent": True,
            "staging_absent_before_creation": True,
            "numba_cache_absent_before_creation": True,
            "no_prior_service_start": True,
            "launcher_survival_proof": "luke-registration-manager-proof-v1.service previously completed detached 25-second dummy job",
        },
        "bindings": bindings,
        "config_path": str(cfg_path),
        "config_sha256": sha(cfg_path),
        "staged_module_sha256_before_and_after": module_hashes,
        "input_receipts": input_receipts,
        "output_dir": str(OUTPUT),
        "output_inventory": output_inventory,
        "scientific_result_exact": {
            "overall_decision": report["decision"],
            "scorecard": scorecard,
            "unavailable_pairs": report["unavailable_pairs"],
            "matching_and_counting_closure": closure,
        },
        "scope": {
            "identity_status": "UNRESOLVED",
            "waveform_status": "UNMEASURED",
            "repaired_B384_status": "UNAVAILABLE_not_run",
            "REF384_repeat_status": "UNAVAILABLE_not_run",
            "saved_outputs_only": True,
            "recording_or_voltage_read": False,
            "sort_training_detection": False,
            "rf_or_holdout_access": False,
            "waveform_capture": False,
            "scientific_interpretation_added_by_h5": False,
        },
    }

    with tempfile.TemporaryDirectory(prefix=".post_qc_eval_packet_", dir=DEST.parent) as td:
        root = Path(td) / DEST.name
        (root / "evidence").mkdir(parents=True)
        (root / "bindings").mkdir()
        (root / "service").mkdir()
        shutil.copytree(OUTPUT, root / "results")
        shutil.copy2(SERVICE, root / "service" / SERVICE.name)
        for name in ("stdout.log", "stderr.log"):
            source = LOGS / name
            if source.exists():
                shutil.copy2(source, root / "service" / name)
            else:
                (root / "service" / name).write_text("")
        for name, packet in (("request", REQUEST), ("source", SOURCE_PACKET), ("traversal_review", TRAVERSAL_REVIEW), ("qc_result", QC_RESULT), ("qc_review", QC_REVIEW)):
            shutil.copy2(packet / "MANIFEST.sha256", root / "bindings" / f"{name}_MANIFEST.sha256")
            shutil.copy2(packet / "COMPLETE.json", root / "bindings" / f"{name}_COMPLETE.json")
        write_json(root / "evidence/EXECUTION_RECEIPT.json", receipt)
        write_json(root / "evidence/INPUT_RECEIPTS.json", input_receipts)
        write_json(root / "evidence/OUTPUT_INVENTORY.json", output_inventory)
        write_json(root / "evidence/MATCHING_COUNTING_CLOSURE.json", closure)
        write_json(root / "RESULT_SUMMARY.json", {
            "overall_decision": report["decision"],
            "pair_id": pair_id,
            "pair_verdict": scorecard["verdict"],
            "pair_status": scorecard["status"],
            "reason_codes": scorecard["reason_codes"],
            "endpoints": scorecard["endpoints"],
            "identity_status": "UNRESOLVED",
            "waveform_status": "UNMEASURED",
            "interpretation": "exact frozen-rule evaluator output; no H5 scientific interpretation added",
        })
        (root / "README.md").write_text(
            "# H5 post-QC evaluator-v3 execution result\n\n"
            "The contract-bound saved-output-only evaluator completed successfully on its only managed-service start. "
            "This packet preserves the exact first output, bindings, source hashes, service receipt, endpoint values, "
            "and matching/counting closures. No retry occurred.\n\n"
            "The frozen report marks the existing REF384 versus existing-corrected-B384 diagnostic scorecard as "
            "`fail` with `POPULATION_COVERAGE_BELOW_0_5`; the four-arm advancement decision remains `inconclusive` "
            "because prospective real-arm outputs are unavailable. These are copied evaluator rules, not a new H5 interpretation.\n\n"
            "Identity is `UNRESOLVED`; waveform evidence is `UNMEASURED`. The run read saved sorter/QC outputs only. "
            "It did not read recording voltage or invoke sorting, training, detection, waveform capture, RF, or holdout.\n\n"
            "## Implementation checks\n\n"
            "- Done: all five prerequisite packet manifests and requested COMPLETE hashes verified before launch.\n"
            "- Done: fresh output/stage/cache checks, byte-identical writable source stage, exact config, saved identities, QC byte hashes, and clock transforms passed.\n"
            "- Done: post-run staged module hashes equal pre-run/accepted hashes; output inventory and exact CSV row closures are preserved.\n"
            "- Done: systemd journal recorded one start, success, exit 0, zero restarts, and 11.148 CPU seconds under the frozen limits.\n"
            "- Not done: waveform/biological identity validation, repaired-arm execution, repeat execution, or sealed RF/holdout evaluation; excluded by contract.\n"
            "- Can establish: frozen evaluator-v3 endpoints for this saved REF384 versus existing-corrected-B384 comparison under the regenerated truncation QC.\n"
            "- Cannot establish: biological identity, waveform quality, prospective repaired-arm efficacy, repeatability, or advancement readiness.\n"
        )

        members = [p for p in sorted(root.rglob("*")) if p.is_file()]
        manifest_lines = [f"{sha(p)}  {p.relative_to(root)}" for p in members]
        (root / "MANIFEST.sha256").write_text("\n".join(manifest_lines) + "\n")
        manifest_sha = sha(root / "MANIFEST.sha256")
        DEST.parent.mkdir(parents=True, exist_ok=True)
        root.rename(DEST)
        write_json(DEST / "COMPLETE.json", {
            "schema": "immutable-result-completion-v1",
            "status": "complete",
            "execution_count": 1,
            "retry_count": 0,
            "manifest_sha256": manifest_sha,
            "source_output_report_sha256": sha(report_path),
            "identity_status": "UNRESOLVED",
            "waveform_status": "UNMEASURED",
            "scientific_interpretation": "none beyond frozen evaluator rules",
            "completion_written_last": True,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        })
        print(json.dumps({"destination": str(DEST), "manifest_sha256": manifest_sha, "complete_sha256": sha(DEST / "COMPLETE.json")}, indent=2))


if __name__ == "__main__":
    main()
