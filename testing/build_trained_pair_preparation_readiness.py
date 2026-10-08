#!/usr/bin/env python3
"""Publish the execution-disabled trained-pair preparation/readiness packet."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path("/home/huklaban5/Documents/SpikeSortingTools/SpikeSortingTools")
SHARED = Path("/mnt/NPX/Luke/DARTsort_motion_experiments")
DEST = SHARED / "en_first_medium_trained_pair_preparation_readiness_20261002_v1_h5"
REVIEW = SHARED / "en_minimum_training_support_mask_trained_entry_ordering_v4_h1_review_20261002_v1_h1"
CANDIDATE = SHARED / "en_minimum_training_support_mask_trained_entry_ordering_repair_candidate_20261002_v4_h5"
POST_QC = SHARED / "en_first_medium_evaluator_post_qc_execution_result_20261002_v1_h5"
QC_RESULT = SHARED / "en_truncation_qc_regeneration_v6_real_result_20261002_v1_h5"
QC_REVIEW = SHARED / "en_truncation_qc_regeneration_v6_postexecution_review_20261002_v1_h1"
EVAL_REQUEST = SHARED / "en_first_medium_evaluator_post_qc_execution_request_20261002_v1_h1"

PLANNED_ROOT = Path("/media/huklaban5/Data/en_first_medium_trained_pair_20261002_v1")
PLANNED_STAGE = Path("/media/huklaban5/Data/en_first_medium_trained_pair_20261002_v1_staging")
PLANNED_CACHE = Path("/media/huklaban5/Data/en_first_medium_trained_pair_20261002_v1_numba_cache")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def binding(packet: Path, manifest: str, complete: str) -> dict:
    observed_manifest = sha(packet / "MANIFEST.sha256")
    observed_complete = sha(packet / "COMPLETE.json")
    assert observed_manifest == manifest, (packet, observed_manifest)
    assert observed_complete == complete, (packet, observed_complete)
    return {"path": str(packet), "manifest_sha256": manifest, "complete_sha256": complete}


def main() -> None:
    assert not DEST.exists(), f"refusing to overwrite {DEST}"
    for path in (PLANNED_ROOT, PLANNED_STAGE, PLANNED_CACHE):
        assert not path.exists(), f"planned namespace is no longer fresh: {path}"

    accepted = json.loads((CANDIDATE / "config/en_minimum_training_support_mask.trained.review_pending.v1.json").read_text())
    repaired = json.loads(json.dumps(accepted))
    repaired["approval"]["h1_review_manifest_sha256"] = "9ff42c9e56251e2659153de194ade4dce986aadcf24a0e419cbb01dd0d16804a"
    repaired["execution_enabled"] = False
    repaired["status"] = "execution_disabled_pending_pair_contract_independent_review"
    repaired["failure_reason"] = "Preparation-only config; enable only in a separately reviewed immutable launch packet"
    repaired["trained"]["run_root"] = str(PLANNED_ROOT / "repaired_B384")
    repaired["trained"]["launch_evidence_dir"] = str(PLANNED_ROOT / "repaired_B384/launch_evidence")
    repaired["trained"]["pre_extraction_snapshot_dir"] = str(PLANNED_ROOT / "repaired_B384/pre_extraction_snapshot")
    repaired["native_invocation"]["results_dir"] = str(PLANNED_ROOT / "repaired_B384/native_results")
    repaired["resources"].update({
        "wall_seconds_max": 1200,
        "ram_bytes_max": 68719476736,
        "cpu_threads_max": 16,
        "gpu_count_max": 1,
        "persistent_output_bytes_max": 8589934592,
    })
    repaired["prohibitions"] = [
        "RF access", "sealed holdout access", "parameter search",
        "preprocessed voltage copy", "unrestricted transfer",
    ]

    base_path = ROOT / "configs/en_common_support_crop_screen.v1.json"
    base = json.loads(base_path.read_text())
    ref_arm = base["arms"]["REF384"]
    repeat = {
        "schema": "en-first-medium-ref384-repeat-proposed-v1",
        "execution_enabled": False,
        "status": "execution_disabled_pending_pair_contract_independent_review",
        "role": "fresh exact REF384 repeatability diagnostic; never an efficacy arm",
        "run_root": str(PLANNED_ROOT / "REF384_repeat"),
        "native_results_dir": str(PLANNED_ROOT / "REF384_repeat/native_results"),
        "pre_extraction_snapshot_dir": str(PLANNED_ROOT / "REF384_repeat/pre_extraction_snapshot"),
        "launch_evidence_dir": str(PLANNED_ROOT / "REF384_repeat/launch_evidence"),
        "input": ref_arm,
        "crop": base["crop"],
        "native_settings": base["sorter"]["consumed_settings"],
        "effective_requirements": {
            "n_chan_bin": 384, "selected_channels": 384, "bad_channels": [],
            "do_CAR": True, "invert_sign": False, "device": "cuda",
            "nblocks": 0, "dshift": None, "save_extra_vars": True,
            "save_preprocessed_copy": False, "clear_cache": True,
            "numpy_seed": 1, "torch_seed": 1, "torch_cuda_all_seed": 1,
            "support_policy_enabled": False,
            "adaptive_state": "recomputed from this arm; no C/W/PCA/template/model reuse",
        },
        "implementation": {
            "historical_runner_path": str(ROOT / "testing/en_common_support_crop_screen.py"),
            "historical_runner_sha256": sha(ROOT / "testing/en_common_support_crop_screen.py"),
            "required_wrapper": "testing.en_first_medium_trained_pair_launch (not yet implemented/reviewed)",
            "terminal_validator_path": str(ROOT / "npx_preprocessing/motion/kilosort_support_mask_trained.py"),
            "terminal_validator_sha256": sha(ROOT / "npx_preprocessing/motion/kilosort_support_mask_trained.py"),
        },
        "required_native_files": accepted["trained"]["required_native_files"],
        "required_snapshot_files": accepted["trained"]["required_snapshot_files"],
        "resources": {
            "wall_seconds_max": 1200, "ram_bytes_max": 68719476736,
            "cpu_threads_max": 16, "gpu_count_max": 1,
            "persistent_output_bytes_max": 8589934592,
        },
    }

    source_bindings = {
        str(path): sha(path)
        for path in (
            ROOT / "npx_preprocessing/motion/kilosort_support_mask_candidate.py",
            ROOT / "npx_preprocessing/motion/prewhitening_support_mask.py",
            ROOT / "npx_preprocessing/motion/kilosort_support_mask_trained.py",
            ROOT / "testing/en_minimum_training_support_mask_trained_launch.py",
            ROOT / "testing/en_common_support_crop_screen.py",
            ROOT / "pipeline/kilosort_compat.py",
            ROOT / "pipeline/sorting.py",
            ROOT / "environments/rescue-production/.venv/lib/python3.12/site-packages/kilosort/io.py",
            ROOT / "environments/rescue-production/.venv/lib/python3.12/site-packages/kilosort/preprocessing.py",
            ROOT / "environments/rescue-production/.venv/lib/python3.12/site-packages/kilosort/spikedetect.py",
            ROOT / "environments/rescue-production/.venv/lib/python3.12/site-packages/kilosort/template_matching.py",
            ROOT / "environments/rescue-production/.venv/lib/python3.12/site-packages/kilosort/run_kilosort.py",
        )
    }
    expected_sources = {
        str(ROOT / "npx_preprocessing/motion/kilosort_support_mask_candidate.py"): "705b4af444699a6e40ddd0e8a189232b619c4e1bbff53f8150b448f5ad160391",
        str(ROOT / "npx_preprocessing/motion/prewhitening_support_mask.py"): "9c981047f94d91a930e17bd51e634fbc96faa5e956c85d0882f3a603f7f42e0f",
        str(ROOT / "npx_preprocessing/motion/kilosort_support_mask_trained.py"): "aafc740beda0ef67fb3e62c37a44d56ba22351b7fa9b1962ed146c0a8b0935ee",
        str(ROOT / "testing/en_minimum_training_support_mask_trained_launch.py"): "78ffe3a29abf9f21f74a3e8c443d7f8d4fd60ee6fb0e72c1f07df717b79ff717",
        str(ROOT / "testing/en_common_support_crop_screen.py"): "800b190b1ad7f0bb832f310c2b86bf4d6802b63ae5990169138bb86ce5d6c897",
        str(ROOT / "environments/rescue-production/.venv/lib/python3.12/site-packages/kilosort/io.py"): "767b76a07647122b45048ba12cdaf49185f87f3fa33334081d4af099825414bd",
        str(ROOT / "environments/rescue-production/.venv/lib/python3.12/site-packages/kilosort/preprocessing.py"): "2329b288ae068361a937632a53461e6cceeb27e30191337e9b9a0fa7404d763d",
        str(ROOT / "environments/rescue-production/.venv/lib/python3.12/site-packages/kilosort/spikedetect.py"): "db1e8f0357319d9db3530a700bd1002c830337c50fde58b3e74e83e7af140892",
        str(ROOT / "environments/rescue-production/.venv/lib/python3.12/site-packages/kilosort/template_matching.py"): "39235cc98428dbb279706718f74a1c3d84568ef31a4f9d1d997a60f7002b1a66",
        str(ROOT / "environments/rescue-production/.venv/lib/python3.12/site-packages/kilosort/run_kilosort.py"): "5c1baee40edef14fa7566e441997da5d2ecc4f7940b8385cf83de40c8bd0f945",
    }
    for path, expected in expected_sources.items():
        assert source_bindings[path] == expected

    provenance = {
        "design_review": binding(REVIEW, "9ff42c9e56251e2659153de194ade4dce986aadcf24a0e419cbb01dd0d16804a", "54af269b287b3cfa06bcf6dc7f28fc5bd9ac3188cbf845baa8a331a37acaf34f"),
        "design_candidate": binding(CANDIDATE, "91bcf5ed0560bf5617815e4929a3d97f82b386f0040ef31dc6a61991a45fd36c", "9ab3e67a41af42b65cbf56509fc64fc407a39c79496a2391852de39aeb1055df"),
        "post_qc_evaluator_result": binding(POST_QC, "e417eceb918f44110c4c43eaac58fe4e33ae4c87a4c4e520f9fc655b4640b8c2", "e77f50d7f4e00b39c9fa6a99bfad1cf00ecb92b9dca7823d4c06c18d29c0a15c"),
        "qc_result": binding(QC_RESULT, "695d0fdb58fbb82c4d1213f665eb3a4a065b4708d3b41c418fce7d193a37db08", "4cd7a0a3bfd4e3c18c92f3a6316e7ef559ed0def63a8bcc345f6266cc18283de"),
        "qc_review": binding(QC_REVIEW, "c2ee013d27540a7facf39fc12373758330d219c1f30a7127503003b0a753c7aa", "a159632006fac3f97df1f8f62f1746990ed1c8e2f8e109e00c87860565b2e3f9"),
        "evaluator_request": binding(EVAL_REQUEST, "933fa3f33956d137e70a3bcc91f9d05ff4b0394ba424713d757f1baaf6d987a3", "c9b0f3790fa7bba46e2bf3c4b6ae1e374e241bad25a3c90b07c0869f01671ae8"),
        "frozen_crop_config": {"path": str(base_path), "sha256": sha(base_path)},
        "source_bindings": source_bindings,
        "inputs": {
            "REF_recording_manifest": {"path": ref_arm["recording_manifest_path"], "sha256": sha(Path(ref_arm["recording_manifest_path"]))},
            "B_recording_manifest": {"path": repaired["recording"]["manifest_path"], "sha256": sha(Path(repaired["recording"]["manifest_path"]))},
            "probe": {"sha256": "ac69e60bfdb6b11e48ebe1d923d3559f8ec9df4fe8ba1865be495ee481d90601"},
            "field": {"path": repaired["field"]["path"], "sha256": sha(Path(repaired["field"]["path"]))},
            "REF_ops": {"path": repaired["comparators"]["REF384"]["root"] + "/sorter_output/ops.npy", "sha256": repaired["comparators"]["REF384"]["ops_sha256"]},
            "B_ops": {"path": repaired["ops"]["path"], "sha256": sha(Path(repaired["ops"]["path"]))},
        },
        "binary_policy": "Do not rehash 241-GB binaries during preparation; require size plus manifest-recorded SHA-256 at launch before opening voltage.",
    }
    assert provenance["frozen_crop_config"]["sha256"] == "bb6dd8f21539e5ee57f4216702b7c9d4c36d1a0feeefe3859ba25449af52ad17"

    intended_diff = {
        "REF384_repeat_vs_historical_REF384": {
            "intended_difference": "fresh execution namespace only",
            "must_match": ["binary identity", "probe/geometry", "crop/clock", "all native settings", "seeds", "no support hook", "adaptive refit policy"],
            "non_scientific_evidence_difference": "stronger terminal validation and immutable launch receipts",
        },
        "repaired_B384_vs_existing_corrected_B384": {
            "intended_difference": "reviewed pointwise support policy after temporal filtering/CAR and before W; all adaptive state refit from the same B input",
            "must_match": ["B binary identity", "probe/geometry", "crop/clock", "Kilosort settings", "seeds", "nblocks=0", "save policy"],
            "known_unavoidable_difference": "fresh execution versus historical comparator; REF repeat diagnoses execution repeatability but is not an efficacy arm",
        },
        "repaired_B384_vs_REF384_repeat": {
            "not_mask_only": True,
            "differences": ["REF versus rounded-remap B input", "support policy disabled versus enabled"],
            "interpretation": "end-to-end prospective contrast only",
        },
        "adaptive_state": "C, W, wPCA, wTEMP, universal detections/features, Wall3, iU/iCC/iCC_mask, learned detections/features, clusters and templates are independently recomputed in both fresh arms; no bank/model/whitening reuse.",
    }

    clock_known_answer = {}
    for arm_id in ("REF384", "B384"):
        sorter = Path("/media/huklaban5/Data/en_common_support_crop_screen_20260930_v1") / arm_id / "sorter_output"
        spike_times = np.load(sorter / "spike_times.npy", mmap_mode="r").reshape(-1)
        full_st = np.load(sorter / "full_st.npy", mmap_mode="r")
        kept = np.load(sorter / "kept_spikes.npy", mmap_mode="r")
        kept_indices = np.flatnonzero(kept) if kept.dtype == np.bool_ else np.asarray(kept, dtype=np.int64)
        delta = np.asarray(spike_times, dtype=np.int64) - np.asarray(full_st[kept_indices, 0], dtype=np.int64)
        assert len(delta) == len(spike_times)
        assert bool(np.all(delta == 208498882))
        clock_known_answer[arm_id] = {
            "rows": int(len(delta)),
            "observed_unique_offset_samples": [208498882],
            "all_rows_equal_crop_imin": True,
            "correct_relation": "spike_times == full_st[kept_spikes,0] + crop.imin",
        }

    rules = {
        "frozen_before_outcomes": True,
        "predictions": [
            "REF384_repeat should reproduce the historical REF384 closely enough to qualify under the frozen evaluator-v3 repeatability rule, subject to required endpoint measurability.",
            "repaired_B384 should increase common-time amplitude measurability relative to existing_corrected_B384 and reach population measurable fraction >=0.50.",
            "If median common-time missingness is measurable, repaired_B384 should improve it by >=5 percentage points in each required efficacy contrast.",
            "The support-aware run should not breach refractory, chance-aware duplicate, boundary, or split/merge guardrails.",
        ],
        "frozen_evaluator_rules": {
            "population_measurable_fraction_min": 0.50,
            "missingness_improvement_pp_min": 5.0,
            "refractory_candidate_minus_baseline_max": 0.01,
            "chance_duplicate_candidate_minus_baseline_max": 0.01,
            "boundary_candidate_minus_baseline_max": 0.01,
            "split_merge_burden_max": 0.02,
            "waveform_instability_max": 0.05,
            "repeat_missingness_absolute_difference_pp_max": 2.0,
            "repeat_exclusive_retention_shortfall_max": 0.02,
        },
        "execution_acceptance": [
            "one managed service start, Restart=no, both fresh namespaces absent at preflight",
            "both arms exit zero and write COMPLETE.json last after semantic validation and full artifact inventories",
            "all source/input/config hashes and effective settings match; dshift is None and nblocks is zero",
            "support-read phase evidence covers covariance/whitening, PCA/template learning, universal detection/features and learned extraction for repaired_B384",
            "strictly increasing integer kept indices (or Boolean flatnonzero), exact full-row ancestry, crop-local full_st nondecreasing, and global spike_times == full_st[kept,0] + imin",
        ],
        "interpretation": {
            "repeat_failure_or_unmeasured": "entire comparison inconclusive; do not attribute repaired-arm effects",
            "required_guardrail_breach": "reject under frozen rule",
            "required_endpoint_unmeasured_or_unresolved": "inconclusive; never substitute Jaccard, rate correlation, counts, or descriptive retention",
            "all_required_pairs_pass_and_repeat_pass": "advance only to the next precommitted validation rung, not production",
            "identity": "UNRESOLVED unless a separately frozen identity endpoint is supplied",
            "waveforms": "UNMEASURED unless separately captured and bound",
            "no_retry_or_tuning": "preserve first failure/result; any redesign uses a fresh namespace and independent review",
        },
    }

    contract = {
        "schema": "en-first-medium-trained-pair-proposed-one-execution-contract-v1",
        "status": "DRAFT_NOT_EXECUTABLE_PENDING_INDEPENDENT_REVIEW_AND_BLOCKERS",
        "execution_enabled": False,
        "milestone": "representative 600-second trained REF-repeat plus support-aware repaired-B pair",
        "decision_changed": "whether the candidate is sufficiently measurable, repeatable and guardrail-safe to progress to a longer development window",
        "cheapest_adequate_test": "two serial fresh 600-second native trained arms; no parameter search",
        "arm_order": ["REF384_repeat", "repaired_B384"],
        "order_policy": "always attempt repaired_B384 after a completed REF384_repeat without inspecting scientific outcomes; stop only on execution/validation failure",
        "planned_paths": {
            "root": str(PLANNED_ROOT), "stage": str(PLANNED_STAGE), "numba_cache": str(PLANNED_CACHE),
            "logs": str(ROOT / "testing/outputs/en_first_medium_trained_pair_20261002_v1"),
        },
        "command_draft": str(ROOT / "environments/rescue-production/.venv/bin/python") + " -m testing.en_first_medium_trained_pair_launch --contract <reviewed-enabled-contract.json> --execute-reviewed-once",
        "service": {
            "name": "en-first-medium-trained-pair-20261002-v1.service",
            "type": "exec", "restart": "no", "start_limit": 1,
            "CPUQuota": "1600%", "AllowedCPUs": "0-15", "MemoryMax": 68719476736,
            "RuntimeMaxSec": 2700, "TasksMax": 256, "GPU_count_max": 1,
            "persistent_output_bytes_max": 17179869184,
            "job_manager": "systemd user service; must pass a detached dummy survival proof before launch",
        },
        "stop_conditions": [
            "any binding/config/source/input/clock/geometry mismatch",
            "any planned output/stage/cache/log namespace already exists",
            "GPU unavailable or free storage below 200 GiB at launch",
            "unexpected dshift, adaptive-bank reuse, mask-order violation, missing native hook phase, semantic validation failure, resource limit, nonzero exit, or missing completion",
            "any RF/holdout access, parameter search, preprocessed voltage copy, or unbounded transfer",
        ],
        "checkpoint_policy": "each arm is atomic; an interruption restarts that arm only after evidence review in a new namespace. A completed first arm is immutable and reusable. This is not within-arm resume.",
        "preservation": {
            "failures": "preserve run_failure, journal, stdout/stderr, partial namespace and no COMPLETE",
            "first_result": "never overwrite or tune; COMPLETE last after inventories",
            "outputs": "hash native outputs, adaptive snapshot, phase audit, effective config, environment, service properties and resource observations",
        },
        "post_run_dependencies": [
            "regenerate truncation QC for both fresh arms under a separately reviewed saved-output-only contract",
            "freeze the four-arm evaluator config with new path-bound identities and QC byte hashes",
            "run evaluator once in a fresh saved-output-only namespace; RF/holdout remain sealed",
        ],
        "completion_condition": "both fresh arms complete and validate in the one service start; this contract does not include QC/evaluator execution",
    }

    blockers = [
        {
            "id": "TRAINED_VALIDATOR_NONZERO_ORIGIN_BUG",
            "status": "OPEN_LAUNCH_STOPPER",
            "evidence": "Kilosort io.py:381 exports spike_times=st[:,0]+imin while io.py:472-478 saves crop-local full_st=st. Accepted trained source lines 443-450 incorrectly require direct equality. Saved REF384 and B384 each have spike_times-full_st[kept,0]==208498882 for every retained row.",
            "resolution": "targeted fresh source/config/fixture packet: validate global spike_times against selected crop-local full_st plus config crop.imin; retain strict kept ordering and crop-local nondecreasing checks; add positive nonzero-origin and negative wrong/double-offset fixtures through the real completion boundary; independently review before any launch.",
        },
        {
            "id": "GPU_DRIVER_UNAVAILABLE",
            "status": "OPEN",
            "evidence": "2026-10-02 preparation nvidia-smi could not communicate with the NVIDIA driver.",
            "resolution": "restore GPU driver/device visibility and record a fresh nvidia-smi/resource receipt before review/launch.",
        },
        {
            "id": "PAIR_LAUNCHER_NOT_IMPLEMENTED_OR_REVIEWED",
            "status": "OPEN",
            "evidence": "the reviewed trained entry always installs the support policy and therefore cannot be used for exact no-mask REF384_repeat; no serial heterogeneous wrapper exists.",
            "resolution": "implement a narrow execution-disabled wrapper using the frozen historical REF run path for REF384_repeat and the accepted v4 trained entry for repaired_B384, add known-answer/no-launch tests, then independently review exact source/config hashes.",
        },
        {
            "id": "COORDINATOR_STRATEGY_NOT_MOUNTED_ON_H5",
            "status": "OPEN",
            "evidence": "C:/Users/Declan/Documents/2026/spikesorting/POST_QC_EVALUATOR_STRATEGY_20261002.md is not mounted at /mnt/c or /mnt/C on H5.",
            "resolution": "publish or copy an immutable strategy packet/hash to shared project storage and reconcile this draft against it before launch review.",
        },
        {
            "id": "DELEGATED_COMPLETE_HASH_TYPO",
            "status": "RESOLVED_WITH_VISIBLE_CORRECTION",
            "evidence": "delegation named 54af...06bc6d..., while the immutable review COMPLETE.json hashes to 54af269b287b3cfa06bcf6dc7f28fc5bd9ac3188cbf845baa8a331a37acaf34f and binds the requested manifest.",
            "resolution": "this packet binds the actual verified hash; later review must acknowledge the correction.",
        },
        {
            "id": "FUTURE_ARM_QC_IDENTITIES_UNKNOWN",
            "status": "EXPECTED_DOWNSTREAM_NOT_SORT_LAUNCH_BLOCKER",
            "evidence": "QC and evaluator identities are path/content bound and cannot exist before the fresh arm outputs.",
            "resolution": "generate QC after successful arms, then freeze exact evaluator identities/hashes before the one evaluator traversal.",
        },
        {
            "id": "CURRENT_EVALUATOR_CANNOT_ADVANCE_WITHOUT_NEW_EVIDENCE_DECISION",
            "status": "OPEN_SCIENTIFIC_DECISION_NOT_SORT_EXECUTION_FAILURE",
            "evidence": "post-QC report has population measurable fraction 0.0, missingness UNMEASURED, identity UNRESOLVED, waveform UNMEASURED and split/merge burden 1.0 versus 0.02. Evaluator-v3 requires measured waveform/repeat guardrails and retains the unresolved identity axis.",
            "resolution": "before outcome access, independently accept the current split/merge definition and freeze a measurable waveform evidence plan, or explicitly freeze that the trained run is evidence-generation only and cannot produce an advancement verdict. Never substitute Jaccard/rate/count proxies for identity.",
        },
    ]

    readiness = {
        "schema": "en-first-medium-trained-pair-preparation-readiness-v1",
        "verdict": "NOT_READY_FOR_EXECUTION",
        "authorization": "standing user authorization applies; no renewed human approval is required solely for arm/window/resource changes",
        "no_execution_performed": True,
        "namespaces_fresh_at_preparation": [str(PLANNED_ROOT), str(PLANNED_STAGE), str(PLANNED_CACHE)],
        "resource_snapshot": {
            "data_volume_free_bytes": 1181955702784,
            "shared_volume_free_bytes": 246455500800,
            "gpu": "UNAVAILABLE_nvidia_smi_driver_communication_failure",
        },
        "blockers": blockers,
        "resolved_preparation": [
            "accepted v4 trained-entry implementation and actual review hash bound",
            "exact prospective namespaces and serial arm order frozen",
            "input, crop, field, ops, source, model/template/whitening provenance policy frozen",
            "intended-factor arm differences made explicit",
            "resource/service/stop/preservation contract drafted",
            "predictions, numeric acceptance thresholds and interpretation rules frozen before outcomes",
            "post-QC evaluator and QC evidence bound; future fresh-arm QC dependency declared",
            "native hook, mask ordering, adaptive refit, completion and row-order preconditions frozen",
            "nonzero-origin terminal-validator defect identified before real execution",
        ],
    }

    unit = f"""[Unit]\nDescription=PROPOSED ONLY - EN first-medium trained pair\nConditionPathExists=!{PLANNED_ROOT}\nConditionPathExists=!{PLANNED_STAGE}\nConditionPathExists=!{PLANNED_CACHE}\n\n[Service]\nType=exec\nRestart=no\nWorkingDirectory=/tmp\nEnvironment=PYTHONPATH={PLANNED_STAGE}/source\nEnvironment=NUMBA_CACHE_DIR={PLANNED_CACHE}\nEnvironment=OMP_NUM_THREADS=16\nEnvironment=MKL_NUM_THREADS=16\nEnvironment=OPENBLAS_NUM_THREADS=16\nAllowedCPUs=0-15\nCPUQuota=1600%\nMemoryMax=68719476736\nRuntimeMaxSec=2700\nTasksMax=256\nExecStart={ROOT}/environments/rescue-production/.venv/bin/python -m testing.en_first_medium_trained_pair_launch --contract <REVIEWED_ENABLED_CONTRACT> --execute-reviewed-once\n\n# DRAFT ONLY: do not install/start until blocker closure and independent review.\n"""

    with tempfile.TemporaryDirectory(prefix=".trained_pair_prep_", dir=DEST.parent) as td:
        packet = Path(td) / DEST.name
        (packet / "config").mkdir(parents=True)
        (packet / "bindings").mkdir()
        write_json(packet / "PREPARATION_READINESS.json", readiness)
        write_json(packet / "PROPOSED_ONE_EXECUTION_CONTRACT.json", contract)
        write_json(packet / "INTENDED_FACTOR_DIFF.json", intended_diff)
        write_json(packet / "PREDICTIONS_ACCEPTANCE_INTERPRETATION.json", rules)
        write_json(packet / "PROVENANCE_BINDINGS.json", provenance)
        write_json(packet / "evidence/CLOCK_ORIGIN_KNOWN_ANSWER.json", clock_known_answer)
        write_json(packet / "config/repaired_B384.execution_disabled.proposed.v1.json", repaired)
        write_json(packet / "config/REF384_repeat.execution_disabled.proposed.v1.json", repeat)
        (packet / "service/PROPOSED_NOT_INSTALLED.service").parent.mkdir()
        (packet / "service/PROPOSED_NOT_INSTALLED.service").write_text(unit)
        for name, source in (
            ("design_review", REVIEW), ("design_candidate", CANDIDATE),
            ("post_qc_result", POST_QC), ("qc_result", QC_RESULT),
            ("qc_review", QC_REVIEW), ("evaluator_request", EVAL_REQUEST),
        ):
            shutil.copy2(source / "MANIFEST.sha256", packet / f"bindings/{name}_MANIFEST.sha256")
            shutil.copy2(source / "COMPLETE.json", packet / f"bindings/{name}_COMPLETE.json")
        (packet / "README.md").write_text(
            "# First-medium trained-pair preparation readiness\n\n"
            "Verdict: **NOT_READY_FOR_EXECUTION**. This is an execution-disabled preparation packet; no recording, voltage, sorting, training, detection, RF, holdout, or waveform operation ran.\n\n"
            "The accepted support-aware trained entry has a real-data clock defect despite its design-ready review: Kilosort exports global `spike_times = full_st[:,0] + imin`, while the validator requires direct equality to crop-local `full_st`. Both historical real arms show the exact +208,498,882 offset on every retained row. A targeted nonzero-origin repair and new independent review are mandatory before launch.\n\n"
            "An exact `REF384_repeat` also must not use the support-aware entry because it unconditionally installs the support policy. The proposed one-start service therefore requires a narrow heterogeneous wrapper: frozen historical no-mask REF path first, repaired-B path second, without inspecting outcomes between arms. That wrapper remains unimplemented and independently unreviewed.\n\n"
            "The GPU is also unavailable at preparation time, and the coordinator's Windows strategy document is not mounted on H5. These are explicit blockers, not requests for renewed human authorization.\n\n"
            "The delegation's COMPLETE hash contained `...06bc6d...`; the actual immutable review file hashes to `54af269b287b3cfa06bcf6dc7f28fc5bd9ac3188cbf845baa8a331a37acaf34f`. This packet visibly binds the actual hash and the requested manifest `9ff42c...`.\n\n"
            "## Implementation checks\n\n"
            "- Done: verified design review/candidate, post-QC evaluator result, QC result/review, evaluator request, all source hashes, input manifests, probe, field, and comparator ops.\n"
            "- Done: inspected the actual native call order and confirmed repaired-B masking occurs after temporal filtering/CAR and before W, with independent C/W/PCA/template/detection/clustering refits.\n"
            "- Done: confirmed strict kept ordering, adaptive-bank snapshot, failure preservation and completion-last protections; found that the accepted row-ancestry check omits the nonzero crop origin and would reject valid real output.\n"
            "- Done: froze exact namespaces, intended-factor differences, service/resource bounds, stop conditions, artifact receipts, predictions, thresholds and interpretation rules.\n"
            "- Not done: targeted clock repair/fixtures/review, heterogeneous pair wrapper implementation/review, GPU recovery receipt, coordinator-strategy hash reconciliation, waveform/split-merge evidence decision, enabled config, service installation, or any data execution.\n"
            "- Can establish: current preparation state and a concrete review target for one later managed two-arm execution.\n"
            "- Cannot establish: launch readiness, successful sorting/training, repeatability, candidate efficacy, identity, waveform quality, or advancement.\n"
        )
        members = [p for p in sorted(packet.rglob("*")) if p.is_file()]
        (packet / "MANIFEST.sha256").write_text("\n".join(f"{sha(p)}  {p.relative_to(packet)}" for p in members) + "\n")
        manifest = sha(packet / "MANIFEST.sha256")
        packet.rename(DEST)
        write_json(DEST / "COMPLETE.json", {
            "schema": "immutable-preparation-completion-v1", "status": "complete",
            "readiness_verdict": "NOT_READY_FOR_EXECUTION", "execution_enabled": False,
            "manifest_sha256": manifest, "completion_written_last": True,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        })
        print(json.dumps({"destination": str(DEST), "manifest_sha256": manifest, "complete_sha256": sha(DEST / "COMPLETE.json")}, indent=2))


if __name__ == "__main__":
    main()
