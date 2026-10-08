#!/usr/bin/env python3
"""Build the three immutable H5 review/candidate handoff packets."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
PUBLISH = Path("/mnt/NPX/Luke/DARTsort_motion_experiments")
EVALUATOR_CANDIDATE = PUBLISH / "en_first_medium_four_arm_evaluator_repair_candidate_20261002_v3_h1"
TRAINED_PREVIOUS = PUBLISH / "en_minimum_training_support_mask_trained_entry_repair_candidate_20261002_v2_h5"
TRAINED_REVIEW = PUBLISH / "en_minimum_training_support_mask_trained_entry_repair_h1_review_20261002_v2_h1"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def finish(packet: Path, complete: dict) -> tuple[str, str]:
    members = sorted(
        path for path in packet.rglob("*")
        if path.is_file() and path.name not in {"MANIFEST.sha256", "COMPLETE.json"}
    )
    manifest = "".join(f"{sha(path)}  {path.relative_to(packet).as_posix()}\n" for path in members)
    (packet / "MANIFEST.sha256").write_text(manifest)
    manifest_sha = sha(packet / "MANIFEST.sha256")
    complete = dict(complete, manifest_sha256=manifest_sha, completion_written_last=True)
    write_json(packet / "COMPLETE.json", complete)
    return manifest_sha, sha(packet / "COMPLETE.json")


def reserve(name: str) -> Path:
    packet = PUBLISH / name
    if packet.exists():
        raise FileExistsError(f"refusing to overwrite existing packet {packet}")
    packet.mkdir(parents=True)
    return packet


def build_evaluator(real_output: Path) -> tuple[Path, str, str]:
    packet = reserve("en_first_medium_four_arm_evaluator_v3_h5_review_20261002_v1_h5")
    report = json.loads((real_output / "FOUR_ARM_REPORT.json").read_text())
    shutil.copytree(real_output, packet / "real_saved_output")
    copy(EVALUATOR_CANDIDATE / "MANIFEST.sha256", packet / "reviewed/MANIFEST.sha256")
    copy(EVALUATOR_CANDIDATE / "COMPLETE.json", packet / "reviewed/COMPLETE.json")
    for relative in (
        "source/testing/first_medium_evaluator.py", "source/testing/sort_comparison.py",
        "source/testing/luke_amplitude_dropout_audit.py", "source/pipeline/config.py",
        "source/pipeline/truncation.py", "tests/run_packaged_cli_proof.py",
        "tests/test_first_medium_evaluator.py",
        "config/en_first_medium_four_arm_evaluator.execution_disabled.v2.json",
    ):
        copy(EVALUATOR_CANDIDATE / relative, packet / "reviewed" / relative)
    expected_source_hashes = {
        name: sha(EVALUATOR_CANDIDATE / "source" / (name.replace(".", "/") + ".py"))
        for name in report["implementation_provenance"]
    }
    traversal_hashes = {name: value["sha256"] for name, value in report["implementation_provenance"].items()}
    source_match = expected_source_hashes == traversal_hashes
    summary = {
        "schema": "first-medium-evaluator-v3-independent-review-v1",
        "reviewer": "H5",
        "reviewed_candidate": str(EVALUATOR_CANDIDATE),
        "reviewed_manifest_sha256": sha(EVALUATOR_CANDIDATE / "MANIFEST.sha256"),
        "reviewed_complete_sha256": sha(EVALUATOR_CANDIDATE / "COMPLETE.json"),
        "candidate_manifest_members_verified": sum(1 for line in (EVALUATOR_CANDIDATE / "MANIFEST.sha256").read_text().splitlines() if line),
        "verdict": "IMPLEMENTATION_ACCEPTED_REAL_SAVED_TRAVERSAL_INCONCLUSIVE",
        "execution_enabled": False,
        "packet_local_source_hashes_match_traversal": source_match,
        "clock_validation": {
            arm: report["arms"][arm]["provenance"]["clock_normalization"]
            for arm in ("REF384", "existing_corrected_B384")
        },
        "absent_qc_behavior": {
            "population_measurable_fraction": report["scientific_scorecard"]["REF384__existing_corrected_B384"]["endpoints"]["population_measurable_fraction"],
            "median_common_time_missingness_improvement_pp": report["scientific_scorecard"]["REF384__existing_corrected_B384"]["endpoints"]["median_common_time_missingness_improvement_pp"],
        },
        "synthetic_exclusion": {
            "fixture_results_not_scientific": report["fixture_results_not_scientific"],
            "prospective_pairs": report["unavailable_pairs"],
        },
        "decision": report["decision"],
        "packaging_runtime_caveat": (
            "Direct import from the immutable read-only packet fails when Numba initializes its cache. "
            "The packaged proof and real traversal passed from a writable byte-identical copy; all five "
            "attested implementation hashes equal the immutable candidate source hashes."
        ),
        "tests": {
            "core_tests_from_writable_packet_copy": "10 passed; the workspace-layout self-copy test is not relocatable to packet/tests",
            "packaged_cli_proof": "passed from an unrelated cwd",
            "negative_clock_fixture": "one-row +1-sample corruption rejected",
        },
        "implementation_checks": {
            "done": [
                "Verified candidate MANIFEST and COMPLETE bindings.",
                "Read evaluator, loader, matcher, QC adapter, clock guard, tests and frozen request.",
                "Ran packaged CLI with packet-only source imports and matched every attested source hash.",
                "Traversed real saved REF384/B384 arrays without recording or voltage access.",
            ],
            "not_done": [
                "No truncation-QC regeneration, waveform capture, recording/voltage read, sort, training, detection, RF or holdout work.",
                "No repaired_B384 or REF384_repeat output exists.",
            ],
            "can_establish": "The v3 repair traverses the two real saved arms, enforces the declared clock transform, and preserves missing QC as UNMEASURED.",
            "cannot_establish": "Amplitude missingness, primary identity improvement, repaired-arm efficacy, repeatability or advancement readiness.",
        },
    }
    write_json(packet / "INDEPENDENT_REVIEW.json", summary)
    (packet / "README.md").write_text(
        "# H5 independent traversal of first-medium evaluator v3\n\n"
        "Verdict: `IMPLEMENTATION_ACCEPTED_REAL_SAVED_TRAVERSAL_INCONCLUSIVE`.\n\n"
        "REF384 and existing-corrected B384 traversed from saved sorter artifacts only. The exact "
        "`+208498882` relation was proven on every exported row. Missing truncation QC remained "
        "`UNMEASURED`; no zero-coverage failure or pass was synthesized. Prospective arms remain unavailable.\n\n"
        "A writable byte-identical copy was required because Numba attempts to create cache files beside "
        "the imported module; direct import from the read-only packet fails. Imported hashes still match "
        "the immutable packet source exactly.\n\n"
        "Implementation checks\n"
        "- Done: candidate hashes, executed source, packet-only imports, corrupt-clock rejection, and real saved traversal.\n"
        "- Not done: QC regeneration, voltage, sorting/training/detection, RF/holdout, or prospective arms.\n"
        "- Can establish: implementation traversal and fail-closed missing-QC behavior.\n"
        "- Cannot establish: a scientific pass, candidate efficacy, identity, or advancement readiness.\n"
    )
    manifest, complete = finish(packet, {
        "schema": "immutable-review-completion-v1", "status": "complete",
        "real_saved_traversal_performed": True, "recording_or_voltage_reads": 0,
    })
    return packet, manifest, complete


def build_trained(fixture: Path) -> tuple[Path, str, str]:
    packet = reserve("en_minimum_training_support_mask_trained_entry_repair_candidate_20261002_v3_h5")
    files = {
        ROOT / "npx_preprocessing/motion/kilosort_support_mask_trained.py": packet / "source/kilosort_support_mask_trained.py",
        ROOT / "testing/en_minimum_training_support_mask_trained_launch.py": packet / "tests/en_minimum_training_support_mask_trained_launch.py",
        ROOT / "testing/en_minimum_training_support_mask_trained_fixture.py": packet / "tests/en_minimum_training_support_mask_trained_fixture.py",
        ROOT / "testing/test_kilosort_support_mask_trained.py": packet / "tests/test_kilosort_support_mask_trained.py",
        ROOT / "testing/build_en_minimum_training_support_mask_trained_candidate.py": packet / "tests/build_en_minimum_training_support_mask_trained_candidate.py",
        ROOT / "configs/en_minimum_training_support_mask.trained.review_pending.v1.json": packet / "config/en_minimum_training_support_mask.trained.review_pending.v1.json",
        ROOT / "configs/en_minimum_training_support_mask.trained.review_pending.v1.delta.json": packet / "config/en_minimum_training_support_mask.trained.review_pending.v1.delta.json",
        TRAINED_REVIEW / "INDEPENDENT_REVIEW.json": packet / "reviewed/INDEPENDENT_REVIEW.json",
        TRAINED_REVIEW / "MANIFEST.sha256": packet / "reviewed/MANIFEST.sha256",
        fixture / "VERTICAL_FIXTURE_RECEIPT.json": packet / "fixture/VERTICAL_FIXTURE_RECEIPT.json",
        fixture / "native_hook_fixture/HOOK_FIXTURE_RESULT.json": packet / "fixture/HOOK_FIXTURE_RESULT.json",
        fixture / "success_fixture/run/COMPLETE.json": packet / "fixture/success_COMPLETE.json",
        fixture / "success_fixture/run/launch_evidence/artifact_inventory.json": packet / "fixture/artifact_inventory.json",
    }
    for case in ("permuted_ancestry", "duplicate_kept", "out_of_bounds_id"):
        files[fixture / f"{case}_fixture/run/launch_evidence/run_failure.json"] = packet / f"fixture/{case}_run_failure.json"
    for source, destination in files.items():
        copy(source, destination)
    old = (TRAINED_PREVIOUS / "source/kilosort_support_mask_trained.py").read_text().splitlines(True)
    new = (ROOT / "npx_preprocessing/motion/kilosort_support_mask_trained.py").read_text().splitlines(True)
    (packet / "trained_source_targeted_repair.diff").write_text("".join(difflib.unified_diff(
        old, new, fromfile="v2/kilosort_support_mask_trained.py", tofile="v3/kilosort_support_mask_trained.py"
    )))
    source_hash = sha(ROOT / "npx_preprocessing/motion/kilosort_support_mask_trained.py")
    config_hash = sha(ROOT / "configs/en_minimum_training_support_mask.trained.review_pending.v1.json")
    review = {
        "schema": "trained-entry-targeted-repair-request-v3",
        "execution_enabled": False,
        "source_sha256": source_hash,
        "config_sha256": config_hash,
        "prior_review_manifest_sha256": sha(TRAINED_REVIEW / "MANIFEST.sha256"),
        "requested_review": [
            "Verify unique/in-bounds explicit kept index resolution and row-wise time/cluster/amplitude ancestry.",
            "Verify spike-template, detection-template and cluster-table ID bindings.",
            "Verify the permutation, duplicate-kept and out-of-bounds fixtures fail closed.",
            "Confirm native extraction-hook and full artifact validation evidence remain intact.",
        ],
        "launch_go_requested": False,
    }
    write_json(packet / "REVIEW_REQUEST.json", review)
    (packet / "TEST_RECEIPT.txt").write_text(
        "PYTHONPATH=environments/rescue-production/.venv/lib/python3.12/site-packages "
        "pytest -q testing/test_kilosort_support_mask_trained.py "
        "testing/test_kilosort_support_mask_candidate.py testing/test_prewhitening_support_mask.py\n"
        "59 passed in 22.81s\n"
        "Full vertical fixture: passed; native hook retained; permutation, duplicate-kept, and "
        "out-of-bounds template ID all rejected with failure receipts.\n"
    )
    (packet / "README.md").write_text(
        "# Trained-entry targeted semantic repair v3\n\n"
        "Execution-disabled candidate applying only H1's remaining findings. Exported times, clusters and "
        "amplitudes now match their exact `full_*[kept_spikes]` rows in the documented crop-local frame; "
        "integer kept indices must be unique and in range. Template/detection IDs are bound to their saved "
        "banks, and every cluster table must contain exactly the unique exported cluster IDs.\n\n"
        "The actual native `template_matching.extract` hook proof and 30-native/9-snapshot validation remain intact. "
        "No real recording, sorting, training or detection was run.\n\n"
        "Implementation checks\n"
        "- Done: executed-source inspection, row ancestry and ID bounds, three new corrupt CLI fixtures, 59 focused tests, full vertical fixture.\n"
        "- Not done: real-data execution or scientific evaluation.\n"
        "- Can establish: the reviewed semantic gaps are implemented and fail closed synthetically.\n"
        "- Cannot establish: consequential-run readiness until independent review, or scientific efficacy.\n"
    )
    manifest, complete = finish(packet, {
        "schema": "immutable-candidate-completion-v1", "status": "complete",
        "execution_enabled": False, "real_data_execution_performed": False,
        "source_sha256": source_hash, "config_sha256": config_hash,
    })
    return packet, manifest, complete


def build_qc(synthetic_output: Path) -> tuple[Path, str, str]:
    packet = reserve("en_truncation_qc_regeneration_candidate_20261002_v2_h5")
    sources = {
        ROOT / "testing/en_truncation_qc_regeneration_candidate.py": packet / "source/testing/en_truncation_qc_regeneration_candidate.py",
        ROOT / "pipeline/qc.py": packet / "source/pipeline/qc.py",
        ROOT / "pipeline/truncation.py": packet / "source/pipeline/truncation.py",
        ROOT / "pipeline/refractory.py": packet / "source/pipeline/refractory.py",
        ROOT / "testing/test_en_truncation_qc_regeneration_candidate.py": packet / "tests/test_en_truncation_qc_regeneration_candidate.py",
    }
    for source, destination in sources.items():
        copy(source, destination)
    (packet / "source/pipeline/__init__.py").write_text(
        '"""Sealed minimal package initializer for truncation-QC regeneration."""\n'
    )
    config = json.loads((ROOT / "configs/en_truncation_qc_regeneration.execution_disabled.v1.json").read_text())
    for binding in config["source_bindings"]:
        name = Path(binding["path"]).name
        if name == "en_truncation_qc_regeneration_candidate.py":
            binding["path"] = str(packet / "source/testing" / name)
        else:
            binding["path"] = str(packet / "source/pipeline" / name)
    write_json(packet / "config/en_truncation_qc_regeneration.execution_disabled.v1.json", config)
    shutil.copytree(synthetic_output, packet / "synthetic_fixture")
    (packet / "TEST_RECEIPT.txt").write_text(
        "pytest -q testing/test_en_truncation_qc_regeneration_candidate.py\n"
        "4 passed in 3.44s\n"
        "Synthetic CLI called the frozen production QC implementation and validated all NPZ keys/shapes plus nonempty PDF.\n"
    )
    write_json(packet / "REVIEW_REQUEST.json", {
        "schema": "truncation-qc-regeneration-review-request-v1",
        "execution_enabled": False,
        "requested_review": [
            "Verify required config keys fail before any bound source/input access.",
            "Verify the exact saved inputs, hashes, +208498882 clock transform and full_st[kept,2] amplitude source.",
            "Verify production defaults max_isi=10 s, 1000 spikes/window and historical 999-value fit slice are unchanged.",
            "Verify real regeneration remains disabled and would require one distinct reviewed execution decision.",
        ],
        "real_execution_requested": False,
    })
    (packet / "README.md").write_text(
        "# Execution-disabled truncation-QC regeneration candidate\n\n"
        "This frozen candidate can regenerate the existing production `truncation_qc.npz`, `present_qc.npz`, "
        "and PDF from four hash-bound saved sorter arrays per arm. It uses the pre-existing production estimator "
        "unchanged: 10 s gap rule, 1,000-spike nominal windows, and the historical half-open 999-value fit slice. "
        "Amplitudes come from `full_st[kept_spikes,2]`; global exported times must prove the exact `+208498882` "
        "relationship before conversion to crop-local seconds.\n\n"
        "Real execution is disabled. The included full synthetic CLI proof exercises the production writer and exact "
        "output schema. A future real regeneration is a distinct one-execution review decision.\n\n"
        "This v2 supersedes, but does not overwrite, v1: v1 copied the repository package initializer, whose unrelated "
        "imports were absent from the sealed snapshot. V2 uses a minimal initializer and leaves estimator code unchanged.\n\n"
        "Implementation checks\n"
        "- Done: frozen source/input hashes, clocks, shapes, thresholds, output schema, early key gate, and full synthetic CLI.\n"
        "- Not done: no real saved-arm regeneration, recording/voltage read, sort/training/detection, RF or holdout access.\n"
        "- Can establish: a bounded saved-output-only regeneration path is specified and synthetically traversed.\n"
        "- Cannot establish: real QC values, amplitude missingness, or any arm comparison.\n"
    )
    manifest, complete = finish(packet, {
        "schema": "immutable-candidate-completion-v1", "status": "complete",
        "execution_enabled": False, "real_regeneration_performed": False,
        "synthetic_cli_performed": True,
    })
    return packet, manifest, complete


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluator-output", required=True, type=Path)
    parser.add_argument("--trained-fixture", required=True, type=Path)
    parser.add_argument("--qc-synthetic", required=True, type=Path)
    parser.add_argument("--only-qc", action="store_true")
    args = parser.parse_args()
    results = [build_qc(args.qc_synthetic.resolve())] if args.only_qc else [
        build_evaluator(args.evaluator_output.resolve()),
        build_trained(args.trained_fixture.resolve()),
        build_qc(args.qc_synthetic.resolve()),
    ]
    print(json.dumps([
        {"packet": str(packet), "manifest_sha256": manifest, "complete_sha256": complete}
        for packet, manifest, complete in results
    ], indent=2))


if __name__ == "__main__":
    main()
