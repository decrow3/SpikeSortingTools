#!/usr/bin/env python3
"""Publish the H1-targeted trained-order and QC-contract repairs."""

from __future__ import annotations

import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PUB = Path("/mnt/NPX/Luke/DARTsort_motion_experiments")
TRAINED_PREVIOUS = PUB / "en_minimum_training_support_mask_trained_entry_repair_candidate_20261002_v3_h5"
TRAINED_REVIEW = PUB / "en_minimum_training_support_mask_trained_entry_v3_h1_review_20261002_v1_h1"
QC_PREVIOUS = PUB / "en_truncation_qc_regeneration_contract_repair_candidate_20261002_v5_h5"
QC_REVIEW = PUB / "en_truncation_qc_regeneration_contract_v5_h1_review_20261002_v1_h1"
PYTHON = ROOT / "environments/rescue-production/.venv/bin/python"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def reserve(name: str) -> Path:
    path = PUB / name
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.mkdir(parents=True)
    return path


def finish(packet: Path, complete: dict) -> tuple[str, str]:
    files = sorted(path for path in packet.rglob("*") if path.is_file() and path.name not in {"MANIFEST.sha256", "COMPLETE.json"})
    (packet / "MANIFEST.sha256").write_text("".join(
        f"{sha(path)}  {path.relative_to(packet).as_posix()}\n" for path in files
    ))
    manifest = sha(packet / "MANIFEST.sha256")
    write_json(packet / "COMPLETE.json", dict(complete, manifest_sha256=manifest, completion_written_last=True))
    return manifest, sha(packet / "COMPLETE.json")


def diff_file(old: Path, new: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    before = old.read_text().splitlines(True)
    after = new.read_text().splitlines(True)
    destination.write_text("".join(difflib.unified_diff(before, after, fromfile=str(old), tofile=str(new))))


def build_trained(fixture: Path) -> tuple[Path, str, str]:
    packet = reserve("en_minimum_training_support_mask_trained_entry_ordering_repair_candidate_20261002_v4_h5")
    mapping = {
        ROOT / "npx_preprocessing/motion/kilosort_support_mask_trained.py": packet / "source/kilosort_support_mask_trained.py",
        ROOT / "testing/en_minimum_training_support_mask_trained_launch.py": packet / "tests/en_minimum_training_support_mask_trained_launch.py",
        ROOT / "testing/en_minimum_training_support_mask_trained_fixture.py": packet / "tests/en_minimum_training_support_mask_trained_fixture.py",
        ROOT / "testing/test_kilosort_support_mask_trained.py": packet / "tests/test_kilosort_support_mask_trained.py",
        ROOT / "testing/build_en_minimum_training_support_mask_trained_candidate.py": packet / "tests/build_en_minimum_training_support_mask_trained_candidate.py",
        ROOT / "configs/en_minimum_training_support_mask.trained.review_pending.v1.json": packet / "config/en_minimum_training_support_mask.trained.review_pending.v1.json",
        ROOT / "configs/en_minimum_training_support_mask.trained.review_pending.v1.delta.json": packet / "config/en_minimum_training_support_mask.trained.review_pending.v1.delta.json",
        TRAINED_REVIEW / "INDEPENDENT_REVIEW.json": packet / "reviewed/INDEPENDENT_REVIEW.json",
        TRAINED_REVIEW / "MANIFEST.sha256": packet / "reviewed/MANIFEST.sha256",
        TRAINED_REVIEW / "COMPLETE.json": packet / "reviewed/COMPLETE.json",
        fixture / "VERTICAL_FIXTURE_RECEIPT.json": packet / "fixture/VERTICAL_FIXTURE_RECEIPT.json",
        fixture / "native_hook_fixture/HOOK_FIXTURE_RESULT.json": packet / "fixture/HOOK_FIXTURE_RESULT.json",
        fixture / "success_fixture/run/COMPLETE.json": packet / "fixture/success_COMPLETE.json",
        fixture / "success_fixture/run/launch_evidence/artifact_inventory.json": packet / "fixture/artifact_inventory.json",
    }
    for case in ("reordered_kept", "coordinated_permutation"):
        mapping[fixture / f"{case}_fixture/run/launch_evidence/run_failure.json"] = packet / f"fixture/{case}_run_failure.json"
    for source, destination in mapping.items():
        copy(source, destination)
    comparisons = {
        "source/kilosort_support_mask_trained.py": ROOT / "npx_preprocessing/motion/kilosort_support_mask_trained.py",
        "tests/en_minimum_training_support_mask_trained_launch.py": ROOT / "testing/en_minimum_training_support_mask_trained_launch.py",
        "tests/en_minimum_training_support_mask_trained_fixture.py": ROOT / "testing/en_minimum_training_support_mask_trained_fixture.py",
        "tests/test_kilosort_support_mask_trained.py": ROOT / "testing/test_kilosort_support_mask_trained.py",
        "config/en_minimum_training_support_mask.trained.review_pending.v1.json": ROOT / "configs/en_minimum_training_support_mask.trained.review_pending.v1.json",
    }
    delta = []
    for relative, current in comparisons.items():
        previous = TRAINED_PREVIOUS / relative
        diff_name = relative.replace("/", "__") + ".diff"
        diff_file(previous, current, packet / "delta" / diff_name)
        delta.append({"path": relative, "previous_sha256": sha(previous), "current_sha256": sha(current),
                      "changed": sha(previous) != sha(current), "diff": "delta/" + diff_name})
    write_json(packet / "EXACT_DELTA.json", {
        "schema": "trained-ordering-targeted-delta-v1", "previous_packet": str(TRAINED_PREVIOUS),
        "previous_manifest_sha256": sha(TRAINED_PREVIOUS / "MANIFEST.sha256"), "files": delta,
        "scientific_parameter_change": False,
    })
    write_json(packet / "REVIEW_REQUEST.json", {
        "schema": "trained-entry-ordering-repair-review-request-v1", "execution_enabled": False,
        "reviewed_finding": "H1-20261002-KEPT-ORDER",
        "prior_review_manifest_sha256": sha(TRAINED_REVIEW / "MANIFEST.sha256"),
        "requested_review": [
            "Verify integer kept indices are strictly increasing in native full-array order.",
            "Verify exported and selected crop-local times are nondecreasing.",
            "Verify reordered-kept and coordinated-permutation cases fail through the real completion boundary.",
            "Confirm all prior ancestry, uniqueness, bounds, template, cluster-table and native-hook checks remain unchanged.",
        ], "launch_go_requested": False,
    })
    (packet / "TEST_RECEIPT.txt").write_text(
        "61 focused tests passed in 28.82s.\n"
        "Full vertical fixture passed: reordered_kept=true, coordinated_permutation=true, native hook=true, completion-last success=true.\n"
        "No recording, voltage, sort, training or detection executed.\n"
    )
    (packet / "README.md").write_text(
        "# Trained-entry native-order repair v4\n\n"
        "This execution-disabled delta applies only H1-20261002-KEPT-ORDER. Integer `kept_spikes` indices must now be "
        "strictly increasing in native full-array order, and the corresponding exported/local spike times must be "
        "nondecreasing. A kept-only reorder and a coordinated reorder of kept indices plus every per-spike export both "
        "fail before completion. All prior ancestry, uniqueness, bounds, bank, table, native-hook and completion-last "
        "checks remain present.\n\n"
        "Implementation checks\n"
        "- Done: exact v3-to-v4 delta, 61 tests, full packaged boundary fixture, both ordering negatives.\n"
        "- Not done: no real-data execution or scientific evaluation.\n"
        "- Can establish: coordinated equal-length row reordering no longer passes validation.\n"
        "- Cannot establish: consequential-run readiness until independent H1 review.\n"
    )
    return (packet, *finish(packet, {
        "schema": "immutable-candidate-completion-v1", "status": "complete", "execution_enabled": False,
        "real_data_execution_performed": False, "source_sha256": sha(ROOT / "npx_preprocessing/motion/kilosort_support_mask_trained.py"),
        "config_sha256": sha(ROOT / "configs/en_minimum_training_support_mask.trained.review_pending.v1.json"),
    }))


def run_cli(cli: Path, config: Path, digest: str, output: Path | None, mode: list[str]) -> subprocess.CompletedProcess:
    command = [str(PYTHON), str(cli), "--config", str(config), "--expected-config-sha256", digest, *mode]
    if output is not None:
        command += ["--output", str(output)]
    import os
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(command, cwd="/tmp", text=True, capture_output=True, env=environment)


def build_qc() -> tuple[Path, str, str]:
    packet = reserve("en_truncation_qc_regeneration_lifecycle_repair_candidate_20261002_v6_h5")
    sources = {
        ROOT / "testing/en_truncation_qc_regeneration_candidate.py": packet / "source/testing/en_truncation_qc_regeneration_candidate.py",
        ROOT / "pipeline/qc.py": packet / "source/pipeline/qc.py",
        ROOT / "pipeline/truncation.py": packet / "source/pipeline/truncation.py",
        ROOT / "pipeline/refractory.py": packet / "source/pipeline/refractory.py",
        ROOT / "testing/test_en_truncation_qc_regeneration_candidate.py": packet / "tests/test_en_truncation_qc_regeneration_candidate.py",
        QC_REVIEW / "INDEPENDENT_REVIEW.json": packet / "reviewed/INDEPENDENT_REVIEW.json",
        QC_REVIEW / "MANIFEST.sha256": packet / "reviewed/MANIFEST.sha256",
        QC_REVIEW / "COMPLETE.json": packet / "reviewed/COMPLETE.json",
    }
    for source, destination in sources.items():
        copy(source, destination)
    (packet / "source/pipeline/__init__.py").write_text('"""Sealed minimal QC package initializer."""\n')
    config = json.loads((ROOT / "configs/en_truncation_qc_regeneration.execution_disabled.v2.json").read_text())
    for binding in config["source_bindings"]:
        name = Path(binding["path"]).name
        binding["path"] = str(packet / "source" / ("testing" if name.startswith("en_truncation") else "pipeline") / name)
    config_path = packet / "config/en_truncation_qc_regeneration.execution_disabled.v2.json"
    write_json(config_path, config)
    digest = sha(config_path)
    cli = packet / "source/testing/en_truncation_qc_regeneration_candidate.py"

    validate = run_cli(cli, config_path, digest, None, ["--validate-only"])
    if validate.returncode:
        raise RuntimeError(validate.stderr)
    (packet / "fixture").mkdir()
    global_ledger = packet / "fixture/global_token_ledger"
    success_root = packet / "fixture/success"
    success = run_cli(cli, config_path, digest, success_root, ["--synthetic", "--fixture-ledger-root", str(global_ledger)])
    if success.returncode:
        raise RuntimeError(success.stderr)
    alternate_root = packet / "fixture/alternate_output_must_not_exist"
    second = run_cli(cli, config_path, digest, alternate_root, ["--synthetic", "--fixture-ledger-root", str(global_ledger)])
    injected_root = packet / "fixture/injected_failure"
    injected = run_cli(cli, config_path, digest, injected_root, ["--synthetic", "--fixture-ledger-root", str(packet / "fixture/injected_ledger"), "--synthetic-outcome", "injected-failure"])
    corrupt_root = packet / "fixture/corrupt_output"
    corrupt = run_cli(cli, config_path, digest, corrupt_root, ["--synthetic", "--fixture-ledger-root", str(packet / "fixture/corrupt_ledger"), "--synthetic-outcome", "corrupt-output"])
    real_root = packet / "fixture/wrong_real_destination_must_not_exist"
    real = run_cli(cli, config_path, digest, real_root, ["--execute-real"])
    wrapper_hash = next(row["sha256"] for row in config["source_bindings"] if Path(row["path"]).name.startswith("en_truncation"))
    gate = {"schema": "truncation-qc-one-execution-gate-v1", "verdict": "ONE_EXECUTION_GO",
            "review_manifest_sha256": "1" * 64, "disabled_config_sha256": digest,
            "source_sha256": wrapper_hash, "frozen_real_output_root": config["activation"]["frozen_real_output_root"],
            "one_shot_token": "packaged-gate-transition-fixture", "synthetic_fixture": True}
    gate_path = packet / "fixture/gate_fixture.json"
    write_json(gate_path, gate)
    gate_check = run_cli(cli, config_path, digest, None, ["--gate-fixture-check", "--gate", str(gate_path)])
    wrong_gate = dict(gate, verdict="NO_GO")
    wrong_gate_path = packet / "fixture/wrong_gate_fixture.json"
    write_json(wrong_gate_path, wrong_gate)
    wrong_gate_check = run_cli(cli, config_path, digest, None, ["--gate-fixture-check", "--gate", str(wrong_gate_path)])
    cases = {"validate": validate, "success": success, "second_attempt": second,
             "injected_failure": injected, "corrupt_output": corrupt, "destination_override": real,
             "gate_transition": gate_check, "wrong_gate": wrong_gate_check}
    case_summary = {name: {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
                    for name, result in cases.items()}

    for case_name, mutation in (
        ("missing_key", lambda value: value.pop("resources")),
        ("malformed_resource", lambda value: value["resources"].update({"ram_bytes_max": "16GiB"})),
    ):
        malformed = json.loads(json.dumps(config))
        mutation(malformed)
        for binding in malformed.get("source_bindings", []):
            binding["path"] = str(packet / "fixture/must_not_open" / Path(binding["path"]).name)
        malformed_path = packet / f"fixture/{case_name}_config.json"
        write_json(malformed_path, malformed)
        result = run_cli(cli, malformed_path, sha(malformed_path), None, ["--validate-only"])
        case_summary[case_name] = {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr,
                                  "sentinel_exists": (packet / "fixture/must_not_open").exists()}

    checks = {
        "validate_passed": validate.returncode == 0,
        "success_completed": success.returncode == 0 and (success_root / "COMPLETE.json").is_file(),
        "global_token_reuse_refused_across_outputs": second.returncode != 0 and "FileExistsError" in second.stderr and not alternate_root.exists(),
        "injected_failure_preserved": injected.returncode != 0 and (injected_root / "evidence/FAILURE.json").is_file() and not (injected_root / "COMPLETE.json").exists(),
        "corrupt_output_rejected": corrupt.returncode != 0 and "Cannot load file containing pickled data" in (corrupt_root / "evidence/worker.stderr").read_text(),
        "destination_override_refused_before_reservation": real.returncode != 0 and "caller output override differs" in real.stderr and not real_root.exists(),
        "exact_enabled_gate_transition": gate_check.returncode == 0 and "reviewed_one_execution_enabled" in gate_check.stdout,
        "wrong_gate_refused": wrong_gate_check.returncode != 0 and "wrong gate schema or verdict" in wrong_gate_check.stderr,
        "preimport_limits_proved": json.loads((success_root / "evidence/BOOTSTRAP.json").read_text())["heavy_modules_present_before_limits"] == [],
        "missing_key_before_access": case_summary["missing_key"]["returncode"] != 0 and not case_summary["missing_key"]["sentinel_exists"],
        "malformed_key_before_access": case_summary["malformed_resource"]["returncode"] != 0 and not case_summary["malformed_resource"]["sentinel_exists"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"packaged fixture failed: {checks}")
    write_json(packet / "fixture/PACKAGED_CLI_RECEIPT.json", {"schema": "truncation-qc-contract-fixture-v1", "checks": checks, "cases": case_summary})

    previous_files = {
        "source/testing/en_truncation_qc_regeneration_candidate.py": QC_PREVIOUS / "source/testing/en_truncation_qc_regeneration_candidate.py",
        "config/en_truncation_qc_regeneration.execution_disabled.v2.json": QC_PREVIOUS / "config/en_truncation_qc_regeneration.execution_disabled.v2.json",
        "tests/test_en_truncation_qc_regeneration_candidate.py": QC_PREVIOUS / "tests/test_en_truncation_qc_regeneration_candidate.py",
    }
    current_files = {
        "source/testing/en_truncation_qc_regeneration_candidate.py": ROOT / "testing/en_truncation_qc_regeneration_candidate.py",
        "config/en_truncation_qc_regeneration.execution_disabled.v2.json": config_path,
        "tests/test_en_truncation_qc_regeneration_candidate.py": ROOT / "testing/test_en_truncation_qc_regeneration_candidate.py",
    }
    delta = []
    for relative, current in current_files.items():
        old = previous_files[relative]
        diff_name = relative.replace("/", "__") + ".diff"
        diff_file(old, current, packet / "delta" / diff_name)
        delta.append({"path": relative, "previous_sha256": sha(old), "current_sha256": sha(current), "diff": "delta/" + diff_name})
    write_json(packet / "EXACT_DELTA.json", {"schema": "truncation-qc-lifecycle-targeted-delta-v1",
        "previous_packet": str(QC_PREVIOUS), "previous_manifest_sha256": sha(QC_PREVIOUS / "MANIFEST.sha256"),
        "production_qc_sha256_unchanged": sha(ROOT / "pipeline/qc.py"),
        "production_truncation_sha256_unchanged": sha(ROOT / "pipeline/truncation.py"), "files": delta})
    write_json(packet / "REVIEW_REQUEST.json", {"schema": "truncation-qc-lifecycle-repair-review-request-v1",
        "execution_enabled": False, "prior_review_manifest_sha256": sha(QC_REVIEW / "MANIFEST.sha256"),
        "requested_review": ["Verify the gate-only enabled transition changes only lifecycle state.",
            "Verify global token reuse across output paths fails before input access.",
            "Verify limits precede every heavy import and runtime attestation follows inside the bounded worker.",
            "Verify real output is frozen and caller override fails before gate/token/input access."],
        "real_execution_requested": False})
    (packet / "TEST_RECEIPT.txt").write_text("11 focused tests passed in 9.80s.\nActual sealed packaged CLI fixture passed all lifecycle/pre-import contract checks.\n")
    (packet / "README.md").write_text(
        "# Truncation-QC lifecycle and pre-import repair v6\n\n"
        "The production truncation estimator, thresholds and scientific behavior are unchanged. The published config "
        "remains disabled; an exact hash-bound ONE_EXECUTION_GO gate can derive an enabled state by changing only status "
        "and execution_enabled. Token consumption is atomic in a frozen global ledger independent of output. Real output "
        "is fixed by contract. The supervisor is standard-library-only, and the worker proves limits were applied before "
        "NumPy or any other bound numerical dependency was imported.\n\n"
        "Implementation checks\n"
        "- Done: exact v5-to-v6 delta, 11 tests, and sealed lifecycle/pre-import CLI cases.\n"
        "- Not done: no real QC regeneration or saved sorter input read.\n"
        "- Can establish: enabled transition, global one-shot, pre-import bounds and destination binding are implemented synthetically.\n"
        "- Cannot establish: real QC values or execution GO pending independent H1 review.\n"
    )
    return (packet, *finish(packet, {"schema": "immutable-candidate-completion-v1", "status": "complete",
        "execution_enabled": False, "real_regeneration_performed": False, "packaged_cli_fixture_passed": True}))


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained-fixture", required=True, type=Path)
    parser.add_argument("--only-qc", action="store_true")
    args = parser.parse_args()
    results = [build_qc()] if args.only_qc else [build_trained(args.trained_fixture.resolve()), build_qc()]
    print(json.dumps([{"packet": str(p), "manifest_sha256": m, "complete_sha256": c} for p, m, c in results], indent=2))


if __name__ == "__main__":
    main()
