#!/usr/bin/env python3
"""Managed downstream QC and frozen REF comparison for the completed imec1 sort."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from pipeline.downstream import (
    build_sort_identity,
    pin_sort_identity,
    run_curation_stage,
    run_matlab_export_stage,
    run_qc_stage,
    run_standard_qc_stage,
)
from testing.luke_improved_rigid_comparison import run as run_comparison
from testing.rigid_comparison_handoff import publish as publish_handoff


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path, block_size: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(partial, path)


def status(root: Path, stage: str, state: str = "running", **extra) -> None:
    atomic_json(root / "STATUS.json", {
        "schema": "en-full-session-part2c-imec1-status-v1",
        "state": state,
        "stage": stage,
        "updated_utc": now(),
        **extra,
    })


def verify_file(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or sha256(path) != expected:
        raise RuntimeError(f"{label} is missing or changed: {path}")


def preflight(contract: dict, config_path: Path, controller_path: Path) -> dict:
    inputs = contract["inputs"]
    execution = contract["execution"]
    verify_file(controller_path, execution["controller_sha256"], "controller")
    verify_file(config_path, execution["comparison_config_sha256"], "comparison config")
    for row in execution["clean_source_files"]:
        verify_file(Path(row["path"]), row["sha256"], "clean release source")
    verify_file(Path(inputs["sort_complete_path"]), inputs["sort_complete_sha256"], "sort completion receipt")
    verify_file(Path(inputs["recording_manifest_path"]), inputs["recording_manifest_sha256"], "recording manifest")
    verify_file(Path(inputs["baseline_curation_receipt"]), inputs["baseline_curation_receipt_sha256"], "baseline curation receipt")
    verify_file(Path(inputs["baseline_qc_receipt"]), inputs["baseline_qc_receipt_sha256"], "baseline QC receipt")
    if Path(inputs["recording_binary"]).stat().st_size != inputs["recording_binary_bytes"]:
        raise RuntimeError("candidate recording byte size changed")
    completion = json.loads(Path(inputs["sort_complete_path"]).read_text())
    if completion.get("status") != "complete_sort_only_downstream_pending" or completion.get("sort_invocations") != 1:
        raise RuntimeError("candidate sort completion state differs")
    if completion.get("recording_manifest_sha256") != inputs["recording_manifest_sha256"]:
        raise RuntimeError("sort receipt is not bound to the candidate recording manifest")
    manifest = json.loads(Path(inputs["recording_manifest_path"]).read_text())
    if manifest.get("complete") is not True or manifest.get("recording_content_sha256") != inputs["recording_content_sha256"]:
        raise RuntimeError("candidate recording manifest differs")
    current = build_sort_identity(Path(inputs["candidate_sort_dir"]))
    if current["identity_digest"] != inputs["candidate_sort_identity_digest"]:
        raise RuntimeError("candidate sort identity changed")
    baseline = json.loads(Path(inputs["baseline_sort_identity"]).read_text())
    if baseline.get("identity_digest") != inputs["baseline_sort_identity_digest"]:
        raise RuntimeError("baseline sort identity changed")
    if shutil.disk_usage(Path(execution["output_root"]).parent).free < execution["minimum_free_bytes"]:
        raise RuntimeError("local free space below frozen output-safety floor")
    return current


def write_candidate_metadata(root: Path, contract: dict, identity: dict, curation: dict) -> None:
    inputs = contract["inputs"]
    manifest = json.loads(Path(inputs["recording_manifest_path"]).read_text())
    application = {
        "schema": "en-imec1-part2c-comparison-application-v1",
        "development_only": True,
        "sort_identity_digest": identity["identity_digest"],
        "duration_s": manifest["num_samples"] / manifest["sampling_frequency_hz"],
        "num_samples": manifest["num_samples"],
        "channels": manifest["num_channels"],
        "position_frame": "corrected-frame saved Kilosort spike_positions from exact-lattice-remapped recording",
        "comparison_depth_handling": "unchanged frozen comparator: each arm is independently classified against identical numeric domains; no cross-frame transform",
        "recording_content_sha256": manifest["recording_content_sha256"],
        "internal_motion_correction": False,
        "release_commit": contract["release_commit"],
    }
    atomic_json(root / "downstream/application_contract.json", application)
    atomic_json(root / "downstream/summary.json", {
        "status": "complete",
        "development_only": True,
        "sort_identity_digest": identity["identity_digest"],
        "sort_summary": identity["sort_summary"],
        "post_curation_summary": curation["summary"],
        "num_samples": manifest["num_samples"],
        "output_channels": manifest["num_channels"],
        "scientific_status": "comparison_pending",
        "correction": "imec1 rounded exact-lattice two-layer field",
        "internal_motion_correction": False,
        "release_commit": contract["release_commit"],
    })


def run(args: argparse.Namespace) -> None:
    contract = json.loads(args.contract.read_text())
    root = Path(contract["execution"]["output_root"])
    if root != args.output:
        raise RuntimeError("output path differs from frozen contract")
    root.mkdir(parents=True, exist_ok=True)
    if any((root / name).exists() for name in ("downstream", "handoff", "comparison", "RESULT.json")):
        raise RuntimeError("downstream output namespace is not fresh")
    status(root, "preflight")
    identity = preflight(contract, args.config, Path(__file__).resolve())
    if args.preflight_only:
        print(json.dumps({"status": "pass", "sort_identity_digest": identity["identity_digest"]}, indent=2))
        return

    inputs = contract["inputs"]
    downstream = root / "downstream"
    status(root, "curation")
    pinned = pin_sort_identity(Path(inputs["candidate_sort_dir"]), downstream / "sort_identity.json")
    curation = run_curation_stage(
        Path(inputs["candidate_sorter_output"]), downstream / "cur", pinned,
        cosine_threshold=0.9, ccg_threshold=0.5,
    )
    status(root, "legacy_qc")
    qc = run_qc_stage(
        Path(inputs["candidate_recording_dir"]), downstream / "cur/cur_output",
        downstream / "qc", pinned, waveform_seed=0,
    )
    status(root, "standard_qc")
    run_standard_qc_stage(
        Path(inputs["candidate_recording_dir"]), downstream / "cur/cur_output",
        downstream / "qc", downstream / "qc/standard", pinned,
    )
    status(root, "matlab_export")
    run_matlab_export_stage(downstream / "cur/cur_output", downstream / "qc", pinned)
    write_candidate_metadata(root, contract, identity, curation)

    status(root, "package_handoff")
    handoff = publish_handoff(downstream, root / "handoff", identity["identity_digest"])
    status(root, "paired_comparison")
    comparison = run_comparison(args.config)
    result = {
        "schema": "en-full-session-part2c-imec1-result-v1",
        "status": "complete",
        "completed_utc": now(),
        "release_commit": contract["release_commit"],
        "candidate_sort_identity_digest": identity["identity_digest"],
        "baseline_sort_identity_digest": inputs["baseline_sort_identity_digest"],
        "curation": curation["summary"],
        "qc": qc["summary"],
        "handoff_manifest_sha256": sha256(root / "handoff/MANIFEST.json"),
        "comparison_summary_sha256": sha256(root / "comparison/comparison_summary.json"),
        "comparison_decision": comparison["decision"],
        "new_sort_launched": False,
        "rf_or_holdout_accessed": False,
        "recording_opened_for_bounded_waveform_qc": True,
        "hard_memory_cap": None,
        "handoff_total_bytes": handoff["total_bytes"],
    }
    atomic_json(root / "RESULT.json", result)
    status(root, "complete", state="complete", result_sha256=sha256(root / "RESULT.json"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    try:
        run(args)
    except BaseException as error:
        root = args.output
        if root.exists():
            atomic_json(root / "FAILURE.json", {
                "schema": "en-full-session-part2c-imec1-failure-v1",
                "status": "failed_preserved",
                "failed_utc": now(),
                "error_type": type(error).__name__,
                "error": str(error),
            })
            status(root, "failed", state="failed", error_type=type(error).__name__, error=str(error))
        raise


if __name__ == "__main__":
    main()
