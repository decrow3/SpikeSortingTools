"""Thin orchestration for standard long-strip sorter arms and shared downstream QC."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pipeline.downstream import (
    pin_sort_identity,
    run_curation_stage,
    run_matlab_export_stage,
    run_qc_stage,
)
from pipeline.preprocess import validate_accepted_recording
from pipeline.runtime import validate_production_environment
from testing.development_ladder import DevelopmentContract
from testing.development_strip import repository_receipt
from testing.ladder_sorter import NAMED_CONFIGS, check_effective_settings, run_sorter_config


def _validated_existing_downstream(root: Path, identity_digest: str) -> tuple[Path, Path, dict, dict]:
    curated = root / "cur/cur_output"
    qc = root / "qc"
    curation_receipt_path = root / "cur/curation_receipt.json"
    qc_receipt_path = qc / "qc_receipt.json"
    if not curation_receipt_path.is_file() or not qc_receipt_path.is_file():
        raise RuntimeError(f"existing downstream output lacks identity-bound receipts: {root}")
    curation_receipt = json.loads(curation_receipt_path.read_text())
    qc_receipt = json.loads(qc_receipt_path.read_text())
    if curation_receipt.get("sort_identity_digest") != identity_digest:
        raise RuntimeError("existing curation belongs to another sort")
    if qc_receipt.get("sort_identity_digest") != identity_digest:
        raise RuntimeError("existing QC belongs to another sort")
    return curated, qc, curation_receipt, qc_receipt


def run_development_arms(
    contract: DevelopmentContract,
    *,
    recording_dir: Path | str,
    output_root: Path | str,
    require_cuda: bool = True,
) -> dict[str, Any]:
    """Run/reuse each named standard arm through identical curation and QC."""
    recording_dir = Path(recording_dir).resolve()
    output_root = Path(output_root).resolve()
    if output_root == Path("/mnt") or output_root.is_relative_to(Path("/mnt")):
        raise ValueError("refusing to write development outputs under /mnt")
    if (
        output_root == recording_dir
        or output_root.is_relative_to(recording_dir)
        or recording_dir.is_relative_to(output_root)
    ):
        raise ValueError("development output must be disjoint from the recording")
    accepted = validate_accepted_recording(recording_dir)
    if accepted.get("request_digest") != contract.raw["recording"]["recording_digest"] and accepted.get("source_recording_request_digest") != contract.raw["recording"]["recording_digest"]:
        raise RuntimeError("development recording is not derived from the contracted accepted recording")
    environment = validate_production_environment(require_cuda=require_cuda)
    repository = repository_receipt()
    output_root.mkdir(parents=True, exist_ok=True)
    summary = {"contract_digest": contract.digest, "recording_dir": str(recording_dir), "arms": {}}
    for candidate in contract.candidates:
        arm_dir = output_root / candidate["name"]
        arm_dir.mkdir(parents=True, exist_ok=True)
        sorter_config = NAMED_CONFIGS[candidate["sorter_config"]]
        existing_sort = candidate.get("existing_sort_dir")
        sort_dir = Path(existing_sort).resolve() if existing_sort else arm_dir / "sort"
        if existing_sort:
            manifest_path = sort_dir / "rescue_sort_manifest.json"
            if not manifest_path.is_file():
                raise FileNotFoundError(f"existing sort manifest missing: {manifest_path}")
            sort_manifest = json.loads(manifest_path.read_text())
        else:
            sort_manifest = run_sorter_config(recording_dir, sort_dir, sorter_config)
        if sort_manifest.get("recording_request_digest") != accepted["request_digest"]:
            raise RuntimeError(f"sort for {candidate['name']} belongs to another recording")
        effective = check_effective_settings(sorter_config.label, sort_manifest)
        identity = pin_sort_identity(sort_dir, arm_dir / "sort_identity.json")

        existing_downstream = candidate.get("existing_downstream_root")
        if existing_downstream:
            curated, qc_dir, curation_receipt, qc_receipt = _validated_existing_downstream(
                Path(existing_downstream).resolve(), identity["identity_digest"]
            )
        else:
            curation_receipt = run_curation_stage(
                sort_dir / "sorter_output", arm_dir / "cur", identity
            )
            curated = arm_dir / "cur/cur_output"
            qc_receipt = run_qc_stage(
                recording_dir, curated, arm_dir / "qc", identity, waveform_seed=0
            )
            qc_dir = arm_dir / "qc"
            run_matlab_export_stage(curated, qc_dir, identity)
        arm_manifest = {
            "schema_version": "longitudinal-development-arm-v1",
            "contract_digest": contract.digest,
            "candidate": candidate,
            "sorter_config_digest": sorter_config.digest,
            "effective_settings": effective,
            "recording_request_digest": accepted["request_digest"],
            "sort_directory": str(sort_dir),
            "sort_identity_digest": identity["identity_digest"],
            "curated_output": str(curated),
            "qc_directory": str(qc_dir),
            "curation_request_digest": curation_receipt["request_digest"],
            "qc_request_digest": qc_receipt["request_digest"],
            "pre_curation_summary": sort_manifest.get("summary"),
            "post_curation_summary": curation_receipt.get("summary"),
            "environment": environment,
            "repository": repository,
            "complete": True,
        }
        manifest_path = arm_dir / "candidate_manifest.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != arm_manifest:
            raise RuntimeError(f"arm manifest changed for {candidate['name']}")
        manifest_path.write_text(json.dumps(arm_manifest, indent=2) + "\n")
        summary["arms"][candidate["name"]] = arm_manifest
    (output_root / "arms_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
