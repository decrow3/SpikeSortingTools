"""Thin orchestration for standard long-strip sorter arms and shared downstream QC."""

from __future__ import annotations

import fcntl
import json
import os
import re
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from pipeline.downstream import (
    completed_stage_receipt,
    pin_sort_identity,
    run_curation_stage,
    run_matlab_export_stage,
    run_qc_stage,
)
from pipeline.preprocess import validate_accepted_recording
from pipeline.runtime import validate_production_environment
from testing.development_ladder import DevelopmentContract
from testing.development_strip import repository_receipt, validate_development_selection
from pipeline.config import fingerprint
from testing.ladder_sorter import NAMED_CONFIGS, _json_safe, check_effective_settings, run_sorter_config


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


@contextmanager
def arm_execution_lock(arm_dir: Path):
    """Prevent two managed workers from executing the same arm concurrently."""
    arm_dir = Path(arm_dir)
    arm_dir.mkdir(parents=True, exist_ok=True)
    lock_path = arm_dir / ".execution.lock"
    with lock_path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"arm is already owned by another worker: {arm_dir.name}") from error
        handle.seek(0)
        handle.truncate()
        json.dump({"pid": os.getpid(), "acquired_at": datetime.now(timezone.utc).isoformat()}, handle)
        handle.write("\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _select_candidates(contract: DevelopmentContract, candidate_names: Iterable[str] | None):
    if candidate_names is None:
        return contract.candidates
    names = tuple(candidate_names)
    if not names:
        raise ValueError("at least one candidate name is required")
    if len(names) != len(set(names)):
        raise ValueError("candidate names must not be repeated")
    by_name = {candidate["name"]: candidate for candidate in contract.candidates}
    unknown = [name for name in names if name not in by_name]
    if unknown:
        raise ValueError(f"unknown candidate name(s): {unknown}")
    return tuple(by_name[name] for name in names)


def finalize_development_arms(
    contract: DevelopmentContract,
    *,
    recording_dir: Path | str,
    output_root: Path | str,
    candidate_names: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Merge completed per-arm manifests after independent workers finish."""
    recording_dir = Path(recording_dir).resolve()
    output_root = Path(output_root).resolve()
    selected = _select_candidates(contract, candidate_names)
    summary = {"contract_digest": contract.digest, "recording_dir": str(recording_dir), "arms": {}}
    for candidate in selected:
        path = output_root / candidate["name"] / "candidate_manifest.json"
        if not path.is_file():
            raise RuntimeError(f"candidate is not complete: {candidate['name']}")
        manifest = json.loads(path.read_text())
        if (manifest.get("schema_version") != "longitudinal-development-arm-v1"
                or manifest.get("contract_digest") != contract.digest
                or manifest.get("candidate") != candidate
                or manifest.get("complete") is not True):
            raise RuntimeError(f"candidate manifest is incompatible or incomplete: {candidate['name']}")
        summary["arms"][candidate["name"]] = manifest
    _atomic_json(output_root / "arms_summary.json", summary)
    return summary


def _validated_existing_downstream(root: Path, identity_digest: str, *, sorter_output: Path, recording_dir: Path) -> tuple[Path, Path, dict, dict]:
    curated = root / "cur/cur_output"
    qc = root / "qc"
    curation_receipt_path = root / "cur/curation_receipt.json"
    qc_receipt_path = qc / "qc_receipt.json"
    if not curation_receipt_path.is_file() or not qc_receipt_path.is_file():
        raise RuntimeError(f"existing downstream output lacks identity-bound receipts: {root}")
    def validate(request_path, receipt_path, stage, settings, required):
        request = json.loads(request_path.read_text())
        expected = dict(schema_version="rescue-downstream-stage-request-v1", stage=stage,
                        sort_identity_digest=identity_digest, settings=settings)
        if any(request.get(k) != v for k, v in expected.items()) or request.get("request_digest") != fingerprint(expected):
            raise RuntimeError(f"existing {stage} settings or source differ from the frozen profile")
        receipt = completed_stage_receipt(receipt_path, request=request, required_files=required)
        if receipt is None or receipt.get("sort_identity_digest") != identity_digest:
            raise RuntimeError(f"existing {stage} belongs to another sort")
        return receipt
    curation_receipt = validate(root / "cur/curation_request.json", curation_receipt_path,
        "legacy_compatible_curation", dict(sorter_output=str(sorter_output.resolve()),
            strategy="run_cur_final_cosine", cosine_threshold=0.90, ccg_threshold=0.5,
            automatic_artifact_pair_merging=False),
        [curated / n for n in ("spike_times.npy", "spike_clusters.npy", "cluster_KSLabel.tsv", "ops.npy")])
    qc_receipt = validate(qc / "qc_request.json", qc_receipt_path, "legacy_compatible_qc",
        dict(recording_dir=str(recording_dir.resolve()), curated_output=str(curated.resolve()),
            waveform_seed=0, waveform_extractor="ordered_chunked_local_memmap_v1",
            waveform_read_chunk_duration_s=1.0, waveforms_per_unit=512, waveform_samples=82),
        [qc / n for n in ("waveforms/waveforms.npz", "refractory/refractory_qc.npz",
            "refractory/refractory_qc.pdf", "amp_truncation/truncation_qc.npz",
            "amp_truncation/present_qc.npz", "amp_truncation/truncation_qc.pdf")])
    return curated, qc, curation_receipt, qc_receipt


def _validate_existing_sort_config(sort_manifest: dict, sorter_config) -> None:
    """Require a reused sort to match every frozen requested sorter parameter."""
    expected_params = _json_safe(sorter_config.params())
    if sort_manifest.get("sorter_params") != expected_params:
        raise RuntimeError("existing sort parameters differ from the frozen candidate configuration")
    saved_digest = sort_manifest.get("config_digest")
    if saved_digest is not None and saved_digest != sorter_config.digest:
        raise RuntimeError("existing sort config digest differs from the frozen candidate configuration")


def run_development_arms(
    contract: DevelopmentContract,
    *,
    recording_dir: Path | str,
    output_root: Path | str,
    require_cuda: bool = True,
    candidate_names: Iterable[str] | None = None,
    group_id: str | None = None,
) -> dict[str, Any]:
    """Run/reuse each named standard arm through identical curation and QC."""
    recording_dir = Path(recording_dir).resolve()
    output_root = Path(output_root).resolve()
    if output_root == Path("/mnt") or output_root.is_relative_to(Path("/mnt")):
        raise ValueError("refusing to write development outputs under /mnt")
    if (
        output_root == recording_dir
        or output_root.is_relative_to(recording_dir)
    ):
        raise ValueError("development arms cannot write inside the recording directory")
    accepted = validate_accepted_recording(recording_dir)
    validate_development_selection(recording_dir, accepted, contract.raw["recording"], contract.raw["spatial_contract"])
    environment = validate_production_environment(require_cuda=require_cuda)
    repository = repository_receipt()
    output_root.mkdir(parents=True, exist_ok=True)
    candidates = _select_candidates(contract, candidate_names)
    if group_id is None:
        group_id = "all-arms" if candidate_names is None else "-".join(candidate["name"] for candidate in candidates)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", group_id):
        raise ValueError("group_id must contain only letters, numbers, dot, underscore, or hyphen")
    summary = {"contract_digest": contract.digest, "recording_dir": str(recording_dir), "arms": {}}
    for candidate in candidates:
        arm_dir = output_root / candidate["name"]
        with arm_execution_lock(arm_dir):
            sorter_config = NAMED_CONFIGS[candidate["sorter_config"]]
            existing_sort = candidate.get("existing_sort_dir")
            sort_dir = Path(existing_sort).resolve() if existing_sort else arm_dir / "sort"
            if existing_sort:
                manifest_path = sort_dir / "rescue_sort_manifest.json"
                if not manifest_path.is_file():
                    raise FileNotFoundError(f"existing sort manifest missing: {manifest_path}")
                sort_manifest = json.loads(manifest_path.read_text())
                _validate_existing_sort_config(sort_manifest, sorter_config)
            else:
                sort_manifest = run_sorter_config(recording_dir, sort_dir, sorter_config)
            if sort_manifest.get("recording_request_digest") != accepted["request_digest"]:
                raise RuntimeError(f"sort for {candidate['name']} belongs to another recording")
            effective = check_effective_settings(sorter_config.label, sort_manifest)
            identity = pin_sort_identity(sort_dir, arm_dir / "sort_identity.json")

            existing_downstream = candidate.get("existing_downstream_root")
            if existing_downstream:
                curated, qc_dir, curation_receipt, qc_receipt = _validated_existing_downstream(
                    Path(existing_downstream).resolve(), identity["identity_digest"],
                    sorter_output=sort_dir / "sorter_output", recording_dir=recording_dir,
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
            _atomic_json(manifest_path, arm_manifest)
            summary["arms"][candidate["name"]] = arm_manifest
    group_receipt = {
        "schema_version": "longitudinal-development-group-receipt-v1",
        "contract_digest": contract.digest,
        "group_id": group_id,
        "candidate_names": [candidate["name"] for candidate in candidates],
        "recording_dir": str(recording_dir),
        "arm_manifest_paths": {
            name: str((output_root / name / "candidate_manifest.json").resolve())
            for name in summary["arms"]
        },
        "complete": True,
    }
    _atomic_json(output_root / "group_receipts" / f"{group_id}.json", group_receipt)
    summary["group_receipt"] = group_receipt
    if candidate_names is None:
        finalize_development_arms(contract, recording_dir=recording_dir, output_root=output_root)
    return summary
