"""Resume auditable downstream stages for the completed Luke nonrigid sort.

This controller never sorts or estimates motion.  Every stage is identity-bound
and restartable through the common pipeline receipts.  Run it under the proven
user systemd manager because waveform QC reads the full corrected recording.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

from pipeline.config import fingerprint
from pipeline.downstream import (
    pin_sort_identity,
    run_completeness_timeline_stage,
    run_curation_stage,
    run_matlab_export_stage,
    run_qc_stage,
    run_standard_qc_stage,
)


SCHEMA = "luke-improved-nonrigid-downstream-v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(partial, path)


def _load_and_validate(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    if config.get("schema_version") != SCHEMA:
        raise ValueError("unsupported downstream config schema")
    required_stages = [
        "pin_sort_identity",
        "curation",
        "legacy_qc",
        "standard_qc",
        "completeness_timeline",
        "matlab_export",
    ]
    if config.get("stages") != required_stages:
        raise ValueError("downstream stage order differs from the frozen common path")
    recording_dir = Path(config["recording_dir"])
    kilosort_dir = Path(config["kilosort_dir"])
    recording_manifest = json.loads(
        (recording_dir / "rescue_recording_manifest.json").read_text()
    )
    sort_manifest = json.loads((kilosort_dir / "rescue_sort_manifest.json").read_text())
    if recording_manifest.get("complete") is not True or sort_manifest.get("complete") is not True:
        raise RuntimeError("recording or sort is incomplete")
    if recording_manifest.get("request_digest") != config["expected_recording_request_digest"]:
        raise RuntimeError("corrected recording identity differs from the frozen config")
    if sort_manifest.get("request_digest") != config["expected_sort_request_digest"]:
        raise RuntimeError("sort identity differs from the frozen config")
    if sort_manifest.get("recording_request_digest") != recording_manifest.get("request_digest"):
        raise RuntimeError("sort does not name the corrected recording")
    if (
        recording_manifest.get("num_samples") != config["expected_num_samples"]
        or recording_manifest.get("num_channels") != config["expected_num_channels"]
    ):
        raise RuntimeError("corrected recording extent differs from the frozen config")
    return config


def run(config_path: Path) -> dict[str, Any]:
    config = _load_and_validate(config_path)
    output = Path(config["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    scientific_config = {key: value for key, value in config.items() if key != "manager"}
    request = {
        "schema_version": SCHEMA,
        "config": scientific_config,
        "config_digest": fingerprint(scientific_config),
    }
    request["request_digest"] = fingerprint(request)
    request_path = output / "request.json"
    if request_path.exists():
        saved = json.loads(request_path.read_text())
        if saved.get("request_digest") != request["request_digest"]:
            saved_config = saved.get("config", {})
            saved_scientific = {
                key: value for key, value in saved_config.items() if key != "manager"
            }
            if saved_scientific != scientific_config:
                raise RuntimeError("existing downstream output belongs to another request")
            _atomic_json(
                request_path,
                {
                    **request,
                    "created_at": saved.get("created_at", _now()),
                    "migrated_at": _now(),
                    "migrated_from_request_digest": saved.get("request_digest"),
                    "migration_reason": "Execution-manager metadata is excluded from scientific request identity.",
                },
            )
    else:
        _atomic_json(request_path, {**request, "created_at": _now()})

    status_path = output / "status.json"

    def stage(name: str) -> None:
        _atomic_json(
            status_path,
            {
                "schema_version": SCHEMA,
                "state": "running",
                "stage": name,
                "request_digest": request["request_digest"],
                "pid": os.getpid(),
                "updated_at": _now(),
            },
        )
        print(name, flush=True)

    recording = Path(config["recording_dir"])
    kilosort = Path(config["kilosort_dir"])
    identity_path = output / "sort_identity.json"
    curated = output / "cur" / "cur_output"
    qc = output / "qc"

    stage("pin_sort_identity")
    identity = pin_sort_identity(kilosort, identity_path)
    stage("curation")
    curation = run_curation_stage(
        kilosort / "sorter_output",
        output / "cur",
        identity,
        cosine_threshold=float(config["curation"]["cosine_threshold"]),
        ccg_threshold=float(config["curation"]["ccg_threshold"]),
    )
    stage("legacy_qc")
    legacy_qc = run_qc_stage(recording, curated, qc, identity)
    stage("standard_qc")
    standard_qc = run_standard_qc_stage(recording, curated, qc, qc / "standard", identity)
    stage("completeness_timeline")
    completeness = run_completeness_timeline_stage(
        curated,
        qc,
        qc / "completeness_timeline",
        identity,
        sampling_frequency=float(config["sampling_frequency_hz"]),
    )
    stage("matlab_export")
    matlab = run_matlab_export_stage(curated, qc, identity)
    summary = {
        "schema_version": SCHEMA,
        "status": "complete",
        "development_only": True,
        "request_digest": request["request_digest"],
        "sort_identity_digest": identity["identity_digest"],
        "curation": curation["summary"],
        "legacy_qc": legacy_qc["summary"],
        "standard_qc": standard_qc["summary"],
        "completeness_timeline": completeness["summary"],
        "matlab_export": matlab["summary"],
        "interpretation": config["interpretation"],
        "completed_at": _now(),
    }
    _atomic_json(output / "summary.json", summary)
    _atomic_json(
        status_path,
        {
            "schema_version": SCHEMA,
            "state": "complete",
            "stage": "complete",
            "request_digest": request["request_digest"],
            "sort_identity_digest": identity["identity_digest"],
            "updated_at": _now(),
        },
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    try:
        run(args.config)
    except BaseException as error:
        try:
            config = json.loads(args.config.read_text())
            _atomic_json(
                Path(config["output_root"]) / "status.json",
                {
                    "schema_version": SCHEMA,
                    "state": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "updated_at": _now(),
                },
            )
        finally:
            raise


if __name__ == "__main__":
    main()
