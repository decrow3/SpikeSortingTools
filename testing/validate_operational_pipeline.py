"""Validate a machine-readable operational spike-sorting pipeline profile."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from pipeline.config import fingerprint


SCHEMA = "luke-operational-pipeline-profile-v1"
REPO = Path(__file__).resolve().parents[1]


def sha256(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO / value


def _receipt(
    value: dict[str, Any], stage: str, identity: str, fallback_dir: Path | None = None,
) -> None:
    if value.get("complete") is not True:
        raise RuntimeError(f"{stage} receipt is incomplete")
    if value.get("stage") != stage:
        raise RuntimeError(f"expected {stage}, got {value.get('stage')!r}")
    if value.get("sort_identity_digest") != identity:
        raise RuntimeError(f"{stage} receipt has the wrong sort identity")
    for required in value.get("required_files", []):
        path = Path(required)
        if not path.is_file() and fallback_dir is not None:
            path = fallback_dir / path.name
        if not path.is_file():
            raise FileNotFoundError(f"{stage} required output is absent: {required}")


def _sidecar_package(manifest_path: Path, value: dict[str, Any], identity: str) -> None:
    if value.get("schema") != "luke-operational-pipeline-sidecars-v1":
        raise RuntimeError("unsupported sidecar package schema")
    if value.get("status") != "complete" or value.get("sort_identity_digest") != identity:
        raise RuntimeError("sidecar package is incomplete or has the wrong identity")
    rows = value.get("files")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("sidecar package inventory is empty")
    observed_total = 0
    for row in rows:
        path = manifest_path.parent / row["path"]
        if not path.is_file() or path.stat().st_size != row.get("bytes") or sha256(path) != row.get("sha256"):
            raise RuntimeError(f"sidecar package artifact changed: {row.get('path')}")
        observed_total += path.stat().st_size
    if observed_total != value.get("total_bytes"):
        raise RuntimeError("sidecar package byte total differs")


def validate(profile_path: Path) -> dict[str, Any]:
    profile_bytes = profile_path.read_bytes()
    profile = json.loads(profile_bytes)
    if profile.get("schema_version") != SCHEMA:
        raise ValueError("unsupported operational profile schema")
    loaded: dict[str, dict[str, Any]] = {}
    artifact_hashes: dict[str, str] = {}
    for name, spec in profile["artifacts"].items():
        path = resolve(spec["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256(path)
        if observed != spec["sha256"]:
            raise RuntimeError(f"operational artifact changed: {name}")
        artifact_hashes[name] = observed
        loaded[name] = json.loads(path.read_text())

    expected = profile["identity"]
    recording = loaded["recording_manifest"]
    sort = loaded["sort_manifest"]
    identity = loaded["sort_identity"]
    if recording.get("complete") is not True:
        raise RuntimeError("accepted recording is incomplete")
    if recording.get("request_digest") != expected["recording_request_digest"]:
        raise RuntimeError("recording request identity differs")
    if recording.get("recording_content_sha256") != expected["recording_content_sha256"]:
        raise RuntimeError("recording content identity differs")
    if recording.get("graph") != profile["processing"]["preprocessing_graph"]:
        raise RuntimeError("preprocessing graph differs")
    for key in ("external_filter", "external_reference", "external_voltage_motion_correction"):
        if recording.get(key) != profile["processing"][key]:
            raise RuntimeError(f"recording {key} differs")
    if sort.get("complete") is not True or sort.get("request_digest") != expected["sort_request_digest"]:
        raise RuntimeError("accepted sort request is incomplete or differs")
    if sort.get("recording_request_digest") != expected["recording_request_digest"]:
        raise RuntimeError("sort is bound to another recording")
    if identity.get("identity_digest") != expected["sort_identity_digest"]:
        raise RuntimeError("pinned sort identity differs")
    if identity.get("request_digest") != expected["sort_request_digest"]:
        raise RuntimeError("sort identity is bound to another request")
    settings = profile["processing"]["critical_sorter_settings"]
    params = sort["sorter_params"]
    for key in ("Th_universal", "Th_learned", "do_correction", "do_CAR", "artifact_threshold"):
        if params.get(key) != settings[key]:
            raise RuntimeError(f"sorter setting differs: {key}")
    if sort["summary"]["critical_saved_settings"].get("effective_nblocks") != settings["effective_nblocks"]:
        raise RuntimeError("effective Kilosort motion setting differs")

    identity_digest = expected["sort_identity_digest"]
    sidecar_manifest_path = resolve(profile["artifacts"]["sidecar_manifest"]["path"])
    _sidecar_package(sidecar_manifest_path, loaded["sidecar_manifest"], identity_digest)
    sidecar_root = sidecar_manifest_path.parent
    stages = {
        "curation_receipt": "legacy_compatible_curation",
        "legacy_qc_receipt": "legacy_compatible_qc",
        "standard_qc_receipt": "standard_unit_quality",
        "completeness_timeline_receipt": "amplitude_completeness_timeline",
    }
    for name, stage in stages.items():
        fallback = None
        if name == "standard_qc_receipt":
            fallback = sidecar_root / "standard_qc"
        elif name == "completeness_timeline_receipt":
            fallback = sidecar_root / "completeness_timeline"
        _receipt(loaded[name], stage, identity_digest, fallback)
    standard = loaded["standard_qc_receipt"]["summary"]
    completeness = loaded["completeness_timeline_receipt"]["summary"]
    policy = profile["downstream_policy"]
    if standard.get("automatic_curation_changes") != policy["standard_qc_automatic_curation_changes"]:
        raise RuntimeError("standard QC changed curation")
    if completeness.get("screening_only") is not policy["completeness_timeline_screening_only"]:
        raise RuntimeError("completeness timeline policy differs")
    if standard.get("unit_count") != completeness.get("unit_count"):
        raise RuntimeError("standard QC and completeness unit inventories differ")
    if standard.get("spike_count") != loaded["curation_receipt"]["summary"].get("spike_count"):
        raise RuntimeError("standard QC and curated spike inventories differ")

    return {
        "schema_version": SCHEMA,
        "status": "pass",
        "profile_sha256": hashlib.sha256(profile_bytes).hexdigest(),
        "profile_digest": fingerprint(profile),
        "sort_identity_digest": identity_digest,
        "recording_content_sha256": expected["recording_content_sha256"],
        "artifacts": artifact_hashes,
        "curated_units": int(standard["unit_count"]),
        "curated_spikes": int(standard["spike_count"]),
        "units_with_any_lab_warning": int(standard["units_with_any_lab_warning"]),
        "completeness_windows": int(completeness["window_count"]),
        "measured_completeness_windows": int(completeness["measured_window_count"]),
        "candidate_status": profile["candidate_status"],
        "validated_at": datetime.now(timezone.utc).isoformat(),
    }


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(partial, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=Path("configs/luke_operational_pipeline.v1.json"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(args.profile)
    if args.output:
        atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
