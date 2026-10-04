#!/usr/bin/env python3
"""Atomically publish the existing, hash-pinned EN AM.3 imec0 payload."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil


SOURCE = Path(
    "/media/huklab/Data/NPX/Ryansorting/Luke/"
    "luke_imec1_medicine_reference_sweep_v2/stage5_imec0"
)
REQUEST = Path(
    "/mnt/NPX/Luke/DARTsort_motion_experiments/"
    "en_am3_imec0_field_request_20260929_v1"
)
DESTINATION = REQUEST / "payload"
EXPECTED_NPZ = "4c769125020aad3263bf6eaa1889cdaa7eccbf19dcf0aa3a068b185221fadb9f"
FILES = (
    "luke0804_imec0_two_layer_motion.npz",
    "luke0804_imec0_two_layer_motion.manifest.json",
    "am3_complete.json",
    "am3_validation_gate.json",
    "censor_mask_v1_imec0.csv",
    "SHA256SUMS",
    "README.md",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_exclusive(path: Path, value: dict) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def verify(payload: Path) -> dict:
    manifest_path = payload / "DELIVERY_MANIFEST.json"
    complete_path = payload / "DELIVERY_COMPLETE.json"
    manifest = json.loads(manifest_path.read_text())
    complete = json.loads(complete_path.read_text())
    if manifest.get("schema") != "en-am3-imec0-payload-v1":
        raise RuntimeError("unexpected delivery manifest schema")
    rows = manifest.get("files")
    if [row.get("path") for row in rows] != list(FILES):
        raise RuntimeError("payload inventory changed")
    for row in rows:
        path = payload / row["path"]
        if path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
            raise RuntimeError(f"payload artifact changed: {row['path']}")
    if rows[0]["sha256"] != EXPECTED_NPZ:
        raise RuntimeError("required field digest changed")
    source_manifest = json.loads(
        (payload / "luke0804_imec0_two_layer_motion.manifest.json").read_text()
    )
    gate = json.loads((payload / "am3_validation_gate.json").read_text())
    am3_complete = json.loads((payload / "am3_complete.json").read_text())
    if source_manifest.get("npz_sha256") != EXPECTED_NPZ:
        raise RuntimeError("source manifest does not bind required field")
    if gate.get("status") != "pass" or am3_complete.get("status") != "complete":
        raise RuntimeError("AM.3 validation/completion sidecar is not complete")
    if complete.get("status") != "complete" or complete.get("manifest_sha256") != sha256(manifest_path):
        raise RuntimeError("delivery completion record does not bind manifest")
    return manifest


def main() -> None:
    if DESTINATION.exists():
        raise FileExistsError(f"refusing to replace {DESTINATION}")
    partial = DESTINATION.with_name(DESTINATION.name + ".partial")
    if partial.exists():
        raise FileExistsError(f"preserve or inspect incomplete payload: {partial}")
    missing = [name for name in FILES if not (SOURCE / name).is_file()]
    if missing:
        raise FileNotFoundError("missing source files: " + ", ".join(missing))
    if sha256(SOURCE / FILES[0]) != EXPECTED_NPZ:
        raise RuntimeError("source field digest is not the requested digest")
    partial.mkdir()
    rows = []
    for name in FILES:
        target = partial / name
        shutil.copyfile(SOURCE / name, target)
        rows.append({"path": name, "bytes": target.stat().st_size, "sha256": sha256(target)})
    manifest = {
        "schema": "en-am3-imec0-payload-v1",
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "producer_host": os.uname().nodename,
        "source_root": str(SOURCE),
        "required_field_sha256": EXPECTED_NPZ,
        "total_bytes": sum(row["bytes"] for row in rows),
        "files": rows,
        "scope": "existing AM.3 field and requested validation sidecars; no voltage",
    }
    write_json_exclusive(partial / "DELIVERY_MANIFEST.json", manifest)
    write_json_exclusive(
        partial / "DELIVERY_COMPLETE.json",
        {
            "schema": "en-am3-imec0-payload-v1",
            "status": "complete",
            "manifest_sha256": sha256(partial / "DELIVERY_MANIFEST.json"),
            "written_last_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    verify(partial)
    os.replace(partial, DESTINATION)
    print(json.dumps(verify(DESTINATION), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
