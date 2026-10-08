"""Restart-safe, single-pass staging of one SpikeGLX stream onto local storage."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any


SOURCE_STAGE_MANIFEST = "source_stage_manifest.json"
SOURCE_STAGE_SCHEMA = "spikeglx-source-stage-v1"
_COPY_BLOCK_BYTES = 16 * 1024 * 1024


def _stream_source_files(source_folder: Path, stream_id: str) -> list[Path]:
    """Find a stream's data file(s) and the metadata needed to reopen them."""
    source_folder = Path(source_folder)
    if not source_folder.is_dir():
        raise FileNotFoundError(f"SpikeGLX source folder does not exist: {source_folder}")
    data_suffixes = (f".{stream_id}.bin", f".{stream_id}.cbin")
    data_files = sorted(
        path
        for path in source_folder.rglob("*")
        if path.is_file() and path.name.endswith(data_suffixes)
    )
    if not data_files:
        raise FileNotFoundError(
            f"No SpikeGLX binary for stream {stream_id!r} found in {source_folder}"
        )
    selected: set[Path] = set()
    for data_path in data_files:
        stream_stem = data_path.name.rsplit(".", 1)[0]
        companions = [
            path
            for path in data_path.parent.iterdir()
            if path.is_file()
            and path.name.rsplit(".", 1)[0] == stream_stem
            and path.suffix in {".bin", ".cbin", ".meta", ".ch"}
        ]
        if not any(path.suffix == ".meta" for path in companions):
            raise FileNotFoundError(f"Missing SpikeGLX metadata beside {data_path}")
        if data_path.suffix == ".cbin" and not any(
            path.suffix == ".ch" for path in companions
        ):
            raise FileNotFoundError(f"Missing compression index beside {data_path}")
        selected.update(companions)
    return sorted(selected, key=lambda path: path.relative_to(source_folder).as_posix())


def _source_inventory(source_folder: Path, stream_id: str) -> list[dict[str, Any]]:
    inventory = []
    for path in _stream_source_files(source_folder, stream_id):
        stat = path.stat()
        inventory.append(
            {
                "relative_path": path.relative_to(source_folder).as_posix(),
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return inventory


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_COPY_BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_or_resume(source: Path, target: Path, expected_size: int) -> str:
    """Resume an interrupted copy without rereading completed bytes remotely."""
    target.parent.mkdir(parents=True, exist_ok=True)
    copied = target.stat().st_size if target.exists() else 0
    if copied > expected_size:
        raise RuntimeError(f"Partial staged file is oversized: {target}")
    digest = hashlib.sha256()
    if copied:
        with target.open("rb") as existing:
            for block in iter(lambda: existing.read(_COPY_BLOCK_BYTES), b""):
                digest.update(block)
    if copied < expected_size:
        with source.open("rb") as reader, target.open("ab") as writer:
            reader.seek(copied)
            while True:
                block = reader.read(_COPY_BLOCK_BYTES)
                if not block:
                    break
                writer.write(block)
                digest.update(block)
            writer.flush()
            os.fsync(writer.fileno())
        shutil.copystat(source, target)
    actual_size = target.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(
            f"Staged file is incomplete: {target} has {actual_size}, expected {expected_size}"
        )
    return digest.hexdigest()


def validate_staged_spikeglx_stream(
    staged_folder: Path, manifest: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Validate local sizes and hashes without rereading the remote binary."""
    staged_folder = Path(staged_folder)
    if manifest is None:
        manifest_path = staged_folder / SOURCE_STAGE_MANIFEST
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing source-stage manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != SOURCE_STAGE_SCHEMA or not manifest.get(
        "complete"
    ):
        raise RuntimeError("Source-stage manifest is incomplete or unsupported")
    for item in manifest.get("staged_files", []):
        path = staged_folder / item["relative_path"]
        if not path.is_file() or path.stat().st_size != item["size_bytes"]:
            raise RuntimeError(f"Staged source file is missing or has changed size: {path}")
        if _hash_file(path) != item["sha256"]:
            raise RuntimeError(f"Staged source file content has changed: {path}")
    return manifest


def stage_spikeglx_stream(
    source_folder: Path, staged_folder: Path, *, stream_id: str
) -> dict[str, Any]:
    """Copy one SpikeGLX stream to local storage once and reuse exact local bytes.

    A ``.partial`` directory is retained after interruption. The next call hashes
    the local prefix, seeks past it in the remote source, and resumes the copy.
    """
    source_folder = Path(source_folder).resolve()
    staged_folder = Path(staged_folder).resolve()
    partial = staged_folder.with_name(staged_folder.name + ".partial")
    if source_folder == staged_folder or source_folder in staged_folder.parents:
        raise ValueError("The local stage must not be inside the SpikeGLX source folder")
    source_inventory = _source_inventory(source_folder, stream_id)
    if staged_folder.exists() and partial.exists():
        raise RuntimeError("Complete and partial source stages coexist")
    if staged_folder.exists():
        manifest = validate_staged_spikeglx_stream(staged_folder)
        expected = {
            "source_folder": str(source_folder),
            "stream_id": stream_id,
            "source_files": source_inventory,
        }
        observed = {key: manifest.get(key) for key in expected}
        if observed != expected:
            raise RuntimeError("Existing local source stage belongs to another source state")
        return manifest
    partial_manifest_path = partial / SOURCE_STAGE_MANIFEST
    partial_request = {
        "schema_version": SOURCE_STAGE_SCHEMA,
        "source_folder": str(source_folder),
        "stream_id": stream_id,
        "source_files": source_inventory,
        "complete": False,
    }
    if partial.exists():
        if not partial_manifest_path.is_file():
            raise RuntimeError(
                f"Partial source stage lacks its resume manifest: {partial}"
            )
        recorded_request = json.loads(partial_manifest_path.read_text())
        if recorded_request != partial_request:
            raise RuntimeError("SpikeGLX source changed after an interrupted stage")
    else:
        partial.mkdir(parents=True)
        partial_manifest_path.write_text(json.dumps(partial_request, indent=2) + "\n")
    remaining = sum(
        max(
            0,
            item["size_bytes"]
            - (
                (partial / item["relative_path"]).stat().st_size
                if (partial / item["relative_path"]).exists()
                else 0
            ),
        )
        for item in source_inventory
    )
    if shutil.disk_usage(partial).free < remaining:
        raise RuntimeError(
            f"Insufficient local free space for source stage: need {remaining} bytes"
        )
    staged_files = []
    for item in source_inventory:
        relative = Path(item["relative_path"])
        digest = _copy_or_resume(
            source_folder / relative,
            partial / relative,
            item["size_bytes"],
        )
        staged_files.append(
            {
                "relative_path": item["relative_path"],
                "size_bytes": item["size_bytes"],
                "sha256": digest,
            }
        )
    if _source_inventory(source_folder, stream_id) != source_inventory:
        raise RuntimeError("SpikeGLX source changed while it was being staged")
    manifest = {
        "schema_version": SOURCE_STAGE_SCHEMA,
        "source_folder": str(source_folder),
        "stream_id": stream_id,
        "source_files": source_inventory,
        "staged_files": staged_files,
        "total_bytes": sum(item["size_bytes"] for item in staged_files),
        "complete": True,
    }
    partial_manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(partial, staged_folder)
    return manifest
