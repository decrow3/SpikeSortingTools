"""Audited compatibility repair for the published Kilosort 4.0.27 wheel."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
from pathlib import Path
from typing import Any


PATCH_ID = "kilosort-4.0.27-empty-clustering-center-v1"
KILOSORT_VERSION = "4.0.27"
ORIGINAL_SOURCE_SHA256 = "7b933e9885f3f6000b6204835e196edd68f2850805024499c88fe2903936965f"
PATCHED_SOURCE_SHA256 = "19ba98a8cc752889a4eb2833410f8d69da6a275688fb9580b840075092fde259"

ORIGINAL_BLOCK = """                Xd, igood, ichan = get_data_cpu(
                    ops, xy, iC, iclust_template, tF, ycent[kk], xcent[jj],
                    dmin=dmin, dminx=dminx, ix=ix,
                    )

                logger.debug(f'Center {ii} | Xd shape: {Xd.shape} | ntemp: {ntemp}')
"""

PATCHED_BLOCK = """                data_result = get_data_cpu(
                    ops, xy, iC, iclust_template, tF, ycent[kk], xcent[jj],
                    dmin=dmin, dminx=dminx, ix=ix,
                    )
                if (
                    len(data_result) == 4
                    and all(value is None for value in data_result)
                ):
                    nearby_chans_empty += 1
                    continue
                Xd, igood, ichan = data_result

                logger.debug(f'Center {ii} | Xd shape: {Xd.shape} | ntemp: {ntemp}')
"""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _kilosort_source_path() -> Path:
    distribution = importlib.metadata.distribution("kilosort")
    return Path(distribution.locate_file("kilosort/clustering_qr.py")).resolve()


def patch_source_text(source: str) -> str:
    """Apply the one reviewed empty-center repair to exact upstream text."""
    if source.count(ORIGINAL_BLOCK) != 1:
        raise RuntimeError("Kilosort empty-center patch target is not unique")
    return source.replace(ORIGINAL_BLOCK, PATCHED_BLOCK, 1)


def kilosort_compatibility_receipt() -> dict[str, Any]:
    source_path = _kilosort_source_path()
    source_hash = _sha256_bytes(source_path.read_bytes())
    return {
        "patch_id": PATCH_ID,
        "kilosort_version": importlib.metadata.version("kilosort"),
        "source_path": str(source_path),
        "source_sha256": source_hash,
        "expected_patched_sha256": PATCHED_SOURCE_SHA256,
        "applied": source_hash == PATCHED_SOURCE_SHA256,
    }


def ensure_kilosort_compatibility() -> dict[str, Any]:
    """Idempotently repair only the known published 4.0.27 wheel source."""
    version = importlib.metadata.version("kilosort")
    if version != KILOSORT_VERSION:
        raise RuntimeError(
            f"Compatibility patch {PATCH_ID} requires Kilosort {KILOSORT_VERSION}, got {version}"
        )
    source_path = _kilosort_source_path()
    original_bytes = source_path.read_bytes()
    observed_hash = _sha256_bytes(original_bytes)
    if observed_hash == PATCHED_SOURCE_SHA256:
        return kilosort_compatibility_receipt()
    if observed_hash != ORIGINAL_SOURCE_SHA256:
        raise RuntimeError(
            f"Refusing to patch unknown Kilosort source {observed_hash} at {source_path}"
        )
    patched_bytes = patch_source_text(original_bytes.decode("utf-8")).encode("utf-8")
    patched_hash = _sha256_bytes(patched_bytes)
    if patched_hash != PATCHED_SOURCE_SHA256:
        raise RuntimeError(
            f"Compatibility patch generated unexpected source hash {patched_hash}"
        )
    temporary = source_path.with_name(source_path.name + ".rescue-patch.tmp")
    temporary.write_bytes(patched_bytes)
    temporary.chmod(source_path.stat().st_mode)
    os.replace(temporary, source_path)
    return kilosort_compatibility_receipt()
