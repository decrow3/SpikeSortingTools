"""Opt-in page-cache eviction for the audited Kilosort 4.0.27 reader.

The patch is deliberately dormant unless ``KILOSORT_FADVISE_DONTNEED_PATH``
names the exact binary being read.  It advises the kernel that the preceding
batch is no longer needed immediately before the next batch is mapped.  This
changes cache residency only; returned samples and sorter settings are
unchanged.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
from pathlib import Path
from typing import Any


PATCH_ID = "kilosort-4.0.27-drop-consumed-input-cache-v1"
KILOSORT_VERSION = "4.0.27"
ORIGINAL_SOURCE_SHA256 = "767b76a07647122b45048ba12cdaf49185f87f3fa33334081d4af099825414bd"
PATCHED_SOURCE_SHA256 = "27acaef1cb60eae2fe5b38fe90a95da169c2f5c70b9c51a6a4437dffaa54a676"

ORIGINAL_BLOCK = """        bstart, bend = self.get_batch_edges(ibatch)
        data = self.file[bstart : bend]
        data = data.T
"""

PATCHED_BLOCK = """        bstart, bend = self.get_batch_edges(ibatch)
        cache_target = os.environ.get('KILOSORT_FADVISE_DONTNEED_PATH')
        if cache_target and self.filename is not None:
            filename = os.path.realpath(os.fspath(self.filename))
            if filename == os.path.realpath(cache_target):
                previous = getattr(self, '_previous_input_cache_region', None)
                if previous is not None:
                    if not hasattr(os, 'posix_fadvise'):
                        raise RuntimeError('POSIX_FADV_DONTNEED is unavailable')
                    with open(filename, 'rb', buffering=0) as cache_stream:
                        os.posix_fadvise(
                            cache_stream.fileno(), previous[0], previous[1],
                            os.POSIX_FADV_DONTNEED,
                            )
                bytes_per_sample = self.n_chan_bin * np.dtype(self.dtype).itemsize
                self._previous_input_cache_region = (
                    int(bstart) * bytes_per_sample,
                    int(bend - bstart) * bytes_per_sample,
                    )
        data = self.file[bstart : bend]
        data = data.T
"""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _source_path() -> Path:
    distribution = importlib.metadata.distribution("kilosort")
    return Path(distribution.locate_file("kilosort/io.py")).resolve()


def patch_source_text(source: str) -> str:
    """Apply the exact opt-in cache repair to exact upstream text."""
    if source.count(ORIGINAL_BLOCK) != 1:
        raise RuntimeError("Kilosort input-cache patch target is not unique")
    return source.replace(ORIGINAL_BLOCK, PATCHED_BLOCK, 1)


def cache_compatibility_receipt() -> dict[str, Any]:
    source_path = _source_path()
    source_hash = _sha256_bytes(source_path.read_bytes())
    return {
        "patch_id": PATCH_ID,
        "kilosort_version": importlib.metadata.version("kilosort"),
        "source_path": str(source_path),
        "source_sha256": source_hash,
        "expected_patched_sha256": PATCHED_SOURCE_SHA256,
        "activation_environment": "KILOSORT_FADVISE_DONTNEED_PATH",
        "applied": source_hash == PATCHED_SOURCE_SHA256,
    }


def ensure_kilosort_cache_compatibility() -> dict[str, Any]:
    """Idempotently install only the frozen opt-in reader repair."""
    version = importlib.metadata.version("kilosort")
    if version != KILOSORT_VERSION:
        raise RuntimeError(
            f"Compatibility patch {PATCH_ID} requires Kilosort {KILOSORT_VERSION}, got {version}"
        )
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        raise RuntimeError("POSIX_FADV_DONTNEED is unavailable")
    source_path = _source_path()
    original_bytes = source_path.read_bytes()
    observed_hash = _sha256_bytes(original_bytes)
    if observed_hash == PATCHED_SOURCE_SHA256:
        return cache_compatibility_receipt()
    if observed_hash != ORIGINAL_SOURCE_SHA256:
        raise RuntimeError(
            f"Refusing to patch unknown Kilosort I/O source {observed_hash} at {source_path}"
        )
    patched_bytes = patch_source_text(original_bytes.decode("utf-8")).encode("utf-8")
    patched_hash = _sha256_bytes(patched_bytes)
    if patched_hash != PATCHED_SOURCE_SHA256:
        raise RuntimeError(
            f"Input-cache patch generated unexpected source hash {patched_hash}"
        )
    temporary = source_path.with_name(source_path.name + ".cache-patch.tmp")
    temporary.write_bytes(patched_bytes)
    temporary.chmod(source_path.stat().st_mode)
    os.replace(temporary, source_path)
    return cache_compatibility_receipt()
