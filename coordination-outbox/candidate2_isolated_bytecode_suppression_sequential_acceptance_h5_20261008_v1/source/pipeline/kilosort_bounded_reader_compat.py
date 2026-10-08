"""Exact-source opt-in bounded pread reader for Kilosort 4.0.27."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
from pathlib import Path
from typing import Any

from .kilosort_cache_compat import (
    ORIGINAL_BLOCK,
    ORIGINAL_SOURCE_SHA256,
    PATCHED_BLOCK as FADVISE_BLOCK,
    PATCHED_SOURCE_SHA256 as FADVISE_SOURCE_SHA256,
)


PATCH_ID = "kilosort-4.0.27-bounded-preadv-input-v1"
KILOSORT_VERSION = "4.0.27"
PATCHED_SOURCE_SHA256 = "c6093e14cae920e8cfaae6c3fed8d4c6ca032b046d705ef9868ec171c7bf3c99"

BOUNDED_BLOCK = """        bstart, bend = self.get_batch_edges(ibatch)
        bounded_target = os.environ.get('KILOSORT_BOUNDED_PREAD_PATH')
        bounded_read = False
        if bounded_target and self.filename is not None:
            filename = os.path.realpath(os.fspath(self.filename))
            if filename == os.path.realpath(bounded_target):
                if not hasattr(os, 'preadv'):
                    raise RuntimeError('os.preadv is unavailable')
                if not hasattr(os, 'posix_fadvise'):
                    raise RuntimeError('POSIX_FADV_DONTNEED is unavailable')
                bytes_per_sample = self.n_chan_bin * np.dtype(self.dtype).itemsize
                byte_offset = int(bstart) * bytes_per_sample
                byte_count = int(bend - bstart) * bytes_per_sample
                buffer = getattr(self, '_bounded_pread_buffer', None)
                if buffer is None or len(buffer) < byte_count:
                    buffer = bytearray(byte_count)
                    self._bounded_pread_buffer = buffer
                view = memoryview(buffer)[:byte_count]
                completed = 0
                with open(filename, 'rb', buffering=0) as read_stream:
                    while completed < byte_count:
                        count = os.preadv(
                            read_stream.fileno(), [view[completed:]],
                            byte_offset + completed,
                            )
                        if count == 0:
                            raise EOFError('bounded Kilosort read ended early')
                        completed += count
                data = np.frombuffer(view, dtype=self.dtype).reshape(
                    int(bend - bstart), self.n_chan_bin,
                    )
                data = data.T
                bounded_read = True
        if not bounded_read:
            data = self.file[bstart : bend]
            data = data.T
"""

ADVISE_BLOCK = """        if bounded_read:
            with open(filename, 'rb', buffering=0) as cache_stream:
                os.posix_fadvise(
                    cache_stream.fileno(), byte_offset, byte_count,
                    os.POSIX_FADV_DONTNEED,
                    )

        inds = [bstart, bend]
"""

ORIGINAL_ADVISE_TARGET = """        inds = [bstart, bend]
"""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _source_path() -> Path:
    distribution = importlib.metadata.distribution("kilosort")
    return Path(distribution.locate_file("kilosort/io.py")).resolve()


def patch_source_text(source: str) -> str:
    """Transform exact upstream or failed-v1 source into the bounded reader."""
    source_hash = _sha256_bytes(source.encode("utf-8"))
    if source_hash == ORIGINAL_SOURCE_SHA256:
        input_block = ORIGINAL_BLOCK
    elif source_hash == FADVISE_SOURCE_SHA256:
        input_block = FADVISE_BLOCK
    else:
        raise RuntimeError(f"Kilosort bounded-reader source is unknown: {source_hash}")
    if source.count(input_block) != 1:
        raise RuntimeError("Kilosort bounded-reader input target is not unique")
    if source.count(ORIGINAL_ADVISE_TARGET) != 1:
        raise RuntimeError("Kilosort bounded-reader advice target is not unique")
    source = source.replace(input_block, BOUNDED_BLOCK, 1)
    return source.replace(ORIGINAL_ADVISE_TARGET, ADVISE_BLOCK, 1)


def bounded_reader_receipt() -> dict[str, Any]:
    source_path = _source_path()
    source_hash = _sha256_bytes(source_path.read_bytes())
    return {
        "patch_id": PATCH_ID,
        "kilosort_version": importlib.metadata.version("kilosort"),
        "source_path": str(source_path),
        "source_sha256": source_hash,
        "expected_patched_sha256": PATCHED_SOURCE_SHA256,
        "activation_environment": "KILOSORT_BOUNDED_PREAD_PATH",
        "applied": source_hash == PATCHED_SOURCE_SHA256,
    }


def ensure_kilosort_bounded_reader() -> dict[str, Any]:
    """Idempotently install only the frozen bounded-reader source repair."""
    version = importlib.metadata.version("kilosort")
    if version != KILOSORT_VERSION:
        raise RuntimeError(
            f"Compatibility patch {PATCH_ID} requires Kilosort {KILOSORT_VERSION}, got {version}"
        )
    for name in ("preadv", "posix_fadvise", "POSIX_FADV_DONTNEED"):
        if not hasattr(os, name):
            raise RuntimeError(f"required OS feature is unavailable: {name}")
    source_path = _source_path()
    source_bytes = source_path.read_bytes()
    observed_hash = _sha256_bytes(source_bytes)
    if observed_hash == PATCHED_SOURCE_SHA256:
        return bounded_reader_receipt()
    if observed_hash not in {ORIGINAL_SOURCE_SHA256, FADVISE_SOURCE_SHA256}:
        raise RuntimeError(
            f"Refusing to patch unknown Kilosort I/O source {observed_hash} at {source_path}"
        )
    patched_bytes = patch_source_text(source_bytes.decode("utf-8")).encode("utf-8")
    patched_hash = _sha256_bytes(patched_bytes)
    if patched_hash != PATCHED_SOURCE_SHA256:
        raise RuntimeError(f"Bounded-reader patch generated unexpected source hash {patched_hash}")
    temporary = source_path.with_name(source_path.name + ".bounded-reader.tmp")
    temporary.write_bytes(patched_bytes)
    temporary.chmod(source_path.stat().st_mode)
    os.replace(temporary, source_path)
    return bounded_reader_receipt()
