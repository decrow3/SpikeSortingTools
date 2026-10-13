"""Thin callable entry into the existing C4 continuation runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


FEATURE_KEYS = ("segment_ownership", "pulse_family")


def run_selected_chunks(
    chunk_starts_samples: Sequence[int],
    feature_switches: Mapping[str, bool],
    output_dir: Path,
    *,
    c4_continuation_runner: Callable[..., Any],
) -> Any:
    """Validate the three A4 arguments and delegate once to the bound runner."""
    chunks = np.asarray(chunk_starts_samples)
    if chunks.ndim != 1 or chunks.size == 0:
        raise ValueError("chunk_starts_samples must be a nonempty one-dimensional array")
    if np.issubdtype(chunks.dtype, np.bool_) or not np.issubdtype(chunks.dtype, np.integer):
        raise ValueError("chunk_starts_samples must have integer dtype")
    if chunks.size and (
        np.any(chunks < 0)
        or np.any(chunks > np.iinfo(np.int64).max)
        or np.any(np.diff(chunks.astype(np.int64, copy=False)) <= 0)
    ):
        raise ValueError("chunk_starts_samples must be nonnegative and strictly increasing")
    chunks = chunks.astype(np.int64, copy=False)
    if set(feature_switches) != set(FEATURE_KEYS):
        raise ValueError(f"feature_switches must contain exactly {FEATURE_KEYS}")
    switches = {key: feature_switches[key] for key in FEATURE_KEYS}
    if any(not isinstance(value, (bool, np.bool_)) for value in switches.values()):
        raise ValueError("feature switches must be boolean")
    switches = {key: bool(value) for key, value in switches.items()}
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(output)
    if not callable(c4_continuation_runner):
        raise TypeError("c4_continuation_runner must be callable")
    return c4_continuation_runner(
        chunk_starts_samples=chunks,
        feature_switches=switches,
        output_dir=output,
    )
