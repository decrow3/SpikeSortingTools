"""Configuration and deterministic fingerprints for the rescue pipeline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping


PIPELINE_VERSION = "spikeglx-ext-ref-rescue-testing-v1"


def fingerprint(value: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 for JSON-compatible configuration data."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RescueConfig:
    """Frozen choices supported by the Luke rescue experiments."""

    saturation_threshold_uv: float = 500.0
    similarity_threshold: float = -0.5
    noise_threshold: float = 0.3
    channel_metric_batches: int = 50
    channel_metric_batch_duration_s: float = 2.0
    channel_metric_seed: int = 1002
    materialize_n_jobs: int = 20
    # PhaseShiftRecording uses an FFT per requested chunk.  Keep this fixed to
    # the 10-s chunking used by the accepted full-duration depth-strip baseline.
    materialize_chunk_duration: str = "10s"

    def __post_init__(self) -> None:
        if self.saturation_threshold_uv <= 0:
            raise ValueError("saturation_threshold_uv must be positive")
        if self.channel_metric_batches < 1:
            raise ValueError("channel_metric_batches must be positive")
        if self.channel_metric_batch_duration_s <= 0:
            raise ValueError("channel_metric_batch_duration_s must be positive")
        if self.materialize_n_jobs < 1:
            raise ValueError("materialize_n_jobs must be positive")

    def as_dict(self) -> dict[str, Any]:
        return {"pipeline_version": PIPELINE_VERSION, **asdict(self)}

    @property
    def digest(self) -> str:
        return fingerprint(self.as_dict())
