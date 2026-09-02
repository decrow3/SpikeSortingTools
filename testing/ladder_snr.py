"""SNR stratification — the last frozen-panel axis (plan §4).

The panel's SNR axis is `high / medium / low` tertiles "by local noise and
template amplitude". This module derives that from the snippet voltage alone —
**no sorter labels, no challenger output** — so it stays inside the §4 selection
discipline and can be re-run from scratch on a second session (§Phase E).

Per snippet:

* **noise floor** — per-channel robust σ (MAD / 0.6745, in µV) over random
  chunks of the highpassed, common-referenced voltage.
* **event amplitude** — the |amplitude| distribution of `detect_peaks`
  (`locally_exclusive`, both polarities) on the same voltage.
* **`snr`** — `event_amp_uv_p90 / noise_uv_median`. This one scalar is the
  stratifier; the rest of the dict is provenance.

`stratify_by_snr` turns a pool of profiles into tertile labels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from pipeline.config import fingerprint
from testing.ladder_snippets import Snippet, load_snippet

SNR_SCHEMA = "luke-ladder-snr-v1"


@dataclass(frozen=True)
class SnrConfig:
    highpass_hz: float = 300.0
    reference: str = "global"
    reference_operator: str = "median"
    detect_method: str = "locally_exclusive"
    detect_threshold: float = 4.0
    peak_sign: str = "both"
    noise_chunks: int = 20
    noise_chunk_s: float = 1.0
    seed: int = 1002
    n_jobs: int = 8

    @property
    def digest(self) -> str:
        return fingerprint({"snr_schema": SNR_SCHEMA, **asdict(self)})


def _conditioned_recording(snippet: Snippet, config: SnrConfig):
    from spikeinterface.preprocessing import common_reference, highpass_filter

    rec = snippet.recording()
    rec = highpass_filter(rec, freq_min=config.highpass_hz, dtype="float32")
    rec = common_reference(
        rec,
        reference=config.reference,
        operator=config.reference_operator,
        dtype="float32",
    )
    return rec


def _noise_uv(rec, config: SnrConfig) -> np.ndarray:
    """Per-channel robust σ in µV over random chunks (method: mad)."""
    fs = rec.get_sampling_frequency()
    n = rec.get_num_samples()
    w = max(int(config.noise_chunk_s * fs), 1)
    rng = np.random.default_rng(config.seed)
    n_chunks = min(config.noise_chunks, max(n // w, 1))
    starts = rng.integers(0, max(n - w, 1), size=n_chunks)
    stacked = np.concatenate(
        [rec.get_traces(start_frame=int(s), end_frame=int(s) + w) for s in starts],
        axis=0,
    )
    return np.median(np.abs(stacked - np.median(stacked, axis=0)), axis=0) / 0.6745


def snr_profile(snippet, config: SnrConfig | None = None) -> dict:
    """Voltage-only SNR characterisation of one snippet."""
    from spikeinterface.sortingcomponents.peak_detection import detect_peaks

    config = config or SnrConfig()
    if not isinstance(snippet, Snippet):
        snippet = load_snippet(snippet)

    rec = _conditioned_recording(snippet, config)
    noise = _noise_uv(rec, config)
    noise_median = float(np.median(noise))

    peaks = detect_peaks(
        rec,
        method=config.detect_method,
        peak_sign=config.peak_sign,
        detect_threshold=config.detect_threshold,
        n_jobs=config.n_jobs,
        progress_bar=False,
    )
    amp = np.abs(np.asarray(peaks["amplitude"], dtype=float))
    dur = snippet.duration_s
    p50, p90, p95 = (
        np.percentile(amp, [50, 90, 95]) if amp.size else (np.nan, np.nan, np.nan)
    )
    snr_event = amp / noise[np.asarray(peaks["channel_index"])] if amp.size else amp

    return {
        "schema": SNR_SCHEMA,
        "snippet_dir": str(snippet.dir),
        "spec_digest": snippet.manifest["spec_digest"],
        "config": asdict(config),
        "config_digest": config.digest,
        "noise_uv_median": noise_median,
        "noise_uv_p10": float(np.percentile(noise, 10)),
        "noise_uv_p90": float(np.percentile(noise, 90)),
        "n_peaks": int(amp.size),
        "peak_rate_hz": float(amp.size / dur) if dur else float("nan"),
        "event_amp_uv_p50": float(p50),
        "event_amp_uv_p90": float(p90),
        "event_amp_uv_p95": float(p95),
        "snr_event_p50": float(np.median(snr_event)) if amp.size else float("nan"),
        "snr_event_p90": float(np.percentile(snr_event, 90)) if amp.size else float("nan"),
        "snr": float(p90 / noise_median) if noise_median > 0 and amp.size else float("nan"),
    }


def stratify_by_snr(
    profiles: dict[str, dict],
    *,
    key: str = "snr",
    labels: tuple[str, str, str] = ("low", "medium", "high"),
) -> dict[str, str]:
    """Assign tertile labels across a pool of profiles (ascending `key`)."""
    values = {name: float(p[key]) for name, p in profiles.items()}
    finite = np.array([v for v in values.values() if np.isfinite(v)])
    if finite.size < 3:
        raise ValueError("need at least 3 finite profiles to form tertiles")
    lo, hi = np.quantile(finite, [1 / 3, 2 / 3])
    out = {}
    for name, v in values.items():
        if not np.isfinite(v):
            out[name] = "unknown"
        elif v <= lo:
            out[name] = labels[0]
        elif v <= hi:
            out[name] = labels[1]
        else:
            out[name] = labels[2]
    return out
