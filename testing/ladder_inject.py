"""Trajectory injection — the reusable core of Phase C / C2.

`docs/pipeline_improvement_plan.md` Phase C step 1 ("wire the sealed injection
scaffold to the L1 runner") and C2 (the paired static-vs-moving identity
challenge). This module builds the injected voltage; `l1_run` sorts it and
`score_sort(truth=...)` scores it.

It never injects into stored `int16` — injection is into the
`Snippet.raw_domain_float32()` µV view, exactly as
`luke_injected_ground_truth_benchmark.inject_float32_raw_domain` requires. The
sealed scaffold's primitives are reused unchanged; this module only schedules
per-spike events along a known trajectory.

C2 in one call:

    static_uv, moving_uv, truth = paired_injection(bg_uv, template, train, ...)
    # write each back as an accepted recording, l1_run both, then:
    penalty = drift_penalty(static_score, moving_score)

`penalty` is the change caused *solely* by motion — Δaccuracy, Δidentities,
Δlabel-switches — the decisive quantity for Phase D's decision tree.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from pipeline.config import PIPELINE_VERSION, fingerprint
from pipeline.preprocess import (
    MANIFEST_NAME,
    RECORDING_MANIFEST_SCHEMA,
    recording_binary_receipt,
)
from testing.ladder_snippets import SNIPPET_SCHEMA
from testing.luke_injected_ground_truth_benchmark import (
    InjectionEvent,
    inject_float32_raw_domain,
    validate_template,
)

INJECT_SCHEMA = "luke-ladder-inject-v1"

TrajectoryFn = Callable[[np.ndarray], np.ndarray]  # seconds -> channel offset


# --------------------------------------------------------------------------- #
# trajectories (channel-index units; exact, no rounding drift in the spec)
# --------------------------------------------------------------------------- #
def static_trajectory() -> TrajectoryFn:
    return lambda t: np.zeros_like(np.asarray(t, dtype=float))


def rigid_ramp(total_channels: float, duration_s: float) -> TrajectoryFn:
    """Linear translation of `total_channels` over the window (rigid drift)."""
    return lambda t: total_channels * (np.asarray(t, dtype=float) / duration_s)


def rigid_oscillation(amplitude_channels: float, period_s: float) -> TrajectoryFn:
    return lambda t: amplitude_channels * np.sin(
        2.0 * np.pi * np.asarray(t, dtype=float) / period_s
    )


def nonrigid_depth_ramp(
    total_channels: float, duration_s: float, depth_gain: float
) -> TrajectoryFn:
    """A ramp whose slope depends on where the template sits (non-rigid).

    `depth_gain` scales the ramp; callers pass a different gain per injected
    unit to impose a depth-varying field.
    """
    return lambda t: (
        depth_gain * total_channels * (np.asarray(t, dtype=float) / duration_s)
    )


def channels_per_um(channel_positions: np.ndarray) -> float:
    """Rough channel-index step per µm of depth, from the strip geometry."""
    y = np.asarray(channel_positions)[:, 1]
    span = float(y.max() - y.min())
    return (len(y) - 1) / span if span > 0 else float("nan")


# --------------------------------------------------------------------------- #
# injection
# --------------------------------------------------------------------------- #
def inject_trajectory(
    background_uv: np.ndarray,
    template: np.ndarray,
    train_samples: np.ndarray,
    *,
    fs: float,
    base_channel: int,
    trajectory: TrajectoryFn,
    amplitude_scale: float = 1.0,
    template_id: str = "inj",
    edge_guard_samples: int = 2,
) -> np.ndarray:
    """Add `template` at every `train_samples` time, its channel following `trajectory`.

    `trajectory(t_seconds)` returns the channel-index offset from `base_channel`.
    The sealed primitive raises if a shifted footprint would cross a time or
    channel boundary, so callers must keep `base_channel + trajectory` within
    `[0, n_channels - template_width]` and the train clear of the window edges.
    """
    background_uv = np.asarray(background_uv, dtype=np.float32)
    template = validate_template(
        np.asarray(template, dtype=np.float32), edge_guard_samples=edge_guard_samples
    )
    train = np.sort(np.asarray(train_samples, dtype=np.int64))
    t_s = train / fs
    offsets = np.rint(np.asarray(trajectory(t_s), dtype=float)).astype(np.int64)

    events = [
        InjectionEvent(
            event_id=f"{template_id}-{i}",
            template_id=template_id,
            sample_index=int(s),
            amplitude_scale=float(amplitude_scale),
            channel_shift=int(base_channel + off),
        )
        for i, (s, off) in enumerate(zip(train, offsets))
    ]
    return inject_float32_raw_domain(
        background_uv,
        {template_id: template},
        events,
        edge_guard_samples=edge_guard_samples,
    )


def paired_injection(
    background_uv: np.ndarray,
    template: np.ndarray,
    train_samples: np.ndarray,
    *,
    fs: float,
    base_channel: int,
    moving_trajectory: TrajectoryFn,
    amplitude_scale: float = 1.0,
    unit_id: str = "inj0",
    edge_guard_samples: int = 2,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """The C2 pair: identical waveform and train, one held still, one moved."""
    static_uv = inject_trajectory(
        background_uv.copy(),
        template,
        train_samples,
        fs=fs,
        base_channel=base_channel,
        trajectory=static_trajectory(),
        amplitude_scale=amplitude_scale,
        template_id=unit_id,
        edge_guard_samples=edge_guard_samples,
    )
    moving_uv = inject_trajectory(
        background_uv.copy(),
        template,
        train_samples,
        fs=fs,
        base_channel=base_channel,
        trajectory=moving_trajectory,
        amplitude_scale=amplitude_scale,
        template_id=unit_id,
        edge_guard_samples=edge_guard_samples,
    )
    truth = {unit_id: np.sort(np.asarray(train_samples, dtype=np.int64))}
    return static_uv, moving_uv, truth


# --------------------------------------------------------------------------- #
# write an injected snippet the L1 runner can consume
# --------------------------------------------------------------------------- #
def write_injected_recording(
    out_dir: Path | str,
    injected_uv: np.ndarray,
    *,
    channel_positions: np.ndarray,
    fs: float,
    gain_uv_per_count: float,
    source_snippet_dir: Path | str | None = None,
    name: str = "injected",
    n_jobs: int = 4,
) -> dict:
    """Quantise µV → int16 and write an accepted-recording folder + manifests."""
    from spikeinterface.core import NumpyRecording

    out_dir = Path(out_dir)
    if str(out_dir).startswith("/mnt/"):
        raise ValueError("refusing to write an injected recording under /mnt")
    counts = np.rint(injected_uv / gain_uv_per_count)
    clipped = int(np.sum((counts < -32768) | (counts > 32767)))
    traces = np.clip(counts, -32768, 32767).astype(np.int16)

    rec = NumpyRecording([traces], sampling_frequency=fs)
    rec.set_dummy_probe_from_locations(
        np.asarray(channel_positions, dtype=np.float64)[:, :2]
    )
    rec.set_channel_gains(gain_uv_per_count)
    if out_dir.exists():
        import shutil

        shutil.rmtree(out_dir)
    rec.save(folder=out_dir, dtype="int16", n_jobs=n_jobs, progress_bar=False)
    np.save(out_dir / "channel_positions.npy", np.asarray(channel_positions))

    receipt = recording_binary_receipt(out_dir)
    n_samples = int(traces.shape[0])
    request = {
        "pipeline_version": PIPELINE_VERSION,
        "kind": "ladder_injected_snippet",
        "name": name,
        "source_snippet_dir": str(source_snippet_dir) if source_snippet_dir else None,
    }
    recording_manifest = {
        "schema_version": RECORDING_MANIFEST_SCHEMA,
        "pipeline_version": PIPELINE_VERSION,
        "kind": "ladder_injected_snippet",
        "complete": True,
        "request_digest": fingerprint(request),
        "num_samples": n_samples,
        "num_channels": int(traces.shape[1]),
        "sampling_frequency_hz": float(fs),
        "dtype": "int16",
        "selected_start_frame": 0,
        "selected_end_frame": n_samples,
        "gain_uv_per_count": float(gain_uv_per_count),
        "expected_binary_bytes": receipt["actual_binary_bytes"],
        "recording_content_sha256": receipt["recording_content_sha256"],
        "recording_binary_files": receipt["recording_binary_files"],
        "injection_clipped_samples": clipped,
    }
    (out_dir / MANIFEST_NAME).write_text(json.dumps(recording_manifest, indent=2) + "\n")

    snippet_manifest = {
        "snippet_schema": SNIPPET_SCHEMA,
        "pipeline_version": PIPELINE_VERSION,
        "name": name,
        "kind": "ladder_injected_snippet",
        "spec_digest": fingerprint(request),
        "axes": {"injected": name},
        "window": {"duration_s": n_samples / fs, "source_start_frame": 0,
                   "source_end_frame": n_samples},
        "channel_count": int(traces.shape[1]),
        "n_samples": n_samples,
        "sampling_frequency_hz": float(fs),
        "gain_uv_per_count": float(gain_uv_per_count),
        "depth_um_range": [
            float(np.asarray(channel_positions)[:, 1].min()),
            float(np.asarray(channel_positions)[:, 1].max()),
        ],
        "content_sha256": receipt["recording_content_sha256"],
        "injection_clipped_samples": clipped,
    }
    (out_dir / "snippet_manifest.json").write_text(
        json.dumps(snippet_manifest, indent=2) + "\n"
    )
    return snippet_manifest


# --------------------------------------------------------------------------- #
# the decisive quantity
# --------------------------------------------------------------------------- #
@dataclass
class DriftPenalty:
    delta_accuracy: float
    delta_n_identities: int
    delta_label_switches: int
    static_accuracy: float
    moving_accuracy: float
    static_recovered: bool
    moving_recovered: bool

    def as_dict(self) -> dict:
        return {"schema": INJECT_SCHEMA, **self.__dict__}


def _unit(score: dict, unit_id: str) -> dict:
    primary = score.get("primary") or {}
    units = {u["truth_unit"]: u for u in primary.get("units", [])}
    if unit_id not in units:
        raise KeyError(f"{unit_id!r} not scored; injected-truth missing from score_sort")
    return units[unit_id]


def drift_penalty(static_score: dict, moving_score: dict, unit_id: str = "inj0") -> dict:
    """Δ(moving − static) on the injected unit: the cost of motion alone."""
    s, m = _unit(static_score, unit_id), _unit(moving_score, unit_id)
    return DriftPenalty(
        delta_accuracy=round(m["accuracy"] - s["accuracy"], 4),
        delta_n_identities=m["n_output_units_capturing"] - s["n_output_units_capturing"],
        delta_label_switches=m["label_switches"] - s["label_switches"],
        static_accuracy=round(s["accuracy"], 4),
        moving_accuracy=round(m["accuracy"], 4),
        static_recovered=bool(s["recovered"]),
        moving_recovered=bool(m["recovered"]),
    ).as_dict()
