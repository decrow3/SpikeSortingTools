"""Known-trajectory ("oracle") motion correction — Phase D candidate 2 support.

C2 showed motion alone costs 30-80 accuracy points and that KS4's *rigid*
internal correction does not recover it. Candidate 2 asks whether a **non-rigid
motion representation** does. That question has two halves:

* can a representation fix it *at all* — the ceiling — tested by handing the
  pipeline the **exact** injected trajectory and correcting the recording with
  it before the frozen rescue sort (`oracle_corrected_recording` here); and
* can it be *estimated* well enough from the data — tested with KS4's own
  non-rigid datashift (`ladder_sorter.NONRIGID`) and, later, an external
  non-rigid estimate.

If even the oracle does not close the penalty, the residual is KS4
clustering/whitening on a moving footprint, not the motion representation, and
Candidate 2 is the wrong lever. If the oracle closes it but the estimators do
not, the lever is estimation quality.

`oracle_corrected_recording` builds a SpikeInterface `Motion` from a ladder
trajectory function, wraps the injected recording in `InterpolateMotionRecording`
(inverse motion, all channels kept), and writes it back as an accepted recording
the L1 runner consumes unchanged. Nothing is written under /mnt.
"""

from __future__ import annotations

import numpy as np

from testing.ladder_inject import write_injected_recording

KNOWN_MOTION_SCHEMA = "luke-ladder-known-motion-v1"


def signed_um_per_channel(channel_positions: np.ndarray) -> float:
    """µm of `y` per unit channel-index step (signed: index order may descend)."""
    y = np.asarray(channel_positions, dtype=np.float64)[:, 1]
    if y.size < 2:
        return float("nan")
    return float((y[-1] - y[0]) / (y.size - 1))


def sampled_displacement(
    trajectory_fn,
    *,
    duration_s: float,
    um_per_channel: float,
    depth_um: float,
    bin_s: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample a ladder trajectory (channel offset vs time) into a µm displacement.

    Returns `(temporal_bins_s, displacement_um, spatial_bins_um)` shaped for
    `spikeinterface.core.motion.Motion`: displacement is `(n_bins, 1)` — the
    trajectory is spatially rigid within the strip; the depth-varying case is a
    separate trajectory, not a separate spatial axis here.
    """
    n_bins = max(2, int(round(duration_s / bin_s)))
    centers = (np.arange(n_bins) + 0.5) * (duration_s / n_bins)
    offset_channels = np.asarray(trajectory_fn(centers), dtype=np.float64)
    disp_um = (offset_channels * um_per_channel).reshape(-1, 1)
    return centers, disp_um, np.array([depth_um], dtype=np.float64)


def oracle_corrected_recording(
    recording_dir,
    out_dir,
    *,
    trajectory_fn,
    duration_s: float,
    fs: float,
    gain_uv_per_count: float,
    sign: float = 1.0,
    bin_s: float = 0.5,
    border_mode: str = "force_extrapolate",
    name: str = "oracle_corrected",
):
    """Correct an injected recording with its KNOWN trajectory, then write it back.

    `sign` flips the correction direction (+1 undoes a trajectory injected with
    the same `trajectory_fn`); expose it so a caller can confirm the convention
    empirically rather than trusting it.
    """
    from spikeinterface.core import load
    from spikeinterface.core.motion import Motion
    from spikeinterface.preprocessing import astype
    from spikeinterface.sortingcomponents.motion import InterpolateMotionRecording

    if str(out_dir).startswith("/mnt/"):
        raise ValueError("refusing to write a corrected recording under /mnt")

    # InterpolateMotionRecording needs a floating dtype; keep counts scale and
    # convert back to µV after (the injected binary is int16 on disk).
    rec = astype(load(recording_dir), "float32")
    positions = np.load(f"{recording_dir}/channel_positions.npy")
    um_per_ch = signed_um_per_channel(positions)
    depth_um = float(np.asarray(positions)[:, 1].mean())

    centers, disp_um, spatial = sampled_displacement(
        trajectory_fn,
        duration_s=duration_s,
        um_per_channel=um_per_ch,
        depth_um=depth_um,
        bin_s=bin_s,
    )
    motion = Motion([sign * disp_um], [centers], spatial, direction="y")
    corrected = InterpolateMotionRecording(rec, motion, border_mode=border_mode)

    traces_uv = corrected.get_traces().astype(np.float32) * np.float32(gain_uv_per_count)
    manifest = write_injected_recording(
        out_dir,
        traces_uv,
        channel_positions=positions,
        fs=fs,
        gain_uv_per_count=gain_uv_per_count,
        source_snippet_dir=str(recording_dir),
        name=name,
    )
    manifest["known_motion_schema"] = KNOWN_MOTION_SCHEMA
    manifest["oracle_motion"] = {
        "sign": sign,
        "bin_s": bin_s,
        "um_per_channel": um_per_ch,
        "max_abs_displacement_um": float(np.abs(disp_um).max()),
        "border_mode": border_mode,
    }
    return manifest
