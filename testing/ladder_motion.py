"""Known-trajectory ("oracle") motion correction — Phase D candidate 2 support.

The original C2/Candidate 2 results are retracted because forward injection and
inverse correction used incompatible geometries. The corrected experiment asks
whether a **non-rigid motion representation** helps. That question has two
halves:

* can a representation fix it *at all* — the ceiling — tested by handing the
  pipeline the **exact** injected trajectory and correcting the recording with
  it before the frozen rescue sort (`oracle_corrected_recording` here); and
* can it be *estimated* well enough from the data — tested with KS4's own
  non-rigid datashift (`ladder_sorter.NONRIGID`) and, later, an external
  non-rigid estimate.

The corrected rerun must establish whether the oracle closes a penalty before
either representation or estimation quality can be identified as the lever.

`oracle_corrected_recording` builds a SpikeInterface `Motion` from a ladder
trajectory function, wraps the injected recording in `InterpolateMotionRecording`
(inverse motion, all channels kept), and writes it back as an accepted recording
the L1 runner consumes unchanged. Nothing is written under /mnt.

For D2b-1 (field-error tolerance) it also accepts a `perturbation` — `perturb_gain`,
`perturb_time_lag`, `perturb_time_smooth`, `perturb_bias`, `perturb_depth_gradient`
— that corrupts the exact field before it is applied, so the runner can find how
far off an *estimated* field can be before correction stops beating no correction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np

from testing.ladder_inject import inject_trajectory, static_trajectory, write_injected_recording

KNOWN_MOTION_SCHEMA = "luke-ladder-known-motion-v1"

# A field perturbation maps (temporal_bins_s, displacement_um, spatial_bins_um)
# to a perturbed triple. D2b-1 uses these to corrupt the exact oracle field and
# measure sensitivity to field errors after the corrected baseline is rerun.
FieldPerturbation = Callable[
    [np.ndarray, np.ndarray, np.ndarray],
    "tuple[np.ndarray, np.ndarray, np.ndarray]",
]


def perturb_none() -> FieldPerturbation:
    return lambda c, d, s: (c, d, s)


def perturb_gain(factor: float) -> FieldPerturbation:
    """Under/over-estimate displacement amplitude (factor 1.0 = exact)."""
    return lambda c, d, s: (c, d * factor, s)


def perturb_bias(bias_um: float) -> FieldPerturbation:
    """A constant residual offset after centering."""
    return lambda c, d, s: (c, d + bias_um, s)


def perturb_time_lag(lag_s: float) -> FieldPerturbation:
    """Clock / alignment error: displacement read `lag_s` late (edge-held)."""

    def f(c, d, s):
        out = np.empty_like(d)
        for k in range(d.shape[1]):
            out[:, k] = np.interp(c - lag_s, c, d[:, k])
        return c, out, s

    return f


def perturb_time_smooth(sigma_s: float) -> FieldPerturbation:
    """Bandwidth loss: Gaussian low-pass of the displacement trace."""

    def f(c, d, s):
        if c.size < 2 or sigma_s <= 0:
            return c, d, s
        from scipy.ndimage import gaussian_filter1d

        bin_s = float(np.median(np.diff(c)))
        return c, gaussian_filter1d(d, sigma_s / bin_s, axis=0, mode="nearest"), s

    return f


def perturb_depth_gradient(
    fraction: float, depth_um: float, span_um: float
) -> FieldPerturbation:
    """Spatial mismatch: impose a spurious depth dependence on a rigid field.

    Displacement is scaled by `1 - fraction/2` at the shallow strip edge and
    `1 + fraction/2` at the deep edge — a wrong non-rigid structure of the size
    a real estimator's spatial over/under-fitting would produce.
    """

    def f(c, d, s):
        base = d[:, [0]]
        new_d = np.concatenate(
            [base * (1.0 - fraction / 2.0), base * (1.0 + fraction / 2.0)], axis=1
        )
        return c, new_d, np.array(
            [depth_um - span_um / 2.0, depth_um + span_um / 2.0], dtype=np.float64
        )

    return f


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


def warp_array_with_known_motion(
    traces_uv: np.ndarray,
    channel_positions: np.ndarray,
    *,
    fs: float,
    trajectory_fn,
    sign: float,
    bin_s: float = 0.5,
    border_mode: str = "force_extrapolate",
    spatial_interpolation_method: str = "kriging",
    sigma_um: float = 20.0,
) -> np.ndarray:
    """Apply a known physical y-motion field to a voltage array.

    Unlike shifting contiguous channel indices, this delegates motion to the
    same geometry-aware SpikeInterface operator used for correction.  This is
    essential on Luke's four-column probe, where adjacent channel indices can
    share y and jump laterally in x.
    """
    from spikeinterface.core import NumpyRecording
    from spikeinterface.core.motion import Motion
    from spikeinterface.sortingcomponents.motion import InterpolateMotionRecording

    traces_uv = np.asarray(traces_uv, dtype=np.float32)
    positions = np.asarray(channel_positions, dtype=np.float64)
    rec = NumpyRecording([traces_uv], sampling_frequency=fs)
    rec.set_dummy_probe_from_locations(positions[:, :2])
    centers, disp_um, spatial = sampled_displacement(
        trajectory_fn,
        duration_s=traces_uv.shape[0] / fs,
        um_per_channel=signed_um_per_channel(positions),
        depth_um=float(positions[:, 1].mean()),
        bin_s=bin_s,
    )
    motion = Motion([sign * disp_um], [centers], spatial, direction="y")
    warped = InterpolateMotionRecording(
        rec,
        motion,
        border_mode=border_mode,
        spatial_interpolation_method=spatial_interpolation_method,
        sigma_um=sigma_um,
    )
    return warped.get_traces().astype(np.float32)


def paired_geometry_motion_injection(
    background_uv: np.ndarray,
    template: np.ndarray,
    train_samples: np.ndarray,
    *,
    fs: float,
    base_channel: int,
    moving_trajectory,
    channel_positions: np.ndarray,
    amplitude_scale: float = 1.0,
    unit_id: str = "inj0",
    edge_guard_samples: int = 2,
    bin_s: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Inject once at rest, then forward-warp the complete voltage geometrically.

    The moving arm is produced with the inverse sign of the oracle correction,
    making forward-motion plus exact correction a genuine operator round trip.
    Warping the complete voltage also avoids the prior confound where only the
    donor moved but correction subsequently resampled the background as well.
    """
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
    moving_uv = warp_array_with_known_motion(
        static_uv,
        channel_positions,
        fs=fs,
        trajectory_fn=moving_trajectory,
        sign=-1.0,
        bin_s=bin_s,
    )
    truth = {unit_id: np.sort(np.asarray(train_samples, dtype=np.int64))}
    return static_uv, moving_uv, truth


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
    spatial_interpolation_method: str = "kriging",
    sigma_um: float = 20.0,
    name: str = "oracle_corrected",
    perturbation: FieldPerturbation | None = None,
    perturbation_label: str = "exact",
):
    """Correct an injected recording with its KNOWN trajectory, then write it back.

    `sign` flips the correction direction (+1 undoes a trajectory injected with
    the same `trajectory_fn`); expose it so a caller can confirm the convention
    empirically rather than trusting it.

    `perturbation` (D2b-1) corrupts the exact field before it is applied —
    `perturb_gain`, `perturb_time_lag`, … — to measure field-error tolerance.
    Pass a distinct `name` per perturbation so each gets its own sort cache leaf.
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
    exact_max = float(np.abs(disp_um).max())
    if perturbation is not None:
        centers, disp_um, spatial = perturbation(centers, disp_um, spatial)
        centers = np.asarray(centers, dtype=np.float64)
        disp_um = np.asarray(disp_um, dtype=np.float64)
        spatial = np.asarray(spatial, dtype=np.float64)
    motion = Motion([sign * disp_um], [centers], spatial, direction="y")
    corrected = InterpolateMotionRecording(
        rec, motion, border_mode=border_mode,
        spatial_interpolation_method=spatial_interpolation_method, sigma_um=sigma_um,
    )

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
        "exact_max_abs_displacement_um": exact_max,
        "applied_max_abs_displacement_um": float(np.abs(disp_um).max()),
        "applied_n_spatial_bins": int(disp_um.shape[1]),
        "perturbation": perturbation_label,
        "border_mode": border_mode,
        "spatial_interpolation_method": spatial_interpolation_method,
        "sigma_um": sigma_um,
    }
    return manifest


# --------------------------------------------------------------------------- #
# waveform preservation — the D2b guardrail (plan §5)
# --------------------------------------------------------------------------- #
def observed_waveform(
    recording_dir,
    train_samples: np.ndarray,
    *,
    base_channel: int,
    width: int,
    n_samples: int,
    max_spikes: int = 1500,
) -> np.ndarray:
    """Spike-triggered average of a recording at the injected times/channels, in µV.

    Alignment is exact: `inject_float32_raw_domain` places a `W`-sample template
    at `[sample_index - W//2 : sample_index - W//2 + W]`, so averaging the same
    window recovers the waveform the sorter will see.
    """
    from spikeinterface.core import load

    recording_dir = Path(recording_dir)
    manifest = json.loads((recording_dir / "rescue_recording_manifest.json").read_text())
    gain = float(manifest["gain_uv_per_count"])
    rec = load(recording_dir)
    total = rec.get_num_samples()
    half = n_samples // 2
    ts = np.sort(np.asarray(train_samples, dtype=np.int64))
    ts = ts[(ts - half >= 0) & (ts - half + n_samples <= total)]
    if ts.size > max_spikes:
        ts = ts[np.linspace(0, ts.size - 1, max_spikes).astype(int)]
    chans = rec.channel_ids[base_channel : base_channel + width]
    acc = np.zeros((n_samples, len(chans)), dtype=np.float64)
    for s in ts:
        acc += rec.get_traces(
            start_frame=int(s - half),
            end_frame=int(s - half + n_samples),
            channel_ids=chans,
        ).astype(np.float64)
    return acc / max(ts.size, 1) * gain


def waveform_preservation(observed: np.ndarray, template: np.ndarray) -> dict:
    """Compare an observed spike-triggered average to the injected template.

    `cosine` and `peak_amp_ratio` measure shape and amplitude retention;
    `peak_channel_shift` is residual mislocalisation (0 = perfectly restored).
    """
    obs = np.asarray(observed, dtype=np.float64)
    tmpl = np.asarray(template, dtype=np.float64)
    a, b = obs.ravel(), tmpl.ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    cosine = float(a @ b / denom) if denom else 0.0
    obs_peak = abs(float(obs.flat[np.argmax(np.abs(obs))]))
    tmpl_peak = abs(float(tmpl.flat[np.argmax(np.abs(tmpl))]))
    return {
        "waveform_cosine": round(cosine, 4),
        "peak_amp_ratio": round(obs_peak / tmpl_peak, 4) if tmpl_peak else 0.0,
        "peak_channel_shift": int(
            np.argmax(np.max(np.abs(obs), axis=0))
            - np.argmax(np.max(np.abs(tmpl), axis=0))
        ),
    }
