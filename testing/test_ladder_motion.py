import numpy as np
import pytest

from testing.ladder_inject import (
    inject_trajectory,
    rigid_ramp,
    static_trajectory,
    write_injected_recording,
)
from testing.ladder_motion import (
    KNOWN_MOTION_SCHEMA,
    oracle_corrected_recording,
    sampled_displacement,
    signed_um_per_channel,
)

FS = 30_000.0


def _geom(n_chan=24, pitch=20.0, ascending=True):
    y = np.arange(n_chan) * pitch
    if not ascending:
        y = y[::-1].copy()
    return np.column_stack([np.zeros(n_chan), y]).astype(np.float64)


def test_signed_um_per_channel_tracks_index_direction():
    assert signed_um_per_channel(_geom(ascending=True)) == pytest.approx(20.0)
    assert signed_um_per_channel(_geom(ascending=False)) == pytest.approx(-20.0)


def test_sampled_displacement_static_is_zero_and_ramp_hits_endpoint():
    centers, disp, spatial = sampled_displacement(
        static_trajectory(), duration_s=10.0, um_per_channel=20.0, depth_um=100.0
    )
    assert disp.shape == (centers.size, 1)
    assert np.allclose(disp, 0.0)
    assert spatial.tolist() == [100.0]

    centers, disp, _ = sampled_displacement(
        rigid_ramp(3.0, 10.0), duration_s=10.0, um_per_channel=20.0, depth_um=0.0,
        bin_s=0.5,
    )
    # 3 channels over the window -> 60 µm near the end
    assert disp[-1, 0] == pytest.approx(3.0 * 20.0 * centers[-1] / 10.0, rel=1e-6)


def _template(n_samp=61, width=7):
    """A narrow (width-channel) template, peak in the centre column."""
    t = np.zeros((n_samp, width), dtype=np.float32)
    prof = np.array([-0.1, -0.5, -1.0, -0.5, -0.1], dtype=np.float32) * 120.0
    peak = width // 2
    for dc, w in ((-1, 0.4), (0, 1.0), (1, 0.4)):
        t[28:33, peak + dc] = prof * w
    return t


def test_oracle_correction_makes_a_ramped_neuron_more_stationary(tmp_path):
    rng = np.random.default_rng(0)
    n_chan, dur_s = 24, 4.0
    n_samp = int(dur_s * FS)
    bg = rng.normal(0, 5, (n_samp, n_chan)).astype(np.float32)
    train = np.arange(int(0.2 * FS), n_samp - int(0.2 * FS), int(FS // 10), dtype=np.int64)
    traj = rigid_ramp(4.0, dur_s)  # 4 channels across the window
    moving = inject_trajectory(
        bg, _template(), train, fs=FS, base_channel=6, trajectory=traj,
    )
    rec_dir = tmp_path / "moving"
    write_injected_recording(
        rec_dir, moving, channel_positions=_geom(n_chan), fs=FS,
        gain_uv_per_count=1.0, n_jobs=1,
    )

    manifest = oracle_corrected_recording(
        rec_dir, tmp_path / "corrected", trajectory_fn=traj, duration_s=dur_s,
        fs=FS, gain_uv_per_count=1.0,
    )
    assert manifest["known_motion_schema"] == KNOWN_MOTION_SCHEMA
    assert manifest["channel_count"] == n_chan  # force_extrapolate keeps all channels

    from spikeinterface.core import load

    def peak_channel_spread(folder):
        tr = load(folder).get_traces()
        halves = np.array_split(np.arange(tr.shape[0]), 6)
        peaks = [int(np.argmax(np.ptp(tr[h], axis=0))) for h in halves]
        return np.ptp(peaks)

    assert peak_channel_spread(tmp_path / "corrected") < peak_channel_spread(rec_dir)


def test_oracle_correction_refuses_mnt(tmp_path):
    rng = np.random.default_rng(1)
    bg = rng.normal(0, 5, (int(FS), 24)).astype(np.float32)
    rec_dir = tmp_path / "r"
    write_injected_recording(
        rec_dir, bg, channel_positions=_geom(24), fs=FS, gain_uv_per_count=1.0, n_jobs=1,
    )
    with pytest.raises(ValueError, match="/mnt"):
        oracle_corrected_recording(
            rec_dir, "/mnt/x", trajectory_fn=static_trajectory(), duration_s=1.0,
            fs=FS, gain_uv_per_count=1.0,
        )
