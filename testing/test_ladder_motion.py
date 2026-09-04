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
    paired_geometry_motion_injection,
    perturb_bias,
    perturb_depth_gradient,
    perturb_gain,
    perturb_time_lag,
    perturb_time_smooth,
    sampled_displacement,
    signed_um_per_channel,
    warp_array_with_known_motion,
    waveform_preservation,
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


def test_field_perturbations_transform_the_displacement_as_specified():
    c = np.linspace(0.25, 9.75, 20)
    d = np.linspace(0.0, 40.0, 20).reshape(-1, 1)
    s = np.array([1000.0])

    _, dg, _ = perturb_gain(0.5)(c, d, s)
    assert np.allclose(dg, d * 0.5)

    _, db, _ = perturb_bias(7.0)(c, d, s)
    assert np.allclose(db, d + 7.0)

    # a 5 s lag pulls earlier (smaller) values to a given time
    _, dl, _ = perturb_time_lag(5.0)(c, d, s)
    assert dl[-1, 0] < d[-1, 0] and dl[0, 0] == pytest.approx(d[0, 0])

    # smoothing cannot exceed the original range and reduces total variation
    _, dsm, _ = perturb_time_smooth(3.0)(c, d, s)
    assert dsm.max() <= d.max() + 1e-9
    assert np.abs(np.diff(dsm, axis=0)).sum() <= np.abs(np.diff(d, axis=0)).sum()

    # a depth gradient makes the field non-rigid: 2 spatial bins, bracketing d
    cc, dgrad, sgrad = perturb_depth_gradient(0.4, depth_um=1000.0, span_um=200.0)(c, d, s)
    assert sgrad.tolist() == [900.0, 1100.0]
    assert dgrad.shape == (20, 2)
    assert np.allclose(dgrad[:, 0], d[:, 0] * 0.8) and np.allclose(dgrad[:, 1], d[:, 0] * 1.2)


def test_waveform_preservation_scores_identity_and_flags_damage():
    rng = np.random.default_rng(0)
    tmpl = np.zeros((61, 11), dtype=np.float64)
    tmpl[28:33, 5] = np.array([-0.2, -0.6, -1.0, -0.6, -0.2]) * 150.0
    tmpl[28:33, 4] = tmpl[28:33, 5] * 0.4
    tmpl[28:33, 6] = tmpl[28:33, 5] * 0.4

    identical = waveform_preservation(tmpl.copy(), tmpl)
    assert identical["waveform_cosine"] == pytest.approx(1.0)
    assert identical["peak_amp_ratio"] == pytest.approx(1.0)
    assert identical["peak_channel_shift"] == 0

    attenuated = waveform_preservation(tmpl * 0.5, tmpl)
    assert attenuated["peak_amp_ratio"] == pytest.approx(0.5)
    assert attenuated["waveform_cosine"] == pytest.approx(1.0)  # shape preserved

    mislocalised = np.roll(tmpl, 2, axis=1)
    assert waveform_preservation(mislocalised, tmpl)["peak_channel_shift"] == 2

    noise = waveform_preservation(rng.normal(0, 50, tmpl.shape), tmpl)
    assert noise["waveform_cosine"] < 0.5


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
    assert manifest["oracle_motion"]["perturbation"] == "exact"

    # a heavily gain-perturbed field (0.0 = no correction) should not out-stabilise
    # the exact field
    m0 = oracle_corrected_recording(
        rec_dir, tmp_path / "corrected0", trajectory_fn=traj, duration_s=dur_s,
        fs=FS, gain_uv_per_count=1.0, perturbation=perturb_gain(0.0),
        perturbation_label="gain_0.0", name="oc0",
    )
    assert m0["oracle_motion"]["perturbation"] == "gain_0.0"
    assert m0["oracle_motion"]["applied_max_abs_displacement_um"] == 0.0

    from spikeinterface.core import load

    def peak_channel_spread(folder):
        tr = load(folder).get_traces()
        halves = np.array_split(np.arange(tr.shape[0]), 6)
        peaks = [int(np.argmax(np.ptp(tr[h], axis=0))) for h in halves]
        return np.ptp(peaks)

    assert peak_channel_spread(tmp_path / "corrected") < peak_channel_spread(rec_dir)


def test_geometry_motion_does_not_turn_depth_drift_into_lateral_channel_jumps():
    # Luke channels are ordered across four staggered columns.  A contiguous
    # index shift would jump x by 16-48 um; physical y-motion must not.
    n_chan, dur_s = 24, 2.0
    x = np.tile([16.0, 48.0, 0.0, 32.0], n_chan // 4)
    y = np.repeat(np.arange(n_chan // 2) * 20.0, 2)
    geom = np.column_stack([x, y])
    train = np.array([600, 1800, 3000, 4200, 5400], dtype=np.int64)
    one_channel = np.zeros((61, 1), dtype=np.float32)
    one_channel[28:33, 0] = np.array([-20, -60, -100, -50, -10], dtype=np.float32)
    static, moving, _ = paired_geometry_motion_injection(
        np.zeros((int(dur_s * 3000), n_chan), dtype=np.float32),
        one_channel,
        train,
        fs=3000.0,
        base_channel=8,
        moving_trajectory=rigid_ramp(4.0, dur_s),
        channel_positions=geom,
    )
    x_centres = []
    y_centres = []
    for sample in train:
        energy = np.max(np.abs(moving[sample - 30 : sample + 30]), axis=0)
        centre = (energy[:, None] * geom).sum(axis=0) / energy.sum()
        x_centres.append(centre[0])
        y_centres.append(centre[1])
    assert np.ptp(x_centres) < 2.0
    assert y_centres[-1] > y_centres[0] + 20.0

    corrected = warp_array_with_known_motion(
        moving,
        geom,
        fs=3000.0,
        trajectory_fn=rigid_ramp(4.0, dur_s),
        sign=1.0,
    )
    # Interpolation is not lossless, but the exact inverse must move the
    # forward-warped voltage closer to the static source.
    assert np.mean((corrected - static) ** 2) < np.mean((moving - static) ** 2)


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


def test_sampled_displacement_um_units_bypass_the_channel_conversion():
    """v4 trajectories are specified in µm; the channel round trip is optional."""
    from testing.ladder_motion import sampled_displacement

    def forty_um(t):
        return np.full_like(np.asarray(t, dtype=float), 40.0)

    _, disp_um, _ = sampled_displacement(
        forty_um, duration_s=10.0, um_per_channel=9.909, depth_um=1900.0,
        bin_s=0.5, trajectory_units="um",
    )
    assert np.all(disp_um == 40.0)  # exact, not 40.000000001

    _, scaled, _ = sampled_displacement(
        forty_um, duration_s=10.0, um_per_channel=9.909, depth_um=1900.0, bin_s=0.5,
    )
    assert np.allclose(scaled, 40.0 * 9.909)  # default stays channel units

    with pytest.raises(ValueError, match="unknown trajectory_units"):
        sampled_displacement(
            forty_um, duration_s=10.0, um_per_channel=1.0, depth_um=0.0,
            trajectory_units="furlongs",
        )
