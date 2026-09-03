import json

import numpy as np
import pytest

from testing.ladder_inject import (
    INJECT_SCHEMA,
    channels_per_um,
    drift_penalty,
    inject_trajectory,
    paired_injection,
    rigid_oscillation,
    rigid_ramp,
    static_trajectory,
    write_injected_recording,
)

FS = 30_000.0
GAIN = 2.34375


def _template(n_samp=21, n_chan=7):
    t = np.zeros((n_samp, n_chan), dtype=np.float32)
    mid = n_chan // 2
    t[8:13, mid] = np.array([-20, -60, -100, -50, -10], np.float32)
    t[8:13, mid - 1] = t[8:13, mid + 1] = np.array([-8, -25, -40, -20, -4], np.float32)
    return t


def _background(seconds=4.0, n_chan=40, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 5.0, size=(int(seconds * FS), n_chan)).astype(np.float32)


def test_trajectory_functions():
    assert np.all(static_trajectory()(np.linspace(0, 4, 9)) == 0)
    assert rigid_ramp(8, 4.0)(np.array([0.0, 2.0, 4.0])).tolist() == [0.0, 4.0, 8.0]
    osc = rigid_oscillation(5.0, 2.0)(np.array([0.0, 0.5, 1.0]))
    assert np.isclose(osc[0], 0.0) and np.isclose(osc[1], 5.0)


def test_inject_trajectory_leaves_background_untouched_away_from_spikes():
    bg = _background()
    train = np.arange(6000, bg.shape[0] - 6000, 6000)
    out = inject_trajectory(
        bg, _template(), train, fs=FS, base_channel=15, trajectory=static_trajectory()
    )
    assert np.array_equal(out[:3000], bg[:3000])
    assert not np.array_equal(out, bg)


def test_moving_injection_migrates_peak_channel_static_does_not():
    bg = _background()
    train = np.arange(3000, bg.shape[0] - 3000, 3000)
    static_uv, moving_uv, truth = paired_injection(
        bg, _template(), train, fs=FS, base_channel=15,
        moving_trajectory=rigid_ramp(total_channels=8, duration_s=4.0),
    )

    def peak_channel(uv, s):
        seg = np.abs(uv[s - 10 : s + 10] - bg[s - 10 : s + 10])
        return int(np.argmax(seg.max(axis=0)))

    assert peak_channel(static_uv, train[0]) == peak_channel(static_uv, train[-1])
    assert peak_channel(moving_uv, train[0]) == 18  # 15 + template centre (3)
    assert peak_channel(moving_uv, train[-1]) == 26  # migrated by 8
    assert truth["inj0"].tolist() == sorted(train.tolist())


def test_static_and_moving_agree_at_the_first_spike_only():
    bg = _background()
    train = np.array([3000, bg.shape[0] - 3000])
    static_uv, moving_uv, _ = paired_injection(
        bg, _template(), train, fs=FS, base_channel=15,
        moving_trajectory=rigid_ramp(8, 4.0),
    )
    # t=0 -> offset 0 -> identical; last spike -> offset 8 -> different
    assert np.array_equal(static_uv[:5000], moving_uv[:5000])
    assert not np.array_equal(static_uv[-5000:], moving_uv[-5000:])


def test_write_injected_recording_is_loadable_and_manifested(tmp_path):
    from spikeinterface.core import load

    bg = _background(seconds=2.0)
    train = np.arange(3000, bg.shape[0] - 3000, 3000)
    uv = inject_trajectory(
        bg, _template(), train, fs=FS, base_channel=15, trajectory=static_trajectory()
    )
    geom = np.column_stack([np.zeros(40), np.arange(40) * 10.0])
    m = write_injected_recording(
        tmp_path / "inj", uv, channel_positions=geom, fs=FS,
        gain_uv_per_count=GAIN, name="c2_static", n_jobs=1,
    )
    assert m["snippet_schema"]
    rec = load(tmp_path / "inj")
    assert rec.get_num_samples() == bg.shape[0]
    manifest = json.loads((tmp_path / "inj" / "rescue_recording_manifest.json").read_text())
    assert manifest["complete"] and manifest["recording_content_sha256"]
    assert manifest["selected_end_frame"] == bg.shape[0]

    # An injected recording carries no SnippetSpec; load_snippet must still
    # verify it (via the sealed binary content hash) and reject a tampered one.
    from testing.ladder_snippets import load_snippet, verify_snippet

    assert verify_snippet(tmp_path / "inj")
    assert load_snippet(tmp_path / "inj").manifest["name"] == "c2_static"
    raw = tmp_path / "inj" / "traces_cached_seg0.raw"
    raw.write_bytes(b"\x01" + raw.read_bytes()[1:])
    assert not verify_snippet(tmp_path / "inj")


def test_write_injected_recording_cache_identity_changes_with_content(tmp_path):
    geom = np.column_stack([np.zeros(4), np.arange(4) * 10.0])
    a = write_injected_recording(
        tmp_path / "inj", np.zeros((100, 4), np.float32),
        channel_positions=geom, fs=FS, gain_uv_per_count=1.0, name="same", n_jobs=1,
    )
    b = write_injected_recording(
        tmp_path / "inj", np.full((100, 4), 50.0, np.float32),
        channel_positions=geom, fs=FS, gain_uv_per_count=1.0, name="same", n_jobs=1,
    )
    assert a["content_sha256"] != b["content_sha256"]
    assert a["spec_digest"] != b["spec_digest"]


def test_write_injected_recording_refuses_mnt():
    with pytest.raises(ValueError, match="/mnt"):
        write_injected_recording(
            "/mnt/x", np.zeros((10, 4), np.float32),
            channel_positions=np.zeros((4, 2)), fs=FS, gain_uv_per_count=GAIN,
        )


def test_drift_penalty_math():
    def score(acc, ident, switches, recovered):
        return {
            "primary": {
                "units": [{
                    "truth_unit": "inj0", "accuracy": acc,
                    "n_output_units_capturing": ident, "label_switches": switches,
                    "recovered": recovered,
                }]
            }
        }

    pen = drift_penalty(score(0.95, 1, 0, True), score(0.6, 3, 4, False))
    assert pen["schema"] == INJECT_SCHEMA
    assert pen["delta_accuracy"] == pytest.approx(-0.35)
    assert pen["delta_n_identities"] == 2
    assert pen["delta_label_switches"] == 4
    assert pen["static_recovered"] is True and pen["moving_recovered"] is False


def test_drift_penalty_raises_when_truth_not_scored():
    with pytest.raises(KeyError, match="not scored"):
        drift_penalty({"primary": {"units": []}}, {"primary": {"units": []}})


def test_channels_per_um():
    geom = np.column_stack([np.zeros(41), np.arange(41) * 10.0])  # 400 µm span, 41 ch
    assert channels_per_um(geom) == pytest.approx(0.1)
