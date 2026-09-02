import json

import numpy as np
import pytest

from testing.ladder_snippets import (
    SNIPPET_SCHEMA,
    SnippetSpec,
    build_snippet,
    freeze_panel,
    load_snippet,
    verify_snippet,
)

FS = 30_000.0
GAIN = 2.34375


@pytest.fixture
def accepted_recording(tmp_path):
    """A tiny stand-in for the accepted rescue recording folder."""
    from spikeinterface.core import NumpyRecording

    rng = np.random.default_rng(4)
    n_samp, n_chan = 20 * int(FS), 24
    traces = rng.integers(-800, 800, size=(n_samp, n_chan)).astype("int16")
    rec = NumpyRecording([traces], sampling_frequency=FS)
    locations = np.column_stack(
        [np.zeros(n_chan), np.arange(n_chan) * 20.0]
    ).astype(np.float64)
    rec.set_dummy_probe_from_locations(locations)

    rec_dir = tmp_path / "recording"
    rec.save(folder=rec_dir, dtype="int16", n_jobs=1, progress_bar=False)
    (rec_dir / "rescue_recording_manifest.json").write_text(
        json.dumps({
            "schema_version": "rescue-recording-manifest-v2",
            "request_digest": "a" * 64,
            "selected_start_frame": 0,
            "selected_end_frame": n_samp,
            "sampling_frequency_hz": FS,
            "gain_uv_per_count": GAIN,
        })
    )
    return rec_dir


def _spec(**kw):
    base = dict(
        name="quiet_mid",
        start_s=2.0,
        duration_s=3.0,
        channel_start=4,
        channel_count=12,
        split="development",
        selection_basis="low input anomaly and low supported motion",
        axes={"motion_regime": "quiet", "depth_strip": "mid"},
    )
    base.update(kw)
    return SnippetSpec(**base)


def test_spec_rejects_label_derived_selection():
    with pytest.raises(ValueError, match="sorter labels"):
        _spec(selection_basis="window with the most ks_good units")
    with pytest.raises(ValueError, match="selection_basis"):
        _spec(selection_basis="   ")
    with pytest.raises(ValueError, match="split"):
        _spec(split="test")


def test_build_snippet_freezes_and_seals(accepted_recording, tmp_path):
    out = tmp_path / "snips"
    m = build_snippet(_spec(), accepted_recording, out, n_jobs=1)

    assert m["snippet_schema"] == SNIPPET_SCHEMA
    assert m["n_samples"] == int(round(3.0 * FS))
    assert m["channel_count"] == 12
    assert m["gain_uv_per_count"] == GAIN
    d = out / (m["name"] + "-" + m["spec_digest"][:10])
    assert (d / "traces_cached_seg0.raw").exists()
    assert (d / "channel_positions.npy").exists()
    assert verify_snippet(d)

    # depth strip really is channels 4..16 of the 20 um pitch probe
    assert m["depth_um_range"] == [80.0, 300.0]


def test_build_snippet_is_idempotent_and_cache_hits(accepted_recording, tmp_path):
    out = tmp_path / "snips"
    first = build_snippet(_spec(), accepted_recording, out, n_jobs=1)
    d = out / (first["name"] + "-" + first["spec_digest"][:10])
    mtime = (d / "traces_cached_seg0.raw").stat().st_mtime_ns
    second = build_snippet(_spec(), accepted_recording, out, n_jobs=1)
    assert second["content_sha256"] == first["content_sha256"]
    assert (d / "traces_cached_seg0.raw").stat().st_mtime_ns == mtime  # not rewritten


def test_verify_snippet_detects_mutation(accepted_recording, tmp_path):
    out = tmp_path / "snips"
    m = build_snippet(_spec(), accepted_recording, out, n_jobs=1)
    d = out / (m["name"] + "-" + m["spec_digest"][:10])
    raw = d / "traces_cached_seg0.raw"
    buf = bytearray(raw.read_bytes())
    buf[0] ^= 0xFF
    raw.write_bytes(buf)
    assert verify_snippet(d) is False


def test_raw_domain_float32_is_scaled_int16(accepted_recording, tmp_path):
    out = tmp_path / "snips"
    m = build_snippet(_spec(), accepted_recording, out, n_jobs=1)
    snip = load_snippet(out / (m["name"] + "-" + m["spec_digest"][:10]))
    i16 = snip.traces_int16()
    uv = snip.raw_domain_float32()
    assert uv.dtype == np.float32
    assert np.allclose(uv, i16.astype(np.float32) * GAIN)
    assert i16.shape == (m["n_samples"], 12)


def test_build_snippet_refuses_mnt(accepted_recording):
    with pytest.raises(ValueError, match="/mnt"):
        build_snippet(_spec(), accepted_recording, "/mnt/whatever", n_jobs=1)


def test_freeze_panel_requires_eight_plus_eight(accepted_recording, tmp_path):
    specs = [
        _spec(name=f"s{i}", start_s=float(i), split="development")
        for i in range(3)
    ]
    with pytest.raises(ValueError, match="8 development"):
        freeze_panel(specs, accepted_recording, tmp_path / "p", n_jobs=1)


def test_freeze_panel_builds_and_seals_small_panel(accepted_recording, tmp_path):
    specs = [
        _spec(name=f"dev{i}", start_s=float(i), split="development")
        for i in range(2)
    ] + [
        _spec(name=f"hold{i}", start_s=float(i), split="held_out")
        for i in range(2)
    ]
    panel = freeze_panel(
        specs, accepted_recording, tmp_path / "p",
        require_balanced=False, n_jobs=1,
    )
    assert panel["n_development"] == 2 and panel["n_held_out"] == 2
    assert len(panel["snippets"]) == 4
    assert panel["panel_digest"]
    assert (tmp_path / "p" / "panel_manifest.json").exists()
