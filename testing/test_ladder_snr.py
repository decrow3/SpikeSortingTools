import json

import numpy as np
import pytest

from testing.ladder_snippets import SnippetSpec, build_snippet, load_snippet
from testing.ladder_snr import SNR_SCHEMA, SnrConfig, snr_profile, stratify_by_snr

FS = 30_000.0
GAIN = 2.34375


@pytest.fixture
def snippet(tmp_path):
    """A snippet with a few embedded high-amplitude events on a mid channel."""
    from spikeinterface.core import NumpyRecording

    rng = np.random.default_rng(1)
    n_samp, n_chan = 12 * int(FS), 24
    counts = rng.normal(0, 6, size=(n_samp, n_chan)).astype("float32")
    # negative spikes every 20 ms on channels 10-13
    for t in range(600, n_samp - 600, 600):
        counts[t, 10:14] -= 120.0
        counts[t - 1, 10:14] -= 60.0
        counts[t + 1, 10:14] -= 60.0
    traces = np.clip(counts, -3000, 3000).astype("int16")
    rec = NumpyRecording([traces], sampling_frequency=FS)
    rec.set_dummy_probe_from_locations(
        np.column_stack([np.zeros(n_chan), np.arange(n_chan) * 20.0]).astype(float)
    )
    rec_dir = tmp_path / "recording"
    rec.save(folder=rec_dir, dtype="int16", n_jobs=1, progress_bar=False)
    (rec_dir / "rescue_recording_manifest.json").write_text(json.dumps({
        "schema_version": "rescue-recording-manifest-v2",
        "request_digest": "b" * 64,
        "selected_start_frame": 0,
        "selected_end_frame": n_samp,
        "sampling_frequency_hz": FS,
        "gain_uv_per_count": GAIN,
    }))
    spec = SnippetSpec(
        name="s", start_s=1.0, duration_s=8.0,
        channel_start=2, channel_count=20,
        split="development",
        selection_basis="quiet input regime",
        axes={"motion_regime": "quiet"},
    )
    m = build_snippet(spec, rec_dir, tmp_path / "snips", n_jobs=1)
    return load_snippet(tmp_path / "snips" / (m["name"] + "-" + m["spec_digest"][:10]))


def test_snr_profile_reports_noise_amplitude_and_ratio(snippet):
    prof = snr_profile(
        snippet, SnrConfig(noise_chunks=6, noise_chunk_s=0.5, n_jobs=1)
    )
    assert prof["schema"] == SNR_SCHEMA
    assert prof["noise_uv_median"] > 0
    assert prof["n_peaks"] > 0
    assert prof["event_amp_uv_p90"] > prof["noise_uv_median"]
    assert prof["snr"] == prof["event_amp_uv_p90"] / prof["noise_uv_median"]
    assert prof["config_digest"]


def test_stratify_by_snr_assigns_tertiles():
    profiles = {
        f"s{i}": {"snr": float(v)}
        for i, v in enumerate([2.0, 3.0, 4.0, 8.0, 9.0, 20.0])
    }
    labels = stratify_by_snr(profiles)
    assert labels["s0"] == "low" and labels["s1"] == "low"
    assert labels["s5"] == "high" and labels["s4"] == "high"
    assert set(labels.values()) == {"low", "medium", "high"}


def test_stratify_marks_non_finite_unknown_and_needs_three():
    with pytest.raises(ValueError, match="at least 3"):
        stratify_by_snr({"a": {"snr": 1.0}, "b": {"snr": 2.0}})
    labels = stratify_by_snr(
        {f"s{i}": {"snr": v} for i, v in enumerate([1.0, 2.0, 3.0, float("nan")])}
    )
    assert labels["s3"] == "unknown"
