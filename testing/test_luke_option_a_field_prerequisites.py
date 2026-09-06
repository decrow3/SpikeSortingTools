"""Known-answer fixtures for the Option A field-prerequisite receipt.

The receipt decides whether a motion field may be applied to voltage at all, so
each test here pins one way it could say yes when it should say no.
"""

import json

import numpy as np
import pytest

from pipeline.motion_coordinates import MOTION_FIELD_SCHEMA, load_qualified_motion_field
from testing.ladder_motion_estimate import FieldGate
from testing.luke_option_a_field_prerequisites import (
    PrerequisiteRefusal,
    _edges,
    assess,
    check_declared_time_reference,
    inter_estimator_disagreement,
    nominate_development_interval,
    probe_operator_polarity,
    read_spikeglx_time_origin,
    support_fraction,
    verify_time_origin,
    window_excursion,
)

FS = 30000.0
FIRST_SAMPLE = 91729810
ORIGIN_S = FIRST_SAMPLE / FS


def _meta(tmp_path, first_sample=FIRST_SAMPLE, fs=FS):
    path = tmp_path / "run_t0.imec0.ap.meta"
    path.write_text(
        f"fileTimeSecs=1000.0\nfirstSample={first_sample}\nimSampRate={fs}\nnSavedChans=385\n"
    )
    return path


def _manifest(tmp_path, t_start=ORIGIN_S):
    path = tmp_path / "binary.json"
    path.write_text(json.dumps(
        {"class": "BinaryRecording",
         "kwargs": {"file_paths": ["traces.raw"], "t_starts": [t_start],
                    "sampling_frequency": FS, "num_channels": 384}}
    ))
    return path


# --------------------------------------------------------------------------- #
# time origin: one source is a claim, two that agree is a mapping
# --------------------------------------------------------------------------- #
def test_the_time_origin_is_recovered_from_two_independent_sources(tmp_path):
    origin = verify_time_origin(_meta(tmp_path), _manifest(tmp_path))
    assert origin.seconds == pytest.approx(ORIGIN_S)
    assert origin.first_sample == FIRST_SAMPLE
    payload = origin.to_dict()
    assert payload["mapping"] == "t_recording_s = t_motion_s - acquisition_time_origin_s"


def test_disagreeing_time_origin_sources_are_a_refusal(tmp_path):
    """Agreement is the evidence; without it there is no verified mapping."""
    with pytest.raises(PrerequisiteRefusal, match="time-origin sources disagree"):
        verify_time_origin(_meta(tmp_path), _manifest(tmp_path, t_start=ORIGIN_S + 5.0))


def test_a_meta_without_the_sample_counter_is_a_refusal(tmp_path):
    path = tmp_path / "bad.meta"
    path.write_text("fileTimeSecs=1000.0\nimSampRate=30000.0\n")
    with pytest.raises(PrerequisiteRefusal, match="firstSample"):
        read_spikeglx_time_origin(path)


# --------------------------------------------------------------------------- #
# a declared time reference that contradicts the values
# --------------------------------------------------------------------------- #
def _field_npz(tmp_path, times, *, reference="selected_recording_start"):
    path = tmp_path / "estimate.npz"
    depths = np.array([100.0, 200.0])
    np.savez(
        path,
        schema_version=MOTION_FIELD_SCHEMA,
        qualification_passed=np.array(True),
        qualification_digest="d", time_reference=reference, depth_reference="probe_y_um",
        displacement_convention="observed_depth_offset_um", estimator="dredge", polarity="pos",
        displacement_um=np.zeros((times.size, depths.size)),
        time_s=times, depth_um=depths,
        support=np.ones((times.size, depths.size)),
        confidence=np.ones((times.size, depths.size)),
    )
    return path


def test_a_field_on_the_acquisition_clock_is_caught_despite_its_declaration(tmp_path):
    """The exact shape of the artifact on disk: right string, wrong clock."""
    duration = 10473.55
    acquisition = np.linspace(ORIGIN_S + 0.5, ORIGIN_S + duration, 50)
    check = check_declared_time_reference(_field_npz(tmp_path, acquisition),
                                          verify_time_origin(_meta(tmp_path), _manifest(tmp_path)),
                                          duration)
    assert check["declared_time_reference"] == "selected_recording_start"
    assert check["verdict"].startswith("MISDECLARED")
    assert "acquisition clock" in check["verdict"]


def test_a_field_genuinely_on_the_recording_clock_passes(tmp_path):
    duration = 10473.55
    check = check_declared_time_reference(
        _field_npz(tmp_path, np.linspace(0.5, duration - 0.5, 50)),
        verify_time_origin(_meta(tmp_path), _manifest(tmp_path)), duration,
    )
    assert check["verdict"] == "consistent"


def test_the_loader_refuses_an_acquisition_clock_field_when_told_the_duration(tmp_path):
    """The guard that makes the declaration checkable rather than trusted."""
    duration = 10473.55
    path = _field_npz(tmp_path, np.linspace(ORIGIN_S + 0.5, ORIGIN_S + duration, 50))

    # without the duration the declared string is all there is, and it passes
    loaded = load_qualified_motion_field(path)
    assert loaded["time_s"][-1] > duration        # plainly off the recording clock

    with pytest.raises(ValueError, match="acquisition clock"):
        load_qualified_motion_field(path, recording_duration_s=duration)

    honest_dir = tmp_path / "ok"
    honest_dir.mkdir()
    honest = _field_npz(honest_dir, np.linspace(0.5, duration - 0.5, 50))
    assert load_qualified_motion_field(honest, recording_duration_s=duration)["time_s"].size == 50


# --------------------------------------------------------------------------- #
# support
# --------------------------------------------------------------------------- #
def test_support_counts_only_cells_with_real_detections(tmp_path):
    """A field is not supported where nothing was detected."""
    time_bins = np.arange(4, dtype=float) + 0.5          # 4 one-second bins
    depth_bins = np.array([10.0, 30.0])                  # 2 depth bins
    # 20 peaks in every cell except the whole second depth bin
    samples, depths = [], []
    for t in time_bins:
        for _ in range(20):
            samples.append(t * FS)
            depths.append(10.0)
    peaks = np.zeros(len(samples), dtype=[("sample_index", "i8")])
    peaks["sample_index"] = np.asarray(samples, dtype=np.int64)
    locations = np.zeros(len(depths), dtype=[("y", "f8")])
    locations["y"] = depths
    np.save(tmp_path / "peaks.npy", peaks)
    np.save(tmp_path / "peak_locations.npy", locations)

    result = support_fraction(tmp_path / "peaks.npy", tmp_path / "peak_locations.npy",
                              time_bins, depth_bins, fs_hz=FS, min_peaks_per_cell=5)
    assert result["support_fraction"] == pytest.approx(0.5)   # one of two depth bins
    assert result["depth_bins_never_supported"] == 1
    assert result["time_bins_fully_supported_fraction"] == 0.0


def test_bin_edges_come_from_centres(tmp_path):
    edges = _edges(np.array([0.5, 1.5, 2.5]))
    assert edges == pytest.approx([0.0, 1.0, 2.0, 3.0])
    with pytest.raises(PrerequisiteRefusal):
        _edges(np.array([1.0]))


# --------------------------------------------------------------------------- #
# scale uncertainty and interval nomination
# --------------------------------------------------------------------------- #
def _field(scale, n=600):
    t = np.arange(n, dtype=float)
    trace = scale * np.sin(2 * np.pi * t / 120.0)
    return {"motion": trace[:, None], "time_recording_s": t,
            "depth_bins_um": np.array([0.0, 100.0])}


def test_estimator_disagreement_is_never_reported_as_a_measured_gain_error():
    """Disagreement says they cannot all be right, not that any one is wrong."""
    fields = {"a": _field(1.0), "b": _field(4.0)}
    windows = [(0.0, 120.0), (120.0, 240.0), (240.0, 360.0)]
    spread = inter_estimator_disagreement(fields, windows)
    assert spread["max_over_min_ratio"] == pytest.approx(4.0, rel=1e-6)
    assert spread["absolute_gain_status"] == "unmeasured"
    assert "NOT an error measurement" in spread["note"]
    assert not any("error_fraction" in key for key in spread)


def test_the_nominated_interval_is_where_every_estimator_agrees_there_is_motion():
    """A window wins on the *minimum* across estimators, not the maximum.

    One estimator alone reporting a large excursion is exactly the disagreement
    decision 0013 records as unresolved; ranking on the minimum makes the
    nomination survive it.
    """
    quiet = np.zeros(600)
    loud = np.zeros(600)
    loud[240:360] = np.linspace(-20, 20, 120)          # motion only in [240, 360)
    disputed = np.zeros(600)
    disputed[120:240] = np.linspace(-50, 50, 120)      # only this estimator sees it
    t = np.arange(600, dtype=float)
    fields = {
        "a": {"motion": (quiet + loud)[:, None], "time_recording_s": t},
        "b": {"motion": (quiet + loud)[:, None], "time_recording_s": t},
        "c": {"motion": (disputed + loud)[:, None], "time_recording_s": t},
    }
    result = nominate_development_interval(fields, [(0.0, 600.0)], window_s=120.0)
    assert result["nominated"]["start_s"] == 240.0
    assert result["quietest_considered"]["min_across_estimators_um"] == 0.0
    assert "no sorter output is consulted" in result["selection_rule"]


def test_a_window_no_estimator_covers_is_not_nominated():
    t = np.arange(100, dtype=float)
    fields = {"a": {"motion": np.zeros((100, 1)), "time_recording_s": t}}
    assert nominate_development_interval(fields, [(500.0, 900.0)])["nominated"] is None


def test_window_excursion_needs_enough_bins():
    t = np.array([0.0, 1.0])
    assert np.isnan(window_excursion(np.zeros((2, 1)), t, 0.0, 120.0))


# --------------------------------------------------------------------------- #
# the receipt fails closed
# --------------------------------------------------------------------------- #
def test_assess_blocks_when_the_evidence_limbs_are_missing(tmp_path):
    """qualify_field fails closed without support, reproducibility and error."""
    motion_root = tmp_path / "motion"
    for name, scale in (("ks-motion", 1.0), ("dredge-motion", 4.0)):
        directory = motion_root / name
        directory.mkdir(parents=True)
        field = _field(scale)
        np.save(directory / "motion.npy", field["motion"])
        np.save(directory / "time_bins.npy", field["time_recording_s"] + ORIGIN_S)
        np.save(directory / "depth_bins.npy", field["depth_bins_um"])

    receipt = assess(
        motion_root=motion_root, meta_path=_meta(tmp_path),
        recording_manifest=_manifest(tmp_path), duration_s=600.0,
        development_windows=[(0.0, 600.0)], gate=FieldGate(),
    )
    assert receipt["can_run_bounded_option_a"] is False
    reasons = {b["prerequisite"]: b for b in receipt["blocking_prerequisites"]}
    assert reasons["split-half field reproducibility"]["status"] == "not_measured"
    assert "estimated gain error within tolerance" in reasons
    for entry in receipt["estimators"].values():
        assert entry["qualification"]["passes"] is False
    # the time axes were mapped back onto the recording clock before use
    assert receipt["time_origin"]["acquisition_time_origin_s"] == pytest.approx(ORIGIN_S)
    assert receipt["nominated_development_interval"]["nominated"] is not None


# --------------------------------------------------------------------------- #
# operator integrity
# --------------------------------------------------------------------------- #
def test_the_polarity_probe_reports_unavailability_rather_than_guessing():
    """An unverifiable sign convention must not be assumed."""
    result = probe_operator_polarity()
    if not result["available"]:
        assert "not importable" in result["reason"]
        return
    # where SpikeInterface is present the probe must actually resolve the sign
    assert result["zero_motion_identity"] is True
    assert result["forward_inverse_symmetric"] is True
    assert result["polarity_resolved"] is True
    assert result["recovered_magnitude_um"] == pytest.approx(result["applied_displacement_um"],
                                                             rel=0.05)
    assert result["passes"] is True
