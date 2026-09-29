import json

import numpy as np
import pandas as pd

from pipeline.completeness_timeline import build_completeness_timeline
from pipeline.downstream import run_completeness_timeline_stage


def fixture(tmp_path):
    curated = tmp_path / "curated"
    qc = tmp_path / "qc"
    (qc / "amp_truncation").mkdir(parents=True)
    curated.mkdir()
    np.save(curated / "spike_times.npy", np.arange(6000, dtype=np.int64))
    np.save(curated / "spike_clusters.npy", np.r_[np.ones(5000, dtype=np.int64), np.full(1000, 2, dtype=np.int64)])
    mpcts = np.array([10.0, 25.0, 50.0, 20.0, np.nan])
    normalization = np.array([10.0, 25.0, 50.0, 0.0, 10.0])
    popts = np.column_stack([
        np.full(5, 10.0), np.ones(5), 1.0 / (1.0 - normalization / 100.0)
    ])
    blocks = np.array([[i * 1000, i * 1000 + 999] for i in range(5)])
    np.savez(qc / "amp_truncation/truncation_qc.npz",
             cid=np.ones(5), window_blocks=blocks, popts=popts, mpcts=mpcts)
    np.savez(qc / "amp_truncation/present_qc.npz",
             cid=np.array([1.0]), valid_blocks=np.array([[0, 5000]]))
    return curated, qc


def test_timeline_separates_measurement_censoring_fit_failure_and_screening(tmp_path):
    curated, qc = fixture(tmp_path)
    timeline, units, policy, summary = build_completeness_timeline(
        curated, qc, sampling_frequency=10.0
    )
    assert timeline.measurement_status.tolist() == [
        "measured", "measured", "censored_at_least_50pct",
        "poor_fit_disagreement", "nonfinite",
    ]
    assert timeline.nominal_window_spikes.tolist() == [1000] * 5
    assert timeline.historical_fit_sample_count.tolist() == [999] * 5
    assert timeline.start_s.tolist() == [0.0, 100.0, 200.0, 300.0, 400.0]
    assert timeline.deterioration_screen.tolist() == [False, True, False, False, False]
    assert units.loc[0, "screening_baseline_missing_pct"] == 11.5
    assert units.loc[0, "eligible_spike_count"] == 5000
    assert bool(units.loc[0, "has_supported_window"])
    unsupported = units[units.unit_id == 2].iloc[0]
    assert unsupported.window_count == 0
    assert not bool(unsupported.has_supported_window)
    assert summary["units_without_supported_windows"] == 1
    assert summary["screening_only"] is True
    assert "not proof of lost spikes" in policy["interpretation"]["deterioration_screen"]


def test_identity_bound_timeline_stage_writes_and_reuses_artifacts(tmp_path):
    curated, qc = fixture(tmp_path)
    output = tmp_path / "timeline"
    identity = {"identity_digest": "sort-identity"}
    first = run_completeness_timeline_stage(
        curated, qc, output, identity, sampling_frequency=10.0
    )
    second = run_completeness_timeline_stage(
        curated, qc, output, identity, sampling_frequency=10.0
    )
    assert first == second
    assert first["summary"]["window_count"] == 5
    rows = pd.read_csv(output / "amplitude_completeness_timeline.csv")
    assert len(rows) == 5
    policy = json.loads((output / "amplitude_completeness_policy.json").read_text())
    assert policy["schema_version"] == "amplitude-completeness-timeline-v2"


def test_units_without_any_cached_fit_are_explicit(tmp_path):
    curated = tmp_path / "curated"
    qc = tmp_path / "qc/amp_truncation"
    curated.mkdir(); qc.mkdir(parents=True)
    np.save(curated / "spike_times.npy", np.array([10, 20, 30]))
    np.save(curated / "spike_clusters.npy", np.array([1, 1, 2]))
    np.savez(qc / "truncation_qc.npz", cid=np.empty(0),
             window_blocks=np.empty((0, 2), int), popts=np.empty((0, 3)), mpcts=np.empty(0))
    np.savez(qc / "present_qc.npz", cid=np.empty(0), valid_blocks=np.empty((0, 2), int))
    timeline, units, _, summary = build_completeness_timeline(
        curated, tmp_path / "qc", sampling_frequency=10.0
    )
    assert timeline.empty
    assert units.unit_id.tolist() == [1, 2]
    assert units.has_supported_window.tolist() == [False, False]
    assert summary["units_without_supported_windows"] == 2
