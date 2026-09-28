import json
import sys
import types

import numpy as np
import pandas as pd

from pipeline.unit_quality import build_unit_quality_tables, write_unit_quality_artifacts


def _write_fixture(root):
    curated = root / "curated"
    legacy = root / "qc"
    curated.mkdir()
    (legacy / "amp_truncation").mkdir(parents=True)
    # Recording: 120 seconds at 1 kHz. Unit 0 occupies both 60-second bins;
    # unit 1 occurs only near the beginning. Unit 0 has one sub-ms pair.
    times = np.array([1000, 1001, 2000, 61000, 62000, 5000, 6000])
    clusters = np.array([0, 0, 0, 0, 0, 1, 1])
    np.save(curated / "spike_times.npy", times)
    np.save(curated / "spike_clusters.npy", clusters)
    np.save(curated / "amplitudes.npy", np.array([10, 11, 12, 13, 14, 4, 6], dtype=float))
    positions = np.c_[np.zeros(len(times)), np.array([50, 51, 52, 60, 61, 5, 6])]
    np.save(curated / "spike_positions.npy", positions)
    templates = np.zeros((2, 5, 3), dtype=float)
    templates[0, 2, 1] = -10
    templates[1, 2, 1] = -9
    np.save(curated / "templates.npy", templates)
    np.save(curated / "similar_templates.npy", np.array([[1, .9], [.9, 1]]))
    np.save(curated / "ops.npy", {"xc": np.array([0, 0, 0]), "yc": np.array([0, 50, 100])})
    pd.DataFrame({"cluster_id": [0, 1], "KSLabel": ["good", "mua"]}).to_csv(
        curated / "cluster_KSLabel.tsv", sep="\t", index=False
    )
    pd.DataFrame({"cluster_id": [0, 1], "ContamPct": [2.0, 12.0]}).to_csv(
        curated / "cluster_ContamPct.tsv", sep="\t", index=False
    )
    np.savez(
        legacy / "amp_truncation/truncation_qc.npz",
        cid=np.array([0, 0]),
        window_blocks=np.array([[0, 999], [1000, 1999]]),
        popts=np.array([[10, 1, 1.02], [8, 1, 2.0]]),
        mpcts=np.array([2.0, 50.0]),
    )
    np.savez(
        legacy / "amp_truncation/present_qc.npz",
        cid=np.array([0]), valid_blocks=np.array([[0, 2000]]),
    )
    return curated, legacy


def test_builds_identity_joined_metrics_without_mislabeling_feature_amplitude(tmp_path):
    curated, legacy = _write_fixture(tmp_path)
    metrics, flags, definitions, policy, summary = build_unit_quality_tables(
        curated, legacy, sampling_frequency=1000, duration_s=120
    )
    unit0 = metrics.set_index("unit_id").loc[0]
    unit1 = metrics.set_index("unit_id").loc[1]
    assert unit0["presence_ratio_60s"] == 1.0
    assert unit1["presence_ratio_60s"] == 0.5
    assert unit0["isi_violations_count_1p5ms"] == 1
    assert unit0["legacy_truncation_saturated_window_count"] == 1
    assert unit0["legacy_truncation_missing_pct_median_uncensored"] == 2.0
    assert unit0["nearby_similar_good_template_count"] == 0
    assert unit1["nearby_similar_good_template_count"] == 1
    assert "ks_feature_amplitude_median" in metrics
    assert "amplitude_uv" not in " ".join(metrics.columns)
    assert definitions["metrics"]["ks_feature_amplitude_median"]["unit"] == "Kilosort feature scale"
    assert set(metrics.columns) == set(definitions["metrics"])
    assert policy["profiles"]["allen_like"]["evaluable"] is False
    assert set(flags["automatic_curation_action"]) == {"none"}
    assert summary["automatic_curation_changes"] == 0


def test_writes_complete_machine_readable_artifact_set(tmp_path):
    curated, legacy = _write_fixture(tmp_path)
    outputs = build_unit_quality_tables(curated, legacy, sampling_frequency=1000, duration_s=120)
    target = tmp_path / "standard"
    write_unit_quality_artifacts(target, *outputs)
    expected = {
        "unit_quality_metrics.csv", "unit_quality_flags.csv", "metric_definitions.json",
        "quality_policy.json", "quality_summary.json",
    }
    assert {path.name for path in target.iterdir()} == expected
    assert json.loads((target / "quality_summary.json").read_text())["unit_count"] == 2


def test_rejects_spikes_outside_recording(tmp_path):
    curated, legacy = _write_fixture(tmp_path)
    np.save(curated / "spike_times.npy", np.array([120000] * 7))
    import pytest
    with pytest.raises(ValueError, match="outside"):
        build_unit_quality_tables(curated, legacy, sampling_frequency=1000, duration_s=120)


def test_refractory_pair_count_includes_exact_boundary():
    from pipeline.unit_quality import _isi_metrics
    result = _isi_metrics(
        np.array([0, 30, 61]), sampling_frequency=30000, duration_s=1,
        isi_threshold_ms=1.5, rp_refractory_ms=1.0, rp_censored_ms=0,
    )
    assert result["rp_violations_count_1ms"] == 1


def test_standard_stage_writes_receipt_and_refuses_changed_inputs(tmp_path, monkeypatch):
    curated, legacy = _write_fixture(tmp_path)
    recording_dir = tmp_path / "recording"
    recording_dir.mkdir()

    class Recording:
        def get_sampling_frequency(self):
            return 1000

        def get_total_duration(self):
            return 120

    core = types.ModuleType("spikeinterface.core")
    core.load = lambda path: Recording()
    package = types.ModuleType("spikeinterface")
    package.core = core
    monkeypatch.setitem(sys.modules, "spikeinterface", package)
    monkeypatch.setitem(sys.modules, "spikeinterface.core", core)

    from pipeline.downstream import run_standard_qc_stage
    identity = {"identity_digest": "sort-id"}
    output = tmp_path / "standard"
    receipt = run_standard_qc_stage(recording_dir, curated, legacy, output, identity)
    assert receipt["complete"] is True
    assert receipt["summary"]["unit_count"] == 2
    assert receipt["sort_identity_digest"] == "sort-id"
    assert run_standard_qc_stage(recording_dir, curated, legacy, output, identity) == receipt

    amplitudes = np.load(curated / "amplitudes.npy")
    amplitudes[0] += 1
    np.save(curated / "amplitudes.npy", amplitudes)
    import pytest
    with pytest.raises(RuntimeError, match="another configuration"):
        run_standard_qc_stage(recording_dir, curated, legacy, output, identity)
