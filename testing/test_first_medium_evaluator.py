import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

from testing.first_medium_evaluator import (
    ARM_ORDER,
    MEASURED,
    UNMEASURED,
    UNRESOLVED,
    ArmInput,
    evaluate_four_arms,
    load_arm_from_spec,
)
from testing.sort_comparison import load_comparison_inputs


def cfg():
    return {
        "sampling_frequency_hz": 1000.0,
        "duration_s": 4.0,
        "correspondence_tolerance_ms": 1.0,
        "minimum_correspondence_overlap": 0.1,
        "primary_retention": 0.5,
        "minimum_valid_amplitude_windows": 2,
        "minimum_common_time_fraction": 0.5,
        "minimum_measurable_unit_fraction": 0.5,
        "coincidence_tolerance_ms": 1.0,
        "coincidence_depth_um": 75.0,
        "coincidence_seed": 4,
        "longitudinal_bin_s": 1.0,
    }


def windows(cluster, missing):
    return pd.DataFrame({
        "cluster_id": cluster,
        "status": "finite_interior",
        "start_s": [0.0, 2.0],
        "end_s": [2.0, 4.0],
        "missing_pct": [missing, missing],
    })


def arm(arm_id, times, clusters, missing, *, kind="synthetic_fixture", depths=True, waveform=False):
    values = np.asarray(times, dtype=np.int64)
    cluster_values = np.asarray(clusters, dtype=np.int64)
    sort = {
        "st": values,
        "cl": cluster_values,
        "amp": np.ones(len(values)),
        "labels": {},
        "identity_digest": f"{arm_id}-identity",
    }
    if depths:
        sort["depth"] = np.full(len(values), 200.0)
    qc = {
        "amplitude_windows": pd.concat(
            [windows(int(cluster), missing) for cluster in np.unique(cluster_values)], ignore_index=True
        ),
        "request_digest": f"{arm_id}-qc",
    }
    evidence = None
    if waveform:
        evidence = {
            int(cluster): {"early": np.array([[1.0, 0.0]]), "middle": np.array([[1.0, 0.0]]), "late": np.array([[1.0, 0.0]])}
            for cluster in np.unique(cluster_values)
        }
    return ArmInput(arm_id, kind, sort, qc, {"fixture": True}, evidence)


def unavailable(arm_id):
    return ArmInput(arm_id, "unavailable", None, None, {"status": "unavailable"})


def base_arms(*, waveform=False):
    times = [100, 200, 1100, 1200, 2100, 2200, 3100, 3200]
    return {
        "REF384": arm("REF384", times, [1, 2] * 4, 20, waveform=waveform),
        "existing_corrected_B384": arm("existing_corrected_B384", times, [11, 12] * 4, 10, waveform=waveform),
        "repaired_B384": arm("repaired_B384", times, [21, 22] * 4, 5, waveform=waveform),
        "REF384_repeat": arm("REF384_repeat", times, [31, 32] * 4, 20, waveform=waveform),
    }


def test_vertical_four_arm_fixture_is_never_scientific(tmp_path):
    output = tmp_path / "out"
    report = evaluate_four_arms(base_arms(waveform=True), cfg(), output_dir=output)
    assert report["scientific_scorecard"] == {}
    assert set(report["fixture_results_not_scientific"]) == {
        "REF384__existing_corrected_B384", "REF384__repaired_B384",
        "existing_corrected_B384__repaired_B384", "REF384__REF384_repeat",
    }
    assert report["decision"]["status"] == "inconclusive"
    assert json.loads((output / "FOUR_ARM_REPORT.json").read_text())["execution_enabled"] is False


def test_lost_baseline_unit_stays_in_population_denominator():
    arms = base_arms(waveform=True)
    repaired = arm(
        "repaired_B384",
        [100, 1100, 2100, 3100],
        [21] * 4,
        5,
        waveform=True,
    )
    arms["repaired_B384"] = repaired
    report = evaluate_four_arms(arms, cfg())
    pair = report["fixture_results_not_scientific"]["REF384__repaired_B384"]
    endpoint = pair["endpoints"]["population_measurable_fraction"]
    assert endpoint["status"] == MEASURED
    assert endpoint["value"] == 0.5
    descriptive = pair["endpoints"]["exclusive_identity_continuity_improvement"][
        "descriptive_exclusive_baseline_event_retention"
    ]
    assert descriptive["value"] == 0.5


def test_controlled_split_and_merge_have_baseline_population_burden():
    arms = base_arms(waveform=True)
    arms["REF384"] = arm("REF384", [100, 200, 300, 400], [1, 1, 2, 2], 20, waveform=True)
    arms["repaired_B384"] = arm("repaired_B384", [100, 200, 300, 400], [9, 9, 9, 9], 5, waveform=True)
    report = evaluate_four_arms(arms, cfg())
    endpoint = report["fixture_results_not_scientific"]["REF384__repaired_B384"]["endpoints"]["split_merge_burden"]
    assert endpoint["status"] == MEASURED
    assert endpoint["value"] == 1.0
    assert endpoint["passes"] is False


def test_unchanged_events_with_shifted_coordinates_keep_correspondence():
    arms = base_arms(waveform=True)
    arms["repaired_B384"].sort["depth"][:] = 900.0
    report = evaluate_four_arms(arms, cfg())
    pair = report["fixture_results_not_scientific"]["REF384__repaired_B384"]
    retention = pair["endpoints"]["exclusive_identity_continuity_improvement"][
        "descriptive_exclusive_baseline_event_retention"
    ]
    assert retention["value"] == 1.0
    assert pair["endpoints"]["exclusive_identity_continuity_improvement"]["status"] == UNRESOLVED


def test_missing_waveform_and_depth_are_unmeasured_not_zero_or_pass():
    report = evaluate_four_arms(base_arms(waveform=False), cfg())
    endpoint = report["fixture_results_not_scientific"]["REF384__repaired_B384"]["endpoints"]["waveform_instability"]
    assert endpoint["status"] == UNMEASURED
    assert endpoint["value"] is None and endpoint["passes"] is None

    arms = base_arms(waveform=True)
    for value in arms.values():
        value.sort.pop("depth")
    report = evaluate_four_arms(arms, cfg())
    boundary = report["fixture_results_not_scientific"]["REF384__repaired_B384"]["endpoints"]["boundary_burden"]
    assert boundary["status"] == UNMEASURED
    assert boundary["value"] is None and boundary["passes"] is None


def test_unavailable_prospective_arms_do_not_create_fixture_or_scientific_rows():
    arms = base_arms(waveform=True)
    arms["REF384"] = ArmInput("REF384", "real_saved", arms["REF384"].sort, arms["REF384"].qc, {"saved": True}, arms["REF384"].waveform_evidence)
    arms["existing_corrected_B384"] = ArmInput(
        "existing_corrected_B384", "real_saved", arms["existing_corrected_B384"].sort,
        arms["existing_corrected_B384"].qc, {"saved": True}, arms["existing_corrected_B384"].waveform_evidence,
    )
    arms["repaired_B384"] = unavailable("repaired_B384")
    arms["REF384_repeat"] = unavailable("REF384_repeat")
    report = evaluate_four_arms(arms, cfg())
    assert set(report["scientific_scorecard"]) == {"REF384__existing_corrected_B384"}
    assert report["unavailable_pairs"]["REF384__repaired_B384"]["verdict"] == "inconclusive"
    assert report["decision"]["reason_codes"] == ["PROSPECTIVE_REAL_ARM_OUTPUTS_UNAVAILABLE"]


def _write_saved_sort(root, arm_id, cluster_offset, missing):
    curated = root / arm_id / "sorter_output"
    qc_dir = root / arm_id / "qc"
    (qc_dir / "amp_truncation").mkdir(parents=True)
    curated.mkdir(parents=True)
    times = np.array([100, 200, 1100, 1200, 2100, 2200, 3100, 3200], dtype=np.int64)
    clusters = np.array([1, 2] * 4, dtype=np.int64) + cluster_offset
    full_st = np.c_[times, np.zeros(len(times)), np.ones(len(times)) * 10]
    np.save(curated / "spike_times.npy", times)
    np.save(curated / "spike_clusters.npy", clusters)
    np.save(curated / "full_st.npy", full_st)
    np.save(curated / "kept_spikes.npy", np.arange(len(times), dtype=np.int64))
    np.save(curated / "spike_positions.npy", np.c_[np.zeros(len(times)), np.ones(len(times)) * 200])
    (curated / "cluster_KSLabel.tsv").write_text(
        "cluster_id\tKSLabel\n" + "".join(f"{value}\tgood\n" for value in np.unique(clusters))
    )
    np.savez(
        qc_dir / "amp_truncation" / "truncation_qc.npz",
        cid=np.repeat(np.unique(clusters), 2),
        window_blocks=np.tile(np.array([[0, 1], [2, 3]], dtype=np.int64), (2, 1)),
        popts=np.tile(np.array([[0.0, 1.0, 1.0]]), (4, 1)),
        mpcts=np.full(4, missing, dtype=float),
    )
    return curated, qc_dir


def clock_spec(origin=0):
    return {
        "saved_frame": "crop_local_samples" if origin == 0 else "global_acquisition_samples",
        "crop_origin_global_frame": origin,
        "local_frames_half_open": [0, 4000],
        "global_frames_half_open": [origin, origin + 4000],
    }


def test_saved_format_loader_to_matcher_endpoint_decision_report(tmp_path):
    ref_curated, ref_qc = _write_saved_sort(tmp_path, "REF384", 0, 20)
    b_curated, b_qc = _write_saved_sort(tmp_path, "existing_corrected_B384", 10, 10)
    ref_loaded, ref_qc_loaded = load_comparison_inputs("REF384", ref_curated, ref_qc, sampling_frequency_hz=1000.0)
    b_loaded, b_qc_loaded = load_comparison_inputs(
        "existing_corrected_B384", b_curated, b_qc, sampling_frequency_hz=1000.0
    )
    bindings = lambda arm_id: {
        "arm_contract_id": arm_id, "input_identity": "fixture-input", "clock_identity": "fixture-clock",
        "geometry_identity": "fixture-geometry", "runtime_identity": "fixture-runtime",
        "config_identity": "fixture-config", "fixture": "saved-format",
    }
    ref = load_arm_from_spec({
        "arm_id": "REF384", "evidence_kind": "real_saved",
        "curated_output": str(ref_curated), "qc_dir": str(ref_qc), "provenance": bindings("REF384"),
        "expected_identity_digest": ref_loaded["identity_digest"],
        "expected_qc_request_digest": ref_qc_loaded["request_digest"],
        "clock_normalization": clock_spec(),
    }, sampling_frequency_hz=1000.0)
    b384 = load_arm_from_spec({
        "arm_id": "existing_corrected_B384", "evidence_kind": "real_saved",
        "curated_output": str(b_curated), "qc_dir": str(b_qc),
        "provenance": bindings("existing_corrected_B384"),
        "expected_identity_digest": b_loaded["identity_digest"],
        "expected_qc_request_digest": b_qc_loaded["request_digest"],
        "clock_normalization": clock_spec(),
    }, sampling_frequency_hz=1000.0)
    arms = {
        "REF384": ref,
        "existing_corrected_B384": b384,
        "repaired_B384": unavailable("repaired_B384"),
        "REF384_repeat": unavailable("REF384_repeat"),
    }
    report = evaluate_four_arms(arms, cfg(), output_dir=tmp_path / "report")
    pair = report["scientific_scorecard"]["REF384__existing_corrected_B384"]
    assert pair["endpoints"]["median_common_time_missingness_improvement_pp"]["value"] == 10.0
    assert pair["endpoints"]["waveform_instability"]["status"] == UNMEASURED
    assert pair["verdict"] == "inconclusive"
    assert (tmp_path / "report" / "pairs" / "REF384__existing_corrected_B384" / "candidate_manifest.json").is_file()


def test_real_saved_loader_rejects_unbound_identity_before_read(tmp_path):
    missing = tmp_path / "does-not-exist"
    try:
        load_arm_from_spec({
            "arm_id": "REF384", "evidence_kind": "real_saved",
            "curated_output": str(missing), "qc_dir": str(missing), "provenance": {},
        }, sampling_frequency_hz=1000.0)
    except ValueError as error:
        assert "unbound real-saved provenance" in str(error)
    else:
        raise AssertionError("unbound real saved arm was accepted")


def test_exact_global_clock_normalization_and_absent_qc_stays_unmeasured(tmp_path):
    curated, qc_dir = _write_saved_sort(tmp_path, "REF384", 0, 20)
    origin = 10_000
    local = np.load(curated / "spike_times.npy")
    np.save(curated / "spike_times.npy", local + origin)
    # Remove only the cached derivative; saved sorter arrays remain intact.
    (qc_dir / "amp_truncation" / "truncation_qc.npz").unlink()
    from pipeline.config import fingerprint
    import hashlib
    from testing.luke_amplitude_dropout_audit import read_curated_arrays
    raw, hashes = read_curated_arrays(curated)
    label_bytes = (curated / "cluster_KSLabel.tsv").read_bytes()
    expected_identity = fingerprint({
        "curated": str(curated.resolve()), "files": hashes,
        "labels_sha256": hashlib.sha256(label_bytes).hexdigest(),
    })
    provenance = {
        "arm_contract_id": "REF384", "input_identity": "fixture-input",
        "clock_identity": "fixture-clock", "geometry_identity": "fixture-geometry",
        "runtime_identity": "fixture-runtime", "config_identity": "fixture-config",
    }
    loaded = load_arm_from_spec({
        "arm_id": "REF384", "evidence_kind": "real_saved",
        "curated_output": str(curated), "qc_dir": str(qc_dir), "provenance": provenance,
        "expected_identity_digest": expected_identity,
        "expected_qc_request_digest": "UNMEASURED_ABSENT_TRUNCATION_QC",
        "clock_normalization": clock_spec(origin),
    }, sampling_frequency_hz=1000.0)
    assert np.array_equal(loaded.sort["st"], local)
    assert loaded.qc["available"] is False
    assert loaded.provenance["clock_normalization"]["observed_exact_offset"] == origin

    comparison = arm("existing_corrected_B384", local, [11, 12] * 4, 10, kind="real_saved", waveform=True)
    pair = evaluate_four_arms({
        "REF384": loaded, "existing_corrected_B384": comparison,
        "repaired_B384": unavailable("repaired_B384"),
        "REF384_repeat": unavailable("REF384_repeat"),
    }, cfg())["scientific_scorecard"]["REF384__existing_corrected_B384"]
    assert pair["endpoints"]["population_measurable_fraction"]["status"] == UNMEASURED
    assert pair["endpoints"]["median_common_time_missingness_improvement_pp"]["status"] == UNMEASURED
    assert pair["verdict"] == "inconclusive"


def test_clock_normalization_rejects_one_row_with_wrong_offset(tmp_path):
    curated, qc_dir = _write_saved_sort(tmp_path, "REF384", 0, 20)
    origin = 10_000
    times = np.load(curated / "spike_times.npy") + origin
    times[3] += 1
    np.save(curated / "spike_times.npy", times)
    from pipeline.config import fingerprint
    import hashlib
    from testing.luke_amplitude_dropout_audit import read_curated_arrays
    _, hashes = read_curated_arrays(curated)
    labels = (curated / "cluster_KSLabel.tsv").read_bytes()
    expected = fingerprint({"curated": str(curated.resolve()), "files": hashes,
                            "labels_sha256": hashlib.sha256(labels).hexdigest()})
    provenance = {
        "arm_contract_id": "REF384", "input_identity": "fixture-input",
        "clock_identity": "fixture-clock", "geometry_identity": "fixture-geometry",
        "runtime_identity": "fixture-runtime", "config_identity": "fixture-config",
    }
    try:
        load_arm_from_spec({
            "arm_id": "REF384", "evidence_kind": "real_saved",
            "curated_output": str(curated), "qc_dir": str(qc_dir), "provenance": provenance,
            "expected_identity_digest": expected,
            "expected_qc_request_digest": "unused", "clock_normalization": clock_spec(origin),
        }, sampling_frequency_hz=1000.0)
    except ValueError as error:
        assert "exact declared offset" in str(error)
    else:
        raise AssertionError("one-row clock corruption was accepted")


def test_packaged_cli_uses_only_snapshot_modules(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    package = tmp_path / "packet" / "source"
    (package / "testing").mkdir(parents=True)
    (package / "pipeline").mkdir()
    (package / "testing" / "__init__.py").write_text("")
    (package / "pipeline" / "__init__.py").write_text("")
    for relative in (
        "testing/first_medium_evaluator.py",
        "testing/sort_comparison.py", "testing/luke_amplitude_dropout_audit.py",
        "pipeline/config.py", "pipeline/truncation.py",
    ):
        shutil.copy2(repo / relative, package / relative)
    ref_curated, ref_qc = _write_saved_sort(tmp_path / "data", "REF384", 0, 20)
    b_curated, b_qc = _write_saved_sort(tmp_path / "data", "existing_corrected_B384", 10, 10)
    ref_loaded, ref_qc_loaded = load_comparison_inputs("REF384", ref_curated, ref_qc, sampling_frequency_hz=1000.0)
    b_loaded, b_qc_loaded = load_comparison_inputs("existing_corrected_B384", b_curated, b_qc, sampling_frequency_hz=1000.0)
    def spec(arm_id, curated, qc_dir, loaded, loaded_qc):
        return {
            "arm_id": arm_id, "evidence_kind": "real_saved",
            "curated_output": str(curated), "qc_dir": str(qc_dir),
            "provenance": {
                "arm_contract_id": arm_id, "input_identity": "fixture-input",
                "clock_identity": "fixture-clock", "geometry_identity": "fixture-geometry",
                "runtime_identity": "fixture-runtime", "config_identity": "fixture-config",
            },
            "expected_identity_digest": loaded["identity_digest"],
            "expected_qc_request_digest": loaded_qc["request_digest"],
            "clock_normalization": clock_spec(),
        }
    request = {
        "execution_enabled": False, "evaluation_config": cfg(),
        "arms": [
            spec("REF384", ref_curated, ref_qc, ref_loaded, ref_qc_loaded),
            spec("existing_corrected_B384", b_curated, b_qc, b_loaded, b_qc_loaded),
            {"arm_id": "repaired_B384", "evidence_kind": "unavailable", "provenance": {}},
            {"arm_id": "REF384_repeat", "evidence_kind": "unavailable", "provenance": {}},
        ],
    }
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request))
    output = tmp_path / "cli-output"
    outside = tmp_path / "outside-workspace"
    outside.mkdir()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(package)
    result = subprocess.run(
        [sys.executable, "-m", "testing.first_medium_evaluator", "--request", str(request_path), "--output", str(output)],
        cwd=outside, env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads((output / "FOUR_ARM_REPORT.json").read_text())
    for receipt in report["implementation_provenance"].values():
        assert Path(receipt["path"]).is_relative_to(package)
