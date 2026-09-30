from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from testing.en_tier1_panel import (
    arm_state_metrics,
    build_field_cells,
    chance_aware_coincidence,
    common_block_multiplicities,
    correspondence,
    correspondence_summary,
    run,
)


def test_field_cells_use_half_open_midpoints_and_merge_equal_state_segments() -> None:
    cells = build_field_cells(
        np.array([1.0, 3.0, 5.0, 7.0]),
        np.array([0.0, 40.0, 40.0, 0.0]),
        duration_s=8.0,
        block_seconds=2.0,
    )
    state, segment = cells.assign(np.array([0, 1999, 2000, 5999, 6000, 7999]), 1000.0)
    assert cells.states_um.tolist() == [0.0, 40.0]
    assert state.tolist() == [0, 0, 1, 1, 0, 0]
    assert segment.tolist() == [0, 0, 1, 1, 2, 2]
    assert cells.state_exposure_s.tolist() == pytest.approx([4.0, 4.0])
    assert cells.block_exposure_s.sum(axis=1).tolist() == pytest.approx([2.0] * 4)


def test_arm_metrics_are_segment_safe_and_report_exact_duplicates() -> None:
    cells = build_field_cells(
        np.array([1.0, 3.0, 5.0, 7.0]),
        np.array([0.0, 40.0, 40.0, 0.0]),
        duration_s=8.0,
        block_seconds=2.0,
    )
    # Unit 1 has a duplicate at 100 samples, a 20-sample short interval in q=0,
    # and a boundary-adjacent 1-sample pair at 1999/2000 that must be excluded.
    times = np.array([100, 100, 120, 1999, 2000, 3000, 3020, 7000, 7020], dtype=np.int64)
    clusters = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2], dtype=np.int64)
    order = np.argsort(times, kind="stable")
    sort = {
        "times": times[order],
        "clusters": clusters[order],
        "depths": np.array([250.0] * 5 + [1000.0] * 4)[order],
        "labels": {1: "good", 2: "mua"},
    }
    multiplicities = common_block_multiplicities(4, draws=40, seed=3)
    result = arm_state_metrics(
        sort,
        cells,
        sampling_frequency_hz=1000.0,
        num_samples=8000,
        multiplicities=multiplicities,
        minimum_reference_events=1,
        processing_depth_um=(200.0, 1200.0),
        scoring_depth_um=(300.0, 1100.0),
        seed=9,
    )
    states = result["state_metrics"].set_index("state_um")
    assert result["summary"]["raw_unit_count"] == 2
    assert result["summary"]["ks_good_unit_count"] == 1
    assert result["summary"]["exact_duplicate_count"] == 1
    assert result["summary"]["units_with_exact_duplicates"] == 1
    assert result["summary"]["exact_duplicate_fraction"] == pytest.approx(1 / 5)
    assert states.loc[0.0, "short_interval_count_1_29"] == 2
    assert states.loc[40.0, "short_interval_count_1_29"] == 1
    assert states.loc[0.0, "exact_duplicate_count"] == 1
    assert result["summary"]["edge_unit_fraction"] == pytest.approx(0.5)


def test_duplicate_fraction_is_specific_to_field_segments() -> None:
    split = build_field_cells(
        np.array([1.0, 3.0, 5.0, 7.0]),
        np.array([0.0, 40.0, 40.0, 0.0]),
        duration_s=8.0,
        block_seconds=2.0,
    )
    flat = build_field_cells(
        np.array([1.0, 3.0, 5.0, 7.0]),
        np.zeros(4),
        duration_s=8.0,
        block_seconds=2.0,
    )
    sort = {
        "times": np.array([100, 100, 1999, 2000], dtype=np.int64),
        "clusters": np.ones(4, dtype=np.int64),
        "depths": np.full(4, 500.0),
        "labels": {1: "good"},
    }
    multiplicities = common_block_multiplicities(4, draws=40, seed=3)
    kwargs = dict(
        sampling_frequency_hz=1000.0,
        num_samples=8000,
        multiplicities=multiplicities,
        minimum_reference_events=1,
        processing_depth_um=(200.0, 1200.0),
        scoring_depth_um=(300.0, 1100.0),
        seed=9,
    )
    split_result = arm_state_metrics(sort, split, **kwargs)
    flat_result = arm_state_metrics(sort, flat, **kwargs)
    assert split_result["summary"]["exact_duplicate_fraction"] == pytest.approx(1 / 2)
    assert flat_result["summary"]["exact_duplicate_fraction"] == pytest.approx(1 / 3)


def test_common_block_multiplicities_resample_exact_block_count() -> None:
    multiplicities = common_block_multiplicities(7, draws=25, seed=4)
    assert multiplicities.shape == (25, 7)
    assert np.all(multiplicities.sum(axis=1) == 7)


def test_correspondence_is_exclusive_and_primary_is_reciprocal() -> None:
    reference = {"times": np.array([10, 20, 30, 100]), "clusters": np.array([1, 1, 1, 2])}
    candidate = {"times": np.array([10, 20, 30, 101]), "clusters": np.array([7, 7, 7, 8])}
    edges = correspondence(
        reference, candidate, tolerance_frames=1, minimum_overlap=0.1, primary_retention=0.5
    )
    primary = edges[edges.primary_match]
    assert set(zip(primary.reference_cluster, primary.candidate_cluster)) == {(1, 7), (2, 8)}
    assert primary.set_index("reference_cluster").loc[1, "matched_events"] == 3
    summary = correspondence_summary(
        edges, np.array([1, 2]), np.array([7, 8]), draws=30, seed=5
    )
    assert summary["primary_matches"] == 2
    assert summary["lost_reference_fraction"] == 0
    assert summary["median_primary_f1"] == pytest.approx(1.0)


def test_correspondence_reference_bootstrap_uses_explicit_common_seed(monkeypatch) -> None:
    seeds = []

    def fake_bootstrap(values, *, draws, seed, statistic):
        seeds.append(seed)
        return 0.0, 0.0, 0.0

    monkeypatch.setattr("testing.en_tier1_panel._unit_bootstrap", fake_bootstrap)
    edges = pd.DataFrame({"primary_match": pd.Series(dtype=bool)})
    for local_seed in (500, 600):
        correspondence_summary(
            edges,
            np.array([1, 2]),
            np.array([7, 8]),
            draws=30,
            seed=local_seed,
            reference_seed=123,
        )
    assert seeds[:5] == [123, 501, 125, 503, 510]
    assert seeds[5:] == [123, 601, 125, 603, 610]


def test_chance_aware_coincidence_marks_nearby_cross_unit_events() -> None:
    sort = {
        "times": np.array([100, 101, 500, 900], dtype=np.int64),
        "clusters": np.array([1, 2, 1, 2], dtype=np.int64),
        "depths": np.array([500.0, 510.0, 500.0, 510.0]),
    }
    units = pd.DataFrame(
        {"unit_id": [1, 2], "in_processing_domain": [True, True]}
    )
    blocks = np.array([0.0, 0.5, 1.0])
    multiplicities = common_block_multiplicities(2, draws=40, seed=6)
    result = chance_aware_coincidence(
        sort, units, num_samples=1000, sampling_frequency_hz=1000.0,
        block_edges_s=blocks, multiplicities=multiplicities,
        tolerance_ms=2.0, depth_tolerance_um=75.0,
        processing_depth_um=(200.0, 1200.0), seed=7,
    )
    assert result["observed_marked_spike_fraction"] == pytest.approx(0.5)
    assert result["eligible_spikes"] == 4


def _write_synthetic_sort(root: Path, identity: str | None) -> None:
    root.mkdir(parents=True)
    times = np.array([100, 101, 120, 1000, 1020, 3000, 3020, 7000, 7020], dtype=np.int64)
    clusters = np.array([1, 2, 1, 1, 1, 2, 2, 1, 2], dtype=np.int32)
    positions = np.c_[np.zeros(len(times)), np.where(clusters == 1, 250.0, 1000.0)].astype(np.float32)
    np.save(root / "spike_times.npy", times[:, None])
    np.save(root / "spike_clusters.npy", clusters)
    np.save(root / "spike_positions.npy", positions)
    (root / "cluster_KSLabel.tsv").write_text("cluster_id\tKSLabel\n1\tgood\n2\tmua\n")
    if identity is not None:
        def receipt(name: str) -> dict[str, object]:
            path = root / name
            return {"size_bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        (root.parents[1] / "sort_identity.json").write_text(
            json.dumps({
                "identity_digest": identity,
                "files": {name: receipt(name) for name in (
                    "spike_times.npy", "spike_clusters.npy", "cluster_KSLabel.tsv"
                )},
            }) + "\n"
        )
    else:
        settings = {
            "do_correction": False, "effective_nblocks": 0,
            "do_CAR": True, "artifact_threshold": "Infinity",
        }
        manifest = {
            "complete": True, "request_digest": "request",
            "summary": {"final_spike_count": len(times), "critical_saved_settings": settings},
        }
        (root.parent / "rescue_sort_manifest.json").write_text(json.dumps(manifest) + "\n")


def test_end_to_end_synthetic_panel(tmp_path: Path) -> None:
    roots = {}
    for arm in ("REF", "XR", "A", "B"):
        root = tmp_path / arm / "sort" / "sorter_output"
        _write_synthetic_sort(root, arm.lower() if arm in {"REF", "XR"} else None)
        roots[arm] = root
    (tmp_path / "XR/COMPLETE.json").write_text(json.dumps({"status": "complete"}) + "\n")
    for arm in ("A", "B"):
        (tmp_path / arm / "COMPLETE.json").write_text(json.dumps({"status": "complete"}) + "\n")
    field_time = np.array([1.0, 3.0, 5.0, 7.0])
    for name, displacement in (
        ("A", np.c_[np.array([0.0, 40.0, 40.0, 0.0]), np.array([0.0, 40.0, 40.0, 0.0])]),
        ("B", np.array([0.0, 40.0, 40.0, 0.0])),
    ):
        np.savez(tmp_path / f"field_{name}.npz", time_s=field_time, displacement_um=displacement)
    config = json.loads(Path("configs/en_tier1_panel.v1.json").read_text())
    config["scope"].update(sampling_frequency_hz=1000.0, num_samples=8000, duration_s=8.0,
                           reference_sort_identity="ref")
    for arm in config["arms"]:
        config["arms"][arm]["sorter_output"] = str(roots[arm])
    config["arms"]["REF"].update(
        sort_identity=str(tmp_path / "REF/sort_identity.json"), expected_sort_identity="ref"
    )
    config["arms"]["XR"].update(
        sort_identity=str(tmp_path / "XR/sort_identity.json"), expected_sort_identity="xr",
        complete_receipt=str(tmp_path / "XR/COMPLETE.json"),
    )
    for arm in ("A", "B"):
        config["arms"][arm]["complete_receipt"] = str(tmp_path / arm / "COMPLETE.json")
        path = tmp_path / f"field_{arm}.npz"
        config["fields"][arm].update(
            path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            time_key="time_s", displacement_key="displacement_um",
        )
    config["uncertainty"].update(bootstrap_draws=30, time_block_seconds=2.0)
    config["state_assignment"]["minimum_reference_state_events_per_unit"] = 1
    config["metrics"]["edge"].update(processing_depth_um=[200.0, 1200.0], scoring_depth_um=[300.0, 1100.0])
    config["metrics"]["chance_aware_coincidence"].update(tolerance_ms=2.0, depth_tolerance_um=75.0)
    config["metrics"]["correspondence"].update(tolerance_ms=2.0)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config) + "\n")
    output = tmp_path / "output"
    result = run(config_path, output)
    assert result["status"] == "complete"
    assert result["arm_summaries"]["B"]["raw_unit_count"] == 2
    assert "exact_duplicate_fraction" not in result["arm_summaries"]["REF"]
    assert result["field_arm_summaries"]["A"]["REF"]["exact_duplicate_fraction"] >= 0
    assert result["field_arm_summaries"]["B"]["REF"]["exact_duplicate_fraction"] >= 0
    assert result["correspondence_to_ref"]["B"]["primary_matches"] == 2
    complete = json.loads((output / "COMPLETE.json").read_text())
    assert complete["manifest_sha256"] == hashlib.sha256((output / "MANIFEST.json").read_bytes()).hexdigest()
    for name in ("TIER1_PANEL.png", "TIER1_PANEL.pdf", "STATE_RATE_RHO.png", "STATE_RATE_RHO.pdf"):
        assert (output / name).stat().st_size > 0
