import json
from pathlib import Path

import numpy as np

from testing import luke_imec1_medicine_stage3_q_v1 as q


def test_exact_per_window_plan():
    for scope in ("pilot", "full"):
        assert all(window["stop_s"]-window["start_s"] == q.WINDOW_S
                   for window in q.window_specs(scope))
    assert len(q.window_specs("pilot")) == 29
    assert len(q.window_specs("full")) == 174
    assert q.window_specs("full")[-1]["terminal_shifted"]


def test_u_block_plan_covers_session_once():
    blocks = q.block_specs()
    assert len(blocks) == 87
    assert blocks[0]["start_s"] == 0
    assert blocks[-1]["stop_s"] == q.SESSION_DURATION_S
    assert blocks[-1]["terminal_extended"]
    for left, right in zip(blocks, blocks[1:]):
        assert left["stop_s"] == right["start_s"]


def test_block_population_concatenation(tmp_path: Path):
    spec = {"id": "full_w001", "start_s": 60.0, "stop_s": 180.0}
    for block in q.block_specs()[:2]:
        target = tmp_path / "peak_cache" / block["id"] / "extraction"
        target.mkdir(parents=True)
        absolute = np.arange(block["start_s"], min(block["stop_s"], 240.0), 10.0)
        relative = absolute - block["start_s"]
        np.savez(target / "population.npz", time_s=relative, depth_um=absolute,
                 x_um=np.zeros(len(absolute)), amplitude=np.ones(len(absolute)))
    population, used = q.population_for_window_from_blocks(tmp_path, spec)
    assert [x["id"] for x in used] == ["full_block_000", "full_block_001"]
    assert np.array_equal(population["time_s"], np.arange(0.0, 120.0, 10.0))
    assert np.array_equal(population["depth_um"], np.arange(60.0, 180.0, 10.0))


def test_stitch_removes_constant_window_offsets(tmp_path: Path):
    for index, window in enumerate(q.window_specs("pilot")):
        target = tmp_path / "fields" / "pilot" / window["id"]
        target.mkdir(parents=True)
        times = np.arange(window["start_s"], window["stop_s"], q.GRID_S)
        depth = np.array([0.0, 100.0])
        common = np.sin(times[:, None] / 20.0)
        motion = common + np.array([[3.0, 8.0]]) + 17.0 * index
        np.savez(target / "field.npz", time_s=times-window["start_s"],
                 session_time_s=times, depth_um=depth, displacement_um=motion)
    result = q.stitch(tmp_path, "pilot")
    seam = json.loads((tmp_path / "seams_pilot.json").read_text())
    assert np.isfinite(result["displacement_um"]).all()
    assert seam["median_abs_seam_discontinuity_um"] < 1e-8


def test_rigid_field_difference_ignores_arbitrary_single_depth_coordinate():
    times = np.arange(10.0, 20.0, q.GRID_S)
    motion = np.sin(times)[:, None]
    a = {"session_time_s": times, "depth_um": np.array([-3.5]), "displacement_um": motion + 12}
    b = {"session_time_s": times, "depth_um": np.array([91.0]), "displacement_um": motion - 7}
    assert q.centered_field_difference(a, b) < 1e-10
