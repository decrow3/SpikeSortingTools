import numpy as np

from testing.luke_medicine_5min_correction_matrix import ARMS, INTERVAL_S, load_field


def test_matrix_has_unwarped_rigid_nonrigid_and_historical_controls():
    assert ARMS["unwarped"] is None
    assert any(v and v["spatial"] == "rigid" for v in ARMS.values())
    assert any(v and v["spatial"] == "nonrigid" and v["p"] == 2 for v in ARMS.values())
    assert ARMS["nonrigid_g100_p1_s20_historical"]["p"] == 1


def test_load_field_accepts_bounded_terminal_nearest_extension(tmp_path):
    path = tmp_path / "field.npz"
    np.savez(path, time_s=np.array([930.25, 1229.75]), depth_um=np.array([0.0, 20.0]),
             displacement_um=np.zeros((2, 2)))
    field = load_field(path)
    assert field["displacement_um"].shape == (2, 2)
    assert INTERVAL_S == (930.0, 1230.0)
