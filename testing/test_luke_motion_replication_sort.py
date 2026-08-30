import json

from testing import luke_motion_replication_sort as replication


def test_shared_motion_path_selects_complete_shared_window(tmp_path, monkeypatch):
    parent = tmp_path / "runs"
    wrong = parent / "full_wrong"
    right = parent / "full_right"
    wrong.mkdir(parents=True)
    right.mkdir(parents=True)
    (wrong / "manifest.json").write_text(json.dumps({"window": {"name": "registration_outlier"}}))
    (wrong / "motion.npy").touch()
    (right / "manifest.json").write_text(json.dumps({"window": {"name": "shared_template"}}))
    (right / "motion.npy").touch()
    monkeypatch.setattr(replication, "MOTION_PARENT", parent)

    assert replication.shared_motion_path() == right


def test_replication_conditions_include_geometry_adapted_quarter_gain():
    assert replication.CONDITION_SIGMA_UM == {
        "no_external_correction": None,
        "rigid_gain_025": 20.0,
        "rigid_gain_025_sigma10": 10.0,
        "rigid_gain_025_p2": 20.0,
        "single_ks_preprocessing": None,
        "single_ks_preprocessing_rigid_gain_025_p2": 20.0,
        "single_ks_preprocessing_dredge_400_400_p2": 20.0,
    }
    assert replication.CONDITIONS == tuple(replication.CONDITION_SIGMA_UM)
