import numpy as np

from testing.luke_cluster21_detection_lineage import detection_template_depths, stage_verdict


def test_stage_verdict_uses_first_material_loss_boundary():
    assert stage_verdict(0.2, 0.3, 0.1) == "loss_at_or_before_full_st_detection"
    assert stage_verdict(0.01, 0.2, 0.1) == "loss_between_full_st_and_retained_sorter_output"
    assert stage_verdict(0.01, 0.09, 0.1) == "stage_attribution_unresolved"


def test_detection_template_depths_uses_iu_channel_map():
    depths = detection_template_depths({"iU": np.array([2, 0]), "yc": np.array([10, 20, 30])})
    assert depths.tolist() == [30.0, 10.0]
