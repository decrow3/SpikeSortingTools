import numpy as np

from testing.luke_dropout_anchor_screen import case_decision


def test_case_advances_only_with_anchor_and_material_loss():
    loss, advances = case_decision(True, 0.9, 0.7, 0.1)
    assert np.isclose(loss, 0.2)
    assert advances
    assert case_decision(False, 0.9, 0.7, 0.1)[1] is False
    assert case_decision(True, 0.9, 0.85, 0.1)[1] is False
