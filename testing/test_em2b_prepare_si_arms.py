from testing.em2b_prepare_si_arms import ARM_SPECS


def test_stage1_is_only_the_primary_and_rounding_bridge():
    assert ARM_SPECS == {
        "unrounded_kriging": {"field": "unrounded", "method": "kriging"},
        "rounded_kriging_bridge": {"field": "rounded", "method": "kriging"},
    }
