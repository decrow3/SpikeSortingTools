import numpy as np
from testing.luke_full_probe_registration_audit import shift_profile_to_reference,profile_correlation


def test_known_positive_physical_motion_requires_negative_native_shift():
    y=np.arange(0,1000,5.)
    reference=np.exp(-.5*((y-400)/25)**2)+.6*np.exp(-.5*((y-620)/15)**2)
    physically_moved=np.exp(-.5*((y-440)/25)**2)+.6*np.exp(-.5*((y-660)/15)**2)
    target=y[(y>=100)&(y<=900)]
    correct=shift_profile_to_reference(physically_moved,y,target,-40)
    wrong=shift_profile_to_reference(physically_moved,y,target,40)
    expected=reference[(y>=100)&(y<=900)]
    assert np.allclose(correct,expected)
    assert profile_correlation(correct,expected)>profile_correlation(wrong,expected)


def test_sampling_beyond_measured_depth_stays_missing():
    y=np.array([0.,10.,20.])
    got=shift_profile_to_reference(np.array([1.,2.,3.]),y,y,-10)
    assert np.array_equal(got[:2],[2.,3.])
    assert np.isnan(got[-1])
