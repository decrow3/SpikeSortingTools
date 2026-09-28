import numpy as np
import pytest
from testing.luke_dh_exact_chunk_motion import ExactChunkLatticeMotion


def test_half_open_boundaries_and_final_partial_chunk():
    m=ExactChunkLatticeMotion([0,-40,-80],sampling_frequency_hz=30_000,chunk_length_samples=7500,n_samples=18_000)
    samples=np.array([0,7499,7500,14999,15000,17999])
    assert np.array_equal(m.disp_at_s(samples/30_000),[0,0,-40,-40,-80,-80])


def test_coordinate_sign_and_grid_shape():
    m=ExactChunkLatticeMotion([-40],sampling_frequency_hz=30_000,chunk_length_samples=7500,n_samples=7000)
    registered=np.array([2000.]); observed=m.uncorrect_s(np.array([.1]),registered)
    assert np.array_equal(observed,[1960.])
    assert np.array_equal(m.correct_s(np.array([.1]),observed),registered)
    assert m.disp_at_s(np.array([.1,.2]),np.array([1000.,2000.]),grid=True).shape==(2,2)


def test_wrong_origin_and_out_of_support_are_not_clipped():
    m=ExactChunkLatticeMotion([0,-40],sampling_frequency_hz=30_000,chunk_length_samples=7500,n_samples=10_000)
    assert m.final_padding_samples == 5000
    assert m.disp_at_s(np.array([14_999/30_000]))[0] == -40
    with pytest.raises(ValueError,match="outside declared support"):
        m.disp_at_s(np.array([-1/30_000]))
    with pytest.raises(ValueError,match="outside declared support"):
        m.disp_at_s(np.array([15_000/30_000]))
    with pytest.raises(ValueError,match="sample clock"):
        m.disp_at_s(np.array([0.10001]))
