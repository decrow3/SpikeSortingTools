import numpy as np
import pytest

from testing.luke_motion_candidate_sort import (
    SOURCE_RECORDING,
    condition_paths,
    field_displacement,
    sorter_params,
)


def test_field_displacement_preserves_native_nonrigid_field():
    displacement = np.arange(12, dtype=float).reshape(3, 4)

    result = field_displacement(displacement, "nonrigid")

    assert result is displacement


def test_field_displacement_preserves_native_medicine_field():
    displacement = np.arange(6, dtype=float).reshape(3, 2)

    result = field_displacement(displacement, "medicine_sigma10")

    assert result is displacement


def test_field_displacement_repeats_depth_mean_for_rigid_field():
    displacement = np.array([[0.0, 2.0, 4.0], [-3.0, 0.0, 3.0]])

    result = field_displacement(displacement, "rigid")

    np.testing.assert_allclose(result, [[2.0, 2.0, 2.0], [0.0, 0.0, 0.0]])
    np.testing.assert_allclose(np.ptp(result, axis=1), 0.0)


def test_field_displacement_returns_zeros_for_identity_field():
    displacement = np.array([[1.0, -2.0], [3.0, 4.0]])

    result = field_displacement(displacement, "identity")

    np.testing.assert_array_equal(result, np.zeros_like(displacement))
    assert result.dtype == displacement.dtype


@pytest.mark.parametrize(
    ("field", "gain"),
    [("rigid_gain_025", 0.25), ("rigid_gain_050", 0.5), ("rigid_gain_075", 0.75)],
)
def test_field_displacement_scales_rigid_mean(field, gain):
    displacement = np.array([[0.0, 2.0, 4.0], [-3.0, 0.0, 3.0]])

    result = field_displacement(displacement, field)

    np.testing.assert_allclose(result, [[2.0 * gain] * 3, [0.0] * 3])


def test_field_displacement_rejects_unknown_field():
    with pytest.raises(ValueError, match="Unknown field"):
        field_displacement(np.zeros((2, 2)), "bad")


def test_internal_rigid_uses_untouched_source_recording():
    condition, recording, sort = condition_paths("ks_internal_rigid")

    assert condition == "kilosort_internal_rigid"
    assert recording == SOURCE_RECORDING
    assert sort.name == "kilosort_internal_rigid"


def test_internal_rigid_enables_kilosort_rigid_correction(monkeypatch):
    monkeypatch.setattr(
        "testing.luke_motion_candidate_sort.build_sorter_params",
        lambda setting: {"do_correction": False, "nblocks": 7, "sentinel": 1},
    )

    params = sorter_params("ks_internal_rigid")

    assert params == {"do_correction": True, "nblocks": 1, "sentinel": 1}
