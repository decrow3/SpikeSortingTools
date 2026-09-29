import numpy as np

from testing.em2a_operator_screen import (
    assign_stepwise,
    effective_kernel,
    interpolation_insertion,
    metric_row,
    render,
)


def test_stepwise_assignment_gives_exact_boundary_to_later_cell():
    centers = np.asarray([0.0, 0.25, 0.5])
    values = np.asarray([1.0, 2.0, 3.0])
    times = np.asarray([0.124999, 0.125, 0.374999, 0.375])

    np.testing.assert_array_equal(assign_stepwise(centers, values, times), [1, 2, 2, 3])


def test_interpolated_bad_channel_is_composed_before_motion_kernel():
    from spikeinterface.preprocessing import get_spatial_interpolation_kernel
    from spikeinterface.preprocessing.preprocessing_tools import get_kriging_channel_weights

    full_ids = ["a", "imec1.ap#AP191", "b"]
    parent_ids = ["a", "b"]
    full_geometry = np.asarray([[0.0, 0.0], [0.0, 20.0], [0.0, 40.0]])
    source_geometry = full_geometry[[0, 2]]
    insertion, receipt = interpolation_insertion(
        parent_ids,
        full_ids,
        source_geometry,
        full_geometry,
        get_kriging_channel_weights,
    )
    kernel, zeros = effective_kernel(
        get_spatial_interpolation_kernel,
        insertion,
        full_geometry,
        full_geometry[[1]],
        0.0,
        {"field": "fixture", "method": "nearest", "dtype": "float32", "force_extrapolate": False},
    )

    np.testing.assert_allclose(kernel[:, 0].sum(), 1.0, rtol=0, atol=1e-6)
    assert np.count_nonzero(kernel[:, 0]) == 2
    assert zeros.tolist() == [False]
    assert receipt["bad_channel_id"] == "imec1.ap#AP191"


def test_render_uses_state_specific_kernels_and_tracks_structural_zeros():
    source = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    kernels = {
        0.0: (np.asarray([[1.0], [0.0]], dtype=np.float32), np.asarray([False])),
        1.0: (np.asarray([[0.0], [0.0]], dtype=np.float32), np.asarray([True])),
    }

    output, zeros = render(source, np.asarray([0.0, 1.0]), kernels)

    np.testing.assert_array_equal(output[:, 0], [1.0, 0.0])
    np.testing.assert_array_equal(zeros[:, 0], [False, True])


def test_trace_metric_detects_nonredundant_arm():
    reference = np.zeros((2, 2), dtype=np.float32)
    candidate = reference.copy()
    candidate[1, 1] = 2.0

    result = metric_row("w", "candidate", "reference", candidate, reference)

    assert result["count_abs_gt_1e_6"] == 1
    assert result["max_abs_uv"] == 2.0
    assert result["rms_uv"] == 1.0
