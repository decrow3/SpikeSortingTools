from pathlib import Path

import numpy as np

from testing.luke_medicine_matched_waveforms import dewhiten_selected


def test_dewhiten_selected_uses_transposed_inverse_for_time_major_templates(
    tmp_path: Path,
):
    """The file-loading consumer must solve a non-symmetric known answer."""
    physical = np.array([
        [[0.0, -1.0, 0.2], [0.1, -3.0, 0.5], [0.0, -1.0, 0.2]],
        [[0.2, 0.0, -0.5], [0.8, 0.1, -2.0], [0.2, 0.0, -0.5]],
    ])
    whitening = np.array([
        [2.0, 1.0, 0.0],
        [0.0, 3.0, 1.0],
        [1.0, 0.0, 4.0],
    ])
    # This is Kilosort's channel-major forward operation expressed without
    # copying the production dewhitening expression.
    stored = np.einsum("ij,ktj->kti", whitening, physical)
    inverse = np.linalg.inv(whitening)
    positions = np.array([[0.0, 0.0], [32.0, 20.0], [0.0, 40.0]])
    np.save(tmp_path / "templates.npy", stored)
    np.save(tmp_path / "whitening_mat_inv.npy", inverse)
    np.save(tmp_path / "channel_positions.npy", positions)

    cluster_ids = np.array([1, 0])
    bank, loaded_positions = dewhiten_selected(tmp_path, cluster_ids)
    wrong = stored @ inverse

    assert np.allclose(bank[1], physical[1], rtol=0, atol=1e-12)
    assert np.allclose(bank[0], physical[0], rtol=0, atol=1e-12)
    assert np.array_equal(loaded_positions, positions)
    assert not np.allclose(wrong, physical, rtol=0, atol=1e-8)
