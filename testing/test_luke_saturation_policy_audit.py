import numpy as np

from testing import luke_saturation_policy_audit as audit


def test_interpolate_masked_intervals_bridges_expanded_run():
    values = np.arange(10, dtype=np.float32)[:, None]
    values[4:6] = 100
    mask = np.zeros_like(values, dtype=bool)
    mask[4:6] = True
    result = audit.interpolate_masked_intervals(values, mask, radius_samples=1)
    assert np.allclose(result[3:7, 0], [3, 4, 5, 6])
    assert np.array_equal(result[:3], values[:3])
    assert np.array_equal(result[7:], values[7:])


def test_interpolate_masked_intervals_handles_edge_interval():
    values = np.arange(6, dtype=np.float32)[:, None]
    mask = np.zeros_like(values, dtype=bool)
    mask[:2] = True
    result = audit.interpolate_masked_intervals(values, mask, radius_samples=0)
    assert result[0, 0] == result[1, 0] == 2


def test_saturation_stages_include_only_motion_free_policies(monkeypatch):
    monkeypatch.setattr(audit.conditioning, "ks_highpass", lambda values, fs: values)
    raw = np.zeros((20, 3), dtype=np.float32)
    phase = raw.copy()
    phase[10, 1] = 20
    stages, mask = audit.saturation_stages(raw, phase, 1000.0, 10.0)
    assert mask[10, 1]
    assert "no_saturation_replacement" in stages
    assert all("motion" not in name for name in stages)
