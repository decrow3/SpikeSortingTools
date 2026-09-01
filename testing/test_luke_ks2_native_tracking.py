import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("luke_ks2_native_tracking.py")
SPEC = importlib.util.spec_from_file_location("luke_ks2_native_tracking", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_patched_batch_constants():
    assert MODULE.NT == 65472
    assert MODULE.NTBUFF == 64
    assert MODULE.STRIDE == 65408
    assert MODULE.ALIGNED_NT - MODULE.NTBUFF == 65536


def test_boundary_ratio_detects_a_trough():
    bases = np.arange(50, dtype=np.int64) * MODULE.STRIDE
    phases = np.linspace(1000, MODULE.STRIDE - 1000, 20, dtype=np.int64)
    broadly_distributed = (bases[:, None] + phases[None, :]).reshape(-1)
    interior = bases + MODULE.STRIDE // 2
    times = np.concatenate((interior, broadly_distributed))
    result = MODULE.boundary_ratio(times)
    assert result["boundary_ratio"] < 0.98


def test_pin_matches_expected_commit():
    import json

    pin = json.loads(MODULE.PIN_PATH.read_text())
    assert pin["commit"] == MODULE.EXPECTED_COMMIT
    assert len(pin["mex_sha256"]) == 8
