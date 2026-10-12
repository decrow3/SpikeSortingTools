"""Replay all immutable v3 adversarial functions against frozen evaluator v4."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prior = load(
    "preserved_v3_adversarial_tests",
    ROOT / "joint_model_k10_evaluator_h1_20261012_v4" / "PRESERVED_V3_ADVERSARIAL_TESTS.py",
)
prior.evaluator = load(
    "frozen_k10_evaluator_v4",
    ROOT / "joint_model_k10_evaluator_h1_20261012_v4" / "EVALUATOR.py",
)

for test_name in sorted(name for name in vars(prior) if name.startswith("test_")):
    globals()[test_name] = getattr(prior, test_name)

