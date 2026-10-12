"""Replay the immutable v1 adversarial functions against frozen evaluator v2.

The author packet's preserved test file still resolves evaluator v1.  This
review-owned harness changes only that module binding, not any test function.
"""

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
    "preserved_v1_adversarial_tests",
    ROOT / "joint_model_k10_evaluator_h1_20261012_v2" / "PRIOR_ADVERSARIAL_TESTS.py",
)
prior.evaluator = load(
    "frozen_k10_evaluator_v2",
    ROOT / "joint_model_k10_evaluator_h1_20261012_v2" / "EVALUATOR.py",
)

# Expose the exact preserved test callables to pytest. Their globals remain in
# the preserved module, whose evaluator binding now names frozen v2 above.
for test_name in sorted(name for name in vars(prior) if name.startswith("test_")):
    globals()[test_name] = getattr(prior, test_name)

