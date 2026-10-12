"""Replay the immutable v1 independent adversarial suite against frozen K7 v2."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
REVIEW_WORK = HERE.parent
REPOSITORY = REVIEW_WORK.parent
V2_PACKET = REVIEW_WORK / "joint_model_k7_frame_score_h1_20261012_v2"
V1_REVIEW = REVIEW_WORK / "joint_model_k7_frame_score_h1_20261012_v1_independent_review_20261012_v1"
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


v2 = _load("frozen_joint_model_k7_v2_rereview", V2_PACKET / "IMPLEMENTATION.py")
prior = _load("immutable_joint_model_k7_v1_adversarial", V1_REVIEW / "ADVERSARIAL_TESTS.py")
prior.k7 = v2

for _name in dir(prior):
    if _name.startswith("test_"):
        globals()[_name] = getattr(prior, _name)

