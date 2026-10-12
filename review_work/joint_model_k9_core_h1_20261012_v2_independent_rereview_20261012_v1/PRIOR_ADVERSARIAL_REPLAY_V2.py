"""Replay the byte-preserved v1 adversarial suite against frozen K9 v2."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
REVIEW_WORK = HERE.parent
V2_PACKET = REVIEW_WORK / "joint_model_k9_core_h1_20261012_v2"
REPOSITORY = REVIEW_WORK.parent
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


v2 = _load("frozen_joint_model_k9_v2_rereview", V2_PACKET / "IMPLEMENTATION.py")
prior = _load(
    "preserved_joint_model_k9_v1_adversarial",
    V2_PACKET / "PRESERVED_V1_ADVERSARIAL_TESTS.py",
)

# The preserved file intentionally retains its original v1 loader. Redirect only
# its implementation binding; all fixtures and assertions remain byte-preserved.
prior.k9 = v2

for _name in dir(prior):
    if _name.startswith("test_"):
        globals()[_name] = getattr(prior, _name)
