#!/usr/bin/env python3
"""Static/API fixture for the isolated runner's exact Kilosort wiring."""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

from kilosort import template_matching
from kilosort.run_kilosort import run_kilosort


RUNNER = Path(__file__).with_name("en_common_support_crop_screen.py")


def main() -> None:
    source = RUNNER.read_text()
    tree = ast.parse(source)
    nested = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "snapshot_extract")
    wrapper_args = [arg.arg for arg in nested.args.args]
    installed_args = list(inspect.signature(template_matching.extract).parameters)
    if wrapper_args != installed_args:
        raise RuntimeError(f"snapshot wrapper signature differs: {wrapper_args} != {installed_args}")
    call = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "run_kilosort"
    )
    keywords = {item.arg for item in call.keywords}
    installed_keywords = set(inspect.signature(run_kilosort).parameters)
    if not keywords <= installed_keywords:
        raise RuntimeError(f"unsupported run_kilosort keywords: {keywords - installed_keywords}")
    required = {"settings", "probe", "filename", "results_dir", "data_dtype", "device", "bad_channels"}
    if not required <= keywords:
        raise RuntimeError(f"missing run_kilosort wiring: {required - keywords}")
    installed_source = inspect.getsource(run_kilosort)
    seeds = {
        "numpy_seed1": "np.random.seed(1)" in installed_source,
        "torch_seed1": "torch.random.manual_seed(1)" in installed_source,
        "torch_cuda_all_seed1": "torch.cuda.manual_seed_all(1)" in installed_source,
    }
    if not all(seeds.values()):
        raise RuntimeError("installed direct Kilosort seed path differs from production")
    restoration = "finally:\n        template_matching.extract = original_extract" in source
    if not restoration or "nblocks=0" not in source:
        raise RuntimeError("snapshot restoration or effective nblocks=0 missing")
    print(json.dumps({
        "schema": "en-common-support-crop-wiring-fixture-v1",
        "status": "pass",
        "snapshot_extract_signature": wrapper_args,
        "run_kilosort_keywords": sorted(keywords),
        "seed_path": seeds,
        "snapshot_monkeypatch_restored_in_finally": restoration,
        "native_nblocks0_explicit": True,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
