#!/usr/bin/env python3
"""CLI boundary for the execution-disabled trained support-mask candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
KS = ROOT / "environments/rescue-production/.venv/lib/python3.12/site-packages"
for path in (ROOT, KS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from kilosort import io, preprocessing
from npx_preprocessing.motion import kilosort_support_mask_trained as trained


def _write_synthetic_native_files(results: Path) -> None:
    arrays = {
        name for name in trained.REQUIRED_NATIVE_FILES
        if name.endswith(".npy")
    }
    for name in arrays:
        np.save(results / name, np.zeros((1,), dtype=np.float32), allow_pickle=False)
    for name in set(trained.REQUIRED_NATIVE_FILES) - arrays:
        (results / name).write_text("orchestration-only synthetic artifact\n")


def _write_synthetic_snapshot(snapshot: Path) -> None:
    for name in trained.REQUIRED_SNAPSHOT_FILES:
        path = snapshot / name
        if name == "RECEIPT.json":
            trained.atomic_json(path, {
                "status": "orchestration_only_substitution",
                "saved_before_learned_batch_extraction": True,
                "wall3_axis0_binding": "spike_detection_templates.npy",
                "not_native_scientific_output": True,
            })
        else:
            np.save(path, np.zeros((1,), dtype=np.float32), allow_pickle=False)


def synthetic_orchestration_runner(config: dict, outcome: str):
    """Exercise native I/O hook and saves without sorting/training/detection."""
    def runner(**kwargs):
        settings = kwargs["settings"]
        n_channels = int(settings["n_chan_bin"])
        batch = int(settings["batch_size"])
        nt = int(settings["nt"])
        frames = max(batch * 4, batch + 2 * nt + 1)
        data = np.random.default_rng(714).normal(
            size=(frames, n_channels)
        ).astype(np.float32)
        bfile = io.BinaryFiltered(
            filename=kwargs["filename"], n_chan_bin=n_channels,
            fs=float(settings["fs"]), NT=batch, nt=nt,
            nt0min=int(settings["nt0min"]), chan_map=np.arange(n_channels),
            whiten_mat=None, do_CAR=bool(kwargs["do_CAR"]),
            artifact_threshold=float("inf"), device=torch.device("cpu"),
            file_object=data,
        )
        geometry = np.asarray(config["orchestration_only"]["geometry_um"], dtype=float)
        whitening = preprocessing.get_whitening_matrix(
            bfile, geometry[:, 0], geometry[:, 1], nskip=25, nrange=32
        )
        bfile.whiten_mat = whitening
        stamped = bfile.support_mask_policy.audit.batches
        for phase_name in (
            "pca_and_universal_template_learning",
            "universal_detection_and_features",
            "learned_template_extraction",
        ):
            stamped.phase.name = phase_name
            bfile.padded_batch_to_torch(0)
        if outcome == "fail-after-whitening":
            raise RuntimeError("injected synthetic downstream failure")
        snapshot = Path(config["trained"]["pre_extraction_snapshot_dir"])
        results = Path(kwargs["results_dir"])
        _write_synthetic_snapshot(snapshot)
        _write_synthetic_native_files(results)
        return {"status": "orchestration_only_success"}
    return runner


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--expected-config-sha256", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--execute-trained", action="store_true")
    parser.add_argument("--synthetic-fixture-root", type=Path)
    parser.add_argument(
        "--synthetic-outcome", choices=("success", "fail-after-whitening"),
        default="success",
    )
    args = parser.parse_args()
    synthetic = args.synthetic_fixture_root is not None
    config, contract = trained.validate_trained_contract(
        args.config.resolve(), expected_config_sha256=args.expected_config_sha256,
        allow_synthetic_orchestration=synthetic,
        fixture_root=args.synthetic_fixture_root,
    )
    report = {
        "status": "validated",
        "config_sha256": args.expected_config_sha256,
        "contract_digest": contract.contract_digest,
        "execution_enabled": contract.execution_enabled,
        "orchestration_only_synthetic": config["orchestration_only"]["synthetic"],
        "recording_binary_opened": False,
        "trained_mode": config["trained"]["mode"],
        "saved_smoke_bank_allowed": config["trained"]["allow_saved_smoke_bank"],
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.validate_only:
        return 0
    runner = None
    if synthetic:
        runner = synthetic_orchestration_runner(config, args.synthetic_outcome)
    result, complete = trained.run_native_trained(
        config_path=args.config.resolve(),
        expected_config_sha256=args.expected_config_sha256,
        allow_synthetic_orchestration=synthetic,
        fixture_root=args.synthetic_fixture_root,
        native_runner=runner,
    )
    print(json.dumps({"result": result, "complete": complete}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
