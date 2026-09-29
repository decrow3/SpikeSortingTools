#!/usr/bin/env python3
"""Build a hash-checked EM.2b scorecard arms manifest.

The frozen S0 and audited exact-DD controls are always included. Later arms are
included only when the production runner published a successful final receipt
that attests the saved sorting bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SOURCES = {
    "field": Path(
        "/media/huklaban5/writable/DARTsort_motion_experiments/incoming/"
        "luke0804_imec1_two_layer_v1/luke0804_imec1_two_layer_motion.npz"
    ),
    "mask": Path(
        "/mnt/NPX/Luke/DARTsort_motion_experiments/"
        "ck_input_intervals_20260927/censor_mask_v1.csv"
    ),
    "catalogue": Path(
        "/media/huklaban5/writable/DARTsort_motion_experiments/incoming/"
        "luke0804_imec1_q_field_v1/episode_catalogue.csv"
    ),
}
CONTROL_SORTS = {
    "none_s0": Path(
        "/mnt/NPX/Luke/DARTsort_motion_experiments/"
        "av_aw_handoff_20260926_v1/static_w2/dartsort_sorting.npz"
    ),
    "rounded_exact_dd": Path(
        "/home/huklaban5/DARTsort_experiment_scratch/em2b_w2_20260929/"
        "rounded_exact_dd_v1/sort/dartsort_sorting.npz"
    ),
}
EXPECTED_CONTROL_HASHES = {
    "none_s0": "be6106ed0cb0f99759fd23629dd3bcb5f50657dbffa238ad3721c15d1b21fba7",
    "rounded_exact_dd": "85f1537a4ffbac430d36e84b1f71e7ed5c0760287338115724ee9d80d052e61a",
}
EXACT_AUDIT = Path(
    "testing/outputs/em2b_w2_rounded_exact_dd_measurement_20260929/AUDIT.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def receipt(path: Path, expected: str | None = None) -> dict[str, object]:
    actual = sha256(path)
    if expected is not None and actual != expected:
        raise RuntimeError(f"hash differs for {path}: {actual} != {expected}")
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": actual}


def attest_runner_sort(run: Path) -> dict[str, object]:
    """Require a successful final production receipt that names the sort hash."""
    final_receipt = run / "receipt.json"
    value = json.loads(final_receipt.read_text())
    if value.get("status") != "complete" or value.get("exit_status") != 0:
        raise RuntimeError(f"runner did not complete successfully: {final_receipt}")
    config = run / "config.json"
    if value.get("config_sha256") != sha256(config):
        raise RuntimeError(f"runner config hash differs: {run}")
    sorting = (run / "sort/dartsort_sorting.npz").resolve()
    sorting_hash = sha256(sorting)
    matching = [
        item
        for item in value.get("stages", {}).get("sort", {}).get("artifacts", [])
        if Path(item.get("resolved_path", "")).resolve() == sorting
    ]
    if len(matching) != 1 or matching[0].get("sha256") != sorting_hash:
        raise RuntimeError(f"final receipt does not attest sorting bytes: {run}")
    return {
        "path": str(sorting),
        "bytes": sorting.stat().st_size,
        "sha256": sorting_hash,
        "run_receipt_path": str(final_receipt.resolve()),
        "run_receipt_sha256": sha256(final_receipt),
        "config_sha256": sha256(config),
    }


def parse_stage(value: str) -> tuple[str, Path]:
    try:
        name, path = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("stage must be NAME=RUN_DIRECTORY") from exc
    if not name or not path:
        raise argparse.ArgumentTypeError("stage must be NAME=RUN_DIRECTORY")
    return name, Path(path)


def build(contract_path: Path, stages: list[tuple[str, Path]]) -> dict[str, object]:
    contract_path = contract_path.resolve()
    contract = json.loads(contract_path.read_text())
    sources = {
        name: receipt(path, contract["inputs"][f"{name}_sha256"])
        for name, path in SOURCES.items()
    }
    arms = {
        name: receipt(path, EXPECTED_CONTROL_HASHES[name])
        for name, path in CONTROL_SORTS.items()
    }
    audit_path = (Path(__file__).resolve().parents[1] / EXACT_AUDIT).resolve()
    audit = json.loads(audit_path.read_text())
    exact = arms["rounded_exact_dd"]
    if not audit.get("scientific_stages_complete") or (
        audit.get("sorting", {}).get("sha256") != exact["sha256"]
    ):
        raise RuntimeError("exact-DD measurement audit does not attest sorting bytes")
    exact["measurement_audit_path"] = str(audit_path)
    exact["measurement_audit_sha256"] = sha256(audit_path)
    known = set(contract["arms"])
    for name, run in stages:
        if name not in known or name in arms:
            raise RuntimeError(f"unknown or duplicate arm: {name}")
        arms[name] = attest_runner_sort(run.resolve())
    return {
        "schema": "em2b-w2-arms-manifest-v1",
        "contract_path": str(contract_path),
        "contract_sha256": sha256(contract_path),
        **sources,
        "arms": arms,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--stage", action="append", default=[], type=parse_stage)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = build(args.contract, args.stage)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
