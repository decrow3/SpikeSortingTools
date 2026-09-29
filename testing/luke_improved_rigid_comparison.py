"""Compare the completed external-rigid sort handoff with the motion-off reference."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

from pipeline.config import fingerprint
from testing.rigid_comparison_handoff import verify
from testing.sort_comparison import compare_sorts, load_comparison_inputs


SCHEMA = "luke-improved-rigid-comparison-v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(partial, path)


def wait_for_handoff(config_path: Path, poll_seconds: float = 30.0) -> None:
    """Wait only for the atomically published manifest, never a partial tree."""
    config = json.loads(config_path.read_text())
    if config.get("schema_version") != SCHEMA:
        raise ValueError("unsupported comparison config schema")
    manifest = Path(config["candidate"]["handoff_dir"]) / "MANIFEST.json"
    while not manifest.is_file():
        print("waiting for verified rigid comparison handoff", flush=True)
        time.sleep(poll_seconds)


def run(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    if config.get("schema_version") != SCHEMA:
        raise ValueError("unsupported comparison config schema")
    baseline_receipt = json.loads(Path(config["baseline"]["curation_receipt"]).read_text())
    if (
        baseline_receipt.get("complete") is not True
        or baseline_receipt.get("sort_identity_digest")
        != config["baseline"]["expected_sort_identity_digest"]
    ):
        raise RuntimeError("baseline curation identity is incomplete or unexpected")
    package = Path(config["candidate"]["handoff_dir"])
    expected_identity = config["candidate"]["expected_sort_identity_digest"]
    handoff = verify(package, expected_identity)

    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    scientific_config = {key: value for key, value in config.items() if key != "manager"}
    request = {
        "schema_version": SCHEMA,
        "config": scientific_config,
        "implementation": {
            "controller_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "generic_comparison_sha256": hashlib.sha256(
                Path(__file__).with_name("sort_comparison.py").read_bytes()
            ).hexdigest(),
            "handoff_verifier_sha256": hashlib.sha256(
                Path(__file__).with_name("rigid_comparison_handoff.py").read_bytes()
            ).hexdigest(),
        },
        "candidate_sort_identity_digest": expected_identity,
        "candidate_handoff_manifest_sha256": hashlib.sha256(
            (package / "MANIFEST.json").read_bytes()
        ).hexdigest(),
    }
    request["request_digest"] = fingerprint(request)
    request_path = output / "request.json"
    if request_path.exists():
        saved = json.loads(request_path.read_text())
        if saved.get("request_digest") != request["request_digest"]:
            raise RuntimeError("existing output belongs to another comparison request")
    else:
        _atomic_json(request_path, {**request, "created_at": _now()})
    _atomic_json(output / "status.json", {
        "schema_version": SCHEMA, "state": "running", "stage": "load",
        "pid": os.getpid(), "updated_at": _now(),
    })
    fs = float(config["evaluation"]["sampling_frequency_hz"])
    baseline_sort, baseline_qc = load_comparison_inputs(
        "rescue_12_9_motion_off", config["baseline"]["curated_output"],
        config["baseline"]["qc_dir"], sampling_frequency_hz=fs,
    )
    candidate_sort, candidate_qc = load_comparison_inputs(
        "external_rigid_12_9", package / "cur/cur_output", package / "qc",
        sampling_frequency_hz=fs,
    )
    _atomic_json(output / "status.json", {
        "schema_version": SCHEMA, "state": "running", "stage": "compare",
        "pid": os.getpid(), "updated_at": _now(),
    })
    spatial = {key: value for key, value in config["common_spatial_domain"].items() if key != "note"}
    report = compare_sorts(
        baseline_sort, candidate_sort, baseline_qc, candidate_qc,
        config["evaluation"], spatial, output_dir=output,
    )
    summary = {
        "schema_version": SCHEMA,
        "status": "complete",
        "development_only": True,
        "request_digest": request["request_digest"],
        "candidate_sort_identity_digest": expected_identity,
        "candidate_handoff_total_bytes": handoff["total_bytes"],
        "comparison": report["summary"],
        "coverage": report["coverage_summary"],
        "split_merge": report["split_merge_summary"],
        "decision": report["decision"],
        "common_spatial_domain": config["common_spatial_domain"],
        "interpretation": config["interpretation"],
        "completed_at": _now(),
    }
    _atomic_json(output / "comparison_summary.json", summary)
    _atomic_json(output / "status.json", {
        "schema_version": SCHEMA, "state": "complete", "stage": "complete",
        "request_digest": request["request_digest"], "updated_at": _now(),
    })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--wait-for-handoff", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args()
    try:
        if args.wait_for_handoff:
            wait_for_handoff(args.config, args.poll_seconds)
        print(json.dumps(run(args.config), indent=2), flush=True)
    except BaseException as error:
        try:
            config = json.loads(args.config.read_text())
            _atomic_json(Path(config["output_dir"]) / "status.json", {
                "schema_version": SCHEMA, "state": "failed",
                "error_type": type(error).__name__, "error": str(error), "updated_at": _now(),
            })
        finally:
            raise


if __name__ == "__main__":
    main()
