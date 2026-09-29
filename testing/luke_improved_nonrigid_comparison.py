"""Compare completed full-session external-nonrigid and motion-off sorts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

from pipeline.config import fingerprint
from testing.sort_comparison import compare_sorts, load_comparison_inputs


SCHEMA = "luke-improved-nonrigid-comparison-v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(partial, path)


def _validated_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = json.loads(path.read_text())
    if config.get("schema_version") != SCHEMA:
        raise ValueError("unsupported comparison config schema")
    baseline_receipt = json.loads(Path(config["baseline"]["curation_receipt"]).read_text())
    if (
        baseline_receipt.get("complete") is not True
        or baseline_receipt.get("sort_identity_digest")
        != config["baseline"]["expected_sort_identity_digest"]
    ):
        raise RuntimeError("baseline curation identity is incomplete or unexpected")
    candidate_summary_path = Path(config["candidate"]["downstream_summary"])
    if not candidate_summary_path.is_file():
        raise RuntimeError("candidate downstream processing is not complete")
    candidate_summary = json.loads(candidate_summary_path.read_text())
    if candidate_summary.get("status") != "complete":
        raise RuntimeError("candidate downstream processing is not complete")
    candidate_request = json.loads(
        (Path(config["candidate"]["curated_output"]).parents[1] / "sort_identity.json").read_text()
    )
    if candidate_request.get("request_digest") != config["candidate"]["expected_sort_request_digest"]:
        raise RuntimeError("candidate sort request identity is unexpected")
    if candidate_summary.get("sort_identity_digest") != candidate_request.get("identity_digest"):
        raise RuntimeError("candidate downstream summary and pinned sort identity differ")
    return config, candidate_summary


def run(config_path: Path) -> dict[str, Any]:
    config, candidate_downstream = _validated_config(config_path)
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    request = {
        "schema_version": SCHEMA,
        "config": config,
        "candidate_sort_identity_digest": candidate_downstream["sort_identity_digest"],
    }
    request["request_digest"] = fingerprint(request)
    request_path = output / "request.json"
    if request_path.exists():
        saved = json.loads(request_path.read_text())
        if saved.get("request_digest") != request["request_digest"]:
            raise RuntimeError("existing comparison output belongs to another request")
    else:
        _atomic_json(request_path, {**request, "created_at": _now()})
    _atomic_json(
        output / "status.json",
        {"schema_version": SCHEMA, "state": "running", "stage": "load", "pid": os.getpid(), "updated_at": _now()},
    )
    fs = float(config["evaluation"]["sampling_frequency_hz"])
    baseline_sort, baseline_qc = load_comparison_inputs(
        "rescue_12_9_motion_off",
        config["baseline"]["curated_output"],
        config["baseline"]["qc_dir"],
        sampling_frequency_hz=fs,
    )
    candidate_sort, candidate_qc = load_comparison_inputs(
        "external_nonrigid_12_9",
        config["candidate"]["curated_output"],
        config["candidate"]["qc_dir"],
        sampling_frequency_hz=fs,
    )
    _atomic_json(
        output / "status.json",
        {"schema_version": SCHEMA, "state": "running", "stage": "compare", "pid": os.getpid(), "updated_at": _now()},
    )
    spatial = {key: value for key, value in config["common_spatial_domain"].items() if key != "note"}
    report = compare_sorts(
        baseline_sort,
        candidate_sort,
        baseline_qc,
        candidate_qc,
        config["evaluation"],
        spatial,
        output_dir=output,
    )
    summary = {
        "schema_version": SCHEMA,
        "status": "complete",
        "development_only": True,
        "request_digest": request["request_digest"],
        "candidate_sort_identity_digest": candidate_downstream["sort_identity_digest"],
        "comparison": report["summary"],
        "coverage": report["coverage_summary"],
        "split_merge": report["split_merge_summary"],
        "decision": report["decision"],
        "common_spatial_domain": config["common_spatial_domain"],
        "interpretation": config["interpretation"],
        "completed_at": _now(),
    }
    _atomic_json(output / "comparison_summary.json", summary)
    _atomic_json(
        output / "status.json",
        {"schema_version": SCHEMA, "state": "complete", "stage": "complete", "request_digest": request["request_digest"], "updated_at": _now()},
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = run(args.config)
        print(json.dumps(result, indent=2), flush=True)
    except BaseException as error:
        try:
            config = json.loads(args.config.read_text())
            _atomic_json(
                Path(config["output_dir"]) / "status.json",
                {"schema_version": SCHEMA, "state": "failed", "error_type": type(error).__name__, "error": str(error), "updated_at": _now()},
            )
        finally:
            raise


if __name__ == "__main__":
    main()
