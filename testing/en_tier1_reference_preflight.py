#!/usr/bin/env python3
"""Full-scale REF/B-field runtime preflight for the frozen EN Tier-1 evaluator."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import time

from testing.en_tier1_panel import (
    arm_state_metrics,
    atomic_json,
    chance_aware_coincidence,
    common_block_multiplicities,
    load_field,
    load_sort,
    sha256,
    validate_arm_input,
    write_table,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/en_tier1_panel.v1.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    config = json.loads(args.config.read_text())
    scope, uncertainty = config["scope"], config["uncertainty"]
    fs, num_samples = float(scope["sampling_frequency_hz"]), int(scope["num_samples"])
    duration = num_samples / fs
    receipt = validate_arm_input("REF", config["arms"]["REF"])
    cells, field_receipt = load_field(
        config["fields"]["B"], duration, float(uncertainty["time_block_seconds"])
    )
    multiplicities = common_block_multiplicities(
        len(cells.block_edges_s) - 1, int(uncertainty["bootstrap_draws"]),
        int(uncertainty["seed"]),
    )
    sort = load_sort(Path(config["arms"]["REF"]["sorter_output"]), num_samples=num_samples)
    edge = config["metrics"]["edge"]
    processing = tuple(float(value) for value in edge["processing_depth_um"])
    scoring = tuple(float(value) for value in edge["scoring_depth_um"])
    state = arm_state_metrics(
        sort, cells, sampling_frequency_hz=fs, num_samples=num_samples,
        multiplicities=multiplicities,
        minimum_reference_events=int(config["state_assignment"]["minimum_reference_state_events_per_unit"]),
        processing_depth_um=processing, scoring_depth_um=scoring,
        seed=int(uncertainty["seed"]),
    )
    coincidence_config = config["metrics"]["chance_aware_coincidence"]
    coincidence = chance_aware_coincidence(
        sort, state["units"], num_samples=num_samples, sampling_frequency_hz=fs,
        block_edges_s=cells.block_edges_s, multiplicities=multiplicities,
        tolerance_ms=float(coincidence_config["tolerance_ms"]),
        depth_tolerance_um=float(coincidence_config["depth_tolerance_um"]),
        processing_depth_um=processing, seed=int(uncertainty["seed"]) + 5000,
    )
    write_table(args.output / "STATE_METRICS_B_REF.csv", state["state_metrics"])
    write_table(args.output / "UNIT_METRICS_B_REF.csv", state["units"])
    result = {
        "schema": "en-tier1-reference-preflight-v1",
        "status": "pass",
        "config_sha256": sha256(args.config),
        "reference_input": receipt,
        "field": field_receipt,
        "summary": state["summary"],
        "chance_aware_coincidence": coincidence,
        "resources": {
            "wall_seconds": time.perf_counter() - started,
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "raw_voltage_bytes_read": 0,
            "sorts_launched": 0,
            "rf_accesses": 0,
        },
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(args.output / "RESULT.json", result)
    products = [
        {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(args.output.iterdir())
        if path.name not in {"MANIFEST.json", "COMPLETE.json"}
    ]
    atomic_json(args.output / "MANIFEST.json", {"products": products})
    atomic_json(
        args.output / "COMPLETE.json",
        {"status": "complete", "manifest_sha256": sha256(args.output / "MANIFEST.json"),
         "written_last_utc": datetime.now(timezone.utc).isoformat()},
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
