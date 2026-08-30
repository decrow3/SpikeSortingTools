"""Wait for the frozen imec0 rescue sort, then audit and score it."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from testing.luke_full_probe_rescue_diagnostics import run as run_diagnostics
from testing.luke_imec0_rescue_acceptance import evaluate


OUTPUT_ROOT = Path(
    "/mnt/NPX/Luke/20250804/"
    "rescue_pipeline_results_Luke0804_V2V1_g0_imec0"
)
DIAGNOSTICS = Path(
    "testing/outputs/luke_full_probe_rescue_diagnostics_imec0_rescue"
)
CRITERIA = Path(
    "testing/outputs/luke_full_probe_rescue_diagnostics_imec0_legacy/"
    "acceptance_criteria.json"
)
DURATION_S = 10473.5537279367


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--diagnostics", type=Path, default=DIAGNOSTICS)
    parser.add_argument("--criteria", type=Path, default=CRITERIA)
    parser.add_argument("--poll-s", type=float, default=60.0)
    parser.add_argument("--timeout-h", type=float, default=24.0)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    args = parse_args()
    manifest_path = args.output_root / "kilosort4" / "rescue_sort_manifest.json"
    sorter = args.output_root / "kilosort4" / "sorter_output"
    status_path = args.diagnostics / "postrun_status.json"
    args.diagnostics.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    status = {
        "state": "waiting_for_sort",
        "started_at": utc_now(),
        "sort_manifest": str(manifest_path),
    }
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    while True:
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("complete"):
                break
        if (time.monotonic() - started) / 3600.0 > args.timeout_h:
            status.update({"state": "timed_out", "finished_at": utc_now()})
            status_path.write_text(json.dumps(status, indent=2) + "\n")
            raise TimeoutError(f"Sort did not complete within {args.timeout_h} hours")
        time.sleep(args.poll_s)

    status.update({"state": "running_diagnostics", "sort_seen_at": utc_now()})
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    summary = run_diagnostics(
        sorter,
        args.diagnostics,
        250,
        20250804,
        probe="imec0",
        duration_override_s=DURATION_S,
    )
    decision = evaluate(args.diagnostics, args.criteria)
    decision_path = args.diagnostics / "acceptance_decision.json"
    decision_path.write_text(json.dumps(decision, indent=2) + "\n")
    status.update(
        {
            "state": "complete",
            "finished_at": utc_now(),
            "n_ks_good": summary["n_ks_good"],
            "decision": decision["decision"],
            "decision_path": str(decision_path),
        }
    )
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
