"""Run the authorized Group 1 long-strip comparisons under a persistent manager."""

import json
import sys
from pathlib import Path

from pipeline.runtime import validate_production_environment
from testing.development_ladder import load_contract
from testing.development_runner import _atomic_json
from testing.luke_group1_job import EXPECTED, PLAN, ROOT
from testing.managed_job import run_managed_command


BASELINE = EXPECTED[0]
CANDIDATES = EXPECTED[1:]


def commands(plan):
    contract = load_contract(ROOT / plan["contract_path"])
    if contract.digest != plan["contract_digest"] or tuple(plan["arms"]) != EXPECTED:
        raise RuntimeError("Group 1 membership or contract changed")
    root = Path(plan["output_root"])
    arms_root = root / "long/arms"
    prefix = [sys.executable, "-u", "-m", "testing.run_development_ladder", "compare-arms"]
    shared = [
        "--config", str(ROOT / plan["contract_path"]),
        "--recording-dir", str(root / "long/recording"),
        "--output-root", str(arms_root),
        "--baseline", BASELINE,
    ]
    return [
        (
            f"{candidate}_vs_{BASELINE}",
            prefix + shared + [
                "--candidate", candidate,
                "--output", str(root / "benchmarks/reports" / f"{candidate}_vs_{BASELINE}.json"),
            ],
        )
        for candidate in CANDIDATES
    ]


def validate_completed_group1(plan):
    contract = load_contract(ROOT / plan["contract_path"])
    root = Path(plan["output_root"])
    status = json.loads((root / "status.json").read_text())
    if status.get("state") != "complete":
        raise RuntimeError("Group 1 long-run controller is not complete")
    receipt = json.loads((root / "long/arms/group_receipts/group_1_motion_axis.json").read_text())
    if (
        receipt.get("complete") is not True
        or receipt.get("contract_digest") != contract.digest
        or tuple(receipt.get("candidate_names", ())) != EXPECTED
    ):
        raise RuntimeError("Group 1 long-run receipt is incomplete or incompatible")
    identities = {}
    for name in EXPECTED:
        manifest = json.loads((root / "long/arms" / name / "candidate_manifest.json").read_text())
        if (
            manifest.get("complete") is not True
            or manifest.get("contract_digest") != contract.digest
            or manifest.get("candidate", {}).get("name") != name
        ):
            raise RuntimeError(f"Group 1 arm is incomplete or incompatible: {name}")
        identities[name] = manifest["sort_identity_digest"]
    return {"group_receipt": receipt, "sort_identity_digests": identities}


def execute(stages, root):
    for stage, command in stages:
        _atomic_json(root / "status.json", {"state": "running", "stage": stage})
        code = run_managed_command(
            command, receipt_path=root / "jobs" / f"{stage}.json", cwd=ROOT
        )
        if code:
            _atomic_json(
                root / "status.json", {"state": "failed", "stage": stage, "returncode": code}
            )
            return code
    _atomic_json(
        root / "status.json",
        {
            "state": "complete",
            "comparisons": [f"{candidate}_vs_{BASELINE}" for candidate in CANDIDATES],
            "interpretation": "Ready for coverage and Pareto review; no composite rank was computed.",
        },
    )
    return 0


def main():
    plan = json.loads(PLAN.read_text())
    validated = validate_completed_group1(plan)
    environment = validate_production_environment(require_cuda=False)
    stages = commands(plan)
    root = Path(plan["output_root"]) / "benchmarks"
    root.mkdir(parents=True, exist_ok=True)
    if (root / "launch.json").exists():
        raise RuntimeError("benchmark launch already exists; inspect the prior attempt")
    _atomic_json(
        root / "launch.json",
        {"plan": plan, "stages": stages, "environment": environment, "validated_group1": validated},
    )
    return execute(stages, root)


if __name__ == "__main__":
    sys.exit(main())
