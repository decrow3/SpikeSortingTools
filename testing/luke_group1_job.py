"""Run the authorized Group 1 long-strip stages under a persistent job manager."""

import json
import shutil
import sys
from pathlib import Path

from pipeline.runtime import validate_production_environment
from testing.development_ladder import load_contract
from testing.development_runner import _atomic_json
from testing.development_smoke import build_smoke_contract
from testing.ladder_sorter import NAMED_CONFIGS
from testing.managed_job import run_managed_command


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "testing/configs/luke0804_group1_huklaban5_v1.json"
EXPECTED = (
    "rescue_12_9_motion_off",
    "rescue_12_9_native_rigid",
    "rescue_12_9_native_nonrigid",
)


def commands(plan):
    contract = load_contract(ROOT / plan["contract_path"])
    if contract.digest != plan["contract_digest"] or tuple(plan["arms"]) != EXPECTED:
        raise RuntimeError("Group 1 membership or contract changed")
    by_name = {candidate["name"]: candidate for candidate in contract.candidates}
    expected_motion = [(False, 1), (True, 1), (True, 6)]
    for name, (do_correction, nblocks) in zip(EXPECTED, expected_motion):
        params = NAMED_CONFIGS[by_name[name]["sorter_config"]].params()
        if ((params["do_correction"], params["nblocks"]) != (do_correction, nblocks)
                or (params["Th_universal"], params["Th_learned"]) != (12, 9)):
            raise RuntimeError("Group 1 must vary only native motion at frozen 12/9 thresholds")
    root = Path(plan["output_root"])
    prefix = [sys.executable, "-u", "-m", "testing.run_development_ladder"]
    config = ["--config", str(ROOT / plan["contract_path"])]
    arms = [value for name in EXPECTED for value in ("--arm", name)]
    return [
        ("prepare_long", prefix + ["prepare-strip"] + config
         + ["--output-root", str(root / "long/recording"), "--n-jobs", "4"]),
        ("run_long", prefix + ["run-arms"] + config + arms
         + ["--recording-dir", str(root / "long/recording"),
            "--output-root", str(root / "long/arms"), "--group-id", plan["group_id"]]),
        ("finalize_long", prefix + ["finalize-arms"] + config + arms
         + ["--recording-dir", str(root / "long/recording"),
            "--output-root", str(root / "long/arms")]),
    ]


def validate_completed_smoke(plan):
    parent = load_contract(ROOT / plan["contract_path"])
    smoke, _ = build_smoke_contract(
        parent,
        candidate_names=EXPECTED,
        start_s=plan["smoke_start_s"],
        duration_s=plan["smoke_duration_s"],
    )
    job = json.loads(Path(plan["smoke_job_receipt"]).read_text())
    if job.get("state") != "complete" or job.get("returncode") != 0:
        raise RuntimeError("Group 1 engineering smoke job did not complete successfully")
    smoke_root = Path(plan["smoke_root"])
    group = json.loads((smoke_root / "group_receipts/group-1-engineering-smoke.json").read_text())
    if (group.get("complete") is not True
            or group.get("contract_digest") != smoke.digest
            or tuple(group.get("candidate_names", ())) != EXPECTED):
        raise RuntimeError("Group 1 engineering smoke receipt is incompatible")
    for name in EXPECTED:
        manifest = json.loads((smoke_root / name / "candidate_manifest.json").read_text())
        if (manifest.get("complete") is not True
                or manifest.get("contract_digest") != smoke.digest
                or manifest.get("candidate", {}).get("name") != name):
            raise RuntimeError(f"Group 1 smoke arm is incomplete or incompatible: {name}")
    return {"job": job, "group": group, "smoke_contract_digest": smoke.digest}


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
        {"state": "complete", "comparison": "Ready for Group 1 comparisons and cross-host Group 2 import"},
    )
    return 0


def main():
    plan = json.loads(PLAN.read_text())
    stages = commands(plan)
    output_root = Path(plan["output_root"])
    mount = Path("/media/huklaban5/Data")
    if not mount.is_mount():
        raise RuntimeError("selected local data disk is not mounted")
    if shutil.disk_usage(mount).free < plan["minimum_free_gib"] * 1024**3:
        raise RuntimeError("insufficient local output capacity")
    manager = json.loads(Path(plan["manager_test_receipt"]).read_text())
    if manager.get("state") != "complete" or manager.get("returncode") != 0:
        raise RuntimeError("independent-manager survival test has not passed")
    smoke = validate_completed_smoke(plan)
    environment = validate_production_environment(require_cuda=True)
    output_root.mkdir(parents=True, exist_ok=True)
    if (output_root / "launch.json").exists():
        raise RuntimeError("launch already exists; inspect prior attempt rather than retry automatically")
    _atomic_json(
        output_root / "launch.json",
        {"plan": plan, "stages": stages, "environment": environment, "validated_smoke": smoke},
    )
    return execute(stages, output_root)


if __name__ == "__main__":
    sys.exit(main())
