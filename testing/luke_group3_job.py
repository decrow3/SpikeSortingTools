"""Run the authorized Group 3 threshold-interaction arms under systemd."""

import json
import shutil
import sys
from pathlib import Path

from pipeline.preprocess import validate_accepted_recording
from pipeline.runtime import validate_production_environment
from testing.development_ladder import load_contract
from testing.development_runner import _atomic_json
from testing.development_strip import validate_development_selection
from testing.ladder_sorter import NAMED_CONFIGS
from testing.managed_job import run_managed_command


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "testing/configs/luke0804_group3_huklaban1_v1.json"
REFERENCE = "rescue_10_9_motion_off"
EXPECTED = (
    "rescue_10_8_motion_off",
    "rescue_10_10_motion_off",
    "rescue_12_10_motion_off",
)
EXPECTED_THRESHOLDS = ((10, 8), (10, 10), (12, 10))


def commands(plan):
    contract = load_contract(ROOT / plan["contract_path"])
    if contract.digest != plan["contract_digest"]:
        raise RuntimeError("Group 3 contract digest changed")
    if plan["reference_arm"] != REFERENCE or tuple(plan["arms"]) != EXPECTED:
        raise RuntimeError("Group 3 membership changed")
    by_name = {candidate["name"]: candidate for candidate in contract.candidates}
    for name, thresholds in zip(EXPECTED, EXPECTED_THRESHOLDS):
        params = NAMED_CONFIGS[by_name[name]["sorter_config"]].params()
        observed = (params["Th_universal"], params["Th_learned"])
        if params["do_correction"] or observed != thresholds:
            raise RuntimeError("Group 3 must vary only thresholds with motion disabled")
    reference = by_name[REFERENCE]
    if not reference.get("existing_sort_dir") or not reference.get("existing_downstream_root"):
        raise RuntimeError("Group 3 reference must reuse the completed Group 2 10/9 arm")

    output_root = Path(plan["output_root"])
    recording_dir = Path(plan["reused_long_recording"])
    prefix = [sys.executable, "-u", "-m", "testing.run_development_ladder"]
    config = ["--config", str(ROOT / plan["contract_path"])]
    new_arms = [value for name in EXPECTED for value in ("--arm", name)]
    long_names = (REFERENCE, *EXPECTED)
    long_arms = [value for name in long_names for value in ("--arm", name)]
    smoke = [
        "--smoke-start-s", str(plan["smoke_start_s"]),
        "--smoke-duration-s", str(plan["smoke_duration_s"]),
    ]
    return [
        (
            "prepare_smoke",
            prefix + ["prepare-smoke"] + config + new_arms + smoke
            + ["--output-root", str(output_root / "smoke/recording")],
        ),
        (
            "run_smoke",
            prefix + ["run-smoke"] + config + new_arms + smoke
            + [
                "--recording-dir", str(output_root / "smoke/recording"),
                "--output-root", str(output_root / "smoke/arms"),
                "--group-id", "group_3_smoke",
            ],
        ),
        (
            "run_long",
            prefix + ["run-arms"] + config + long_arms
            + [
                "--recording-dir", str(recording_dir),
                "--output-root", str(output_root / "long/arms"),
                "--group-id", plan["group_id"],
            ],
        ),
        (
            "finalize_long",
            prefix + ["finalize-arms"] + config + long_arms
            + [
                "--recording-dir", str(recording_dir),
                "--output-root", str(output_root / "long/arms"),
            ],
        ),
    ]


def execute(stages, output_root):
    for stage, command in stages:
        _atomic_json(output_root / "status.json", {"state": "running", "stage": stage})
        code = run_managed_command(
            command,
            receipt_path=output_root / "jobs" / f"{stage}.json",
            cwd=ROOT,
        )
        if code:
            _atomic_json(
                output_root / "status.json",
                {"state": "failed", "stage": stage, "returncode": code},
            )
            return code
    _atomic_json(
        output_root / "status.json",
        {
            "state": "complete",
            "comparison": "Compare the two completed interaction cells; upper cell awaits Group 1 12/9",
        },
    )
    return 0


def main():
    plan = json.loads(PLAN.read_text())
    contract = load_contract(ROOT / plan["contract_path"])
    stages = commands(plan)
    output_root = Path(plan["output_root"])
    recording_dir = Path(plan["reused_long_recording"])

    data_mount = Path("/media/huklab/Data")
    if not data_mount.is_mount():
        raise RuntimeError("selected local data disk is not mounted")
    if shutil.disk_usage(data_mount).free < plan["minimum_free_gib"] * 1024**3:
        raise RuntimeError("insufficient local output capacity")

    accepted = validate_accepted_recording(recording_dir)
    validate_development_selection(
        recording_dir,
        accepted,
        contract.raw["recording"],
        contract.raw["spatial_contract"],
    )
    for label, code in (("success", 0), ("failure", 7)):
        evidence = json.loads(
            (ROOT / f"testing/outputs/group2_preparation/{label}.json").read_text()
        )
        expected_state = "complete" if code == 0 else "failed"
        if evidence.get("returncode") != code or evidence.get("state") != expected_state:
            raise RuntimeError("the proven systemd launch method lacks valid dummy evidence")

    environment = validate_production_environment(require_cuda=True)
    output_root.mkdir(parents=True, exist_ok=True)
    if (output_root / "launch.json").exists():
        raise RuntimeError("launch already exists; inspect the prior attempt instead of retrying")
    _atomic_json(
        output_root / "launch.json",
        {
            "plan": plan,
            "stages": stages,
            "environment": environment,
            "validated_reused_recording_request_digest": accepted["request_digest"],
            "manager": "systemd user service luke-group3-threshold-interactions-v1.service",
        },
    )
    return execute(stages, output_root)


if __name__ == "__main__":
    sys.exit(main())
