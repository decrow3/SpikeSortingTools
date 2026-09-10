"""Wait for the active rigid-sort service, then fit MEDiCINe for 30k steps."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from testing.luke_full_session_medicine import save, seal, sealed, sha
from testing.luke_medicine_rigid_queue import dependency_decision, read_optional, service_state
from testing.managed_job import run_managed_command


def verify(cfg):
    for path, digest in cfg["pinned_inputs"].items():
        if sha(path) != digest:
            raise RuntimeError(f"Pinned input changed: {path}")
    sealed(Path(cfg["input"]))
    if cfg["medicine_settings"]["training_steps"] != 30000:
        raise ValueError("This queue is specifically the authorized 30k-step diagnostic")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--dummy", action="store_true")
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    out = Path(cfg["output"])
    out.mkdir(parents=True, exist_ok=False)
    save(out / "queue_request.json", cfg)

    def stage(name, **extra):
        value = dict(stage=name, updated_unix=time.time(), pid=os.getpid(), **extra)
        save(out / "status.json", value)
        print(json.dumps(value), flush=True)

    if args.dummy:
        stage("dummy_waiting")
        time.sleep(20)
        stage("dummy_complete")
        save(out / "summary.json", {"status": "dummy_complete"})
        return

    verify(cfg)
    dep_job = Path(cfg["dependency_job"])
    stage("queued_waiting_for_rigid_sort")
    while True:
        state = service_state(cfg["dependency_unit"])
        decision = dependency_decision(
            state, read_optional(dep_job / "receipt.json"),
            read_optional(dep_job / "service_result.json"))
        save(out / "dependency_state.json", dict(checked_unix=time.time(), decision=decision, systemd=state))
        if decision == "ready":
            break
        time.sleep(30)

    verify(cfg)
    stage("medicine_full_session_30000_steps")
    fit_dir = out / "fit"
    fit_dir.mkdir()
    command = [cfg["medicine_python"], "-u", "-m", "testing.luke_full_session_medicine",
               "--config", str(args.config), "--phase", "fit", "--input", cfg["input"],
               "--output", str(fit_dir)]
    save(fit_dir / "command.json", command)
    code = run_managed_command(command, receipt_path=fit_dir / "receipt.json",
                               cwd=Path(__file__).resolve().parents[1])
    if code:
        raise RuntimeError(f"30k MEDiCINe fit failed with exit {code}; preserve evidence")
    seal(fit_dir)
    verify(cfg)

    new = np.load(fit_dir / "field.npz")
    old = np.load(cfg["baseline_field"])
    lo, hi = max(new["time_s"][0], old["time_s"][0]), min(new["time_s"][-1], old["time_s"][-1])
    sample = np.linspace(lo, hi, 10000)
    differences = []
    for depth in np.linspace(max(new["depth_um"][0], old["depth_um"][0]),
                             min(new["depth_um"][-1], old["depth_um"][-1]), 16):
        def curve(z):
            values = np.array([np.interp(depth, z["depth_um"], row) for row in z["displacement_um"]])
            values -= np.median(values[(z["time_s"] >= 930) & (z["time_s"] < 940)])
            return np.interp(sample, z["time_s"], values)
        differences.append(curve(new) - curve(old))
    differences = np.concatenate(differences)
    save(out / "comparison_to_10k.json", dict(
        common_support_s=[float(lo), float(hi)], sampled_depths=16, sampled_times=10000,
        median_abs_difference_um=float(np.median(abs(differences))),
        p95_abs_difference_um=float(np.quantile(abs(differences), .95)),
        max_abs_difference_um=float(abs(differences).max()),
        interpretation="Numerical fit sensitivity only; not motion accuracy or lighthouse validation"))
    save(out / "summary.json", dict(status="complete", training_steps=30000,
        input_reused=True, extraction_rerun=False, scientific_status="requires_lighthouse_review",
        fit_audit=json.loads((fit_dir / "fit_audit.json").read_text())))
    stage("complete")


if __name__ == "__main__":
    main()
