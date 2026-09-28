#!/usr/bin/env python
"""Persist the W GPU probe/optional trial and five-block throughput report."""
from __future__ import annotations

import argparse, csv, json, os, time
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def wait_for(predicate, description, interval=2):
    while True:
        value = predicate()
        if value:
            return value
        time.sleep(interval)


def completed_audits(output):
    result = []
    for receipt in output.glob("peak_cache/full_block_*/extraction/complete.json"):
        audit = receipt.with_name("audit.json")
        if audit.exists():
            result.append((receipt, read(audit)))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    output = args.output.resolve()
    schedule = read(output / "w_schedule.json")

    logs = wait_for(lambda: sorted((output / "w_gpu_logs").glob("full_block_*.csv"))[:2]
                    if len(list((output / "w_gpu_logs").glob("full_block_*.csv"))) >= 2 else None,
                    "two GPU probe logs")
    samples = []
    per_log = []
    for path in logs:
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        util = [float(x["utilization_percent"]) for x in rows]
        memory = [float(x["memory_mib"]) for x in rows]
        samples.extend((u, m) for u, m in zip(util, memory))
        per_log.append({"block": path.stem, "samples": len(rows),
                        "mean_utilization_percent": sum(util) / len(util),
                        "peak_memory_mib": max(memory)})
    mean_util = sum(x[0] for x in samples) / len(samples)
    peak_memory = max(x[1] for x in samples)
    trial = mean_util < 60 and peak_memory < 10240
    probe = {"status": "dual_trial" if trial else "single_gpu_kept",
             "logs": per_log, "samples": len(samples),
             "mean_utilization_percent": mean_util, "peak_memory_mib": peak_memory,
             "criteria": {"mean_utilization_below_percent": 60, "peak_memory_below_mib": 10240},
             "decided_at": time.time()}
    write(output / "w_gpu_probe_decision.json", probe)

    if trial:
        counter = output / "w_gpu_trial_counter.txt"
        if counter.exists(): counter.unlink()
        write(output / "w_gpu_mode.json", {"mode": "dual_trial", "reason": "W.2 probe passed", "updated_at": time.time()})
        trial_rows = wait_for(lambda: [(r, a) for r, a in completed_audits(output)
                                       if a.get("gpu_schedule", {}).get("mode") == "dual_trial"]
                              if len([(r, a) for r, a in completed_audits(output)
                                      if a.get("gpu_schedule", {}).get("mode") == "dual_trial"]) >= 2 else None,
                              "two validated dual-trial blocks")[:2]
        probe_rows = wait_for(lambda: [(r, a) for r, a in completed_audits(output)
                                       if a.get("gpu_probe")]
                              if len([(r, a) for r, a in completed_audits(output) if a.get("gpu_probe")]) >= 2 else None,
                              "two validated single-GPU probe blocks")[:2]
        def rate(rows):
            start = min(a["gpu_schedule"]["start_epoch"] for _, a in rows)
            stop = max(a["gpu_schedule"]["stop_epoch"] for _, a in rows)
            return len(rows) * 3600 / (stop - start)
        baseline_rate, trial_rate = rate(probe_rows), rate(trial_rows)
        keep = trial_rate >= 1.2 * baseline_rate
        write(output / "w_gpu_mode.json", {"mode": "dual_keep" if keep else "single_locked",
            "reason": "W.2 two-block trial improved throughput >=20%" if keep else "W.2 two-block trial improved throughput <20%",
            "baseline_blocks_per_hour": baseline_rate, "trial_blocks_per_hour": trial_rate,
            "validated_trial_blocks": [r.parts[-3] for r, _ in trial_rows], "updated_at": time.time()})
        probe["trial"] = {"baseline_blocks_per_hour": baseline_rate, "trial_blocks_per_hour": trial_rate,
                          "improvement_fraction": trial_rate / baseline_rate - 1, "kept": keep,
                          "validated_blocks": [r.parts[-3] for r, _ in trial_rows]}
        write(output / "w_gpu_probe_decision.json", probe)
    else:
        write(output / "w_gpu_mode.json", {"mode": "single_locked", "reason": "W.2 utilization/memory condition not met", "updated_at": time.time()})

    ids = schedule["five_pipeline_blocks"]
    targets = [output / f"peak_cache/full_block_{i:03d}/extraction" for i in ids]
    wait_for(lambda: all((p / "complete.json").exists() for p in targets), "five W pipeline blocks")
    audits = [read(p / "audit.json") for p in targets]
    starts = [(p / "complete.json").stat().st_mtime - a["seconds"] for p, a in zip(targets, audits)]
    stops = [(p / "complete.json").stat().st_mtime for p in targets]
    blocks_per_hour = len(targets) * 3600 / (max(stops) - min(starts))
    completed = len(list(output.glob("peak_cache/full_block_*/extraction/complete.json")))
    remaining = 87 - completed
    pilot_fit = [read(p)["runtime_s"] for p in output.glob("fields/pilot/*/receipt.json")]
    median_fit = sorted(pilot_fit)[len(pilot_fit)//2]
    projected = time.time() + remaining / blocks_per_hour * 3600 + 174 * median_fit
    report = {"status": "complete", "pipeline_blocks": [f"full_block_{i:03d}" for i in ids],
              "blocks_per_hour": blocks_per_hour, "completed_blocks_at_report": completed,
              "mean_phase_seconds": {key: sum(a["phase_seconds"].get(key, 0) for a in audits) / len(audits)
                                     for key in ("raw_read", "preprocess", "gpu_lock_wait", "detection_and_denoiser_fit")},
              "gpu_probe": read(output / "w_gpu_probe_decision.json"),
              "gpu_mode": read(output / "w_gpu_mode.json"),
              "projected_finish_unix": projected,
              "projected_finish_local": time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(projected)),
              "gpu_hours_used": sum(read(p)["runtime_s"] for p in output.glob("fields/*/*/receipt.json")) / 3600,
              "scientific_parameters_changed": False, "completed_blocks_rerun": False,
              "reported_at": time.time()}
    write(output / "w_first_5_pipeline_report.json", report)


if __name__ == "__main__":
    main()
