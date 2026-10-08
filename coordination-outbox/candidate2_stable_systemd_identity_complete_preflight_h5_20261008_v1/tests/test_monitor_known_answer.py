from __future__ import annotations

import importlib.util
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "source/testing/candidate2_cache_managed_monitor.py"
SPEC = importlib.util.spec_from_file_location("candidate2_monitor_under_test", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def healthy_sample() -> dict:
    return {
        "target": {
            "returncode": "0", "LoadState": "loaded", "ActiveState": "activating",
            "InvocationID": "abc", "ControlGroup": "/user.slice/example.service",
        },
        "target_cgroup": {"path": "/sys/fs/cgroup/user.slice/example.service"},
        "gpu": {"gpu_returncode": 0, "gpu_rows": ["0, GPU, 24564, 0, 24564, 0"], "compute_returncode": 0},
        "host_memory": {"MemAvailable_bytes": 200_000_000_000},
        "host_pressure": {"some": {"avg10": 0.0}},
    }


def main() -> None:
    assert MODULE.ready_failures(healthy_sample()) == []
    for path, value in (
        (("target", "returncode"), "1"),
        (("target", "LoadState"), "not-found"),
        (("target", "InvocationID"), ""),
        (("gpu", "gpu_returncode"), 1),
        (("gpu", "compute_returncode"), 1),
    ):
        sample = healthy_sample()
        sample[path[0]][path[1]] = value
        assert MODULE.ready_failures(sample), path
    cases = [
        ({"Result": "success", "ExecMainStatus": "0"}, {}, "success"),
        ({"Result": "oom-kill", "ExecMainStatus": "9"}, {}, "oomd_or_kernel_oom"),
        ({"Result": "signal", "ExecMainStatus": "15"}, {}, "external_or_signal_termination"),
        ({"Result": "exit-code", "ExecMainStatus": "1"}, {}, "managed_job_failure_or_preflight_refusal"),
        ({"Result": "failed", "ExecMainStatus": "1"}, {"oom_kill": 1}, "oomd_or_kernel_oom"),
        ({"Result": "timeout", "ExecMainStatus": "1"}, {}, "terminal_unclassified"),
    ]
    for target, events, expected in cases:
        assert MODULE.classify(target, events) == expected
    print("11 monitor readiness/classification known-answer cases passed")


if __name__ == "__main__":
    main()
