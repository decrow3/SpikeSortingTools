from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


EXPECTED_INTERPRETER = Path(
    "/home/huklaban5/Documents/SpikeSortingTools/SpikeSortingTools/"
    "environments/rescue-production/.venv/bin/python"
)
DEFAULT_SOURCE = Path(__file__).resolve().parents[1] / "source/testing/candidate2_cache_managed_preflight.py"
MINIMUM_FREE = 21_474_836_480
ENOUGH_FREE = 25_091_112_960


class Query:
    def __init__(self, returncode: int, stdout: str, stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def load_source(path: Path):
    spec = importlib.util.spec_from_file_location("candidate2_preflight_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def fixture(source: Path) -> None:
    sys.dont_write_bytecode = True
    module = load_source(source)
    self_pid = 424242

    def evaluate(rows: str, *, free: int = ENOUGH_FREE, returncode: int = 0, stderr: str = ""):
        gpu = module.gpu_metrics(
            mem_get_info=lambda: (free, 25_422_594_048),
            compute_query=lambda: Query(returncode, rows, stderr),
            current_pid=self_pid,
            device_name=lambda: "synthetic RTX A5000",
        )
        return gpu, module.gpu_gate_failures(gpu, MINIMUM_FREE)

    self_only, failures = evaluate(f"{self_pid}, python, 202\n")
    assert failures == []
    assert self_only["compute_apps_raw"] == [f"{self_pid}, python, 202"]
    assert [row["pid"] for row in self_only["self_compute_apps"]] == [self_pid]
    assert self_only["external_compute_apps"] == []

    self_external, failures = evaluate(
        f"{self_pid}, python, 202\n515151, external-sorter, 4096\n"
    )
    assert failures == ["external_gpu_compute_apps"]
    assert [row["pid"] for row in self_external["external_compute_apps"]] == [515151]

    _, failures = evaluate(f"{self_pid}, python, 202\n", free=MINIMUM_FREE - 1)
    assert failures == ["gpu_free_memory"]

    try:
        evaluate("", returncode=7, stderr="query failed")
    except RuntimeError as exc:
        assert "compute-app query failed" in str(exc)
    else:
        raise AssertionError("query error did not fail closed")

    malformed_rows = (
        "not-a-pid, python, 202\n",
        "123, [Not Found], 202\n",
        "123, python\n",
        "123, python, 202\n123, python, 202\n",
    )
    for rows in malformed_rows:
        try:
            evaluate(rows)
        except RuntimeError as exc:
            assert "malformed" in str(exc) or "unknown or duplicate" in str(exc)
        else:
            raise AssertionError(f"malformed rows did not fail closed: {rows!r}")

    print(json.dumps({
        "pass": True,
        "interpreter": sys.executable,
        "source": str(source),
        "cases": ["self_only", "self_plus_external", "low_memory", "query_error", "malformed_data"],
    }, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--fixture-child", action="store_true")
    parser.add_argument("--live-child", action="store_true")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    if args.fixture_child:
        assert Path(sys.executable).resolve() == EXPECTED_INTERPRETER.resolve()
        fixture(args.source.resolve())
        return
    if args.live_child:
        assert Path(sys.executable).resolve() == EXPECTED_INTERPRETER.resolve()
        sys.dont_write_bytecode = True
        module = load_source(args.source.resolve())
        gpu = module.gpu_metrics()
        failures = module.gpu_gate_failures(gpu, MINIMUM_FREE)
        print(json.dumps({
            "pass": failures == [],
            "interpreter": sys.executable,
            "source": str(args.source.resolve()),
            "gpu": gpu,
            "failures": failures,
        }, sort_keys=True))
        return
    child_mode = "--live-child" if args.live else "--fixture-child"
    completed = subprocess.run(
        [
            str(EXPECTED_INTERPRETER), "-I", "-u", str(Path(__file__).resolve()),
            child_mode, "--source", str(args.source.resolve()),
        ],
        text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["pass"] is True
    if args.live:
        assert result["gpu"]["external_compute_apps"] == []
        assert result["gpu"]["preflight_pid"] > 0
        assert result["gpu"]["compute_query_returncode"] == 0
        print("actual-interpreter live GPU metric excluded only its exact self PID and passed")
        return
    assert result["cases"] == [
        "self_only", "self_plus_external", "low_memory", "query_error", "malformed_data"
    ]
    print("actual-interpreter GPU self-observation fixture passed 5 case groups")


if __name__ == "__main__":
    main()
