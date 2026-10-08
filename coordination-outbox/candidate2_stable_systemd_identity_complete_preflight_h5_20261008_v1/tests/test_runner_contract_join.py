from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess


EXPECTED_INTERPRETER = Path(
    "/home/huklaban5/Documents/SpikeSortingTools/SpikeSortingTools/"
    "environments/rescue-production/.venv/bin/python"
)
DEFAULT_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    source = root / "source"
    contract = root / "CONTRACT.json"
    manifest = root / "SOURCE_MANIFEST.json"
    missing_preflight = root / "tests/DELIBERATELY_ABSENT_PREFLIGHT.json"
    assert not missing_preflight.exists()
    code = (
        "import sys; sys.dont_write_bytecode=True; root=sys.argv.pop(1); sys.path.insert(0,root); "
        "sys.argv[0]=\'testing.candidate2_cache_managed_full\'; "
        "from testing.candidate2_cache_managed_full import main; main()"
    )
    command = [
        str(EXPECTED_INTERPRETER), "-I", "-u", "-c", code, str(source),
        "--contract", str(contract), "--contract-sha256", sha256(contract),
        "--source-manifest-sha256", sha256(manifest),
        "--preflight", str(missing_preflight),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    assert completed.returncode != 0
    assert "wrong reviewed parent contract" not in completed.stderr
    assert "FileNotFoundError" in completed.stderr, completed.stderr
    assert str(missing_preflight) in completed.stderr
    print("actual isolated runner accepted the frozen contract join and stopped at absent preflight")


if __name__ == "__main__":
    main()
