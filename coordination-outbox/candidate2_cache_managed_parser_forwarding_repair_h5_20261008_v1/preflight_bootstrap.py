"""Stdlib-only source verifier before importing the project preflight."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import os
import sys


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source(root: Path, manifest_path: Path, manifest_sha256: str) -> None:
    if sha256(manifest_path) != manifest_sha256:
        raise RuntimeError("runtime source manifest hash mismatch")
    expected = json.loads(manifest_path.read_text()).get("files")
    if not isinstance(expected, dict) or not expected:
        raise RuntimeError("runtime source manifest is invalid")
    observed = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"runtime source contains symlink: {path}")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            observed.add(relative)
            if path.suffix in {".pyc", ".pyo"} or "__pycache__" in path.parts:
                raise RuntimeError(f"runtime source contains bytecode: {relative}")
    if observed != set(expected):
        raise RuntimeError("runtime source inventory mismatch")
    for relative, wanted in expected.items():
        if sha256(root / relative) != wanted:
            raise RuntimeError(f"runtime source content mismatch: {relative}")


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--self-sha256", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha256", required=True)
    args, forwarded = parser.parse_known_args()
    if sha256(Path(__file__).resolve()) != args.self_sha256:
        raise RuntimeError("preflight bootstrap self hash mismatch")
    if sha256(args.contract) != args.contract_sha256:
        raise RuntimeError("contract hash mismatch")
    verify_source(args.source_root.resolve(), args.source_manifest.resolve(), args.source_manifest_sha256)
    os.environ.pop("PYTHONPATH", None)
    code = (
        "import sys; root=sys.argv.pop(1); sys.path.insert(0,root); "
        "sys.argv[0]='testing.candidate2_cache_managed_preflight'; "
        "from testing.candidate2_cache_managed_preflight import main; main()"
    )
    command = [
        sys.executable, "-I", "-u", "-c", code, str(args.source_root.resolve()),
        "--contract", str(args.contract), "--contract-sha256", args.contract_sha256,
        "--source-root", str(args.source_root), "--source-manifest", str(args.source_manifest),
        "--source-manifest-sha256", args.source_manifest_sha256, *forwarded,
    ]
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
