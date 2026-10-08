from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--extra-kilosort-root", type=Path, required=True)
    parser.add_argument("--dependency-site", type=Path)
    parser.add_argument("--mode", choices=("append", "prepend"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    root = str(args.extra_kilosort_root.resolve())
    if args.mode == "append":
        sys.path.append(root)
    else:
        sys.path.insert(0, root)
    if args.dependency_site is not None:
        sys.path.append(str(args.dependency_site.resolve()))

    import dartsort
    import kilosort
    from dartsort.util import main_util, preprocess_util
    from kilosort.io import BinaryFiltered

    paths = {
        "dartsort_package": Path(dartsort.__file__).resolve(),
        "dartsort_main": Path(dartsort.__file__).resolve().parent / "main.py",
        "dartsort_main_util": Path(sys.modules[main_util.__name__].__file__).resolve(),
        "dartsort_preprocess_util": Path(sys.modules[preprocess_util.__name__].__file__).resolve(),
        "kilosort_package": Path(kilosort.__file__).resolve(),
        "kilosort_io": Path(sys.modules[BinaryFiltered.__module__].__file__).resolve(),
    }
    result = {
        "schema": "candidate3-import-identity-probe-v1",
        "interpreter": str(Path(sys.executable).resolve()),
        "mode": args.mode,
        "extra_kilosort_root": root,
        "dependency_site": str(args.dependency_site.resolve()) if args.dependency_site else None,
        "kilosort_version": getattr(kilosort, "__version__", None),
        "paths": {name: str(path) for name, path in paths.items()},
        "sha256": {name: sha256(path) for name, path in paths.items()},
        "bytecode_disabled": bool(sys.dont_write_bytecode),
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
