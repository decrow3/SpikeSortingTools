"""Verify the exact rescue-production runtime before processing data."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from pipeline.runtime import production_environment_receipt, validate_production_environment
from pipeline.kilosort_compat import ensure_kilosort_compatibility


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    validate_production_environment(require_cuda=args.require_cuda)
    compatibility_patch = ensure_kilosort_compatibility()
    from neo.rawio.spikeglxrawio import SpikeGLXRawIO

    if "load_sync_channel" not in inspect.signature(SpikeGLXRawIO.__init__).parameters:
        raise RuntimeError(
            "Neo SpikeGLXRawIO lacks load_sync_channel; the production reader "
            "contract is incompatible"
        )
    receipt = production_environment_receipt(check_cuda=True)
    receipt["compatibility_patches"] = [compatibility_patch]
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
