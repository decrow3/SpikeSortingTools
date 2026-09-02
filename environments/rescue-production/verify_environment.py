"""Verify the rescue-production runtime, and optionally the host, before processing data.

Two independent layers are checked:

*Runtime* — interpreter, pinned package versions, lockfile identity, CUDA, and
the Neo reader contract. This is the portable software contract and is always
checked.

*Host* — server mount, source-stream discovery, output writability, and NVMe
scratch space. These differ per machine and are the usual cause of a first run
failing on a newly set up host. They are checked only for the paths you pass.

Examples
--------
Runtime only::

    python environments/rescue-production/verify_environment.py --require-cuda

Runtime plus the host preconditions for a specific recording::

    python environments/rescue-production/verify_environment.py \
        --require-cuda \
        --data-dir /mnt/NPX/Luke/20250804/Luke0804_V2V1_g0 \
        --stream-id imec0.ap \
        --output-dir /mnt/NPX/Luke/20250804/rescue_results_imec0 \
        --local-work-dir /local/nvme/Luke0804_imec0_rescue

Exit status is non-zero if either layer fails.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from pipeline.preflight import format_preflight, preflight_report
from pipeline.runtime import production_environment_receipt, validate_production_environment
from pipeline.kilosort_compat import ensure_kilosort_compatibility


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument(
        "--data-dir", type=Path, help="SpikeGLX source folder to check for reachability"
    )
    parser.add_argument(
        "--stream-id", help="Stream to locate under --data-dir, e.g. imec0.ap"
    )
    parser.add_argument("--output-dir", type=Path, help="Results directory to check")
    parser.add_argument(
        "--local-work-dir", type=Path, help="NVMe scratch directory to check"
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=None,
        help="Override the scratch free-space floor",
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit the combined receipt as JSON only"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

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

    host_requested = any(
        value is not None
        for value in (args.data_dir, args.output_dir, args.local_work_dir)
    )
    host = None
    if host_requested:
        kwargs = {
            "data_dir": args.data_dir,
            "output_dir": args.output_dir,
            "local_work_dir": args.local_work_dir,
            "stream_id": args.stream_id,
        }
        if args.min_free_gb is not None:
            kwargs["min_free_gb"] = args.min_free_gb
        host = preflight_report(**kwargs)
        receipt["host_preflight"] = host

    if args.json:
        print(json.dumps(receipt, indent=2))
    else:
        print(json.dumps(receipt, indent=2))
        if host is not None:
            print("\nHost preflight:")
            print(format_preflight(host))

    if args.data_dir is not None and args.stream_id is None:
        print(
            "\nNote: --data-dir given without --stream-id; the source stream was "
            "not verified.",
            file=sys.stderr,
        )

    return 0 if host is None or host["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
