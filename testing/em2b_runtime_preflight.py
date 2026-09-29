#!/usr/bin/env python3
"""No-voltage runtime preflight for a serialized EM.2b recording."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    import spikeinterface as si
    import torch

    config = json.loads(args.config.read_text())
    torch.ones(1, device=config["device"]).sum().item()
    recording = si.load(config["recording"]["path"])
    signature = {
        "frames": int(recording.get_num_samples()),
        "channels": int(recording.get_num_channels()),
        "dtype": str(recording.get_dtype()),
        "sampling_frequency_hz": float(recording.get_sampling_frequency()),
    }
    if signature != {
        "frames": 10_199_918,
        "channels": 182,
        "dtype": "float32",
        "sampling_frequency_hz": 29999.759166666667,
    }:
        raise RuntimeError(f"unexpected stage-1 recording signature: {signature}")
    module_path = Path(sys.modules[type(recording).__module__].__file__).resolve()
    receipt = {
        "schema": "em2b-runtime-preflight-v1",
        "status": "complete",
        "pid": os.getpid(),
        "python": sys.executable,
        "spikeinterface": si.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "recording_class": f"{type(recording).__module__}.{type(recording).__name__}",
        "recording_module_path": str(module_path),
        "recording_module_sha256": hashlib.sha256(module_path.read_bytes()).hexdigest(),
        "recording_signature": signature,
        "voltage_read": False,
        "sort_launched": False,
        "finished_epoch": time.time(),
    }
    temporary = args.output / "runtime-preflight.json.partial"
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output / "runtime-preflight.json")


if __name__ == "__main__":
    main()
