#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np
import torch
from kilosort.io import BinaryFiltered
from kilosort.preprocessing import get_highpass_filter


def produce(kind: str, output: Path) -> None:
    rg = np.random.default_rng(7021 if kind == "ordinary" else 7022)
    x = rg.integers(-180, 181, size=(4, 512), dtype=np.int16)
    if kind == "transition":
        x[:, 256:] = np.roll(x[:, 256:], shift=1, axis=0)
        x[0, 256:] = 0
    obj = object.__new__(BinaryFiltered)
    obj.chan_map = np.arange(4)
    obj.invert_sign = False
    obj.do_CAR = True
    obj.hp_filter = get_highpass_filter(fs=30000, cutoff=300, device=torch.device("cpu"))
    obj.artifact_threshold = np.inf
    obj.whiten_mat = None
    obj.dshift = None
    obj.device = torch.device("cpu")
    y = BinaryFiltered.filter(obj, torch.from_numpy(x).float()).numpy(force=True)
    np.save(output / f"{kind}_integer_input.npy", x, allow_pickle=False)
    np.save(output / f"{kind}_prewhitening.npy", y, allow_pickle=False)
    (output / f"{kind}_producer.json").write_text(json.dumps({
        "kind": kind,
        "input_shape": list(x.shape),
        "output_shape": list(y.shape),
        "output_dtype": str(y.dtype),
        "finite": bool(np.isfinite(y).all()),
    }, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    produce(sys.argv[1], Path(sys.argv[2]))
