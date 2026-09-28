#!/usr/bin/env python3
"""Focused DQ fixture for EnforceDecrease's zero-PTP rescaling algebra."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

import dartsort
from dartsort.transform.enforce_decrease import EnforceDecrease
from dartsort.util import spiketorch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "testing/outputs/dq_enforce_decrease_zero_ptp_20260928"
SOURCE = Path(dartsort.transform.enforce_decrease.__file__)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def proposed_forward(module, waveforms, channels):
    """Exact current algebra plus the minimal finite zero/zero guard."""
    waveforms = waveforms.clone()
    ptps = spiketorch.ptp(waveforms).to(module.b.parents_index.device)
    pad_ptps = F.pad(ptps, (0, 1), value=torch.inf)
    parent_min = torch.zeros_like(ptps)
    n = len(waveforms)
    for bs in range(0, n, module.batch_size):
        be = min(n, bs + module.batch_size)
        parent_ptps = pad_ptps[
            torch.arange(bs, be)[:, None, None],
            module.b.parents_index[channels[bs:be]],
        ]
        torch.amin(parent_ptps, dim=2, out=parent_min[bs:be])
    zero_zero = (parent_min == 0) & (ptps == 0)
    rescaling = parent_min.div(ptps)
    rescaling.masked_fill_(zero_zero, 1.0).clamp_(max=1.0)
    return waveforms.mul_(rescaling.to(waveforms.device)[:, None, :]), ptps, parent_min, rescaling


def enc(x):
    a = x.detach().cpu().numpy()
    return [[None if not np.isfinite(v) else float(v) for v in row] for row in a.reshape(-1, a.shape[-1])]


def main():
    started = time.monotonic()
    cpu0 = time.process_time()
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.mkdir(parents=True)

    # Four local waveform slots, with the last slot padded for channel 0.
    geom = np.c_[np.zeros(4), np.arange(4) * 20.0]
    channel_index = np.array(
        [[0, 1, 2, 4], [0, 1, 2, 4], [0, 1, 2, 4], [0, 1, 2, 3]],
        dtype=np.int64,
    )
    module = EnforceDecrease(channel_index, geom)
    module.precompute()
    channels = torch.zeros(4, dtype=torch.long)
    t = 5
    waveforms = torch.empty((4, t, 4), dtype=torch.float32)
    waveforms[0] = 0.0
    waveforms[0, :, 3] = torch.nan
    waveforms[1] = torch.tensor([1.0, 2.0, 3.0, torch.nan])[None, :]
    # parent positive / child PTP zero on slot 1; padded slot remains NaN.
    waveforms[2, :, 0] = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    waveforms[2, :, 1] = 5.0
    waveforms[2, :, 2] = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
    waveforms[2, :, 3] = torch.nan
    # Ordinary finite, spatially decreasing waveform.
    base = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])[:, None]
    waveforms[3] = base * torch.tensor([1.0, 0.75, 0.5, 0.25])[None, :]

    current = module(waveforms.clone(), channels=channels)
    proposed, ptps, parent_min, scaling = proposed_forward(module, waveforms, channels)

    checks = {
        "current_exact_zero_has_child_nans": bool(torch.isnan(current[0, :, 1:3]).all()),
        "proposed_exact_zero_all_finite_nonpad": bool(torch.isfinite(proposed[0, :, :3]).all()),
        "proposed_exact_zero_stays_zero": bool(torch.equal(proposed[0, :, :3], torch.zeros_like(proposed[0, :, :3]))),
        "ptp0_constant_is_not_waveform_zero": bool(torch.count_nonzero(waveforms[1, :, :3]).item()),
        "current_nonzero_constant_has_child_nans": bool(torch.isnan(current[1, :, 1:3]).all()),
        "proposed_nonzero_constant_unchanged": bool(torch.equal(proposed[1, :, :3], waveforms[1, :, :3])),
        "padded_channel_remains_nan": bool(torch.isnan(proposed[:3, :, 3]).all()),
        "positive_parent_zero_child_scale_is_one": bool(ptps[2, 1] == 0 and parent_min[2, 1] > 0 and scaling[2, 1] == 1),
        "ordinary_finite_output": bool(torch.isfinite(proposed[3]).all()),
        "ordinary_path_unchanged": bool(torch.equal(current[3], proposed[3])),
        "ordinary_scaling_finite": bool(torch.isfinite(scaling[3]).all()),
    }
    if not all(checks.values()):
        raise RuntimeError(checks)

    result = {
        "status": "focused_fixture_pass_actual_h5_cause_unproven",
        "installed_version": getattr(dartsort, "__version__", "unknown"),
        "installed_source": str(SOURCE),
        "installed_source_sha256": sha(SOURCE),
        "source_lines": {"division": 91, "waveform_multiply": 93},
        "checks": checks,
        "case_ptps": enc(ptps),
        "case_parent_min_ptps": enc(parent_min),
        "case_proposed_scaling": enc(scaling),
        "recommendation": (
            "Immediately before division, save zero_zero = (parent_min_ptps == 0) & (ptps == 0); "
            "after division, set only zero_zero entries to 1, then clamp max=1. This preserves exact-zero "
            "and nonzero-constant channels, leaves genuine padded NaNs as NaN, preserves positive/zero -> "
            "+inf -> clamp(1), and leaves the ordinary finite path unchanged."
        ),
        "non_recommendations": [
            "Do not equate PTP==0 with an all-zero waveform; temporally constant nonzero channels also have PTP zero.",
            "Do not blanket-reject every zero denominator; positive/zero legitimately yields scale 1 after the existing clamp.",
            "Do not nan_to_num the full ratio because padded-channel NaNs are expected structural values.",
        ],
        "interpretation_limit": (
            "This reproduces the algebraic hazard in h1's installed DARTsort 0.5.16 only. The h5 source and actual failing input were not accessible here, so this is a compatible mechanism and focused fix recommendation, not independent proof of the h5 failure cause."
        ),
    }
    (OUT / "FIXTURE.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (OUT / "RECOMMENDATION.md").write_text(
        "# DQ EnforceDecrease zero-PTP check\n\n"
        "## Verdict\n\n"
        "The h1-installed DARTsort 0.5.16 source has a real finite `0/0` hazard at "
        "`parent_min_ptps / ptps`. The focused fixture reproduces NaNs for both an exact-zero "
        "waveform and a temporally constant nonzero waveform. This is compatible with the h5 "
        "failure hypothesis, but it does not independently prove that h5 encountered the same input or source.\n\n"
        "## Minimal handling\n\n"
        "Save a mask for finite `parent_min_ptps == 0` and `ptps == 0`, perform the existing "
        "division, replace only those masked ratios with 1, and retain the existing maximum-1 "
        "clamp. This leaves structural padded-channel NaNs intact, preserves the legitimate "
        "positive/zero to infinity to 1 path, and is identical on the ordinary finite fixture.\n\n"
        "No held h5 payload, voltage, full fit, donor qualification, broad test suite, GPU work, "
        "or sort was accessed or run.\n"
    )
    receipt = {
        "status": result["status"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cpu_s_measured": time.process_time() - cpu0,
        "wall_s_measured": time.monotonic() - started,
        "conservative_active_charge_s": 60.0,
        "threads_max": 2,
        "readers_max": 1,
        "source_bytes_read": SOURCE.stat().st_size,
        "raw_voltage_bytes": 0,
        "gpu_s": 0,
        "sorts": 0,
        "prior_h1_cumulative_active_s": 18653.22,
        "new_h1_cumulative_active_s": 18713.22,
        "overall_ceiling_active_s": 26000.0,
    }
    (OUT / "RESOURCE_RECEIPT.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    files = []
    for p in sorted(OUT.iterdir()):
        if p.name not in {"MANIFEST.json", "COMPLETE.json"}:
            files.append({"path": p.name, "bytes": p.stat().st_size, "sha256": sha(p)})
    (OUT / "MANIFEST.json").write_text(json.dumps({"files": files}, indent=2, sort_keys=True) + "\n")
    (OUT / "COMPLETE.json").write_text(
        json.dumps({"status": result["status"], "manifest_sha256": sha(OUT / "MANIFEST.json"), "written_last": True}, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"status": result["status"], "output": str(OUT), "charge_s": 60.0}, sort_keys=True))


if __name__ == "__main__":
    main()
