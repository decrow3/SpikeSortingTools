"""Characterise and freeze the evaluation-ladder snippet panel (plan §4).

The panel is 16 time-window × depth-strip snippets, 8 development + 8 held out,
**selected from input-side and estimator-side signatures only** and split before
any sorter result is seen (plan §4 rules).

Two phases:

* `characterise()` — builds each candidate snippet and records its input-side
  signatures: motion regime (from `luke_motion_regime_windows.py`, which never
  reads labels), SNR profile (from `ladder_snr.py`, voltage only), and
  >500 µV artifact-point density in the window × strip (from the accepted
  recording's `raw_over_500uv.h5` sidecar). No sorter is run.
* `freeze()` — applies the **written split rule** below and calls
  `ladder_snippets.freeze_panel`.

**Split rule, fixed 2026-09-02 before any characterisation was read:** order the
15 (regime, strip) cells regime-major (quiet, rapid_motion, sustained_noise,
support_dropout, noise_plus_motion) then strip-minor (shallow, mid, deep); the
16th cell is a second quiet window (shallow). Assign odd positions (1,3,…) to
**development**, even positions to **held_out**. This is deterministic, spans
all five regimes and all three strips in both halves, and cannot be tuned to a
result because no result exists yet.

    python testing/luke_ladder_panel.py --characterise
    python testing/luke_ladder_panel.py --freeze     # after reviewing the table

Outputs to testing/outputs/luke_ladder_panel/. Snippets go to the configured
snippet root (local disk, never /mnt).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from testing.ladder_snippets import SnippetSpec, build_snippet, freeze_panel, snippet_root
from testing.ladder_snr import SnrConfig, snr_profile

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "testing/outputs/luke_ladder_panel"
PROBE = "imec0"
RECORDING_DIR = Path(
    f"/mnt/NPX/Luke/20250804/rescue_pipeline_results_Luke0804_V2V1_g0_{PROBE}/recording"
)
ARTIFACT_SIDECAR = Path(
    f"/mnt/NPX/Luke/20250804/rescue_pipeline_results_Luke0804_V2V1_g0_{PROBE}"
    "/artifacts/raw_over_500uv.h5"
)
DURATION_S = 120.0

# Motion regimes: start_s + selection_basis from luke_motion_regime_windows.py
# (manifest.json, sorter_labels_accessed=False, motion_applied=False).
REGIMES = {
    "quiet": (4080.0, "low input anomaly and low supported motion"),
    "rapid_motion": (5910.0, "high DREDGE/decentralized-supported change with normal input"),
    "sustained_noise": (1170.0, "persistent synchrony/amplitude/channel anomaly, little motion"),
    "support_dropout": (6540.0, "unstable or depleted peak support, little supported motion"),
    "noise_plus_motion": (7230.0, "input anomaly coincident with a supported estimate change"),
}
# A second quiet window for the 16th cell (an earlier quiet stretch, input-side).
SECOND_QUIET = (3480.0, "low input anomaly and low supported motion (earlier window)")

# 112-channel contiguous depth strips, non-overlapping, edge margins kept.
STRIPS = {"shallow": 8, "mid": 136, "deep": 260}
STRIP_COUNT = 112

REGIME_ORDER = [
    "quiet",
    "rapid_motion",
    "sustained_noise",
    "support_dropout",
    "noise_plus_motion",
]
STRIP_ORDER = ["shallow", "mid", "deep"]


def _cells() -> list[dict]:
    cells = []
    for regime in REGIME_ORDER:
        start_s, basis = REGIMES[regime]
        for strip in STRIP_ORDER:
            cells.append({"regime": regime, "start_s": start_s, "basis": basis, "strip": strip})
    cells.append(
        {"regime": "quiet", "start_s": SECOND_QUIET[0], "basis": SECOND_QUIET[1], "strip": "shallow"}
    )
    for i, cell in enumerate(cells):
        cell["position"] = i + 1
        cell["split"] = "development" if (i % 2 == 0) else "held_out"
        cell["name"] = f"{cell['regime']}_{cell['strip']}_p{cell['position']:02d}"
    return cells


def _spec(cell: dict) -> SnippetSpec:
    return SnippetSpec(
        name=cell["name"],
        start_s=cell["start_s"],
        duration_s=DURATION_S,
        channel_start=STRIPS[cell["strip"]],
        channel_count=STRIP_COUNT,
        split=cell["split"],
        selection_basis=(
            f"luke_motion_regime_windows {cell['regime']} window "
            f"({cell['basis']}); {cell['strip']} depth strip"
        ),
        axes={
            "motion_regime": cell["regime"],
            "depth_strip": cell["strip"],
            "snr": "unmeasured",
            "artifact_proximity": "unmeasured",
        },
    )


def _artifact_density(spec: SnippetSpec, fs: float) -> dict:
    if not ARTIFACT_SIDECAR.exists():
        return {"artifact_points_in_window_strip": None, "artifact_proximity": "unknown"}
    import h5py

    start = int(round(spec.start_s * fs))
    stop = start + int(round(spec.duration_s * fs))
    ch_lo, ch_hi = spec.channel_start, spec.channel_start + spec.channel_count
    with h5py.File(ARTIFACT_SIDECAR, "r") as f:
        samp = f["sample_index"][:]
        chan = f["channel_index"][:]
    m = (samp >= start) & (samp < stop) & (chan >= ch_lo) & (chan < ch_hi)
    n = int(m.sum())
    rate = n / spec.duration_s
    return {
        "artifact_points_in_window_strip": n,
        "artifact_point_rate_hz": round(rate, 1),
        "artifact_proximity": "near" if rate >= 50.0 else "clear",
    }


def characterise() -> pd.DataFrame:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    root = snippet_root()
    rows = []
    for cell in _cells():
        spec = _spec(cell)
        manifest = build_snippet(spec, RECORDING_DIR, root, n_jobs=12)
        snip_dir = root / spec.directory_name
        fs = float(manifest["sampling_frequency_hz"])
        prof = snr_profile(snip_dir, SnrConfig(n_jobs=12))
        art = _artifact_density(spec, fs)
        rows.append({
            "position": cell["position"],
            "name": cell["name"],
            "split": cell["split"],
            "regime": cell["regime"],
            "strip": cell["strip"],
            "start_s": cell["start_s"],
            "depth_um_lo": manifest["depth_um_range"][0],
            "depth_um_hi": manifest["depth_um_range"][1],
            "noise_uv_median": round(prof["noise_uv_median"], 2),
            "event_amp_uv_p90": round(prof["event_amp_uv_p90"], 1),
            "peak_rate_hz": round(prof["peak_rate_hz"], 0),
            "snr": round(prof["snr"], 2),
            **art,
        })
    df = pd.DataFrame(rows)
    # SNR tertiles across the panel (emergent axis, measured not chosen)
    lo, hi = df["snr"].quantile([1 / 3, 2 / 3])
    df["snr_tertile"] = np.where(
        df["snr"] <= lo, "low", np.where(df["snr"] > hi, "high", "medium")
    )
    df.to_csv(OUTPUT / "panel_characterisation.csv", index=False)

    _write_balance(df)
    return df


def _write_balance(df: pd.DataFrame) -> dict:
    def by_split(col: str) -> dict:
        return {
            split: sub[col].value_counts().to_dict()
            for split, sub in df.groupby("split")
        }

    balance = {
        "split_counts": df["split"].value_counts().to_dict(),
        "regime_by_split": by_split("regime"),
        "strip_by_split": by_split("strip"),
        "snr_tertile_by_split": by_split("snr_tertile"),
        "artifact_by_split": by_split("artifact_proximity"),
    }
    (OUTPUT / "panel_balance.json").write_text(
        json.dumps(balance, indent=2, default=str) + "\n"
    )
    return balance


def freeze() -> dict:
    char_path = OUTPUT / "panel_characterisation.csv"
    if not char_path.exists():
        raise SystemExit("run --characterise first and review panel_characterisation.csv")
    char = pd.read_csv(char_path).set_index("name")
    root = snippet_root()
    specs = []
    for cell in _cells():
        spec = _spec(cell)
        row = char.loc[cell["name"]]
        # fill the emergent axes measured during characterisation
        spec.axes["snr"] = str(row["snr_tertile"])
        spec.axes["artifact_proximity"] = str(row["artifact_proximity"])
        specs.append(spec)
    panel = freeze_panel(specs, RECORDING_DIR, root, n_jobs=12)
    (OUTPUT / "panel_manifest.json").write_text(json.dumps(panel, indent=2) + "\n")
    return panel


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--characterise", action="store_true")
    ap.add_argument("--freeze", action="store_true")
    args = ap.parse_args()
    if args.characterise:
        df = characterise()
        print(df.to_string(index=False))
        print(f"\nwrote {OUTPUT}")
    if args.freeze:
        panel = freeze()
        print(json.dumps(panel, indent=2))


if __name__ == "__main__":
    main()
