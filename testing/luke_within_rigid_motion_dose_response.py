"""Within-Luke rigid-motion dose--response.

Prespec: ``docs/luke_within_rigid_motion_dose_response_plan.md`` (frozen
2026-09-03). Motivated by the failed Luke<->Yates motion-overlap gate
(``luke_yates_stable_window_overlap_result.md``): Luke imec0 has no motion-quiet
subset, so instead of matching it against Yates we ask whether sorting quality
degrades *within Luke* as rigid motion rises across its own ~4-23 um / 120 s
range.

Phases, run separately:

* ``--select``  reads increment 1's ``window_signatures.csv``, applies the frozen
  rank rule, converts native motion-clock starts to the frame-relative clock the
  accepted recording uses, and writes
  ``docs/luke_within_rigid_motion_windows.frozen.json``. Deterministic, cheap.
  **Implemented + tested below.**
* ``--run``     builds each selected 120 s full-probe snippet from the accepted
  RESCUE recording and sorts it with the frozen RESCUE graph (no motion
  correction). Heavy (~1 h).
* ``--endpoints`` computes the 8 KSLabel-free endpoints (prespec §4) per window.
* ``--analyze`` the Spearman dose--response vs rigid excursion / speed, with
  session-time partials and cross-estimator sensitivity (prespec §5).

  > The ``--run`` / ``--endpoints`` / ``--analyze`` code below is a SKELETON
  > pending the code review the user asked for before anything is launched. Phase
  > 1 is the reviewed-and-frozen part. See the per-function TODO notes.

No endpoint depends on ``KSLabel``. Nothing here tunes the pipeline.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "luke-within-rigid-motion-dose-response-v1"

INCREMENT1_CSV = (
    REPO_ROOT / "testing/outputs/luke_yates_stable_window_overlap/window_signatures.csv"
)
FROZEN_WINDOWS = REPO_ROOT / "docs/luke_within_rigid_motion_windows.frozen.json"
OUTPUT_DIR = REPO_ROOT / "testing/outputs/luke_within_rigid_motion_dose_response"

ACCEPTED_RECORDING = Path(
    "/mnt/NPX/Luke/20250804/rescue_pipeline_results_Luke0804_V2V1_g0_imec0/recording"
)

PRIMARY_ESTIMATOR = "medicine"
SENSITIVITY_ESTIMATORS = ("ks-motion", "dredge-motion")
DOSE_AXES = ("rigid_excursion_um", "rigid_speed_um_s")

WINDOW_S = 120.0
N_WINDOWS = 24
N_WINDOWS_MAX = 28  # after the speed-coverage top-up
FULL_PROBE_CHANNELS = 384

BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20260903


# ======================================================================= #
# phase 1 -- window selection  (implemented + tested)
# ======================================================================= #
def frame_relative_start(native_start_s: float, time_origin_native_s: float,
                         dt_median_s: float) -> float:
    """Native motion-clock start -> frame-relative seconds.

    Matches ``testing/luke_motion_regime_windows.relative_times``:
    ``acquisition_start = native_times[0] - dt/2``. The frozen 16-snippet panel
    and every ``SnippetSpec.start_s`` in this repo use that convention, so
    snippets built here line up with existing ones.
    """
    return float(native_start_s - (time_origin_native_s - dt_median_s / 2.0))


@dataclass(frozen=True)
class SelectedWindow:
    rank: int                       # 0..(n_avail-1) position in the excursion ordering
    native_start_s: float
    frame_start_s: float
    rigid_excursion_um: float
    rigid_speed_um_s: float
    added_for_speed_coverage: bool


def _increment1_luke_imec0(csv_path: Path, estimator: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[
        (df["dataset"] == "Luke")
        & (df["probe"] == "imec0")
        & (df["estimator"] == estimator)
    ].copy()
    # increment 1 QC (same thresholds as the feasibility gate)
    ok = (df["finite_fraction"] >= 0.90) & (
        df["max_time_gap_s"] <= 3.0 * df["dt_median_s"]
    )
    ok &= np.isfinite(df["rigid_excursion_um"]) & np.isfinite(df["rigid_speed_um_s"])
    df = df[ok]
    return df.sort_values(
        ["rigid_excursion_um", "window_start_native_s"], kind="stable"
    ).reset_index(drop=True)


def select_windows(csv_path: Path = INCREMENT1_CSV) -> list[SelectedWindow]:
    """The frozen rank rule (prespec §2). Deterministic; reads no sorter output."""
    ranked = _increment1_luke_imec0(csv_path, PRIMARY_ESTIMATOR)
    n_avail = len(ranked)
    if n_avail < N_WINDOWS:
        raise RuntimeError(f"only {n_avail} qualifying Luke imec0 windows; need {N_WINDOWS}")

    def _mk(i: int, top_up: bool) -> SelectedWindow:
        row = ranked.iloc[int(i)]
        return SelectedWindow(
            rank=int(i),
            native_start_s=float(row["window_start_native_s"]),
            frame_start_s=frame_relative_start(
                row["window_start_native_s"], row["time_origin_native_s"], row["dt_median_s"]
            ),
            rigid_excursion_um=float(row["rigid_excursion_um"]),
            rigid_speed_um_s=float(row["rigid_speed_um_s"]),
            added_for_speed_coverage=top_up,
        )

    pick_idx = np.unique(np.round(np.linspace(0, n_avail - 1, N_WINDOWS)).astype(int))
    chosen: dict[int, SelectedWindow] = {int(i): _mk(int(i), False) for i in pick_idx}

    # speed-coverage top-up (prespec §2): cover the 10th-90th pct of the full
    # speed distribution, adding nearest-decile windows up to N_WINDOWS_MAX.
    speed_all = ranked["rigid_speed_um_s"].to_numpy(float)
    lo, hi = np.percentile(speed_all, [10, 90])
    tol = (hi - lo) / 16.0
    covered = [chosen[k].rigid_speed_um_s for k in chosen]
    for d in np.linspace(lo, hi, 9):
        if any(abs(c - d) <= tol for c in covered):
            continue
        if len(chosen) >= N_WINDOWS_MAX:
            break
        cand = int(np.argmin(np.abs(speed_all - d)))
        if cand in chosen:
            continue
        chosen[cand] = _mk(cand, True)
        covered.append(chosen[cand].rigid_speed_um_s)

    return [chosen[k] for k in sorted(chosen)]


def write_frozen_list(windows: list[SelectedWindow], path: Path = FROZEN_WINDOWS) -> None:
    if path.exists():
        raise RuntimeError(f"{path} already exists; the window list is frozen once")
    path.write_text(json.dumps(
        {
            "schema": SCHEMA,
            "prespec": "docs/luke_within_rigid_motion_dose_response_plan.md",
            "source_csv": str(INCREMENT1_CSV),
            "estimator": PRIMARY_ESTIMATOR,
            "window_s": WINDOW_S,
            "n_windows": len(windows),
            "windows": [asdict(w) for w in windows],
        },
        indent=2,
    ))


def load_frozen_list(path: Path = FROZEN_WINDOWS) -> list[SelectedWindow]:
    return [SelectedWindow(**w) for w in json.loads(path.read_text())["windows"]]


# ======================================================================= #
# phase 4 -- statistics  (implemented + tested; endpoint sourcing is skeleton)
# ======================================================================= #
def spearman_ci(x: np.ndarray, y: np.ndarray, n_boot: int = BOOTSTRAP,
                seed: int = BOOTSTRAP_SEED) -> dict:
    from scipy.stats import spearmanr

    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 4 or np.unique(x).size < 3 or np.unique(y).size < 2:
        return {"rho": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "n": int(x.size)}
    rho = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, x.size, x.size)
        if np.unique(x[idx]).size < 3 or np.unique(y[idx]).size < 2:
            continue
        boot.append(float(spearmanr(x[idx], y[idx]).statistic))
    lo, hi = (np.nanpercentile(boot, [2.5, 97.5]) if boot else (np.nan, np.nan))
    return {"rho": rho, "ci_lo": float(lo), "ci_hi": float(hi), "n": int(x.size)}


def mann_kendall(x: np.ndarray, y: np.ndarray) -> dict:
    """Trend of y ordered by x (normal approximation, no tie correction)."""
    from scipy.stats import norm

    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    y = y[ok][np.argsort(x[ok], kind="stable")]
    n = y.size
    if n < 4:
        return {"tau": float("nan"), "p": float("nan"), "n": int(n)}
    s = sum(int(np.sum(np.sign(y[i + 1:] - y[i]))) for i in range(n - 1))
    var = n * (n - 1) * (2 * n + 5) / 18.0
    z = (s - np.sign(s)) / np.sqrt(var) if var > 0 else 0.0
    return {"tau": float(2 * s / (n * (n - 1))), "p": float(2 * (1 - norm.cdf(abs(z)))),
            "S": int(s), "n": int(n)}


def partial_spearman(y: np.ndarray, x: np.ndarray, z: np.ndarray) -> float:
    """Spearman partial correlation of y and x controlling for z (rank residuals)."""
    from scipy.stats import spearmanr

    y, x, z = (np.asarray(v, float) for v in (y, x, z))
    ok = np.isfinite(y) & np.isfinite(x) & np.isfinite(z)
    if ok.sum() < 4:
        return float("nan")
    ry, rx, rz = (pd.Series(v[ok]).rank().to_numpy() for v in (y, x, z))

    def resid(a, b):
        return a - np.polyval(np.polyfit(b, a, 1), b)

    return float(spearmanr(resid(ry, rz), resid(rx, rz)).statistic)


ENDPOINT_KEYS = (
    "E1_compact_events_per_mm_per_s",
    "E2_fraction_events_assigned",
    "E3_qualified_units_per_mm",
    "E4_refractory_burden_median",
    "E5_similar_pairs_per_qualified_unit",
    "E6_waveform_stability_median",
    "E7_qualified_rate_hz_median",
    "E8_fragmentation_index",
)


def dose_response(endpoints: pd.DataFrame) -> dict:
    """Spearman + MK + session-time partials for every endpoint x dose axis (prespec §5)."""
    out: dict = {"schema": SCHEMA, "n_windows": int(len(endpoints)), "by_endpoint": {}}
    if "frame_start_s" not in endpoints:
        raise ValueError("endpoints frame needs a frame_start_s column for the session-time partial")
    time_col = endpoints["frame_start_s"].to_numpy(float)
    for ep in ENDPOINT_KEYS:
        if ep not in endpoints:
            continue
        y = endpoints[ep].to_numpy(float)
        rec: dict = {}
        for axis in DOSE_AXES:
            x = endpoints[axis].to_numpy(float)
            rec[axis] = {
                "spearman": spearman_ci(x, y),
                "mann_kendall": mann_kendall(x, y),
                "partial_given_session_time": partial_spearman(y, x, time_col),
                "session_time_partial_given_dose": partial_spearman(y, time_col, x),
            }
        out["by_endpoint"][ep] = rec
    return out


# ======================================================================= #
# phase 2 -- build + sort  (SKELETON -- pending design review)
# ======================================================================= #
def run_window(window: SelectedWindow, out_root: Path | None = None) -> dict:
    """Build the full-probe 120 s snippet and sort it with the frozen RESCUE graph.

    TODO (review first):
    - confirm ``ACCEPTED_RECORDING`` covers ``window.frame_start_s + 120`` for
      every selected rank (the motion estimate spans ~10.5 ks frame-relative;
      the accepted recording may be shorter);
    - confirm full-probe (384 ch) ``build_snippet`` + ``l1_run`` is within a sane
      per-window budget, else fall back to a single fixed 112 ch strip and say so
      in the prespec;
    - ``SnippetSpec.split`` value is metadata only here (not a panel entry) --
      check ``freeze_panel`` / digest interplay is irrelevant for a standalone
      snippet.
    """
    from testing.ladder_snippets import SnippetSpec, build_snippet
    from testing.ladder_l1 import l1_run

    spec = SnippetSpec(
        name=f"rigid_dose_rank{window.rank:02d}",
        start_s=window.frame_start_s,
        duration_s=WINDOW_S,
        channel_start=0,
        channel_count=FULL_PROBE_CHANNELS,
        split="development",
        selection_basis=(
            f"luke_within_rigid_motion_dose_response: rigid_excursion_um rank "
            f"{window.rank} under the medicine motion estimate "
            "(input/estimator-side only; no sorter labels)"
        ),
        axes={"motion_regime": f"rigid_excursion_{window.rigid_excursion_um:.1f}um"},
    )
    manifest = build_snippet(spec, ACCEPTED_RECORDING)
    snippet_dir = Path(manifest.get("snippet_dir") or manifest["dir"])
    return {"window": asdict(window), "snippet_dir": str(snippet_dir),
            "l1": l1_run(snippet_dir, out_root=out_root)}


# ======================================================================= #
# phase 3 -- endpoints  (SKELETON -- pending design review)
# ======================================================================= #
# Each endpoint is defined in prespec §4. They operate over ALL curated clusters
# (never ``sort["good"]``). The functions below fix the *interfaces*; the
# trace-level computations (E1 compactness, E6 half-split waveform cosine) are
# marked and will be wired against the built snippet recording after review.

def _refractory_fraction(train: np.ndarray, fs: float, refractory_ms: float = 1.5) -> float:
    if train.size < 2:
        return float("nan")
    isi_ms = np.diff(np.sort(train)) / fs * 1000.0
    return float((isi_ms < refractory_ms).mean())


def qualify_units(spike_times: np.ndarray, spike_clusters: np.ndarray,
                  template_amp_uv: dict[int, float], fs: float, duration_s: float,
                  *, amplitude_uv: float = 15.0, rv_reference: float = 0.001,
                  rv_multiple: float = 2.0, presence_frac: float = 0.60,
                  presence_bin_s: float = 20.0) -> dict:
    """E3 -- KSLabel-free unit qualification (amplitude / refractory / presence)."""
    n_bins = max(int(round(duration_s / presence_bin_s)), 1)
    edges = np.linspace(0.0, duration_s * fs, n_bins + 1)
    per_unit, qualified = [], []
    for c in np.unique(spike_clusters):
        train = spike_times[spike_clusters == c]
        if train.size < 2:
            continue
        amp = float(template_amp_uv.get(int(c), np.nan))
        rv = _refractory_fraction(train, fs)
        presence = float((np.histogram(train, bins=edges)[0] > 0).mean())
        ok = (amp >= amplitude_uv) and (np.isfinite(rv) and rv <= rv_multiple * rv_reference) \
            and (presence >= presence_frac)
        per_unit.append({"cluster": int(c), "n_spikes": int(train.size), "amp_uv": amp,
                         "rv_fraction": rv, "presence_frac": presence,
                         "rate_hz": float(train.size / duration_s) if duration_s else float("nan"),
                         "qualified": bool(ok)})
        if ok:
            qualified.append(int(c))
    return {"qualified": qualified, "per_unit": per_unit}


def window_endpoints(run_record: dict) -> dict:
    """E1-E8 for one window (prespec §4). SKELETON: fills what is computable from
    the curated arrays; E1/E2/E6 need the conditioned snippet traces (wired after
    review). Returns a flat dict keyed by ENDPOINT_KEYS + provenance."""
    raise NotImplementedError(
        "phase 3 endpoint extraction is a skeleton pending review of the prespec "
        "§4 definitions and the trace-level compactness / waveform-stability wiring"
    )


# ======================================================================= #
# CLI
# ======================================================================= #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--select", action="store_true", help="phase 1: freeze the window list")
    ap.add_argument("--run", action="store_true", help="phase 2: build + sort each window")
    ap.add_argument("--endpoints", action="store_true", help="phase 3: per-window endpoints")
    ap.add_argument("--analyze", action="store_true", help="phase 4: dose-response")
    ap.add_argument("--csv", type=Path, default=INCREMENT1_CSV)
    ap.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.select:
        windows = select_windows(args.csv)
        write_frozen_list(windows)
        print(f"froze {len(windows)} windows -> {FROZEN_WINDOWS}")
        for w in windows:
            tag = " (speed top-up)" if w.added_for_speed_coverage else ""
            print(f"  rank {w.rank:2d}  frame {w.frame_start_s:8.1f}s  "
                  f"excursion {w.rigid_excursion_um:6.2f}um  speed {w.rigid_speed_um_s:.3f}um/s{tag}")
        return 0

    if args.run:
        rows = [run_window(w) for w in load_frozen_list()]
        (args.out_dir / "run_index.json").write_text(json.dumps(rows, indent=2, default=str))
        print(f"ran {len(rows)} windows")
        return 0

    if args.endpoints:
        raise SystemExit("phase 3 is a skeleton pending review")

    if args.analyze:
        ep_path = args.out_dir / "window_endpoints.csv"
        if not ep_path.exists():
            raise SystemExit(f"{ep_path} missing; run --endpoints first")
        result = dose_response(pd.read_csv(ep_path))
        (args.out_dir / "dose_response.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return 0

    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
