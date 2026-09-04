"""Within-Luke rigid-motion dose--response.

Prespec: ``docs/luke_within_rigid_motion_dose_response_plan.md`` (frozen
2026-09-03, revised after Codex review). Motivated by the failed Luke<->Yates
motion-overlap gate (``luke_yates_stable_window_overlap_result.md``): Luke imec0
has no motion-quiet 120 s window, so instead of matching it against Yates we ask
whether sorting metrics *covary* with estimated rigid motion within Luke's own
~2-25 um / 120 s range.

Motion is an OBSERVED EXPOSURE, not a manipulation. No causal or "quiet windows
are healthy" claims -- that is C2 v4. The four motion estimators disagree on
which windows are quiet (MEDiCINe vs the others: rank rho ~0.1), so the primary
dose is the consensus percentile rank across the three mutually concordant
estimators {ks-motion, dredge-motion, decentralized-motion}; MEDiCINe is a
sensitivity arm.

Phases (run separately):
* ``--select``  consensus-dose window selection -> frozen JSON. Deterministic.
* ``--pilot``   build + sort ONE window, report wall-clock + disk (finding: run
                the storage/runtime pilot before the batch).
* ``--run``     build + sort all frozen windows (RESCUE graph, no motion corr).
* ``--endpoints`` the 8 KSLabel-free endpoints (prespec §5) per window.
* ``--analyze`` the dose-response (prespec §6): one primary test (E3 vs consensus
                excursion), the rest supportive; session-time partials;
                cross-estimator exposure validity.

No endpoint reads ``KSLabel``. Nothing here tunes the pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "luke-within-rigid-motion-dose-response-v2"

INCREMENT1_CSV = (
    REPO_ROOT / "testing/outputs/luke_yates_stable_window_overlap/window_signatures.csv"
)
FROZEN_WINDOWS = REPO_ROOT / "docs/luke_within_rigid_motion_windows.frozen.json"
OUTPUT_DIR = REPO_ROOT / "testing/outputs/luke_within_rigid_motion_dose_response"

ACCEPTED_RECORDING = Path(
    "/mnt/NPX/Luke/20250804/rescue_pipeline_results_Luke0804_V2V1_g0_imec0/recording"
)

# prespec §2 -- MEDiCINe is the rank outlier and is demoted to sensitivity only.
CONCORDANT_ESTIMATORS = ("ks-motion", "dredge-motion", "decentralized-motion")
SENSITIVITY_ESTIMATOR = "medicine"
ALL_ESTIMATORS = (SENSITIVITY_ESTIMATOR, *CONCORDANT_ESTIMATORS)

WINDOW_S = 120.0
N_WINDOWS = 24
FULL_PROBE_CHANNELS = 384
LOW_POWER_QUALIFIED_UNITS = 8   # prespec §5 -- flags the primary test as underpowered

# --- E3 qualification (prespec §5, frozen) ------------------------------- #
QUAL_MIN_SPIKES = 150
QUAL_AMPLITUDE_UV = 15.0
QUAL_RV_CEILING = 0.01           # flat 1% contamination ceiling
QUAL_PRESENCE_BINS = 12          # 10 s sub-bins over 120 s
QUAL_PRESENCE_MIN_BINS = 9
REFRACTORY_MS = 1.5
# --- E5 / E6 / E8 --------------------------------------------------------- #
SIMILAR_COSINE = 0.8
SIMILAR_DEPTH_UM = 100.0
E6_MIN_SPIKES_PER_HALF = 60
FRAG_DEPTH_UM = 40.0
FRAG_COINCIDENCE_FRAC = 0.05
FRAG_TOL_MS = 0.5
ASSIGN_TOL_MS = 0.5             # C2
ASSIGN_TOL_UM = 40.0

BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20260903
EXPOSURE_NEGLIGIBLE_RHO = 0.10   # |consensus rho| below this -> 'no_association'
PARTIAL_SURVIVES_MIN = 0.10      # a partial 'survives' iff same sign as rho and |.| >= this
STA_NEIGHBOURHOOD_CH = 4         # +/- channels around the dewhitened peak for the bandpass STA
PERM_P_LAPLACE = True            # Monte-Carlo p = (k+1)/(n_perm+1)
COVARIATES_AVAILABLE = ()        # frozen: no independent stimulus/running/pupil source for 20250804

ENDPOINT_KEYS = (
    "E3_qualified_units_per_mm",     # primary
    "E4_refractory_burden_median",
    "E5_similar_pairs_per_qualified_unit",
    "E6_waveform_stability_median",
    "E7_qualified_rate_hz_median",
    "E8_fragmentation_index",
    "C1_detected_events_per_mm_per_s",
    "C2_fraction_events_near_qualified",
)
PRIMARY_ENDPOINT = "E3_qualified_units_per_mm"
PRIMARY_DOSE = "exc_consensus_rank"
SUPPORTIVE_EXPECTED_SIGN = {   # prespec §6; None = no prediction
    "E4_refractory_burden_median": +1,
    "E5_similar_pairs_per_qualified_unit": +1,
    "E6_waveform_stability_median": -1,
    "E7_qualified_rate_hz_median": None,
    "E8_fragmentation_index": +1,
    "C1_detected_events_per_mm_per_s": None,
    "C2_fraction_events_near_qualified": -1,
}


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


# ======================================================================= #
# phase 1 -- consensus-dose window selection
# ======================================================================= #
def _qc_pass(df: pd.DataFrame) -> pd.Series:
    return (
        (df["finite_fraction"] >= 0.90)
        & (df["max_time_gap_s"] <= 3.0 * df["dt_median_s"])
        & np.isfinite(df["rigid_excursion_um"])
        & np.isfinite(df["rigid_speed_um_s"])
    )


def consensus_dose_table(csv_path: Path = INCREMENT1_CSV) -> pd.DataFrame:
    """Per-`time_interval_id` dose table (prespec §2--3).

    An interval survives only if every concordant estimator passes increment 1
    QC there. Columns: ``window_start_recording_s``, per-estimator ``exc_*`` /
    ``spd_*``, and the consensus percentile ranks over the concordant three.
    """
    raw = pd.read_csv(csv_path)
    li = raw[(raw["dataset"] == "Luke") & (raw["probe"] == "imec0")].copy()

    per_est: dict[str, pd.DataFrame] = {}
    for est in ALL_ESTIMATORS:
        e = li[li["estimator"] == est].copy()
        e = e[_qc_pass(e)]
        e = e.set_index("time_interval_id")
        per_est[est] = e[["window_start_recording_s", "rigid_excursion_um", "rigid_speed_um_s"]]

    concordant_ids = set.intersection(*(set(per_est[e].index) for e in CONCORDANT_ESTIMATORS))
    ids = sorted(concordant_ids)
    if len(ids) < N_WINDOWS:
        raise RuntimeError(f"only {len(ids)} intervals pass QC on all concordant estimators; need {N_WINDOWS}")

    out = pd.DataFrame(index=pd.Index(ids, name="time_interval_id"))
    # recording-relative start: take it from a concordant estimator (they agree
    # to well under a bin); assert consistency.
    starts = np.column_stack([
        per_est[e].loc[ids, "window_start_recording_s"].to_numpy(float) for e in CONCORDANT_ESTIMATORS
    ])
    if starts.ptp(axis=1).max() > 2.0:
        raise RuntimeError("concordant estimators disagree on window_start_recording_s by > 2 s")
    out["window_start_recording_s"] = starts.mean(axis=1)

    for est in ALL_ESTIMATORS:
        e = per_est[est].reindex(ids)
        out[f"exc_{est}"] = e["rigid_excursion_um"].to_numpy(float)
        out[f"spd_{est}"] = e["rigid_speed_um_s"].to_numpy(float)

    exc = out[[f"exc_{e}" for e in CONCORDANT_ESTIMATORS]]
    spd = out[[f"spd_{e}" for e in CONCORDANT_ESTIMATORS]]
    out["exc_consensus_rank"] = exc.rank(pct=True).mean(axis=1)
    out["spd_consensus_rank"] = spd.rank(pct=True).mean(axis=1)
    return out.reset_index()


@dataclass(frozen=True)
class SelectedWindow:
    time_interval_id: int
    snippet_start_s: float          # recording-relative boundary (0, 120, 240, ...)
    exc_consensus_rank: float
    spd_consensus_rank: float
    exc_by_estimator: dict
    spd_by_estimator: dict


def select_windows(csv_path: Path = INCREMENT1_CSV) -> list[SelectedWindow]:
    """24 windows at even consensus rigid-excursion ranks (prespec §3). Deterministic."""
    table = consensus_dose_table(csv_path).sort_values(
        ["exc_consensus_rank", "time_interval_id"], kind="stable"
    ).reset_index(drop=True)
    pick = np.unique(np.round(np.linspace(0, len(table) - 1, N_WINDOWS)).astype(int))
    out = []
    for i in pick:
        r = table.iloc[int(i)]
        out.append(SelectedWindow(
            time_interval_id=int(r["time_interval_id"]),
            snippet_start_s=float(r["window_start_recording_s"]),
            exc_consensus_rank=float(r["exc_consensus_rank"]),
            spd_consensus_rank=float(r["spd_consensus_rank"]),
            exc_by_estimator={e: float(r[f"exc_{e}"]) for e in ALL_ESTIMATORS},
            spd_by_estimator={e: float(r[f"spd_{e}"]) for e in ALL_ESTIMATORS},
        ))
    return out


def write_frozen_list(windows: list[SelectedWindow], csv_path: Path = INCREMENT1_CSV,
                      path: Path = FROZEN_WINDOWS) -> None:
    if path.exists():
        raise RuntimeError(f"{path} exists; the window list is frozen once")
    ids = [w.time_interval_id for w in windows]
    if len(set(ids)) != len(ids):
        raise RuntimeError("duplicate time_interval_id in selection")
    path.write_text(json.dumps({
        "schema": SCHEMA,
        "prespec": "docs/luke_within_rigid_motion_dose_response_plan.md",
        "git_commit": _git_commit(),
        "source_csv": str(csv_path),
        "source_csv_sha256": _sha256(csv_path),
        "primary_dose": PRIMARY_DOSE,
        "concordant_estimators": list(CONCORDANT_ESTIMATORS),
        "window_s": WINDOW_S,
        "n_windows": len(windows),
        "time_interval_ids": ids,
        "windows": [asdict(w) for w in windows],
    }, indent=2))


def load_frozen_list(path: Path = FROZEN_WINDOWS, csv_path: Path = INCREMENT1_CSV,
                     *, verify_source: bool = True) -> list[SelectedWindow]:
    """Load the frozen window list and revalidate every contract field.

    With ``verify_source`` (default), the source CSV must be present, its SHA-256
    must match, and re-running the selection from it must reproduce the frozen
    records exactly. Snippet starts must be whole 120 s boundaries.
    """
    payload = json.loads(path.read_text())
    if payload.get("schema") != SCHEMA:
        raise RuntimeError(f"frozen list schema {payload.get('schema')!r} != {SCHEMA!r}")
    for k in ("git_commit", "source_csv", "source_csv_sha256", "primary_dose",
              "concordant_estimators", "window_s", "time_interval_ids", "windows"):
        if k not in payload:
            raise RuntimeError(f"frozen list missing field {k!r}")
    if payload["primary_dose"] != PRIMARY_DOSE:
        raise RuntimeError(f"frozen primary_dose {payload['primary_dose']!r} != {PRIMARY_DOSE!r}")
    if tuple(payload["concordant_estimators"]) != CONCORDANT_ESTIMATORS:
        raise RuntimeError("frozen concordant_estimators != current")
    if payload["window_s"] != WINDOW_S:
        raise RuntimeError("frozen window_s != current")
    if not (payload["n_windows"] == len(payload["windows"]) == N_WINDOWS):
        raise RuntimeError("frozen list N mismatch")
    ids = [w["time_interval_id"] for w in payload["windows"]]
    if ids != payload["time_interval_ids"] or len(set(ids)) != len(ids):
        raise RuntimeError("frozen list interval ids inconsistent / duplicated")

    frozen = [SelectedWindow(**w) for w in payload["windows"]]
    for w in frozen:
        if abs(w.snippet_start_s / WINDOW_S - round(w.snippet_start_s / WINDOW_S)) > 1e-6:
            raise RuntimeError(f"iv {w.time_interval_id}: snippet_start_s not a 120 s boundary")

    if verify_source:
        if not Path(csv_path).exists():
            raise RuntimeError(f"source CSV {csv_path} not found; pass verify_source=False to skip")
        if _sha256(csv_path) != payload["source_csv_sha256"]:
            raise RuntimeError("source CSV changed since the window list was frozen")
        recomputed = {w.time_interval_id: w for w in select_windows(csv_path)}
        if set(recomputed) != {w.time_interval_id for w in frozen}:
            raise RuntimeError("re-running selection from the source does not reproduce the frozen id set")
        for w in frozen:
            r = recomputed[w.time_interval_id]
            if (abs(r.snippet_start_s - w.snippet_start_s) > 1e-3
                    or abs(r.exc_consensus_rank - w.exc_consensus_rank) > 1e-9
                    or abs(r.spd_consensus_rank - w.spd_consensus_rank) > 1e-9):
                raise RuntimeError(f"iv {w.time_interval_id}: frozen record != recomputed")
    return frozen


# ======================================================================= #
# phase 4 -- statistics
# ======================================================================= #
def spearman_ci(x, y, n_boot: int = BOOTSTRAP, seed: int = BOOTSTRAP_SEED) -> dict:
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


def block_bootstrap_spearman(x, y, order, block: int = 3, n_boot: int = 1000,
                             seed: int = BOOTSTRAP_SEED) -> dict:
    """Bootstrap CI over contiguous blocks of `order` (prespec §6 robustness)."""
    from scipy.stats import spearmanr

    x, y, order = (np.asarray(v, float) for v in (x, y, order))
    o = np.argsort(order, kind="stable")
    xs, ys = x[o], y[o]
    n = xs.size
    if n < 2 * block:
        return {"ci_lo": float("nan"), "ci_hi": float("nan")}
    starts = np.arange(0, n - block + 1)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        chunks = [np.arange(s, s + block) for s in rng.choice(starts, size=int(np.ceil(n / block)))]
        idx = np.concatenate(chunks)[:n]
        if np.unique(xs[idx]).size < 3 or np.unique(ys[idx]).size < 2:
            continue
        boot.append(float(spearmanr(xs[idx], ys[idx]).statistic))
    lo, hi = (np.nanpercentile(boot, [2.5, 97.5]) if boot else (np.nan, np.nan))
    return {"ci_lo": float(lo), "ci_hi": float(hi)}


def mann_kendall(x, y, n_perm: int = 5000, seed: int = BOOTSTRAP_SEED) -> dict:
    """Mann-Kendall trend of y against x. S from pairwise sign(dx)*sign(dy) so x
    ties are handled; tau_b with both tie terms; permutation p = (k+1)/(n_perm+1).
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = y.size
    if n < 4:
        return {"tau_b": float("nan"), "p_perm": float("nan"), "S": float("nan"),
                "n": int(n), "x_has_ties": bool(np.unique(x).size < n)}

    def _S(xx, yy):
        tot = 0.0
        for i in range(xx.size - 1):
            tot += np.sum(np.sign(xx[i + 1:] - xx[i]) * np.sign(yy[i + 1:] - yy[i]))
        return float(tot)

    def _ties(v):
        _, c = np.unique(v, return_counts=True)
        return float(np.sum(c * (c - 1) / 2.0))

    s = _S(x, y)
    n0 = n * (n - 1) / 2.0
    denom = np.sqrt((n0 - _ties(x)) * (n0 - _ties(y)))
    tau_b = s / denom if denom > 0 else float("nan")

    rng = np.random.default_rng(seed)
    null = np.array([_S(x, rng.permutation(y)) for _ in range(n_perm)])
    k = int(np.sum(np.abs(null) >= abs(s)))
    p = (k + 1) / (n_perm + 1) if PERM_P_LAPLACE else k / n_perm
    return {"tau_b": float(tau_b), "p_perm": float(p), "S": float(s), "n": int(n),
            "x_has_ties": bool(np.unique(x).size < n)}


def _rank(v):
    return pd.Series(np.asarray(v, float)).rank().to_numpy()


def partial_corr_spearman(y, x, z) -> float:
    """Spearman partial correlation of y,x controlling for z.

    Standard rank-residual method: Pearson correlation of the residuals of
    rank(y)~rank(z) and rank(x)~rank(z).
    """
    from scipy.stats import pearsonr

    y, x, z = (np.asarray(v, float) for v in (y, x, z))
    ok = np.isfinite(y) & np.isfinite(x) & np.isfinite(z)
    if ok.sum() < 4:
        return float("nan")
    ry, rx, rz = _rank(y[ok]), _rank(x[ok]), _rank(z[ok])
    if np.unique(rz).size < 2:
        from scipy.stats import spearmanr
        return float(spearmanr(y[ok], x[ok]).statistic)

    def resid(a, b):
        return a - np.polyval(np.polyfit(b, a, 1), b)

    r_y, r_x = resid(ry, rz), resid(rx, rz)
    if np.std(r_y) == 0 or np.std(r_x) == 0:
        return float("nan")
    return float(pearsonr(r_y, r_x).statistic)


def partial_corr_quadratic(y, x, z) -> float:
    """As partial_corr_spearman but detrending on z with a quadratic (prespec §6)."""
    from scipy.stats import pearsonr

    y, x, z = (np.asarray(v, float) for v in (y, x, z))
    ok = np.isfinite(y) & np.isfinite(x) & np.isfinite(z)
    if ok.sum() < 5:
        return float("nan")
    ry, rx = _rank(y[ok]), _rank(x[ok])
    zc = (z[ok] - z[ok].mean()) / (z[ok].std() or 1.0)

    def resid(a):
        return a - np.polyval(np.polyfit(zc, a, 2), zc)

    r_y, r_x = resid(ry), resid(rx)
    if np.std(r_y) == 0 or np.std(r_x) == 0:
        return float("nan")
    return float(pearsonr(r_y, r_x).statistic)


def _validate_cohort(endpoints: pd.DataFrame, frozen: "list | None") -> None:
    required = {"time_interval_id", "snippet_start_s", PRIMARY_DOSE, "spd_consensus_rank"}
    missing = required - set(endpoints.columns)
    if missing:
        raise ValueError(f"endpoints frame missing columns: {sorted(missing)}")
    if endpoints["time_interval_id"].duplicated().any():
        raise ValueError("duplicate time_interval_id in endpoints frame")
    for e in ALL_ESTIMATORS:
        if f"exc_{e}" not in endpoints.columns:
            raise ValueError(f"endpoints frame missing estimator column exc_{e}")
        if not np.isfinite(endpoints[f"exc_{e}"].to_numpy(float)).all():
            raise ValueError(f"non-finite values in exc_{e}")
    if not np.isfinite(endpoints[PRIMARY_DOSE].to_numpy(float)).all():
        raise ValueError(f"non-finite {PRIMARY_DOSE}")
    if not np.isfinite(endpoints[PRIMARY_ENDPOINT].to_numpy(float)).all():
        raise ValueError(f"non-finite {PRIMARY_ENDPOINT}")
    if frozen is None:
        return
    if len(endpoints) != N_WINDOWS:
        raise ValueError(f"expected {N_WINDOWS} windows, got {len(endpoints)}")
    want = {w.time_interval_id for w in frozen}
    got = set(endpoints["time_interval_id"].astype(int))
    if want != got:
        raise ValueError(f"endpoint interval-id set != frozen set (missing {want - got}, extra {got - want})")
    by_iv = {w.time_interval_id: w for w in frozen}
    for _, row in endpoints.iterrows():
        w = by_iv[int(row["time_interval_id"])]
        if abs(float(row["snippet_start_s"]) - w.snippet_start_s) > 1e-3:
            raise ValueError(f"iv {w.time_interval_id}: snippet_start_s drift")
        if abs(float(row[PRIMARY_DOSE]) - w.exc_consensus_rank) > 1e-6:
            raise ValueError(f"iv {w.time_interval_id}: consensus dose != frozen value")


def _dose_rank_cross_estimator(endpoints: pd.DataFrame) -> dict:
    from scipy.stats import spearmanr

    out = {}
    for i, a in enumerate(ALL_ESTIMATORS):
        for b in ALL_ESTIMATORS[i + 1:]:
            out[f"{a}~{b}"] = float(spearmanr(endpoints[f"exc_{a}"], endpoints[f"exc_{b}"]).statistic)
    return out


def dose_response(endpoints: pd.DataFrame, *, frozen_windows: "list | Path | None" = None,
                  require_all_endpoints: bool = True) -> dict:
    """Prespec §6. One primary test with a full pre-committed decision; the rest
    supportive/descriptive; cross-estimator exposure-validity."""
    if isinstance(frozen_windows, (str, Path)):
        frozen_windows = load_frozen_list(Path(frozen_windows))
    _validate_cohort(endpoints, frozen_windows)
    present = [k for k in ENDPOINT_KEYS if k in endpoints.columns]
    if require_all_endpoints and set(present) != set(ENDPOINT_KEYS):
        raise ValueError(f"missing endpoint columns: {sorted(set(ENDPOINT_KEYS) - set(present))}")

    tcol = endpoints["snippet_start_s"].to_numpy(float)
    exc = endpoints[PRIMARY_DOSE].to_numpy(float)
    spd = endpoints["spd_consensus_rank"].to_numpy(float)
    est_exc = {e: endpoints[f"exc_{e}"].to_numpy(float) for e in ALL_ESTIMATORS}
    low_power = bool(("n_qualified" in endpoints.columns)
                     and (endpoints["n_qualified"] < LOW_POWER_QUALIFIED_UNITS).any())

    out: dict = {"schema": SCHEMA, "n_windows": int(len(endpoints)),
                 "primary_endpoint": PRIMARY_ENDPOINT, "primary_dose": PRIMARY_DOSE,
                 "low_power": low_power,
                 "dose_rank_cross_estimator": _dose_rank_cross_estimator(endpoints)}

    def _endpoint_block(y, expected_sign):
        from scipy.stats import spearmanr
        sp = spearman_ci(exc, y)
        rho = sp["rho"]
        primary_sign = int(np.sign(rho)) if np.isfinite(rho) else 0
        per_est_rho, per_est_sign = {}, {}
        for e, v in est_exc.items():
            r = spearmanr(v, y).statistic
            per_est_rho[e] = float(r) if np.isfinite(r) else float("nan")
            per_est_sign[e] = int(np.sign(r)) if np.isfinite(r) else 0

        concordant_signs = {per_est_sign[e] for e in CONCORDANT_ESTIMATORS}
        medicine_agrees = per_est_sign[SENSITIVITY_ESTIMATOR] == primary_sign and primary_sign != 0
        if not np.isfinite(rho) or abs(rho) < EXPOSURE_NEGLIGIBLE_RHO:
            validity = "no_association"
        elif len(concordant_signs) == 1 and primary_sign in concordant_signs and medicine_agrees:
            validity = "resolved"
        else:
            validity = "unresolved"

        lin = partial_corr_spearman(y, exc, tcol)
        quad = partial_corr_quadratic(y, exc, tcol)

        def _survives(pv):
            return bool(np.isfinite(pv) and np.sign(pv) == primary_sign
                        and abs(pv) >= PARTIAL_SURVIVES_MIN and primary_sign != 0)

        block = {
            "spearman_vs_excursion": sp,
            "spearman_vs_speed": spearman_ci(spd, y),
            "mann_kendall_excursion": mann_kendall(exc, y),
            "partial_session_time_linear": lin,
            "partial_session_time_quadratic": quad,
            "session_time_partial_given_dose": partial_corr_spearman(y, tcol, exc),
            "block_bootstrap_ci": block_bootstrap_spearman(exc, y, tcol),
            "per_estimator_excursion_rho": per_est_rho,
            "per_estimator_excursion_sign": per_est_sign,
            "exposure_validity": validity,
            "medicine_sign_agrees": bool(medicine_agrees),
            "linear_partial_survives": _survives(lin),
            "quadratic_partial_survives": _survives(quad),
        }
        if expected_sign is not None:
            block["matches_prereg_direction"] = bool(primary_sign == expected_sign)
        return block

    pb = _endpoint_block(endpoints[PRIMARY_ENDPOINT].to_numpy(float), expected_sign=-1)
    sp = pb["spearman_vs_excursion"]
    reasons = {
        "rho_negative": bool(np.isfinite(sp["rho"]) and sp["rho"] < 0),
        "ci_excludes_zero": bool(np.isfinite(sp["ci_lo"]) and np.isfinite(sp["ci_hi"])
                                 and sp["ci_lo"] < 0 and sp["ci_hi"] < 0),
        "linear_partial_survives": pb["linear_partial_survives"],
        "quadratic_partial_survives": pb["quadratic_partial_survives"],
        "exposure_resolved": pb["exposure_validity"] == "resolved",
        "not_low_power": not low_power,
    }
    pb["primary_decision_reasons"] = reasons
    pb["primary_supported"] = bool(all(reasons.values()))
    out["primary"] = pb

    out["supportive"] = {
        k: _endpoint_block(endpoints[k].to_numpy(float), SUPPORTIVE_EXPECTED_SIGN.get(k))
        for k in present if k != PRIMARY_ENDPOINT
    }
    moved = [k for k, b in out["supportive"].items() if b.get("matches_prereg_direction")]
    predicted = [k for k, sgn in SUPPORTIVE_EXPECTED_SIGN.items() if sgn is not None and k in present]
    out["concordance_summary"] = f"{len(moved)}/{len(predicted)} predicted-direction supportive endpoints"
    return out


# ======================================================================= #
# phase 2 -- build + sort one window (RESCUE graph, no motion correction)
# ======================================================================= #
def _spec_for(window: SelectedWindow):
    from testing.ladder_snippets import SnippetSpec

    return SnippetSpec(
        name=f"rigid_dose_iv{window.time_interval_id}",
        start_s=window.snippet_start_s,
        duration_s=WINDOW_S,
        channel_start=0,
        channel_count=FULL_PROBE_CHANNELS,
        split="development",
        selection_basis=(
            "luke_within_rigid_motion_dose_response: consensus rigid-excursion "
            f"rank {window.exc_consensus_rank:.3f} across "
            f"{'+'.join(CONCORDANT_ESTIMATORS)} motion estimates "
            "(input-side / estimator-side signature only)"
        ),
        axes={"motion_regime": f"exc_consensus_rank_{window.exc_consensus_rank:.2f}"},
    )


def _snippet_dir_for(spec) -> Path:
    from testing.ladder_snippets import snippet_root

    return snippet_root() / spec.directory_name


def run_window(window: SelectedWindow, *, l1_out_root: Path | None = None) -> dict:
    from testing.ladder_snippets import build_snippet
    from testing.ladder_l1 import l1_run

    spec = _spec_for(window)
    build_snippet(spec, ACCEPTED_RECORDING)
    snippet_dir = _snippet_dir_for(spec)
    result = l1_run(snippet_dir, out_root=l1_out_root)  # sorter=None -> frozen RESCUE
    return {"window": asdict(window), "snippet_dir": str(snippet_dir), "l1": result}


# ======================================================================= #
# phase 3 -- endpoints (prespec §5; all KSLabel-free, over ALL clusters)
# ======================================================================= #
def _refractory_fraction(train: np.ndarray, fs: float) -> float:
    if train.size < 2:
        return float("nan")
    isi_ms = np.diff(np.sort(train)) / fs * 1000.0
    return float((isi_ms < REFRACTORY_MS).mean())


def qualify_units(spike_times: np.ndarray, spike_clusters: np.ndarray,
                  template_amp_uv: dict[int, float], cluster_depth: dict[int, float],
                  fs: float, duration_s: float) -> dict:
    """E3 -- KSLabel-free qualification (prespec §5, frozen thresholds)."""
    edges = np.linspace(0.0, duration_s * fs, QUAL_PRESENCE_BINS + 1)
    per_unit, qualified = [], []
    for c in np.unique(spike_clusters):
        train = np.sort(spike_times[spike_clusters == c])
        amp = float(template_amp_uv.get(int(c), np.nan))
        rv = _refractory_fraction(train, fs)
        presence_bins = int((np.histogram(train, bins=edges)[0] > 0).sum())
        ok = (
            train.size >= QUAL_MIN_SPIKES
            and amp >= QUAL_AMPLITUDE_UV
            and np.isfinite(rv) and rv <= QUAL_RV_CEILING
            and presence_bins >= QUAL_PRESENCE_MIN_BINS
        )
        per_unit.append({
            "cluster": int(c), "n_spikes": int(train.size), "amp_uv": amp,
            "rv_fraction": rv, "presence_bins": presence_bins,
            "depth_um": float(cluster_depth.get(int(c), np.nan)),
            "rate_hz": float(train.size / duration_s) if duration_s else float("nan"),
            "qualified": bool(ok),
        })
        if ok:
            qualified.append(int(c))
    return {"qualified": qualified, "per_unit": per_unit}


def _similar_pairs(qualified: list[int], templates: np.ndarray,
                   depth: dict[int, float]) -> int:
    pairs = 0
    for i, a in enumerate(qualified):
        for b in qualified[i + 1:]:
            if not (np.isfinite(depth.get(a, np.nan)) and np.isfinite(depth.get(b, np.nan))):
                continue
            if abs(depth[a] - depth[b]) > SIMILAR_DEPTH_UM:
                continue
            ta, tb = templates[a].ravel(), templates[b].ravel()
            denom = np.linalg.norm(ta) * np.linalg.norm(tb)
            if denom > 0 and float(np.dot(ta, tb) / denom) >= SIMILAR_COSINE:
                pairs += 1
    return pairs


def _fragmentation(qualified: list[int], trains: dict[int, np.ndarray],
                   depth: dict[int, float], fs: float) -> dict:
    from testing.ladder_score import coincident_mask

    tol = int(round(FRAG_TOL_MS * fs / 1000.0))
    flagged: set[int] = set()
    pairs = 0
    for i, a in enumerate(qualified):
        for b in qualified[i + 1:]:
            if not (np.isfinite(depth.get(a, np.nan)) and np.isfinite(depth.get(b, np.nan))):
                continue
            if abs(depth[a] - depth[b]) > FRAG_DEPTH_UM:
                continue
            ta, tb = trains[a], trains[b]
            smaller = ta if ta.size <= tb.size else tb
            larger = tb if ta.size <= tb.size else ta
            co = coincident_mask(smaller, larger, tol)
            if co.size and co.mean() > FRAG_COINCIDENCE_FRAC:
                continue
            union = np.sort(np.concatenate([ta, tb]))
            if _refractory_fraction(union, fs) <= QUAL_RV_CEILING:
                flagged |= {a, b}
                pairs += 1
    n = len(qualified)
    return {"E8_fragmentation_index": float(len(flagged) / n) if n else float("nan"),
            "n_fragment_pairs": pairs}


def _frozen_conditioned(snippet):
    """The frozen 300-6000 Hz bandpass + whole-probe common-median reference
    (prespec §5). Fails closed -- callers must not substitute other units."""
    from spikeinterface.preprocessing import bandpass_filter, common_reference

    rec = bandpass_filter(snippet.recording(), freq_min=300.0, freq_max=6000.0, dtype="float32")
    return common_reference(rec, reference="global", operator="median", dtype="float32")


def _dewhitened_peak_channel(t_row: np.ndarray, winv: "np.ndarray | None") -> int:
    dw = t_row @ winv if winv is not None else t_row
    return int(np.argmax(np.abs(dw).max(axis=0)))


def _sta(band, samples: np.ndarray, channels, fs: float, cap: int = 400) -> "np.ndarray | None":
    win = int(round(1.3e-3 * fs))
    n = band.get_num_samples()
    take = np.sort(samples[(samples - win >= 0) & (samples + win < n)])
    if take.size < 30:
        return None
    if take.size > cap:
        take = take[np.linspace(0, take.size - 1, cap).astype(int)]
    ids = [band.channel_ids[c] for c in channels]
    acc = np.zeros((2 * win, len(ids)))
    for x in take:
        acc += band.get_traces(start_frame=int(x) - win, end_frame=int(x) + win, channel_ids=ids)
    return acc / take.size


def _cluster_amp_and_depth(cl, st, templates, winv, geom, band, gain, fs):
    """Depth = y of the de-whitened template peak channel. Amplitude = bandpass
    STA peak on +/- STA_NEIGHBOURHOOD_CH channels around that peak, x gain -> uV.
    NaN amplitude (fails the E3 gate) when the STA is not estimable."""
    n_ch = geom.shape[0]
    depth, amp, peak_ch = {}, {}, {}
    for c in np.unique(cl):
        c = int(c)
        if templates is None or c >= templates.shape[0]:
            depth[c] = amp[c] = float("nan")
            continue
        pc = _dewhitened_peak_channel(np.asarray(templates[c], float), winv)
        peak_ch[c] = pc
        depth[c] = float(geom[pc, 1])
    if band is None:
        for c in peak_ch:
            amp[c] = float("nan")
        return amp, depth, peak_ch
    for c, pc in peak_ch.items():
        lo, hi = max(0, pc - STA_NEIGHBOURHOOD_CH), min(n_ch, pc + STA_NEIGHBOURHOOD_CH + 1)
        sta = _sta(band, np.sort(st[cl == c]), range(lo, hi), fs)
        amp[c] = float(np.abs(sta).max() * gain) if sta is not None else float("nan")
    return amp, depth, peak_ch


def _waveform_stability_median(qualified, trains, peak_ch, band, geom, fs) -> float:
    """E6 -- cosine of first-half vs second-half mean waveform on ONE frozen
    channel set per unit (top 8 |amplitude| channels of the full-window STA)."""
    if band is None:
        return float("nan")
    half = None
    vals = []
    n_ch = geom.shape[0]
    for c in qualified:
        tr = trains[c]
        if half is None:
            half = tr.max() / 2.0 if tr.size else 0.0
        a, b = tr[tr < half], tr[tr >= half]
        if a.size < E6_MIN_SPIKES_PER_HALF or b.size < E6_MIN_SPIKES_PER_HALF:
            continue
        pc = peak_ch.get(c, 0)
        lo, hi = max(0, pc - 8), min(n_ch, pc + 9)
        full = _sta(band, tr, range(lo, hi), fs)
        if full is None:
            continue
        keep = np.argsort(np.abs(full).max(axis=0))[::-1][:8]
        chan = [list(range(lo, hi))[k] for k in keep]
        wa, wb = _sta(band, a, chan, fs), _sta(band, b, chan, fs)
        if wa is None or wb is None:
            continue
        vals.append(_cosine(wa, wb))
    return float(np.median(vals)) if vals else float("nan")


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 0 else float("nan")


def _context_event_metrics(band, qualified_trains, qualified_depths, geom, probe_mm, fs):
    """C1 detected-event density /mm/s on the frozen conditioning; C2 fraction of
    detected events within +/-0.5 ms AND <= ASSIGN_TOL_UM of an E3-qualified unit."""
    from testing.ladder_score import coincident_mask
    from spikeinterface.sortingcomponents.peak_detection import detect_peaks

    if band is None:
        return float("nan"), float("nan")
    peaks = detect_peaks(band, method="locally_exclusive", peak_sign="both",
                         detect_threshold=4.0, n_jobs=8, progress_bar=False)
    p_s = np.asarray(peaks["sample_index"], np.int64)
    p_ch = np.asarray(peaks["channel_index"], np.int64)
    dur = band.get_num_samples() / fs
    c1 = float(p_s.size / dur / probe_mm) if probe_mm else float("nan")
    if p_s.size == 0 or not qualified_trains:
        return c1, float("nan")
    p_depth = geom[p_ch, 1]
    near = np.zeros(p_s.size, dtype=bool)
    tol = int(round(ASSIGN_TOL_MS * fs / 1000.0))
    for c, tr in qualified_trains.items():
        ud = qualified_depths.get(c, np.nan)
        if not np.isfinite(ud):
            continue
        near |= coincident_mask(p_s, np.sort(tr), tol) & (np.abs(p_depth - ud) <= ASSIGN_TOL_UM)
    return c1, float(near.mean())


def _depth_stratified_e3(qualified, depth, geom):
    y = geom[:, 1]
    edges = np.quantile(y, [0.0, 1 / 3, 2 / 3, 1.0])
    span_mm = np.diff(edges) / 1000.0
    out = {}
    for name, lo, hi, mm in zip(("shallow", "mid", "deep"), edges[:-1], edges[1:], span_mm):
        n = sum(1 for c in qualified if lo <= depth.get(c, np.nan) < hi
                or (hi == edges[-1] and depth.get(c, np.nan) == hi))
        out[f"E3_{name}_per_mm"] = float(n / mm) if mm > 0 else float("nan")
    return out


def window_endpoints(run_record: dict) -> dict:
    """E3-E8 + C1/C2 for one window (prespec §5). Requires the built snippet + curated sort."""
    from testing.ladder_snippets import load_snippet

    snippet = load_snippet(run_record["snippet_dir"])
    fs, dur = snippet.fs, snippet.duration_s
    gain = float(snippet.gain_uv_per_count)
    curated = Path(run_record["l1"]["score"]["sorter_output"])

    st = np.load(curated / "spike_times.npy").reshape(-1).astype(np.int64)
    cl = np.load(curated / "spike_clusters.npy").reshape(-1).astype(np.int64)
    templates = np.load(curated / "templates.npy") if (curated / "templates.npy").exists() else None
    winv = np.load(curated / "whitening_mat_inv.npy") if (curated / "whitening_mat_inv.npy").exists() else None
    geom = np.load(snippet.dir / "channel_positions.npy")
    probe_mm = (float(geom[:, 1].ptp()) + _site_pitch(geom)) / 1000.0

    band = _frozen_conditioned(snippet)  # fail closed: no template-unit fallback

    amp_uv, depth, peak_ch = _cluster_amp_and_depth(cl, st, templates, winv, geom, band, gain, fs)
    trains = {int(c): np.sort(st[cl == c]) for c in np.unique(cl)}
    qual = qualify_units(st, cl, amp_uv, depth, fs, dur)
    q = qual["qualified"]
    by_c = {u["cluster"]: u for u in qual["per_unit"]}

    e4 = float(np.nanmedian([by_c[c]["rv_fraction"] for c in q])) if q else float("nan")
    e5 = float(_similar_pairs(q, templates, depth) / len(q)) if (q and templates is not None) else float("nan")
    e6 = _waveform_stability_median(q, {c: trains[c] for c in q}, peak_ch, band, geom, fs)
    rates = np.array([by_c[c]["rate_hz"] for c in q], float)
    e7 = float(np.median(rates)) if rates.size else float("nan")
    e7_iqr = float(np.subtract(*np.percentile(rates, [75, 25]))) if rates.size else float("nan")
    e8 = _fragmentation(q, {c: trains[c] for c in q}, depth, fs)
    c1, c2 = _context_event_metrics(band, {c: trains[c] for c in q},
                                    {c: depth.get(c, np.nan) for c in q}, geom, probe_mm, fs)

    row = {
        "time_interval_id": run_record["window"]["time_interval_id"],
        "snippet_start_s": run_record["window"]["snippet_start_s"],
        "exc_consensus_rank": run_record["window"]["exc_consensus_rank"],
        "spd_consensus_rank": run_record["window"]["spd_consensus_rank"],
        **{f"exc_{e}": run_record["window"]["exc_by_estimator"][e] for e in ALL_ESTIMATORS},
        **{f"spd_{e}": run_record["window"]["spd_by_estimator"][e] for e in ALL_ESTIMATORS},
        "n_qualified": len(q),
        "E3_qualified_units_per_mm": float(len(q) / probe_mm) if probe_mm else float("nan"),
        "E4_refractory_burden_median": e4,
        "E5_similar_pairs_per_qualified_unit": e5,
        "E6_waveform_stability_median": e6,
        "E7_qualified_rate_hz_median": e7,
        "E7_qualified_rate_hz_iqr": e7_iqr,
        "E8_fragmentation_index": e8["E8_fragmentation_index"],
        "C1_detected_events_per_mm_per_s": c1,
        "C2_fraction_events_near_qualified": c2,
    }
    row.update(_depth_stratified_e3(q, depth, geom))
    return row


def _site_pitch(geom: np.ndarray) -> float:
    ys = np.unique(geom[:, 1])
    return float(np.min(np.diff(ys))) if ys.size > 1 else 20.0


# ======================================================================= #
# CLI
# ======================================================================= #
def _endpoints_frame(records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(records)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--pilot", action="store_true", help="build+sort ONE window; report cost")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--endpoints", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--csv", type=Path, default=INCREMENT1_CSV)
    ap.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.select:
        windows = select_windows(args.csv)
        write_frozen_list(windows, args.csv)
        print(f"froze {len(windows)} windows -> {FROZEN_WINDOWS}")
        for w in windows:
            print(f"  iv {w.time_interval_id:5d}  start {w.snippet_start_s:8.1f}s  "
                  f"exc_rank {w.exc_consensus_rank:.3f}  "
                  f"ks={w.exc_by_estimator['ks-motion']:.1f} dredge={w.exc_by_estimator['dredge-motion']:.1f} "
                  f"med={w.exc_by_estimator['medicine']:.1f}")
        return 0

    if args.pilot:
        import time
        windows = load_frozen_list(csv_path=args.csv)
        w = sorted(windows, key=lambda z: z.exc_consensus_rank)[len(windows) // 2]
        print(f"pilot: interval {w.time_interval_id}, exc_rank {w.exc_consensus_rank:.3f}")
        t0 = time.time()
        rec = run_window(w)
        dt = time.time() - t0
        sd = Path(rec["snippet_dir"])
        disk = sum(f.stat().st_size for f in sd.rglob("*") if f.is_file()) / 1e9
        print(f"  wall {dt/60:.1f} min   snippet {disk:.2f} GB   -> extrapolated batch "
              f"{dt*N_WINDOWS/3600:.1f} h, {disk*N_WINDOWS:.0f} GB (snippets only)")
        (args.out_dir / "pilot.json").write_text(json.dumps(
            {"interval": w.time_interval_id, "wall_s": dt, "snippet_gb": disk, "record": rec},
            indent=2, default=str))
        return 0

    if args.run:
        rows = [run_window(w) for w in load_frozen_list(csv_path=args.csv)]
        (args.out_dir / "run_index.json").write_text(json.dumps(rows, indent=2, default=str))
        print(f"ran {len(rows)} windows")
        return 0

    if args.endpoints:
        run_index = json.loads((args.out_dir / "run_index.json").read_text())
        recs = [window_endpoints(r) for r in run_index]
        _endpoints_frame(recs).to_csv(args.out_dir / "window_endpoints.csv", index=False)
        print(f"endpoints for {len(recs)} windows -> window_endpoints.csv")
        return 0

    if args.analyze:
        ep = pd.read_csv(args.out_dir / "window_endpoints.csv")
        result = dose_response(ep, frozen_windows=FROZEN_WINDOWS)
        (args.out_dir / "dose_response.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return 0

    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
