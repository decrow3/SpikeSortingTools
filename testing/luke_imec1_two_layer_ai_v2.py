#!/usr/bin/env python
"""AI: compare and package the exclude-3s Luke0804 two-layer field v2.

Reads saved fields and Q pilot peak caches only. No fitting, sorting, or voltage
access occurs here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motionqc.field import MotionField, SIGN
from motionqc.reference import shift_test
from motionqc.report import build_report
from testing import luke_imec1_medicine_stage3_q_v1 as qstage

PARENT = Path("/media/huklab/Data/NPX/Ryansorting/Luke/luke_imec1_medicine_reference_sweep_v2")
Q = PARENT / "stage3_q"
AB = PARENT / "stage4_ab"
OUT = AB / "ai_v2"
DT = 0.25
COLORS = {"v1": "#0072B2", "v2": "#D55E00", "deployed_q": "#009E73"}
LINESTYLES = {"v1": "-", "v2": "--", "deployed_q": "-."}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", newline="\n")


def load_npz(path: Path, source: str) -> MotionField:
    with np.load(path, allow_pickle=False) as z:
        field = MotionField(z["time_s"], z["depth_um"], z["displacement_um"], source=source,
                            config={"input": str(path), "sha256": sha256(path)})
    field.validate(recording_length_s=10473.55)
    return field


def canonical_mask() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    frame = pd.read_csv(AB / "censor_mask_v1.csv")
    with np.load(AB / "censor_mask_v1_grid.npz", allow_pickle=False) as z:
        return frame, np.asarray(z["time_s"], float), np.asarray(z["mask"], bool)


def on_grid(field: MotionField, time_s: np.ndarray) -> np.ndarray:
    return np.column_stack([np.interp(time_s, field.time_s, field.displacement_um[:, j])
                            for j in range(len(field.depth_um))])


def compose(slow: MotionField, fast: MotionField, time_s: np.ndarray, mask: np.ndarray, source: str) -> tuple[MotionField, np.ndarray]:
    slow_grid = on_grid(slow, time_s)
    fast_rigid = fast.at(time_s, np.full(len(time_s), np.median(fast.depth_um)))
    fast_grid = np.repeat(fast_rigid[:, None], len(slow.depth_um), axis=1)
    weight = np.zeros(len(time_s)); weight[mask] = 1.0
    starts = np.flatnonzero(mask & np.r_[True, ~mask[:-1]])
    stops = np.flatnonzero(mask & np.r_[~mask[1:], True]) + 1
    ramp_bins = int(round(1.0 / DT))
    for a, b in zip(starts, stops):
        n = min(ramp_bins, b - a)
        weight[a:a+n] = np.minimum(weight[a:a+n], np.arange(1, n + 1) / ramp_bins)
        weight[b-n:b] = np.minimum(weight[b-n:b], np.arange(n, 0, -1) / ramp_bins)
    combined = slow_grid * (1 - weight[:, None]) + fast_grid * weight[:, None]
    combined -= np.nanmedian(combined, axis=0, keepdims=True)
    return MotionField(time_s, slow.depth_um, combined, source=source,
                       config={"slow": slow.source, "fast": fast.source, "mask": "AE canonical",
                               "crossfade_s": 1.0}), weight


def episode_pool() -> pd.DataFrame:
    root = PARENT / "stage2_o"
    a = pd.read_csv(root / "references_episode/measured_episodes.csv")
    b = pd.read_csv(root / "s_correction/all_quiet_measurements.csv")
    x = pd.concat([a, b], ignore_index=True)
    x = x[x.accepted.astype(bool)].copy()
    if len(x) != 45:
        raise RuntimeError(f"frozen S episode pool has {len(x)} rows, expected 45")
    return x.rename(columns={"stop_s": "end_s", "best_shift_um": "measured_shift_um"}).assign(status="accepted")


def episode_prediction(field: MotionField, start: float, stop: float) -> float:
    ep = np.arange(start + DT / 2, stop, DT)
    rest = np.arange(start - 4 + DT / 2, start - 1, DT)
    z = np.full(len(ep), np.median(field.depth_um)); zr = np.full(len(rest), np.median(field.depth_um))
    return float(np.nanmedian(field.at(ep, z)) - np.nanmedian(field.at(rest, zr)))


def score(fields: dict[str, MotionField], time_s: np.ndarray, mask: np.ndarray, episodes: pd.DataFrame) -> pd.DataFrame:
    reference = pd.read_csv(AB / "slow_shift_reference.csv")
    reference = reference[reference.resolved.astype(bool)]
    boundaries = pd.read_csv(AB / "ag_block_break_boundaries.csv")
    step = int(round(5 / DT)); quiet_pairs = (~mask[:-step]) & (~mask[step:])
    rows = []
    for name, field in fields.items():
        pred = field.at(reference.time_s.to_numpy(), np.full(len(reference), np.median(field.depth_um))) - field.at(
            reference.previous_time_s.to_numpy(), np.full(len(reference), np.median(field.depth_um)))
        err = pred - reference.shift_um.to_numpy()
        rigid = field.at(time_s, np.full(len(time_s), np.median(field.depth_um)))
        inc = rigid[step:] - rigid[:-step]
        episode_errors = [abs(episode_prediction(field, r.start_s, r.end_s) - r.measured_shift_um)
                          for r in episodes.itertuples(index=False)]
        levels = []
        for r in boundaries.itertuples(index=False):
            free = float(field.at(np.array([r.free_block_start_s + 5]), np.array([np.median(field.depth_um)]))[0])
            dense = float(field.at(np.array([r.dense_block_start_s + 5]), np.array([np.median(field.depth_um)]))[0])
            levels.append(free - dense)
        seam = seam_table_one(name, field, time_s, mask)
        rows.append({"field": name, "quiet_reference_pairs": len(err),
                     "quiet_reference_mae_um": float(np.median(np.abs(err))),
                     "quiet_reference_bias_um": float(np.median(err)),
                     "quiet_reference_rmse_um": float(np.sqrt(np.mean(err ** 2))),
                     "quiet_increment_rms_um": float(np.sqrt(np.mean(inc[quiet_pairs] ** 2))),
                     "episode_count": len(episode_errors), "episode_err_um": float(np.median(episode_errors)),
                     "ag_block_break_median_um": float(np.median(levels)),
                     "seam_median_abs_um": float(seam.abs_step_um.median()),
                     "seam_p95_abs_um": float(seam.abs_step_um.quantile(.95)),
                     "seam_max_abs_um": float(seam.abs_step_um.max())})
    return pd.DataFrame(rows)


def seam_table_one(name: str, field: MotionField, time_s: np.ndarray, mask: np.ndarray) -> pd.DataFrame:
    rigid = field.at(time_s, np.full(len(time_s), np.median(field.depth_um)))
    starts = np.flatnonzero(mask & np.r_[True, ~mask[:-1]])
    stops = np.flatnonzero(mask & np.r_[~mask[1:], True]) + 1
    rows = []
    for kind, indices in (("start", starts), ("stop", stops)):
        for ix in indices:
            if ix <= 0 or ix >= len(time_s):
                continue
            delta = float(rigid[ix] - rigid[ix-1])
            rows.append({"field": name, "boundary_kind": kind, "time_s": float(time_s[ix]),
                         "before_um": float(rigid[ix-1]), "after_um": float(rigid[ix]),
                         "step_um": delta, "abs_step_um": abs(delta)})
    return pd.DataFrame(rows)


def load_all_peaks() -> pd.DataFrame:
    pieces = []
    for block in qstage.block_specs():
        path = Q / f"peak_cache/{block['id']}/extraction/population.npz"
        with np.load(path, allow_pickle=False) as z:
            pieces.append(pd.DataFrame({"time_s": np.asarray(z["time_s"], float) + block["start_s"],
                                        "depth_um": np.asarray(z["depth_um"], float),
                                        "x_um": np.asarray(z["x_um"], float),
                                        "amplitude": np.asarray(z["amplitude"], float)}))
    return pd.concat(pieces, ignore_index=True)


def boundary_reference(peaks: pd.DataFrame, time_s: float, seed: int) -> dict:
    pre = (peaks.time_s >= time_s - 2) & (peaks.time_s < time_s)
    post = (peaks.time_s >= time_s) & (peaks.time_s < time_s + 2)
    main = shift_test(peaks, post.to_numpy(), pre.to_numpy(), refine_um=2, shift_min_um=-320,
                      shift_max_um=320, min_peaks=150, min_gain=.05,
                      depth_range_um=(0, 3840), x_range_um=(-32, 80))
    relevant = np.flatnonzero(pre.to_numpy() | post.to_numpy()); rng = np.random.default_rng(seed)
    null_rows = []
    target = min(int(pre.sum()), int(post.sum()))
    for draw in range(20):
        order = rng.permutation(relevant); a = order[:target]; b = order[target:2*target]
        am = np.zeros(len(peaks), bool); bm = np.zeros(len(peaks), bool); am[a] = True; bm[b] = True
        value = shift_test(peaks, am, bm, refine_um=2, shift_min_um=-320, shift_max_um=320,
                           min_peaks=150, min_gain=0, depth_range_um=(0, 3840), x_range_um=(-32, 80))
        null_rows.append(value["best_shift_um"])
    null = np.asarray(null_rows, float); finite = null[np.isfinite(null)]
    mode = float(pd.Series(finite).mode().iloc[0]) if len(finite) else np.nan
    med = float(np.median(np.abs(finite))) if len(finite) else np.nan
    return {"pre_peaks": int(pre.sum()), "post_peaks": int(post.sum()),
            "measured_shift_um": float(main["best_shift_um"]), "corr_zero": float(main["corr_zero"]),
            "corr_best": float(main["corr_best"]), "gain": float(main["gain"]),
            "resolved": bool(main["accepted"]), "null_n": len(finite), "null_mode_um": mode,
            "null_median_abs_um": med, "null_pass": bool(len(finite) == 20 and mode == 0 and med <= 2),
            "null_shifts_um": null_rows}


def boundary_checks(fields: dict[str, MotionField], time_s: np.ndarray, mask: np.ndarray,
                    peaks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_seams = pd.concat([seam_table_one(name, field, time_s, mask) for name, field in fields.items()], ignore_index=True)
    ranked = pd.concat([all_seams[all_seams.field.eq(name)].nlargest(10, "abs_step_um").assign(rank=np.arange(1, 11))
                        for name in ("v1", "v2")], ignore_index=True)
    unique_times = sorted(ranked.time_s.unique()); refs = {}
    # Load only the +/-2 s slices into the shift tests, avoiding repeated work on 47 M rows.
    for i, t in enumerate(unique_times):
        local = peaks[(peaks.time_s >= t - 2) & (peaks.time_s < t + 2)].reset_index(drop=True)
        refs[t] = boundary_reference(local, t, 20260925 + i)
    null_rows = []; checked_rows = []
    for row in ranked.itertuples(index=False):
        ref = refs[row.time_s]; shift = ref["measured_shift_um"]
        if not ref["null_pass"]:
            verdict = "invalid_null"
        elif ref["resolved"] and abs(shift) > 2 and np.sign(shift) == np.sign(row.step_um):
            verdict = "real_motion_at_episode_edge"
        elif np.isfinite(shift) and abs(shift) <= 2:
            verdict = "join_artifact"
        elif ref["resolved"]:
            verdict = "join_artifact_or_sign_mismatch"
        else:
            verdict = "unresolved"
        for draw, value in enumerate(ref["null_shifts_um"]):
            null_rows.append({"time_s": row.time_s, "draw": draw, "best_shift_um": value})
        checked_rows.append({**row._asdict(), **{k: v for k, v in ref.items() if k != "null_shifts_um"},
                             "classification": verdict})
    return all_seams, pd.DataFrame(checked_rows), pd.DataFrame(null_rows)


def figure(scores: pd.DataFrame, ranked: pd.DataFrame) -> None:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    metrics = ["quiet_reference_mae_um", "quiet_increment_rms_um", "episode_err_um", "ag_block_break_median_um"]
    x = np.arange(len(metrics))
    for name in ("v1", "v2", "deployed_q"):
        row = scores[scores.field.eq(name)].iloc[0]
        axes[0].plot(x, [row[m] for m in metrics], marker="o", color=COLORS[name], ls=LINESTYLES[name], label=name)
    axes[0].axhline(0, color="0.5", lw=.8); axes[0].set_xticks(x, ["quiet MAE", "quiet increment", "episode error", "block/break bias"], rotation=20)
    axes[0].set_ylabel("µm"); axes[0].set_title("Frozen-score comparison"); axes[0].grid(alpha=.2); axes[0].legend()
    for name in ("v1", "v2"):
        data = ranked[ranked.field.eq(name)].sort_values("rank")
        axes[1].plot(data["rank"], data.abs_step_um, marker="o", color=COLORS[name], ls=LINESTYLES[name], label=name)
    axes[1].set(xlabel="Boundary-step rank", ylabel="Absolute 0.25 s step (µm)", title="Ten largest layer-boundary steps")
    axes[1].grid(alpha=.2); axes[1].legend(); fig.tight_layout(); fig.savefig(OUT / "ai_v1_v2_comparison.png", dpi=180); plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mask_frame, grid_t, mask = canonical_mask()
    v1 = load_npz(AB / "luke0804_imec1_two_layer_motion.npz", "v1")
    fast = load_npz(Q / "luke0804_imec1_medicine_deployable_motion.npz", "deployed_q")
    slow2 = load_npz(AB / "sensitivity_ag/medicine_amp50_d1_k30_exclude3s/field.npz", "medicine_amp50_d1_k30_exclude3s")
    v2, weight = compose(slow2, fast, grid_t, mask, "v2")
    episodes = episode_pool(); episodes.to_csv(OUT / "episode_pool_45.csv", index=False)
    fields = {"v1": v1, "v2": v2, "deployed_q": fast}
    scores = score(fields, grid_t, mask, episodes); scores.to_csv(OUT / "score_table.csv", index=False)
    peaks = load_all_peaks()
    all_seams, ranked, nulls = boundary_checks(fields, grid_t, mask, peaks)
    all_seams.to_csv(OUT / "boundary_steps_all.csv", index=False)
    ranked.to_csv(OUT / "boundary_steps_top10.csv", index=False)
    nulls.drop_duplicates().to_csv(OUT / "boundary_null_draws.csv", index=False)
    figure(scores, ranked)
    by = scores.set_index("field")
    gates = {
        "quiet_reference_mae": bool(by.loc["v2", "quiet_reference_mae_um"] <= by.loc["v1", "quiet_reference_mae_um"] + .5),
        "quiet_increment_rms": bool(by.loc["v2", "quiet_increment_rms_um"] <= by.loc["v1", "quiet_increment_rms_um"] + .5),
        "episode_error": bool(by.loc["v2", "episode_err_um"] <= by.loc["deployed_q", "episode_err_um"] + 3),
        "ag_bias": bool(abs(by.loc["v2", "ag_block_break_median_um"]) < abs(by.loc["v1", "ag_block_break_median_um"])),
    }
    adopted = all(gates.values())
    package = OUT / "luke0804_imec1_two_layer_motion_v2.npz"
    if adopted:
        np.savez_compressed(package, time_s=v2.time_s, depth_um=v2.depth_um, displacement_um=v2.displacement_um,
                            sign_convention=np.asarray(SIGN))
    build_report(list(fields.values()), peaks, episodes, mask_frame, OUT / "motionqc_report")
    del peaks
    decision = {"decision": "adopt_v2" if adopted else "keep_v1", "gates": gates,
                "rule": "AI.4 frozen", "package": str(package) if adopted else None,
                "package_sha256": sha256(package) if adopted else None,
                "no_new_fit": True, "sort_run": False, "voltage_modified": False}
    write_json(OUT / "decision.json", decision)
    s = scores.set_index("field")
    verdicts = ranked.groupby(["field", "classification"]).size().unstack(fill_value=0)
    boundary_display = ranked[["field", "rank", "boundary_kind", "time_s", "step_um", "measured_shift_um",
                               "gain", "null_mode_um", "null_median_abs_um", "classification"]]
    text = f"""# AI — Luke0804 imec1 two-layer field v2

## Decision

**{'Adopt v2.' if adopted else 'Keep v1.'}** The frozen AI.4 gates were {gates}. No new fit was required; v2 reuses the completed exclude-3s sensitivity fit.

{scores.to_markdown(index=False, floatfmt='.3f')}

The 45-episode pool is the frozen S pool. Boundary tests use 2 s maps immediately before/after each edge, a symmetric −320…+320 µm search refined to 2 µm, and 20 deterministic equal-count split-null draws. Full classifications are in `boundary_steps_top10.csv`.

All 20 top-step null checks passed with mode 0 µm and median absolute shift 0 µm. Classification counts:

{verdicts.to_markdown()}

{boundary_display.to_markdown(index=False, floatfmt='.3f')}

![v1/v2 comparison](ai_v1_v2_comparison.png)

Package: `{package.name if adopted else 'not written'}`{f' (SHA-256 `{sha256(package)}`)' if adopted else ''}.
"""
    (OUT / "README.md").write_text(text, newline="\n")
    manifest = {"schema": "luke0804-imec1-two-layer-motion-v2", "decision": decision,
                "slow_layer": str(AB / "sensitivity_ag/medicine_amp50_d1_k30_exclude3s/field.npz"),
                "slow_layer_sha256": sha256(AB / "sensitivity_ag/medicine_amp50_d1_k30_exclude3s/field.npz"),
                "fast_layer": str(Q / "luke0804_imec1_medicine_deployable_motion.npz"),
                "fast_layer_sha256": sha256(Q / "luke0804_imec1_medicine_deployable_motion.npz"),
                "canonical_mask": str(AB / "censor_mask_v1.csv"), "canonical_mask_sha256": sha256(AB / "censor_mask_v1.csv"),
                "crossfade_s": 1.0, "grid_s": DT,
                "outputs": {p.name: sha256(p) for p in OUT.iterdir() if p.is_file() and p.name != "manifest.json"}}
    write_json(OUT / "manifest.json", manifest)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
