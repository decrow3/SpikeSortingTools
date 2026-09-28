#!/usr/bin/env python
"""Direct cached replication audit across independently seeded imec1 banks.

Candidate choice uses only seed evidence: the original bank's sole localized
candidate p06_f000 and the 300--320 s bank's top globally distinctive,
seed-qualified motion-scale candidate p09_f020. Motion estimates are not read.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from testing.luke_imec1_dots_lighthouse_direct_check import MOTION_WINDOWS
from testing.luke_imec1_sorterfree_phase_audit_v1 import lagged_overlap_cosine

ROOT = Path(__file__).resolve().parents[1]
FIRST = ROOT / "testing/outputs/luke_imec1_dots_sorterfree_waveform_discovery_v3"
SECOND = ROOT / "testing/outputs/luke_imec1_dots_sorterfree_waveform_discovery_v3_s300"
OUTPUT = ROOT / "testing/outputs/luke_imec1_cross_seed_replication_v1"
PRIMARY = (("s036", "p06_f000"), ("s300", "p09_f020"))
DRAW_SEED = 20260912
N_DRAWS = 2000
PAIR_TOLERANCE_S = 0.25
SAMPLE_TOLERANCE_S = 1 / 29999.835983263598 + 1e-9


def reciprocal_pairs(a: pd.DataFrame, b: pd.DataFrame, tolerance: float = PAIR_TOLERANCE_S) -> pd.DataFrame:
    if not len(a) or not len(b):
        return pd.DataFrame(columns=["time_a", "time_b", "relative_a_um", "relative_b_um"])
    ta, tb = a.time_s.to_numpy(), b.time_s.to_numpy()
    nearest_b = np.argmin(np.abs(ta[:, None] - tb[None, :]), axis=1)
    nearest_a = np.argmin(np.abs(tb[:, None] - ta[None, :]), axis=1)
    rows = []
    for i, j in enumerate(nearest_b):
        if nearest_a[j] == i and abs(ta[i] - tb[j]) <= tolerance:
            rows.append({"time_a": ta[i], "time_b": tb[j], "relative_a_um": a.relative_um.iloc[i], "relative_b_um": b.relative_um.iloc[j]})
    return pd.DataFrame(rows)


def shift_within_windows(table: pd.DataFrame, offsets: np.ndarray) -> pd.DataFrame:
    out = table.copy()
    shifted = out.time_s.to_numpy().copy()
    for offset, (start, stop) in zip(offsets, MOTION_WINDOWS):
        inside = (shifted >= start) & (shifted < stop)
        shifted[inside] = start + np.mod(shifted[inside] - start + offset, stop - start)
    out["time_s"] = shifted
    return out.sort_values("time_s").reset_index(drop=True)


def pair_metrics(pairs: pd.DataFrame) -> dict:
    if len(pairs) < 4:
        return {"pairs": len(pairs), "correlation": np.nan, "centered_median_absolute_difference_um": np.nan,
                "same_direction_fraction": np.nan}
    a = pairs.relative_a_um.to_numpy(); b = pairs.relative_b_um.to_numpy()
    return {"pairs": len(pairs), "correlation": float(np.corrcoef(a, b)[0, 1]),
            "centered_median_absolute_difference_um": float(np.median(np.abs((a - np.median(a)) - (b - np.median(b))))),
            "same_direction_fraction": float(np.mean(np.sign(a) == np.sign(b)))}


def load_bank(path: Path, tag: str):
    families = pd.read_csv(path / "families_after_depth_reveal.csv").set_index("family_id")
    events = pd.read_csv(path / "heldout_events.csv")
    saved = np.load(path / "family_templates.npz")
    templates = saved["waveforms"]
    ids = pd.read_csv(path / "families_after_depth_reveal.csv").family_id.tolist()
    return {"path": path, "tag": tag, "families": families, "events": events, "templates": templates,
            "relative_geometry": saved["relative_geometry"], "template_index": {fid: i for i, fid in enumerate(ids)}}


def candidate(bank, family_id: str) -> pd.DataFrame:
    q = bank["events"][(bank["events"].family_id == family_id) & (bank["events"].status == "strict")].copy()
    q["relative_um"] = q.waveform_centroid_um - bank["families"].loc[family_id].seed_depth_median_um
    return q.sort_values("time_s").reset_index(drop=True)


def exact_overlap_count(a: pd.DataFrame, b: pd.DataFrame) -> int:
    if not len(a) or not len(b): return 0
    times = np.sort(b.time_s.to_numpy()); query = a.time_s.to_numpy(); i = np.searchsorted(times, query)
    lo = np.abs(query - times[np.clip(i - 1, 0, len(times) - 1)])
    hi = np.abs(query - times[np.clip(i, 0, len(times) - 1)])
    return int((np.minimum(lo, hi) <= SAMPLE_TOLERANCE_S).sum())


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    banks = {"s036": load_bank(FIRST, "s036"), "s300": load_bank(SECOND, "s300")}
    cohort = [("s036", "p06_f000"), ("s300", "p09_f020"), ("s300", "p09_f038"), ("s300", "p09_f024")]
    rows, pair_tables, null_values = [], {}, {}
    rng = np.random.default_rng(DRAW_SEED)
    for i, (tag_a, fid_a) in enumerate(cohort):
        for tag_b, fid_b in cohort[i + 1:]:
            ba, bb = banks[tag_a], banks[tag_b]
            a, b = candidate(ba, fid_a), candidate(bb, fid_b)
            ia, ib = ba["template_index"][fid_a], bb["template_index"][fid_b]
            fa, fb = ba["families"].loc[fid_a], bb["families"].loc[fid_b]
            cosine, coverage, common = lagged_overlap_cosine(ba["templates"][ia], int(fa.phase_id), bb["templates"][ib], int(fb.phase_id), ba["relative_geometry"])
            pairs = reciprocal_pairs(a, b); metrics = pair_metrics(pairs)
            null = []
            for _ in range(N_DRAWS):
                offsets = np.asarray([rng.uniform(0, stop - start) for start, stop in MOTION_WINDOWS])
                value = pair_metrics(reciprocal_pairs(a, shift_within_windows(b, offsets)))["correlation"]
                if np.isfinite(value): null.append(value)
            null = np.asarray(null)
            p_value = float((1 + np.sum(null >= metrics["correlation"])) / (1 + len(null))) if len(null) and np.isfinite(metrics["correlation"]) else np.nan
            key = f"{tag_a}:{fid_a}__{tag_b}:{fid_b}"
            pair_tables[key] = pairs; null_values[key] = null
            rows.append({"pair": key, "template_cosine": cosine, "overlap_energy_fraction": coverage,
                         "common_channels": common, "strict_events_a": len(a), "strict_events_b": len(b),
                         "exact_same_detection_count_a_to_b": exact_overlap_count(a, b), **metrics,
                         "circular_shift_valid_draws": len(null), "correlation_upper_tail_p": p_value})
    audit = pd.DataFrame(rows)
    audit.to_csv(OUTPUT / "pair_audit.csv", index=False)
    for key, table in pair_tables.items(): table.to_csv(OUTPUT / (key.replace(":", "_").replace("__", "--") + ".csv"), index=False)

    primary_key = "s036:p06_f000__s300:p09_f020"
    a = candidate(banks["s036"], "p06_f000"); b = candidate(banks["s300"], "p09_f020")
    pairs = pair_tables[primary_key]; primary = audit.set_index("pair").loc[primary_key]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    for q, color, label in [(a, "#0072B2", "p06_f000, seed 36--56 s"), (b, "#D55E00", "p09_f020, seed 300--320 s")]:
        axes[0, 0].scatter(q.time_s, q.relative_um, s=17, alpha=.8, color=color, label=label)
    for start, stop in MOTION_WINDOWS: axes[0, 0].axvspan(start, stop, color="#eeeeee", zorder=-2)
    axes[0, 0].axhline(0, color="black", lw=.6); axes[0, 0].legend(fontsize=8)
    axes[0, 0].set(xlabel="Recording time (s)", ylabel="Depth relative to independent seed median (um)", title="Strict observations; no motion estimate")
    axes[0, 1].scatter(pairs.relative_a_um, pairs.relative_b_um, s=32, color="#009E73")
    axes[0, 1].set(xlabel="p06_f000 relative depth (um)", ylabel="p09_f020 relative depth (um)", title=f"Reciprocal pairs <=250 ms: n={len(pairs)}, r={primary.correlation:.3f}")
    lo = min(axes[0, 1].get_xlim()[0], axes[0, 1].get_ylim()[0]); hi = max(axes[0, 1].get_xlim()[1], axes[0, 1].get_ylim()[1]); axes[0, 1].plot([lo, hi], [lo, hi], color="black", ls=":", lw=1)
    null = null_values[primary_key]; axes[1, 0].hist(null, bins=40, color="#999999")
    axes[1, 0].axvline(primary.correlation, color="#D55E00", lw=2, label=f"observed r={primary.correlation:.3f}")
    axes[1, 0].legend(); axes[1, 0].set(xlabel="Correlation after independent within-window circular shifts", ylabel="Draws", title=f"Temporal-alignment null; p={primary.correlation_upper_tail_p:.4f}")
    axes[1, 1].axis("off")
    axes[1, 1].text(0, 1, "\n".join([f"Template cosine: {primary.template_cosine:.3f}", f"Exact shared detections: {int(primary.exact_same_detection_count_a_to_b)}", f"Centered median absolute difference: {primary.centered_median_absolute_difference_um:.1f} um", f"Same-sign relative observations: {100*primary.same_direction_fraction:.1f}%", "Candidates were chosen from seed evidence before this comparison.", "This establishes local waveform-track replication,", "not biological identity or calibrated physical displacement."]), va="top", family="monospace")
    fig.suptitle("Independent-seed imec1 lighthouse replication: p06_f000 versus p09_f020")
    for ext in ("png", "pdf"): fig.savefig(OUTPUT / f"01_primary_replication.{ext}", dpi=160)
    plt.close(fig)
    summary = {"schema": "luke0804-imec1-cross-seed-replication-v1", "status": "complete", "motion_estimate_read": False,
               "candidate_selection_used_replication": False, "primary_pair": primary_key,
               "primary_metrics": {k: (int(v) if k in {"pairs", "strict_events_a", "strict_events_b", "exact_same_detection_count_a_to_b", "circular_shift_valid_draws"} else float(v)) for k, v in primary.items() if k != "pair"},
               "interpretation": "Positive local replication lead between distinct seed-selected waveform templates. Treat p09_f020 as motion-scale provisional, not yet certified identity or displacement ground truth."}
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__": main()
