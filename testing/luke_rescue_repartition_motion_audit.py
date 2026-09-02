"""Phase A2 — is rescue's re-partitioning of legacy units motion-structured?

`docs/pipeline_improvement_plan.md` Phase A2. Phase A established that the −127
lost legacy-good units and the +200 gains are all relabelling — no detection
loss. It split the −127 into 27 label-threshold demotions and **100 re-clustered
units**, and named a mechanistic hypothesis for the 100 (and the dispersed
gains):

> Legacy partially stabilised moving neurons by resampling voltage. Rescue
> preserves the voltage but leaves KS4 to represent a moving waveform footprint,
> which it can only do by splitting it across templates.

This audit tests that, on existing sorts only, with the discriminator **fixed
before looking** (`PRESPEC` below, also written to `prespec.json`; the script
refuses to run against a different frozen prespec).

For each strongly-dispersed legacy↔rescue family it asks the two questions the
plan fixes in advance:

| Observation | Reading |
|---|---|
| Fragments occupy **successive** epochs, follow a coherent depth trajectory (tracking estimated motion or monotonic), and `S` merges **without** refractory violations | **motion fragmentation** |
| Fragments **coexist** at the same times and motion state | **over-splitting** |

Runs on **imec0 and imec1** (`--probe`). imec0's DREDGE rigid estimate is small
and QC-unqualified; imec1's motion is larger. The output reports the *mix* per
probe — counts per class, not a verdict — which sets the Phase D priority tree.

Outputs to testing/outputs/luke_rescue_repartition_motion_audit/<probe>/.
Nothing is written under /mnt.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from testing.luke_rescue_unique_units_audit import load_sort, nearest_hit

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "testing/outputs/luke_rescue_repartition_motion_audit"
LUKE_ROOT = Path("/mnt/NPX/Luke/20250804")

PRESPEC = {
    "schema": "luke-rescue-repartition-motion-v1",
    "frozen": "2026-09-02",
    "question": (
        "Are rescue's dispersed re-partitions of legacy units temporally "
        "complementary and motion-tracking (motion fragmentation), or coexisting "
        "at the same motion state (over-splitting)?"
    ),
    "family_sampling": {
        "sides": ["legacy_lost_dispersed", "rescue_gained_dispersed"],
        "legacy_lost_dispersed": (
            "legacy KS-good unit with no mutual-best match to a rescue good "
            "unit, whose single best rescue partner captures < 0.60 of its train"
        ),
        "rescue_gained_dispersed": (
            "rescue KS-good unit with no mutual-best match to a legacy good "
            "unit, whose single best legacy partner captures < 0.25 of its train"
        ),
        "min_spikes": 300,
        "min_fragments": 2,
        "fragment_capture_floor": 0.05,
        "max_fragments_scored": 5,
        "subsampling": "none -- every qualifying family is scored",
        "probes": ["imec0", "imec1"],
    },
    "bin_s": 30.0,
    "min_spikes_per_bin": 5,
    "discriminator": {
        "temporal_overlap_successive_max": 0.20,
        "temporal_overlap_coexist_min": 0.50,
        "motion_tracking_abs_corr_min": 0.50,
        "motion_tracking_depth_range_um_min": 5.0,
        "monotonic_abs_spearman_min": 0.70,
        "clean_merge_rv_frac_max": 0.015,
        "refractory_ms": 1.5,
        "match_tolerance_samples": 15,
    },
    "classification_rule": {
        "motion_fragmentation": (
            "temporal_overlap < successive_max AND (motion_tracking OR "
            "monotonic_depth) AND clean_merge"
        ),
        "over_splitting": "temporal_overlap > coexist_min",
        "successive_clean_no_motion_signal": (
            "temporal_overlap < successive_max AND clean_merge AND no depth "
            "trajectory -- reported separately, forced into neither bucket"
        ),
        "ambiguous": "none of the above",
    },
    "reporting": "counts per class per probe + distributions; the mix, not a verdict",
}

BIN_S = PRESPEC["bin_s"]
D = PRESPEC["discriminator"]
TOL = D["match_tolerance_samples"]
REFRACTORY_MS = D["refractory_ms"]
CAPTURE_FLOOR = PRESPEC["family_sampling"]["fragment_capture_floor"]
MIN_SPIKES = PRESPEC["family_sampling"]["min_spikes"]
MAX_FRAGMENTS = PRESPEC["family_sampling"]["max_fragments_scored"]
MIN_BIN = PRESPEC["min_spikes_per_bin"]


def _probe_paths(probe: str) -> dict:
    root = LUKE_ROOT
    legacy = root / f"pipeline_results_Luke0804_V2V1_g0_{probe}/cur/cur_sorter_output"
    rescue_curated = (
        root / f"rescue_pipeline_results_Luke0804_V2V1_g0_{probe}/cur/cur_output"
    )
    rescue_raw = (
        root
        / f"rescue_pipeline_results_Luke0804_V2V1_g0_{probe}/kilosort4/sorter_output"
    )
    rescue = rescue_curated if rescue_curated.exists() else rescue_raw
    motion = (
        root
        / f"dredge_pipeline_results_Luke0804_V2V1_g0_{probe}/motion/dredge-motion"
    )
    return {
        "legacy": legacy,
        "rescue": rescue,
        "rescue_is_curated": rescue_curated.exists(),
        "motion": motion,
    }


def _sample_rate(path: Path) -> float:
    ns: dict = {}
    exec((path / "params.py").read_text(), {}, ns)
    return float(ns["sample_rate"])


def _spike_depth_native(path: Path) -> np.ndarray:
    return np.load(path / "spike_positions.npy")[:, 1].astype(np.float64)


def _rigid_motion_relative(motion_dir: Path):
    """DREDGE rigid trace on a frame-relative second clock (or None)."""
    if not (motion_dir / "motion.npy").exists():
        return None
    field = np.load(motion_dir / "motion.npy")
    times = np.load(motion_dir / "time_bins.npy").astype(float)
    dt = float(np.median(np.diff(times)))
    rel = times - (times[0] - dt / 2.0)
    rigid = np.nanmedian(np.asarray(field, dtype=float), axis=1)
    return rel, rigid


def _mutual_best_good(a: dict, b: dict) -> tuple[set, set]:
    """Return (a good ids matched to b, b good ids matched to a)."""
    a_mask = np.isin(a["cl"], list(a["good"]))
    b_mask = np.isin(b["cl"], list(b["good"]))
    a_st, a_cl = a["st"][a_mask], a["cl"][a_mask]
    b_st, b_cl = b["st"][b_mask], b["cl"][b_mask]
    hit, hit_cl = nearest_hit(a_st, b_st, b_cl)
    a_ids, b_ids = np.unique(a_cl), np.unique(b_cl)
    ai = {c: i for i, c in enumerate(a_ids)}
    bi = {c: i for i, c in enumerate(b_ids)}
    counts = np.zeros((len(a_ids), len(b_ids)), dtype=np.int64)
    np.add.at(
        counts, ([ai[c] for c in a_cl[hit]], [bi[c] for c in hit_cl[hit]]), 1
    )
    n_a = np.array([(a_cl == c).sum() for c in a_ids])
    n_b = np.array([(b_cl == c).sum() for c in b_ids])
    frac = counts / np.maximum(np.minimum(n_a[:, None], n_b[None, :]), 1)
    a_matched, b_matched = set(), set()
    for i, ca in enumerate(a_ids):
        j = int(np.argmax(frac[i]))
        if frac[i, j] >= 0.5 and int(np.argmax(frac[:, j])) == i:
            a_matched.add(int(ca))
            b_matched.add(int(b_ids[j]))
    return a_matched, b_matched


def _fragments(anchor_st: np.ndarray, other: dict) -> list[tuple[int, float]]:
    """Other-sort clusters capturing >= CAPTURE_FLOOR of the anchor train."""
    hit, hit_cl = nearest_hit(anchor_st, other["st"], other["cl"])
    n = anchor_st.size
    if not hit.any():
        return []
    ids, cnt = np.unique(hit_cl[hit], return_counts=True)
    order = np.argsort(cnt)[::-1]
    return [
        (int(ids[k]), float(cnt[k] / n))
        for k in order
        if cnt[k] / n >= CAPTURE_FLOOR
    ][:MAX_FRAGMENTS]


def _assign_to_fragments(
    anchor_st: np.ndarray, other: dict, fragment_ids: list[int]
) -> np.ndarray:
    """Per anchor spike, the fragment index it hits, or -1."""
    hit, hit_cl = nearest_hit(anchor_st, other["st"], other["cl"])
    idx = {c: i for i, c in enumerate(fragment_ids)}
    out = np.full(anchor_st.size, -1, dtype=np.int64)
    for k, c in enumerate(hit_cl):
        if hit[k] and int(c) in idx:
            out[k] = idx[int(c)]
    return out


def _score_family(
    anchor_st: np.ndarray,
    anchor_depth: np.ndarray,
    other: dict,
    fs: float,
    motion,
) -> dict | None:
    order = np.argsort(anchor_st)
    st, depth = anchor_st[order], anchor_depth[order]
    frags = _fragments(st, other)
    if len(frags) < 2:
        return None
    frag_ids = [f for f, _ in frags]
    assign = _assign_to_fragments(st, other, frag_ids)

    t_s = st / fs
    edges = np.arange(t_s[0], t_s[-1] + BIN_S, BIN_S)
    if edges.size < 3:
        return None
    b_idx = np.clip(np.digitize(t_s, edges) - 1, 0, edges.size - 2)

    n_bins = edges.size - 1
    share = np.zeros((n_bins, len(frag_ids)))
    bin_depth = np.full(n_bins, np.nan)
    bin_time = (edges[:-1] + edges[1:]) / 2.0
    valid = np.zeros(n_bins, dtype=bool)
    for b in range(n_bins):
        m = b_idx == b
        if m.sum() < MIN_BIN:
            continue
        valid[b] = True
        bin_depth[b] = np.median(depth[m])
        a = assign[m]
        for f in range(len(frag_ids)):
            share[b, f] = np.mean(a == f)

    if valid.sum() < 3:
        return None

    owner = np.where(valid, np.argmax(share, axis=1), -1)
    seq = owner[valid]
    n_switches = int(np.sum(seq[1:] != seq[:-1]))
    _vt_all = bin_time[valid]
    span_hr = float(_vt_all[-1] - _vt_all[0]) / 3600.0
    switch_per_hr = round(n_switches / span_hr, 2) if span_hr > 0 else None

    top2 = np.argsort(share.sum(axis=0))[::-1][:2]
    s1, s2 = share[valid][:, top2[0]], share[valid][:, top2[1]]
    denom = np.sum(np.maximum(s1, s2))
    temporal_overlap = float(np.sum(np.minimum(s1, s2)) / denom) if denom else 1.0

    vd = bin_depth[valid]
    vt = bin_time[valid]
    depth_range = float(np.percentile(vd, 90) - np.percentile(vd, 10))
    if np.std(vd) > 0 and np.std(vt) > 0:
        sp = spearmanr(vt, vd)
        monotonic_r = float(sp.statistic if hasattr(sp, "statistic") else sp[0])
    else:
        monotonic_r = 0.0

    motion_corr = np.nan
    if motion is not None:
        m_at = np.interp(vt, motion[0], motion[1], left=np.nan, right=np.nan)
        ok = np.isfinite(m_at) & np.isfinite(vd)
        if ok.sum() >= 3 and np.std(vd[ok]) > 0 and np.std(m_at[ok]) > 0:
            motion_corr = float(np.corrcoef(vd[ok], m_at[ok])[0, 1])

    isi_ms = np.diff(st) / fs * 1000.0
    merged_rv = float((isi_ms < REFRACTORY_MS).mean()) if st.size > 1 else np.nan

    successive = temporal_overlap < D["temporal_overlap_successive_max"]
    coexist = temporal_overlap > D["temporal_overlap_coexist_min"]
    motion_tracking = (
        np.isfinite(motion_corr)
        and abs(motion_corr) >= D["motion_tracking_abs_corr_min"]
        and depth_range >= D["motion_tracking_depth_range_um_min"]
    )
    monotonic = (
        abs(monotonic_r) >= D["monotonic_abs_spearman_min"]
        and depth_range >= D["motion_tracking_depth_range_um_min"]
    )
    clean_merge = np.isfinite(merged_rv) and merged_rv <= D["clean_merge_rv_frac_max"]

    if successive and (motion_tracking or monotonic) and clean_merge:
        klass = "motion_fragmentation"
    elif coexist:
        klass = "over_splitting"
    elif successive and clean_merge:
        klass = "successive_clean_no_motion_signal"
    else:
        klass = "ambiguous"

    return {
        "n_spikes": int(st.size),
        "n_fragments": len(frag_ids),
        "fragment_capture": [round(c, 3) for _, c in frags],
        "n_bins_scored": int(valid.sum()),
        "n_ownership_switches": n_switches,
        "ownership_switch_per_hr": switch_per_hr,
        "temporal_overlap": round(temporal_overlap, 3),
        "depth_range_um": round(depth_range, 2),
        "depth_time_spearman": round(monotonic_r, 3),
        "motion_corr": None if not np.isfinite(motion_corr) else round(motion_corr, 3),
        "merged_rv_frac": None if not np.isfinite(merged_rv) else round(merged_rv, 5),
        "median_depth_um": round(float(np.median(vd)), 1),
        "successive": bool(successive),
        "coexist": bool(coexist),
        "motion_tracking": bool(motion_tracking),
        "monotonic_depth": bool(monotonic),
        "clean_merge": bool(clean_merge),
        "classification": klass,
    }


def _families_for_side(
    side: str, anchor: dict, other: dict, anchor_dir: Path
) -> list[dict]:
    anchor_matched, _ = _mutual_best_good(anchor, other)
    best_partner_ceiling = 0.60 if side == "legacy_lost_dispersed" else 0.25

    clu_native = np.load(anchor_dir / "spike_clusters.npy").reshape(-1)
    st_native = np.load(anchor_dir / "spike_times.npy").reshape(-1).astype(np.int64)
    depth_native = _spike_depth_native(anchor_dir)

    rows = []
    for cid in sorted(anchor["good"] - anchor_matched):
        m = clu_native == cid
        if m.sum() < MIN_SPIKES:
            continue
        a_st = st_native[m]
        frags = _fragments(np.sort(a_st), other)
        if not frags or frags[0][1] >= best_partner_ceiling:
            continue
        rows.append({"cid": int(cid), "st": a_st, "depth": depth_native[m]})
    return rows


def _freeze_prespec(probe_out: Path) -> None:
    path = OUTPUT / "prespec.json"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != PRESPEC:
            raise SystemExit(
                f"{path} differs from the frozen PRESPEC in this module. "
                "Phase A2 is a run-once analysis: do not change the prespec "
                "after results exist. Delete the output tree to re-freeze."
            )
    else:
        path.write_text(json.dumps(PRESPEC, indent=2) + "\n")
    probe_out.mkdir(parents=True, exist_ok=True)


def run_probe(probe: str) -> dict:
    paths = _probe_paths(probe)
    probe_out = OUTPUT / probe
    _freeze_prespec(probe_out)

    legacy = load_sort(paths["legacy"])
    rescue = load_sort(paths["rescue"])
    fs_legacy = _sample_rate(paths["legacy"])
    fs_rescue = _sample_rate(paths["rescue"])
    motion = _rigid_motion_relative(paths["motion"])

    sides = {
        "legacy_lost_dispersed": (legacy, rescue, paths["legacy"], fs_legacy),
        "rescue_gained_dispersed": (rescue, legacy, paths["rescue"], fs_rescue),
    }

    all_rows = []
    for side, (anchor, other, anchor_dir, fs) in sides.items():
        families = _families_for_side(side, anchor, other, anchor_dir)
        for fam in families:
            scored = _score_family(fam["st"], fam["depth"], other, fs, motion)
            if scored is None:
                continue
            scored.update({"side": side, "anchor_cluster": fam["cid"]})
            all_rows.append(scored)

    df = pd.DataFrame(all_rows)
    df.to_csv(probe_out / "families_classified.csv", index=False)

    summary = {
        "probe": probe,
        "rescue_is_curated": paths["rescue_is_curated"],
        "motion_available": motion is not None,
        "n_families": len(df),
    }
    if len(df):
        depth_tercile = pd.qcut(
            df["median_depth_um"], 3, labels=["shallow", "mid", "deep"],
            duplicates="drop",
        )
        summary["mix"] = {
            k: int(v) for k, v in df["classification"].value_counts().items()
        }
        summary["mix_by_side"] = {
            side: {
                k: int(v)
                for k, v in sub["classification"].value_counts().items()
            }
            for side, sub in df.groupby("side")
        }
        summary["mix_by_depth"] = {
            str(t): {
                k: int(v)
                for k, v in df.loc[depth_tercile == t, "classification"]
                .value_counts()
                .items()
            }
            for t in depth_tercile.dropna().unique()
        }
        summary["distributions"] = {
            col: {
                "median": round(float(df[col].median()), 3),
                "p10": round(float(df[col].quantile(0.1)), 3),
                "p90": round(float(df[col].quantile(0.9)), 3),
            }
            for col in [
                "temporal_overlap",
                "ownership_switch_per_hr",
                "depth_range_um",
                "merged_rv_frac",
            ]
            if df[col].notna().any()
        }
        summary["headline"] = {
            "coexisting_fragment_fraction": round(
                float((df["temporal_overlap"] > 0.5).mean()), 3
            ),
            "successive_fragment_fraction": round(
                float((df["temporal_overlap"] < 0.2).mean()), 3
            ),
            "clean_merge_fraction": round(float(df["clean_merge"].mean()), 3),
            "median_ownership_switch_per_hr": round(
                float(df["ownership_switch_per_hr"].median()), 1
            ),
            "median_abs_motion_corr": (
                round(float(df["motion_corr"].abs().median()), 3)
                if df["motion_corr"].notna().any()
                else None
            ),
        }
        mc = df["motion_corr"].dropna()
        if len(mc):
            summary["distributions"]["abs_motion_corr"] = {
                "median": round(float(mc.abs().median()), 3),
                "p90": round(float(mc.abs().quantile(0.9)), 3),
            }
    (probe_out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--probe", choices=["imec0", "imec1", "both"], default="both")
    args = ap.parse_args()
    probes = ["imec0", "imec1"] if args.probe == "both" else [args.probe]
    for probe in probes:
        summary = run_probe(probe)
        print(f"\n=== {probe} ===")
        print(json.dumps(summary, indent=2))
    print(f"\nwrote {OUTPUT}")


if __name__ == "__main__":
    main()
