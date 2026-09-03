"""What are the KS-good units the rescue sort has and the legacy sort does not?

decisions/0009 retracted the amplitude-completeness deficit and left the real
promotion question open: rescue reports 301 KS-good units against legacy's 228,
and it was not known whether the difference is new neurons, promoted MUA,
fragments, or noise.

This is the corrected v2 audit.  The original results are retracted because
cross-sort identity reused target events and whole-probe coincidence had an
87--89% chance baseline.  V2 uses exclusive identity matching plus a spatially
constrained, circular-shift null.  No empirical conclusion is valid until this
version's outputs are regenerated.

Outputs to testing/outputs/luke_rescue_unique_units_audit_v2/ (untracked, local).
Nothing is written under /mnt.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path("/mnt/NPX/Luke/20250804")
OUTPUT = REPO_ROOT / "testing/outputs/luke_rescue_unique_units_audit_v2"

RESCUE = ROOT / "rescue_pipeline_results_Luke0804_V2V1_g0_imec0/cur/cur_output"
LEGACY = ROOT / "pipeline_results_Luke0804_V2V1_g0_imec0/cur/cur_sorter_output"

TOL = 15              # 0.5 ms at 30 kHz
MATCH_THRESHOLD = 0.5  # coincident fraction of the smaller unit
DURATION_S = 10473.55
FS = 29999.835983263598
REFRACTORY_MS = 1.5
# Gate definition copied from testing/luke_full_probe_rescue_diagnostics.py
SIMILARITY_THRESHOLD = 0.8
DEPTH_WINDOW_UM = 100.0
DETECTION_EXCESS_FLOOR = 0.10
DETECTION_OBSERVED_FLOOR = 0.25
NULL_SHIFT_S = (137.0, 431.0, 997.0)


def load_sort(path: Path) -> dict:
    st = np.load(path / "spike_times.npy").reshape(-1).astype(np.int64)
    cl = np.load(path / "spike_clusters.npy").reshape(-1).astype(np.int64)
    order = np.argsort(st, kind="stable")
    labels = pd.read_csv(path / "cluster_KSLabel.tsv", sep="\t")
    col = next(c for c in labels.columns if c != "cluster_id")
    lab = dict(zip(labels["cluster_id"], labels[col].astype(str).str.strip().str.lower()))
    return {"st": st[order], "cl": cl[order], "label": lab,
            "good": {c for c, v in lab.items() if v == "good"}}


def nearest_hit(a_st, b_st, b_cl):
    """Nearest target event for each source event (distribution diagnostics only).

    This helper is intentionally not used for cross-sort unit identity because
    it can reuse one target event.  Use :func:`exclusive_event_pairs` there.
    """
    if len(b_st) == 0:
        return np.zeros(np.asarray(a_st).shape, dtype=bool), np.full(
            np.asarray(a_st).shape, -1, dtype=np.int64
        )
    idx = np.searchsorted(b_st, a_st)
    best = np.full(a_st.shape, -1, dtype=np.int64)
    dist = np.full(a_st.shape, np.iinfo(np.int64).max, dtype=np.int64)
    for shift in (-1, 0):
        j = np.clip(idx + shift, 0, len(b_st) - 1)
        ok = (idx + shift >= 0) & (idx + shift < len(b_st))
        d = np.where(ok, np.abs(a_st - b_st[j]), np.iinfo(np.int64).max)
        take = d < dist
        dist, best = np.where(take, d, dist), np.where(take & ok, j, best)
    hit = (dist <= TOL) & (best >= 0)
    out = np.full(a_st.shape, -1, dtype=np.int64)
    out[hit] = b_cl[best[hit]]
    return hit, out


def exclusive_event_pairs(a_st, b_st, tolerance: int = TOL):
    """Maximum-cardinality, one-to-one pairs of time-ordered events.

    Each event can be used once.  The interval-order two-pointer algorithm is
    deterministic and gives the same pairing after swapping ``a`` and ``b``.
    Returned indices refer to the original input arrays.
    """
    a = np.asarray(a_st, dtype=np.int64)
    b = np.asarray(b_st, dtype=np.int64)
    if a.size == 0 or b.size == 0:
        empty = np.array([], dtype=np.int64)
        return empty, empty
    ao = np.argsort(a, kind="stable")
    bo = np.argsort(b, kind="stable")
    sa, sb = a[ao], b[bo]
    ai: list[int] = []
    bi: list[int] = []
    i = j = 0
    while i < sa.size and j < sb.size:
        delta = int(sa[i]) - int(sb[j])
        if delta < -tolerance:
            i = max(i + 1, int(np.searchsorted(sa, sb[j] - tolerance)))
        elif delta > tolerance:
            j = max(j + 1, int(np.searchsorted(sb, sa[i] - tolerance)))
        else:
            ai.append(int(ao[i]))
            bi.append(int(bo[j]))
            i += 1
            j += 1
    return np.asarray(ai, dtype=np.int64), np.asarray(bi, dtype=np.int64)


def mutual_best_matches(a, b):
    """Mutual-best good-unit matches from exclusive spike-event pairs."""
    a_mask = np.isin(a["cl"], list(a["good"]))
    b_mask = np.isin(b["cl"], list(b["good"]))
    a_st, a_cl = a["st"][a_mask], a["cl"][a_mask]
    b_st, b_cl = b["st"][b_mask], b["cl"][b_mask]
    a_hit, b_hit = exclusive_event_pairs(a_st, b_st)

    a_ids, b_ids = np.unique(a_cl), np.unique(b_cl)
    ai = {c: i for i, c in enumerate(a_ids)}
    bi = {c: i for i, c in enumerate(b_ids)}
    counts = np.zeros((len(a_ids), len(b_ids)), dtype=np.int64)
    if a_hit.size:
        np.add.at(
            counts,
            ([ai[c] for c in a_cl[a_hit]], [bi[c] for c in b_cl[b_hit]]),
            1,
        )
    n_a = np.array([(a_cl == c).sum() for c in a_ids])
    n_b = np.array([(b_cl == c).sum() for c in b_ids])
    frac = counts / np.maximum(np.minimum(n_a[:, None], n_b[None, :]), 1)
    pairs = []
    for i, ca in enumerate(a_ids):
        j = int(np.argmax(frac[i]))
        if frac[i, j] >= MATCH_THRESHOLD and int(np.argmax(frac[:, j])) == i:
            pairs.append({"rescue_cluster": int(ca), "legacy_cluster": int(b_ids[j]),
                          "coincident_fraction": float(frac[i, j])})
    return pd.DataFrame(
        pairs,
        columns=["rescue_cluster", "legacy_cluster", "coincident_fraction"],
    )


def template_depth_by_cluster(path: Path) -> dict[int, float]:
    """Cluster depth for spatially constrained cross-sort evidence."""
    spike_positions = path / "spike_positions.npy"
    spike_clusters = path / "spike_clusters.npy"
    if spike_positions.exists() and spike_clusters.exists():
        depth = np.load(spike_positions, mmap_mode="r")[:, 1]
        clusters = np.load(spike_clusters, mmap_mode="r").reshape(-1)
        if depth.shape[0] == clusters.shape[0]:
            n = int(np.max(clusters)) + 1 if clusters.size else 0
            count = np.bincount(clusters, minlength=n)
            mean_depth = np.bincount(clusters, weights=depth, minlength=n) / np.maximum(
                count, 1
            )
            return {
                int(cluster): float(mean_depth[cluster])
                for cluster in np.flatnonzero(count)
            }
    # Fallback for compact fixtures or sorter exports without per-spike
    # positions. Kilosort's unmerged cluster ids are template-row ids.
    templates = np.load(path / "templates.npy", mmap_mode="r")
    positions = np.load(path / "channel_positions.npy")
    peak = np.argmax(np.max(np.abs(templates), axis=1), axis=1)
    return {int(c): float(positions[ch, 1]) for c, ch in enumerate(peak)}


def spatial_null_distribution(
    anchor_st: np.ndarray,
    anchor_depth_um: float,
    target: dict,
    target_depth: dict[int, float],
    *,
    fs: float = FS,
    duration_s: float = DURATION_S,
    depth_window_um: float = DEPTH_WINDOW_UM,
) -> tuple[float, list[tuple[int, float, str]], dict]:
    """Spatially plausible coincidence plus circular time-shift null evidence.

    Nearest-event lookup remains useful within one anchor train, but whole-probe
    coincidence is not evidence at these spike densities.  We therefore retain
    only target clusters within a depth window and require observed coincidence
    to exceed fixed circular-shift nulls before calling shared detection.
    """
    anchor = np.sort(np.asarray(anchor_st, dtype=np.int64))
    if anchor.size == 0 or not np.isfinite(anchor_depth_um):
        return 0.0, [], {
            "observed_spatial_fraction": 0.0,
            "null_median_fraction": None,
            "coincidence_excess": None,
            "shared_detection_supported": False,
        }

    max_cluster = max(target_depth, default=-1)
    depth_lookup = np.full(max_cluster + 1, np.nan, dtype=np.float64)
    for cluster, depth in target_depth.items():
        depth_lookup[int(cluster)] = float(depth)
    target_st = np.asarray(target["st"], dtype=np.int64)
    target_cl = np.asarray(target["cl"], dtype=np.int64)

    def spatial_hits(times):
        times = np.asarray(times, dtype=np.int64)
        lo = np.searchsorted(target_st, times - TOL, side="left")
        hi = np.searchsorted(target_st, times + TOL, side="right")
        hit_cl = np.full(len(times), -1, dtype=np.int64)
        used_target: set[int] = set()
        # Earliest-feasible interval matching, restricted to spatially
        # plausible clusters, is one-to-one and maximum-cardinality.
        for anchor_idx, (start, stop) in enumerate(zip(lo, hi)):
            for target_idx in range(int(start), int(stop)):
                if target_idx in used_target:
                    continue
                cluster = int(target_cl[target_idx])
                if not (0 <= cluster < depth_lookup.size):
                    continue
                depth = depth_lookup[cluster]
                if np.isfinite(depth) and abs(depth - anchor_depth_um) <= depth_window_um:
                    hit_cl[anchor_idx] = cluster
                    used_target.add(target_idx)
                    break
        return hit_cl >= 0, hit_cl

    observed_hit, observed_cl = spatial_hits(anchor)
    ids, cnt = (
        np.unique(observed_cl[observed_hit], return_counts=True)
        if observed_hit.any()
        else (np.array([], dtype=np.int64), np.array([], dtype=np.int64))
    )
    order = np.argsort(cnt)[::-1]
    ranked = [
        (int(ids[k]), float(cnt[k] / anchor.size), target["label"].get(int(ids[k]), "?"))
        for k in order
    ]

    total = max(int(round(duration_s * fs)), 1)
    null = []
    for shift_s in NULL_SHIFT_S:
        shifted = np.sort((anchor + int(round(shift_s * fs))) % total)
        shifted_hit, _ = spatial_hits(shifted)
        null.append(float(shifted_hit.mean()))
    observed = float(observed_hit.mean())
    null_median = float(np.median(null))
    excess = observed - null_median
    evidence = {
        "observed_spatial_fraction": observed,
        "null_fractions": null,
        "null_median_fraction": null_median,
        "coincidence_excess": excess,
        "shared_detection_supported": bool(
            observed >= DETECTION_OBSERVED_FLOOR
            and excess >= DETECTION_EXCESS_FLOOR
        ),
        "depth_window_um": depth_window_um,
    }
    return observed, ranked, evidence


def nearby_similar_good_pairs(path: Path):
    sim = np.load(path / "similar_templates.npy")
    templates = np.load(path / "templates.npy")
    positions = np.load(path / "channel_positions.npy")
    labels = pd.read_csv(path / "cluster_KSLabel.tsv", sep="\t")
    col = next(c for c in labels.columns if c != "cluster_id")
    good_ids = labels.loc[labels[col].astype(str).str.strip().str.lower() == "good", "cluster_id"]
    n = sim.shape[0]
    is_good = np.zeros(n, dtype=bool)
    is_good[list(good_ids)] = True
    depth = positions[np.argmax(np.max(np.abs(templates), axis=1), axis=1), 1]
    upper = np.triu(np.ones((n, n), dtype=bool), 1)
    nearby = np.abs(depth[:, None] - depth[None, :]) <= DEPTH_WINDOW_UM
    first, second = np.where(upper & nearby & (sim >= SIMILARITY_THRESHOLD))
    both = is_good[first] & is_good[second]
    return first[both], second[both]


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rescue, legacy = load_sort(RESCUE), load_sort(LEGACY)
    rescue_depth = template_depth_by_cluster(RESCUE)
    legacy_depth = template_depth_by_cluster(LEGACY)
    print(f"KS-good units -- rescue {len(rescue['good'])}, legacy {len(legacy['good'])}\n")

    matches = mutual_best_matches(rescue, legacy)
    matches.to_csv(OUTPUT / "matched_all_good.csv", index=False)
    unique_ids = sorted(rescue["good"] - set(matches.rescue_cluster))
    print(f"rescue good units matched to a legacy good unit : {len(matches)}")
    print(f"rescue good units unique to rescue              : {len(unique_ids)}")
    print(f"legacy good units NOT matched by rescue         : "
          f"{len(legacy['good']) - len(matches)}")
    print(f"  net KS-good difference: +{len(unique_ids)} / "
          f"-{len(legacy['good']) - len(matches)} = "
          f"{len(rescue['good']) - len(legacy['good']):+d}\n")

    rows = []
    for cid in unique_ids:
        a = rescue["st"][rescue["cl"] == cid]
        hit, _ = nearest_hit(a, legacy["st"], legacy["cl"])
        _, ranked, evidence = spatial_null_distribution(
            a, rescue_depth.get(int(cid), np.nan), legacy, legacy_depth
        )
        n = len(a)
        if ranked:
            top, frac, label = ranked[0]
        else:
            top, frac, label = -1, 0.0, "none"
        isi = np.diff(np.sort(a)) / FS * 1000.0
        rows.append({"rescue_cluster": int(cid), "n_spikes": n, "rate_hz": n / DURATION_S,
                     "frac_found_in_legacy": float(hit.mean()),
                     **evidence,
                     "best_legacy_cluster": top, "best_legacy_label": label,
                     "best_legacy_frac": float(frac),
                     "rv_frac": float((isi < REFRACTORY_MS).mean()) if n > 1 else np.nan})
    df = pd.DataFrame(rows)

    def classify(r):
        if not r.shared_detection_supported:
            return "detection status unresolved"
        if r.best_legacy_frac < 0.25:
            return "dispersed across legacy clusters"
        return f"legacy {r.best_legacy_label} relabelled good"

    df["classification"] = df.apply(classify, axis=1)
    df.to_csv(OUTPUT / "rescue_unique_all_good_classified.csv", index=False)

    print("=== Where the rescue-unique good units' spikes live in the legacy sort ===")
    print(df.groupby("classification")
            .agg(n=("rescue_cluster", "size"),
                 median_found_in_legacy=("frac_found_in_legacy", "median"),
                 median_best_partner=("best_legacy_frac", "median"),
                 median_rate_hz=("rate_hz", "median"),
                 median_rv=("rv_frac", "median"),
                 frac_rv_over_1pct=("rv_frac", lambda x: (x > 0.01).mean()))
            .sort_values("n", ascending=False)
            .to_string(float_format=lambda v: f"{v:.3g}"))

    first, second = nearby_similar_good_pairs(RESCUE)
    involved = set(np.concatenate([first, second]).tolist()) if len(first) else set()
    print(f"\n=== Nearby similar good-good pairs (sim >= {SIMILARITY_THRESHOLD}, "
          f"|dz| <= {DEPTH_WINDOW_UM:g} um): {len(first)} ===")
    cls = {int(r.rescue_cluster): r.classification for r in df.itertuples()}
    for c in matches.rescue_cluster:
        cls.setdefault(int(c), "matched to a legacy good unit")
    out = []
    for name in sorted(set(cls.values())):
        members = [c for c, k in cls.items() if k == name]
        k = sum(1 for c in members if c in involved)
        out.append({"class": name, "n_units": len(members),
                    "in_similar_pair": k, "pct": 100 * k / len(members)})
    print(pd.DataFrame(out).sort_values("pct", ascending=False)
            .to_string(index=False, float_format=lambda v: f"{v:.1f}"))

    disp = [c for c, k in cls.items() if k.startswith("dispersed")]
    rest = [c for c, k in cls.items() if not k.startswith("dispersed")]
    table = [[sum(c in involved for c in disp), len(disp) - sum(c in involved for c in disp)],
             [sum(c in involved for c in rest), len(rest) - sum(c in involved for c in rest)]]
    odds, p = fisher_exact(table)
    print(f"\ndispersed vs rest: {table}  odds ratio {odds:.2f}  p = {p:.3g}")
    print("Association with the similar-pair gate is descriptive, not causal.")
    print(f"\nwrote {OUTPUT}")


if __name__ == "__main__":
    main()
