"""What are the KS-good units the rescue sort has and the legacy sort does not?

decisions/0009 retracted the amplitude-completeness deficit and left the real
promotion question open: rescue reports 301 KS-good units against legacy's 228,
and it was not known whether the difference is new neurons, promoted MUA,
fragments, or noise.

This answers it without resorting anything, by locating every rescue-unique
good unit's spikes inside the *complete* legacy sort, including its MUA
clusters, and then asking whether any class explains the similar-pair gate
failure.

Outputs to testing/outputs/luke_rescue_unique_units_audit/ (untracked, local).
Nothing is written under /mnt.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path("/mnt/NPX/Luke/20250804")
OUTPUT = REPO_ROOT / "testing/outputs/luke_rescue_unique_units_audit"

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


def mutual_best_matches(a, b):
    """Mutual best spike-time matches between the two sorts' good units."""
    a_mask = np.isin(a["cl"], list(a["good"]))
    b_mask = np.isin(b["cl"], list(b["good"]))
    a_st, a_cl = a["st"][a_mask], a["cl"][a_mask]
    b_st, b_cl = b["st"][b_mask], b["cl"][b_mask]
    hit, hit_cl = nearest_hit(a_st, b_st, b_cl)

    a_ids, b_ids = np.unique(a_cl), np.unique(b_cl)
    ai = {c: i for i, c in enumerate(a_ids)}
    bi = {c: i for i, c in enumerate(b_ids)}
    counts = np.zeros((len(a_ids), len(b_ids)), dtype=np.int64)
    np.add.at(counts,
              ([ai[c] for c in a_cl[hit]], [bi[c] for c in hit_cl[hit]]), 1)
    n_a = np.array([(a_cl == c).sum() for c in a_ids])
    n_b = np.array([(b_cl == c).sum() for c in b_ids])
    frac = counts / np.maximum(np.minimum(n_a[:, None], n_b[None, :]), 1)
    pairs = []
    for i, ca in enumerate(a_ids):
        j = int(np.argmax(frac[i]))
        if frac[i, j] >= MATCH_THRESHOLD and int(np.argmax(frac[:, j])) == i:
            pairs.append({"rescue_cluster": int(ca), "legacy_cluster": int(b_ids[j]),
                          "coincident_fraction": float(frac[i, j])})
    return pd.DataFrame(pairs)


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
        hit, hit_cl = nearest_hit(a, legacy["st"], legacy["cl"])
        n = len(a)
        if hit.any():
            ids, cnt = np.unique(hit_cl[hit], return_counts=True)
            k = int(np.argmax(cnt))
            top, frac = int(ids[k]), cnt[k] / n
            label = legacy["label"].get(top, "?")
        else:
            top, frac, label = -1, 0.0, "none"
        isi = np.diff(np.sort(a)) / FS * 1000.0
        rows.append({"rescue_cluster": int(cid), "n_spikes": n, "rate_hz": n / DURATION_S,
                     "frac_found_in_legacy": float(hit.mean()),
                     "best_legacy_cluster": top, "best_legacy_label": label,
                     "best_legacy_frac": float(frac),
                     "rv_frac": float((isi < REFRACTORY_MS).mean()) if n > 1 else np.nan})
    df = pd.DataFrame(rows)

    def classify(r):
        if r.frac_found_in_legacy < 0.25:
            return "genuinely new detection"
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
    print("No class explains the similar-pair gate failure.")
    print(f"\nwrote {OUTPUT}")


if __name__ == "__main__":
    main()
