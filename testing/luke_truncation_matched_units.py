"""Compare amplitude-truncation estimates on the SAME neurons across sorts.

The population comparison in the post-curation evaluation is confounded: the
three configurations admit different unit populations, and estimated
missingness depends strongly on unit amplitude. The original matched-unit
numbers are retracted because their matcher could reuse target events. This v2
implementation uses exclusive event pairs; its empirical output must be rerun.

Outputs to testing/outputs/luke_truncation_fitter_audit/ (gitignored, local).
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from testing.luke_rescue_unique_units_audit import exclusive_event_pairs

REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path("/mnt/NPX/Luke/20250804")
OUTPUT = REPO_ROOT / "testing/outputs/luke_truncation_fitter_audit"
MATCHED_OUTPUT = OUTPUT / "matched_units_v2.csv"

CUR = {
    "rescue": ROOT / "rescue_pipeline_results_Luke0804_V2V1_g0_imec0/cur/cur_output",
    "legacy": ROOT / "pipeline_results_Luke0804_V2V1_g0_imec0/cur/cur_sorter_output",
    "claim_mask": ROOT / "patched_pipeline_results_Luke0804_V2V1_g0_imec0/cur/cur_sorter_output",
}
TOLERANCE_SAMPLES = 15  # 0.5 ms at 30 kHz
MATCH_THRESHOLD = 0.5   # coincident fraction of the smaller unit


def good_spikes(method: str, units: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    path = CUR[method]
    st = np.load(path / "spike_times.npy").reshape(-1).astype(np.int64)
    cl = np.load(path / "spike_clusters.npy").reshape(-1).astype(np.int64)
    keep_ids = set(units.loc[units.method == method, "cluster_id"])
    mask = np.isin(cl, list(keep_ids))
    order = np.argsort(st[mask], kind="stable")
    return st[mask][order], cl[mask][order]


def coincidence_matrix(a_st, a_cl, b_st, b_cl):
    """Count exclusive coincident events for each cluster pair."""
    a_hit, b_hit = exclusive_event_pairs(
        a_st, b_st, tolerance=TOLERANCE_SAMPLES
    )

    a_ids = np.unique(a_cl)
    b_ids = np.unique(b_cl)
    a_pos = {c: i for i, c in enumerate(a_ids)}
    b_pos = {c: i for i, c in enumerate(b_ids)}
    counts = np.zeros((len(a_ids), len(b_ids)), dtype=np.int64)
    ai = np.array([a_pos[c] for c in a_cl[a_hit]])
    bi = np.array([b_pos[c] for c in b_cl[b_hit]])
    if len(ai):
        np.add.at(counts, (ai, bi), 1)
    return a_ids, b_ids, counts


def match(method_a, method_b, units):
    a_st, a_cl = good_spikes(method_a, units)
    b_st, b_cl = good_spikes(method_b, units)
    a_ids, b_ids, counts = coincidence_matrix(a_st, a_cl, b_st, b_cl)
    n_a = np.array([(a_cl == c).sum() for c in a_ids])
    n_b = np.array([(b_cl == c).sum() for c in b_ids])
    denom = np.minimum(n_a[:, None], n_b[None, :])
    frac = counts / np.maximum(denom, 1)

    rows = []
    for i, ca in enumerate(a_ids):
        j = int(np.argmax(frac[i]))
        # mutual best match, above threshold
        if frac[i, j] >= MATCH_THRESHOLD and int(np.argmax(frac[:, j])) == i:
            rows.append(
                {
                    "cluster_a": int(ca),
                    "cluster_b": int(b_ids[j]),
                    "coincident_fraction": float(frac[i, j]),
                    "n_spikes_a": int(n_a[i]),
                    "n_spikes_b": int(n_b[j]),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    units = pd.read_csv(OUTPUT / "unit_truncation_enriched.csv")
    cohort = units[units.is_good & units.eligible & (units.rate_hz > 1)]

    print(f"Matching good, eligible, >1 Hz units. tolerance={TOLERANCE_SAMPLES} samples "
          f"(0.5 ms), mutual best match, coincident fraction >= {MATCH_THRESHOLD}\n")
    all_pairs = []
    for ma, mb in combinations(CUR, 2):
        m = match(ma, mb, cohort)
        if m.empty:
            print(f"  {ma} vs {mb}: no matches")
            continue
        left = cohort[cohort.method == ma].set_index("cluster_id")
        right = cohort[cohort.method == mb].set_index("cluster_id")
        m["mpct_a"] = m.cluster_a.map(left["median_mpct"])
        m["mpct_b"] = m.cluster_b.map(right["median_mpct"])
        m["amp_a"] = m.cluster_a.map(left["ks_amplitude"])
        m["amp_b"] = m.cluster_b.map(right["ks_amplitude"])
        m = m.dropna(subset=["mpct_a", "mpct_b"])
        m["pair"] = f"{ma}_vs_{mb}"
        all_pairs.append(m)

        d = m.mpct_a - m.mpct_b
        stat = wilcoxon(m.mpct_a, m.mpct_b) if len(m) > 5 else None
        print(f"  {ma} vs {mb}: {len(m)} matched neurons")
        print(f"      median missing%  {ma}: {m.mpct_a.median():.2f}   {mb}: {m.mpct_b.median():.2f}")
        print(f"      paired difference median: {d.median():+.2f} pp"
              + (f"   Wilcoxon p={stat.pvalue:.3g}" if stat is not None else ""))
        print(f"      median KS amplitude  {ma}: {m.amp_a.median():.1f}   {mb}: {m.amp_b.median():.1f}")
        print(f"      unmatched: {ma} {len(left)-len(m)}, {mb} {len(right)-len(m)}\n")

    if all_pairs:
        out = pd.concat(all_pairs, ignore_index=True)
        out.to_csv(MATCHED_OUTPUT, index=False)
        print(f"wrote {MATCHED_OUTPUT}")


if __name__ == "__main__":
    main()
