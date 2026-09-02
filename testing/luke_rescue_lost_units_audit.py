"""Where do the legacy KS-good units the rescue sort does not reproduce go?

[`decisions/0010`](../docs/decisions/0010-rescue-yield-is-relabelling-not-detection.md)
established the rescue-vs-legacy KS-good difference on Luke0804 imec0 is
+200 / -127, a two-way relabelling. The +200 side was classified in
`luke_rescue_unique_units_audit.py`: none is a new detection. The -127 side --
legacy good units rescue does not reproduce -- had never been examined. They are
a symmetric risk to the 200 gained and no gate measures them.

This closes Phase A of `docs/pipeline_improvement_plan.md`: it locates every
one of the 127 legacy-good units' spikes inside the *complete* rescue sort
(MUA clusters included, pre- and post-curation) and classifies each into

    absent at detection | preserved as MUA | merged into a rescue good unit
    | split across rescue clusters | dispersed across rescue clusters

The decisive class is **absent at detection**: a legacy good unit whose spikes
the rescue sort never produced. If a substantial share of the 127 are clean,
well-formed neurons lost at detection, that is a regression the yield narrative
hid, and per Checkpoint A it becomes the top-priority defect.

The coincidence machinery is imported unchanged from
`luke_rescue_unique_units_audit.py` so the two sides of the +200 / -127 table
are computed the same way.

Outputs to testing/outputs/luke_rescue_lost_units_audit/ (untracked, local).
Nothing is written under /mnt.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from testing.luke_rescue_unique_units_audit import (
    DURATION_S,
    FS,
    LEGACY,
    REFRACTORY_MS,
    RESCUE,
    TOL,
    load_sort,
    mutual_best_matches,
    nearest_hit,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "testing/outputs/luke_rescue_lost_units_audit"

# A legacy good unit counts as recovered somewhere in rescue if this fraction
# of its spikes land within TOL of any rescue spike (any cluster, any label).
DETECTION_FLOOR = 0.25
# A single rescue cluster this dominant is the legacy unit's counterpart.
DOMINANT_FRAC = 0.5
# Two rescue clusters each at least this large is a split.
SPLIT_FRAC = 0.25


def load_full_sort(path: Path) -> dict:
    """Every detected spike, before curation dropped any, with labels."""
    st = np.load(path / "full_st.npy")
    st = (st[:, 0] if st.ndim == 2 else st).reshape(-1).astype(np.int64)
    cl = np.load(path / "full_clu.npy").reshape(-1).astype(np.int64)
    order = np.argsort(st, kind="stable")
    labels = pd.read_csv(path / "cluster_KSLabel.tsv", sep="\t")
    col = next(c for c in labels.columns if c != "cluster_id")
    lab = dict(
        zip(
            labels["cluster_id"],
            labels[col].astype(str).str.strip().str.lower(),
        )
    )
    return {"st": st[order], "cl": cl[order], "label": lab}


def cluster_amplitude(path: Path) -> dict:
    amp = pd.read_csv(path / "cluster_Amplitude.tsv", sep="\t")
    col = next(c for c in amp.columns if c != "cluster_id")
    return dict(zip(amp["cluster_id"], amp[col].astype(float)))


def spike_distribution(a_st: np.ndarray, sort: dict) -> tuple:
    """(frac of a_st found anywhere in sort, ordered (cluster, frac, label))."""
    hit, hit_cl = nearest_hit(a_st, sort["st"], sort["cl"])
    n = len(a_st)
    if not hit.any():
        return 0.0, []
    ids, cnt = np.unique(hit_cl[hit], return_counts=True)
    order = np.argsort(cnt)[::-1]
    ranked = [
        (int(ids[k]), float(cnt[k] / n), sort["label"].get(int(ids[k]), "?"))
        for k in order
    ]
    return float(hit.mean()), ranked


def classify(frac_found: float, ranked: list, merge_targets: set) -> str:
    if frac_found < DETECTION_FLOOR or not ranked:
        return "absent at detection"
    top_cl, p1, top_label = ranked[0]
    p2 = ranked[1][1] if len(ranked) > 1 else 0.0
    if p1 >= DOMINANT_FRAC:
        if top_cl in merge_targets:
            return "merged into a rescue good unit"
        return "preserved as MUA" if top_label == "mua" else (
            "merged into a rescue good unit"
        )
    if p1 >= SPLIT_FRAC and p2 >= SPLIT_FRAC:
        return "split across rescue clusters"
    return "dispersed across rescue clusters"


def dominant_partner_counts(legacy: dict, rescue_curated: dict,
                            legacy_good: list) -> dict:
    """For every legacy good unit, its dominant rescue-curated cluster.

    A rescue cluster that is the dominant partner (>= DOMINANT_FRAC) of two or
    more legacy good units is absorbing a merge.
    """
    claims: dict[int, int] = {}
    for cid in legacy_good:
        a = legacy["st"][legacy["cl"] == cid]
        _, ranked = spike_distribution(a, rescue_curated)
        if ranked and ranked[0][1] >= DOMINANT_FRAC:
            claims[ranked[0][0]] = claims.get(ranked[0][0], 0) + 1
    return {cl for cl, k in claims.items() if k >= 2}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    legacy = load_sort(LEGACY)
    rescue = load_sort(RESCUE)
    rescue_full = load_full_sort(RESCUE)
    legacy_amp = cluster_amplitude(LEGACY)

    matches = mutual_best_matches(rescue, legacy)
    matched_legacy = set(matches.legacy_cluster)
    lost = sorted(legacy["good"] - matched_legacy)
    print(
        f"legacy KS-good units: {len(legacy['good'])}\n"
        f"  matched to a rescue good unit : {len(matched_legacy)}\n"
        f"  NOT reproduced by rescue      : {len(lost)}\n"
    )

    merge_targets = dominant_partner_counts(
        legacy, rescue, sorted(legacy["good"])
    )

    rows = []
    for cid in lost:
        a = legacy["st"][legacy["cl"] == cid]
        n = len(a)
        frac_curated, ranked_curated = spike_distribution(a, rescue)
        frac_full, ranked_full = spike_distribution(a, rescue_full)

        top_cl, p1, top_label = (ranked_curated[0] if ranked_curated
                                 else (-1, 0.0, "none"))
        isi = np.diff(np.sort(a)) / FS * 1000.0
        rv = float((isi < REFRACTORY_MS).mean()) if n > 1 else np.nan

        cls = classify(frac_curated, ranked_curated, merge_targets)
        # A unit absent from the curated sort but present pre-curation was
        # detected then dropped by curation, not missed at detection.
        dropped_by_curation = (
            cls == "absent at detection"
            and frac_full >= DETECTION_FLOOR
        )
        if dropped_by_curation:
            cls = "removed by curation"

        rows.append({
            "legacy_cluster": int(cid),
            "n_spikes": n,
            "rate_hz": n / DURATION_S,
            "amplitude_uv": legacy_amp.get(int(cid), np.nan),
            "rv_frac": rv,
            "frac_found_in_rescue_curated": frac_curated,
            "frac_found_in_rescue_full": frac_full,
            "best_rescue_cluster": top_cl,
            "best_rescue_label": top_label,
            "best_rescue_frac": p1,
            "second_rescue_frac": (ranked_curated[1][1]
                                   if len(ranked_curated) > 1 else 0.0),
            "n_rescue_clusters_touched": len(ranked_curated),
            "classification": cls,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT / "legacy_lost_good_classified.csv", index=False)

    print("=== Where the 127 legacy-good units rescue drops actually go ===")
    summary = (
        df.groupby("classification")
        .agg(
            n=("legacy_cluster", "size"),
            median_found_curated=("frac_found_in_rescue_curated", "median"),
            median_found_full=("frac_found_in_rescue_full", "median"),
            median_best_partner=("best_rescue_frac", "median"),
            median_rate_hz=("rate_hz", "median"),
            median_amp_uv=("amplitude_uv", "median"),
            median_rv=("rv_frac", "median"),
            frac_rv_over_1pct=("rv_frac", lambda x: float((x > 0.01).mean())),
        )
        .sort_values("n", ascending=False)
    )
    print(summary.to_string(float_format=lambda v: f"{v:.3g}"))
    summary.to_csv(OUTPUT / "classification_summary.csv")

    absent = df[df.classification == "absent at detection"]
    print(
        f"\n=== The decisive class: absent at detection (n={len(absent)}) ===\n"
        "Checkpoint A: are these clean, well-formed neurons lost before the\n"
        "sorter, or were they marginal in legacy too?\n"
    )
    if len(absent):
        clean = absent[(absent.rv_frac <= 0.01)]
        print(
            f"  refractory-clean (rv <= 1%)      : {len(clean)} / {len(absent)}\n"
            f"  median rate                      : {absent.rate_hz.median():.3g} Hz\n"
            f"  median amplitude                 : {absent.amplitude_uv.median():.4g} uV\n"
            f"  median refractory violation      : {absent.rv_frac.median():.3g}\n"
            f"  n with rate > 0.5 Hz and rv<=1%  : "
            f"{int(((absent.rate_hz > 0.5) & (absent.rv_frac <= 0.01)).sum())}"
        )
        absent.sort_values("rate_hz", ascending=False).to_csv(
            OUTPUT / "absent_at_detection.csv", index=False
        )

    # The symmetric +200 / -127 table, both sides classified.
    plus = _read_plus_side()
    if plus is not None:
        symmetric = _symmetric_table(plus, df)
        symmetric.to_csv(OUTPUT / "symmetric_200_minus_127.csv", index=False)
        print("\n=== Symmetric relabelling table (+ gains / - losses) ===")
        print(symmetric.to_string(index=False))

    print(f"\nwrote {OUTPUT}")


def _read_plus_side() -> pd.DataFrame | None:
    path = (REPO_ROOT / "testing/outputs/luke_rescue_unique_units_audit"
            / "rescue_unique_all_good_classified.csv")
    if not path.exists():
        print(f"\n(skip symmetric table: {path} not found; "
              "run luke_rescue_unique_units_audit.py first)")
        return None
    return pd.read_csv(path)


def _symmetric_table(plus: pd.DataFrame, minus: pd.DataFrame) -> pd.DataFrame:
    rows = [{"side": "+ gained", "classification": k, "n": int(v)}
            for k, v in plus.classification.value_counts().items()]
    rows += [{"side": "- lost", "classification": k, "n": int(v)}
             for k, v in minus.classification.value_counts().items()]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
