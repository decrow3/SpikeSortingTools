"""Motion-aware family stitching — the first Phase D candidate.

A2 + C2 (`docs/pipeline_improvement_plan.md`): rescue's KS-good deficit against
legacy is not over-splitting and not detection loss — it is one clean neuron
whose spikes are partitioned across several templates as it drifts. The
fragments are **temporally complementary** (they own successive epochs, ~0 %
co-fire) and **refractory-clean when merged**. Production curation
(`pipeline.curation`) will not merge them: its CCG gate *requires* the two units
to co-fire cleanly, which temporally-complementary fragments do not.

This module adds the missing pass. It merges a group of KS-good units when all
of:

* **spatial** — peak channels within `depth_window_um` (wide enough to span a
  drift trajectory), and the depth-aligned template cosine ≥ `min_shifted_cosine`;
* **temporal** — pairwise `temporal_overlap` ≤ `max_temporal_overlap` (the A2
  "successive" metric: Σ min / Σ max of per-bin spike shares — 0 = strictly one
  at a time);
* **refractory** — the merged spike train's ISI-violation fraction ≤
  `max_merged_rv_frac`.

Units become nodes; an edge is drawn only between *mutual best partners* (each
one's highest-cosine stitchable partner is the other), because on a full session
the pairwise gates alone link hundreds of unrelated low-rate pairs and a
transitive closure collapses the probe into a few mega-components. Connected
components of those edges (capped at `max_family_size`) become families.
`apply_stitch` writes a new curated output with each family relabelled to its
largest member.

`stitch_families` is a pure function of an existing curated sort — no sorter, no
`/mnt`. `l1_run(stitch=…)` runs it as a post-curation cache leaf.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.config import fingerprint
from testing.ladder_score import coincident_mask

STITCH_SCHEMA = "luke-ladder-stitch-v1"


@dataclass(frozen=True)
class StitchConfig:
    # Wide enough to span a drift trajectory; the C2 experiment showed genuine
    # fragments of one 40 µm-drifting neuron land ~140 µm apart in peak-channel
    # depth once KS4's low-count templates are involved.
    depth_window_um: float = 150.0
    # A low floor only — filters obviously-unrelated templates. Fragment
    # templates of the same drifting neuron are unreliable (few spikes,
    # whitened), so this cannot be the primary gate.
    min_shifted_cosine: float = 0.30
    max_temporal_overlap: float = 0.25
    # The primary gate: merging genuine fragments of one neuron stays
    # refractory-clean; merging in a contaminated cluster does not.
    max_merged_rv_frac: float = 0.010
    max_coincident_frac: float = 0.03
    refractory_ms: float = 1.5
    min_spikes: int = 40
    include_mua: bool = True
    bin_s: float = 30.0
    max_family_size: int = 4

    @property
    def digest(self) -> str:
        return fingerprint({"stage": "stitch", **asdict(self)})


def _load(sort_dir: Path) -> dict:
    sort_dir = Path(sort_dir)
    st = np.load(sort_dir / "spike_times.npy").reshape(-1).astype(np.int64)
    cl = np.load(sort_dir / "spike_clusters.npy").reshape(-1).astype(np.int64)
    templates = np.load(sort_dir / "templates.npy")  # (n_units, n_samp, n_chan)
    positions = np.load(sort_dir / "channel_positions.npy")
    labels = pd.read_csv(sort_dir / "cluster_KSLabel.tsv", sep="\t")
    col = next(c for c in labels.columns if c != "cluster_id")
    lab = dict(
        zip(labels["cluster_id"], labels[col].astype(str).str.strip().str.lower())
    )
    fs = _sample_rate(sort_dir)
    # Per-cluster spike times, computed once — the pairwise pass would otherwise
    # re-scan the whole (millions-long) spike array for every candidate pair.
    order = np.argsort(cl, kind="stable")
    cl_sorted, st_sorted = cl[order], st[order]
    bounds = np.searchsorted(cl_sorted, np.arange(cl_sorted.max() + 2)) if cl.size else np.array([0])
    by_cluster = {
        int(c): np.sort(st_sorted[bounds[c]:bounds[c + 1]])
        for c in range(len(bounds) - 1)
        if bounds[c + 1] > bounds[c]
    }
    return {
        "st": st, "cl": cl, "templates": templates, "positions": positions,
        "label": lab, "good": [c for c, v in lab.items() if v == "good"],
        "mua": [c for c, v in lab.items() if v == "mua"], "fs": fs,
        "by_cluster": by_cluster,
    }


def _sample_rate(sort_dir: Path) -> float:
    ns: dict = {}
    exec((sort_dir / "params.py").read_text(), {}, ns)
    return float(ns["sample_rate"])


def _peak_channel(template: np.ndarray) -> int:
    return int(np.argmax(np.max(np.abs(template), axis=0)))


def _peak_depth(template: np.ndarray, positions: np.ndarray) -> float:
    return float(positions[_peak_channel(template), 1])


def shifted_cosine(t_a: np.ndarray, t_b: np.ndarray) -> float:
    """Template cosine after aligning `t_b` onto `t_a` by their peak channels.

    `t_b`'s peak sits `shift` channels from `t_a`'s; comparing `t_a[:, c]` with
    `t_b[:, c + shift]` over the channel range both templates cover.
    """
    shift = _peak_channel(t_b) - _peak_channel(t_a)
    n_chan = t_a.shape[1]
    lo, hi = max(0, -shift), min(n_chan, n_chan - shift)
    if hi <= lo:
        return 0.0
    a = t_a[:, lo:hi].ravel()
    b = t_b[:, lo + shift : hi + shift].ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


def temporal_overlap(st_a: np.ndarray, st_b: np.ndarray, fs: float, bin_s: float) -> float:
    """A2 metric: Σ min / Σ max of the two units' per-bin shares of their union."""
    if st_a.size == 0 or st_b.size == 0:
        return 1.0
    lo = min(st_a.min(), st_b.min())
    hi = max(st_a.max(), st_b.max())
    edges = np.arange(lo, hi + bin_s * fs, bin_s * fs)
    if edges.size < 2:
        return 1.0
    ha, _ = np.histogram(st_a, bins=edges)
    hb, _ = np.histogram(st_b, bins=edges)
    ra = ha / max(st_a.size, 1)
    rb = hb / max(st_b.size, 1)
    denom = np.sum(np.maximum(ra, rb))
    return float(np.sum(np.minimum(ra, rb)) / denom) if denom else 1.0


def merged_rv_fraction(st_a: np.ndarray, st_b: np.ndarray, fs: float, refractory_ms: float) -> float:
    merged = np.sort(np.concatenate([st_a, st_b]))
    if merged.size < 2:
        return 0.0
    isi_ms = np.diff(merged) / fs * 1000.0
    return float((isi_ms < refractory_ms).mean())


def _pair_qualifies(a: int, b: int, sort: dict, config: StitchConfig) -> dict | None:
    ta, tb = sort["templates"][a], sort["templates"][b]
    dz = abs(_peak_depth(ta, sort["positions"]) - _peak_depth(tb, sort["positions"]))
    if dz > config.depth_window_um:
        return None
    empty = np.empty(0, dtype=np.int64)
    st_a = sort["by_cluster"].get(a, empty)
    st_b = sort["by_cluster"].get(b, empty)
    if st_a.size < config.min_spikes or st_b.size < config.min_spikes:
        return None
    # genuinely simultaneous units are different neurons, not fragments
    if coincident_mask(st_a, st_b, int(0.5e-3 * sort["fs"])).mean() > config.max_coincident_frac:
        return None
    cos = shifted_cosine(ta, tb)
    ovl = temporal_overlap(st_a, st_b, sort["fs"], config.bin_s)
    rv = merged_rv_fraction(st_a, st_b, sort["fs"], config.refractory_ms)
    ok = (
        cos >= config.min_shifted_cosine
        and ovl <= config.max_temporal_overlap
        and rv <= config.max_merged_rv_frac
    )
    return {
        "a": a, "b": b, "depth_gap_um": round(dz, 1),
        "shifted_cosine": round(cos, 3), "temporal_overlap": round(ovl, 3),
        "merged_rv_frac": round(rv, 5), "stitch": bool(ok),
    }


def _connected_components(edges: list[tuple[int, int]], nodes: set[int]) -> list[list[int]]:
    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        parent[find(a)] = find(b)
    groups: dict[int, list[int]] = {}
    for n in nodes:
        groups.setdefault(find(n), []).append(n)
    return [sorted(g) for g in groups.values() if len(g) > 1]


def stitch_families(sort_dir: Path | str, config: StitchConfig | None = None) -> dict:
    config = config or StitchConfig()
    sort = _load(sort_dir)
    good_set = set(sort["good"])
    nodes = sorted(good_set | (set(sort["mua"]) if config.include_mua else set()))
    depths = {c: _peak_depth(sort["templates"][c], sort["positions"]) for c in nodes}

    pair_rows, stitchable = [], []
    for i, a in enumerate(nodes):
        for b in nodes[i + 1 :]:
            if abs(depths[a] - depths[b]) > config.depth_window_um:
                continue
            row = _pair_qualifies(a, b, sort, config)
            if row is None:
                continue
            pair_rows.append(row)
            if row["stitch"]:
                stitchable.append(row)

    # Mutual-best-partner edges only. On a full session the pairwise gates
    # (successive epochs, refractory-clean merge, close depth) are satisfied by
    # hundreds of unrelated low-rate pairs, and a transitive closure over them
    # collapses the probe into a few mega-components. Keeping an edge only when
    # each endpoint's *best* stitchable partner (by template cosine) is the
    # other keeps families to genuine drift chains.
    best: dict[int, tuple[float, int]] = {}
    for row in stitchable:
        for u, v in ((row["a"], row["b"]), (row["b"], row["a"])):
            if row["shifted_cosine"] > best.get(u, (-1.0, -1))[0]:
                best[u] = (row["shifted_cosine"], v)
    edges = [
        (row["a"], row["b"])
        for row in stitchable
        if best[row["a"]][1] == row["b"] and best[row["b"]][1] == row["a"]
    ]

    families = _connected_components(edges, set(nodes))
    families = [
        f for f in families
        if len(f) <= config.max_family_size and any(u in good_set for u in f)
    ]
    good = sorted(good_set)

    good_absorbed = sum(
        sum(1 for u in f if u in good_set) - 1 for f in families
    )
    return {
        "schema": STITCH_SCHEMA,
        "sort_dir": str(sort_dir),
        "config": asdict(config),
        "config_digest": config.digest,
        "include_mua": config.include_mua,
        "n_good_before": len(good),
        "n_pairs_scored": len(pair_rows),
        "n_stitch_edges": len(edges),
        "families": families,
        "n_families": len(families),
        "n_units_absorbed": sum(len(f) - 1 for f in families),
        "n_good_absorbed": good_absorbed,
        "n_good_after": len(good) - good_absorbed,
        "pairs": pair_rows,
    }


def apply_stitch(
    sort_dir: Path | str, out_dir: Path | str, config: StitchConfig | None = None
) -> dict:
    """Write a curated output with each family relabelled to its largest member."""
    sort_dir, out_dir = Path(sort_dir), Path(out_dir)
    if str(out_dir).startswith("/mnt/"):
        raise ValueError("refusing to write a stitched sort under /mnt")
    result = stitch_families(sort_dir, config)

    st = np.load(sort_dir / "spike_times.npy").reshape(-1)
    cl = np.load(sort_dir / "spike_clusters.npy").reshape(-1).astype(np.int64)
    labels = pd.read_csv(sort_dir / "cluster_KSLabel.tsv", sep="\t")
    col = next(c for c in labels.columns if c != "cluster_id")
    good_set = set(
        labels.loc[labels[col].astype(str).str.strip().str.lower() == "good", "cluster_id"]
    )
    counts = pd.Series(cl).value_counts()

    remap = {}
    absorbed = set()
    for family in result["families"]:
        # keep a good member (the largest); the merged unit is a good unit
        good_members = [u for u in family if u in good_set] or family
        keep = max(good_members, key=lambda u: int(counts.get(u, 0)))
        for u in family:
            if u != keep:
                remap[u] = keep
                absorbed.add(u)
    new_cl = np.array([remap.get(int(c), int(c)) for c in cl], dtype=np.int64)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(sort_dir, out_dir)
    np.save(out_dir / "spike_clusters.npy", new_cl)

    labels = labels[~labels["cluster_id"].isin(absorbed)].reset_index(drop=True)
    labels.to_csv(out_dir / "cluster_KSLabel.tsv", sep="\t", index=False)
    if (out_dir / "cluster_group.tsv").exists():
        labels.rename(columns={col: "KSLabel"}).to_csv(
            out_dir / "cluster_group.tsv", sep="\t", index=False
        )
    (out_dir / "stitch_result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
