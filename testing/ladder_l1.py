"""L1 runner — one snippet, the full pipeline, scored, cached (plan §3).

`l1_run(snippet_dir)` takes a frozen snippet through
`sort → curate → score` and returns the same score dictionary `score_sort`
produces at L4 (plan §3 rule 1). The stages are cached on content so an
iteration only recomputes what changed (plan §3 caching contract):

    <l1_root>/<snippet spec digest>/
        sort/                     ← per snippet; reused across curation variants
        cur-<curation digest>/    ← per curation config
            l1_result.json

A curation-parameter change reuses the sort. Only a snippet change invalidates
the sort — and a snippet change is a new spec digest, so it lands in a new
directory by construction.

Per §3 rule 2, every run also records cheap per-stage observables (detection
counts, unit counts, amplitude spread) so per-stage auditing is a by-product of
every end-to-end iteration, not a separate project.

Nothing is written under `/mnt`.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from pipeline.config import PIPELINE_VERSION, fingerprint
from pipeline.downstream import build_sort_identity, run_curation_stage
from pipeline.sorting import SORT_MANIFEST, run_kilosort4
from testing.ladder_score import score_sort
from testing.ladder_snippets import load_snippet, snippet_root

L1_SCHEMA = "luke-ladder-l1-v1"
DEFAULT_L1_ROOT = Path("/media/huklab/Data/ladder_l1")
L1_WALL_BUDGET_S = 300.0  # §3: L1 target < 5 min


def l1_root() -> Path:
    env = os.environ.get("LADDER_L1_ROOT")
    if env:
        return Path(env)
    cfg = Path(__file__).resolve().parents[1] / "configs" / "ladder.toml"
    if cfg.is_file():
        import tomllib

        data = tomllib.loads(cfg.read_text())
        if "l1_root" in data:
            return Path(data["l1_root"])
    return DEFAULT_L1_ROOT


@dataclass(frozen=True)
class CurationConfig:
    """The curation knobs Phase D is allowed to vary (plan §6 Phase D)."""

    cosine_threshold: float = 0.90
    ccg_threshold: float = 0.5

    @property
    def digest(self) -> str:
        return fingerprint({"stage": "curation", **asdict(self)})


def _amplitude_observables(sorter_output: Path) -> dict:
    path = sorter_output / "amplitudes.npy"
    if not path.exists():
        return {}
    amp = np.abs(np.load(path, mmap_mode="r").reshape(-1).astype(np.float64))
    if amp.size == 0:
        return {}
    p10, p50, p90 = np.percentile(amp, [10, 50, 90])
    return {
        "amplitude_p10": float(p10),
        "amplitude_p50": float(p50),
        "amplitude_p90": float(p90),
    }


def _stage_observables(sort_dir: Path, curated: Path) -> dict:
    """Cheap per-stage diagnostics, recorded on every run (plan §3 rule 2)."""
    sort_manifest = json.loads((sort_dir / SORT_MANIFEST).read_text())
    obs = {"sort_summary": sort_manifest.get("summary", {})}
    obs.update(_amplitude_observables(sort_dir / "sorter_output"))
    clu = np.load(curated / "spike_clusters.npy", mmap_mode="r").reshape(-1)
    obs["curated_spike_count"] = int(clu.size)
    obs["curated_unit_count"] = int(np.unique(clu).size)
    return obs


def l1_run(
    snippet_dir: Path | str,
    *,
    sorter: "SorterConfig | None" = None,
    curation: CurationConfig | None = None,
    reference: Path | str | None = None,
    truth: dict | None = None,
    out_root: Path | str | None = None,
    wall_budget_s: float = L1_WALL_BUDGET_S,
) -> dict:
    """Run one snippet through sort → curate → score, cached. Returns the score.

    `sorter` defaults to the frozen rescue config (`pipeline.sorting.run_kilosort4`,
    the production path). Pass a `ladder_sorter.SorterConfig` for a comparator or
    Phase D candidate — it caches at its own `sort-<digest>/` leaf.
    """
    curation = curation or CurationConfig()
    snippet = load_snippet(snippet_dir)
    out_root = Path(out_root) if out_root is not None else l1_root()
    if str(out_root).startswith("/mnt/"):
        raise ValueError("refusing to write L1 outputs under /mnt (plan §3)")

    is_rescue = sorter is None or getattr(sorter, "label", "rescue") == "rescue"
    work = out_root / snippet.manifest["spec_digest"][:16]
    sort_dir = work / ("sort" if is_rescue else f"sort-{sorter.digest[:12]}")
    cur_dir = work / (
        f"cur-{curation.digest[:12]}"
        if is_rescue
        else f"cur-{sorter.digest[:12]}-{curation.digest[:12]}"
    )
    work.mkdir(parents=True, exist_ok=True)

    # --- stage 1: sort (cached per snippet × sorter config) ---------------- #
    t0 = time.time()
    sort_cached = (sort_dir / SORT_MANIFEST).exists()
    if not sort_cached:
        if is_rescue:
            run_kilosort4(snippet.dir, sort_dir)
        else:
            from testing.ladder_sorter import run_sorter_config

            run_sorter_config(snippet.dir, sort_dir, sorter)
    sort_wall = 0.0 if sort_cached else time.time() - t0
    sorter_output = sort_dir / "sorter_output"

    # --- stage 2: curation (cached per curation config) -------------------- #
    identity = build_sort_identity(sort_dir)
    t1 = time.time()
    cur_cached = (cur_dir / "cur_output" / "cluster_KSLabel.tsv").exists()
    run_curation_stage(
        sorter_output,
        cur_dir,
        identity,
        cosine_threshold=curation.cosine_threshold,
        ccg_threshold=curation.ccg_threshold,
    )
    cur_wall = 0.0 if cur_cached else time.time() - t1
    curated = cur_dir / "cur_output"

    # --- stage 3: score (§3 rule 1: identical dict at L1 and L4) ----------- #
    pipeline_wall = sort_wall + cur_wall
    window = snippet.manifest.get("window", {})
    ref_window = None
    if reference is not None and "source_start_frame" in window:
        ref_window = (window["source_start_frame"], window["source_end_frame"])
    ref_depth = tuple(snippet.manifest["depth_um_range"]) if (
        reference is not None and "depth_um_range" in snippet.manifest
    ) else None
    t2 = time.time()
    score = score_sort(
        curated,
        truth=truth,
        reference=reference,
        reference_window=ref_window,
        reference_depth_range=ref_depth,
        runtime_s=pipeline_wall or None,
        fs=snippet.fs,
        duration_s=snippet.duration_s,
    )
    score_wall = time.time() - t2

    result = {
        "schema": L1_SCHEMA,
        "pipeline_version": PIPELINE_VERSION,
        "snippet_dir": str(snippet.dir),
        "spec_digest": snippet.manifest["spec_digest"],
        "snippet_axes": snippet.manifest.get("axes", {}),
        "sorter_config": "rescue" if is_rescue else sorter.label,
        "sorter_overrides": {} if is_rescue else dict(sorter.overrides),
        "curation_config": asdict(curation),
        "curation_digest": curation.digest,
        "wall_clock": {
            "sort_s": sort_wall,
            "curation_s": cur_wall,
            "score_s": score_wall,
            "pipeline_s": pipeline_wall,
            "sort_was_cached": sort_cached,
            "curation_was_cached": cur_cached,
        },
        "within_l1_budget": bool(
            (pipeline_wall if not (sort_cached and cur_cached) else 0.0)
            <= wall_budget_s
        ),
        "stage_observables": _stage_observables(sort_dir, curated),
        "score": score,
    }
    (cur_dir / "l1_result.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    return result


def _cli(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("snippet_dir", type=Path)
    ap.add_argument("--reference", type=Path)
    ap.add_argument("--cosine-threshold", type=float, default=0.90)
    ap.add_argument("--ccg-threshold", type=float, default=0.5)
    ap.add_argument("--out-root", type=Path)
    args = ap.parse_args(argv)
    result = l1_run(
        args.snippet_dir,
        curation=CurationConfig(
            cosine_threshold=args.cosine_threshold, ccg_threshold=args.ccg_threshold
        ),
        reference=args.reference,
        out_root=args.out_root,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
