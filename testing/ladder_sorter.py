"""Config-parametrised Kilosort 4 for the ladder — the Phase D / C2 comparator.

`pipeline.sorting.run_kilosort4` runs the **frozen** rescue configuration and
validates that the rescue invariants held. The ladder needs to run *other*
configurations on the same snippet — the legacy-style motion-corrected KS4 as
C2's comparator arm, and Phase D candidate variants — without touching the
locked `pipeline/` extraction.

This module does exactly that: it starts from `build_kilosort4_params()` (so a
"config" is a small, explicit diff from the rescue baseline), runs `run_sorter`,
and writes a manifest keyed by the config digest. The cache leaf per config is
what makes `l1_run` reuse a sort across curation variants but recompute it when
the sorter config changes (plan §3 caching contract).

The only configs with a name are the ones the plan compares directly:

* `RESCUE` — the frozen baseline, no override.
* `LEGACY_STYLE` — KS4 with rigid internal drift correction and the legacy
  detection thresholds, on the *same* conditioned input. The controlled
  contrast for "what does representing motion buy?" (`ops.npy`:
  legacy `nblocks=1, Th_universal=9, Th_learned=8` vs rescue `nblocks=0, 12, 9`).
* `RESCUE_RIGID` — the controlled internal-correction arm C2 v4 needs: the
  rescue detection thresholds unchanged, with KS4's own *rigid* datashift turned
  on (`do_correction=True, nblocks=1`). `LEGACY_STYLE` changes correction *and*
  thresholds together, so it cannot isolate what `nblocks=1` buys; this can.
  Note the rescue baseline already carries `nblocks=1` but `do_correction=False`,
  and KS4 forces the effective value to 0 — so the arms must be validated on the
  *saved effective* settings, never on the requested parameter dict.
* `NONRIGID` — Phase D candidate 2: KS4's own *non-rigid* datashift
  (`do_correction=True, nblocks=6`) with the rescue detection thresholds
  unchanged. The controlled contrast against `LEGACY_STYLE` isolates
  non-rigid vs rigid; against `RESCUE`, motion representation vs none.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from pipeline.config import PIPELINE_VERSION, fingerprint
from pipeline.kilosort_compat import ensure_kilosort_compatibility
from pipeline.preprocess import MANIFEST_NAME
from pipeline.sorting import SORT_MANIFEST, build_kilosort4_params

LADDER_SORT_MANIFEST = SORT_MANIFEST  # same filename; downstream reads it unchanged


@dataclass(frozen=True)
class SorterConfig:
    """A named diff from the frozen rescue KS4 parameters."""

    label: str
    overrides: dict[str, Any] = field(default_factory=dict)

    def params(self) -> dict[str, Any]:
        merged = dict(build_kilosort4_params())
        merged.update(self.overrides)
        return merged

    @property
    def digest(self) -> str:
        safe = {
            k: ("Infinity" if isinstance(v, float) and math.isinf(v) else v)
            for k, v in self.overrides.items()
        }
        return fingerprint({"stage": "kilosort4", "label": self.label, "overrides": safe})


RESCUE = SorterConfig("rescue", {})
LEGACY_STYLE = SorterConfig(
    "legacy_style",
    {"do_correction": True, "nblocks": 1, "Th_universal": 9, "Th_learned": 8},
)
RESCUE_RIGID = SorterConfig(
    "rescue_rigid",
    {"do_correction": True, "nblocks": 1},
)
NONRIGID = SorterConfig(
    "nonrigid",
    {"do_correction": True, "nblocks": 6},
)
RESCUE_10_9 = SorterConfig(
    "rescue_10_9",
    {"Th_universal": 10, "Th_learned": 9},
)
RESCUE_9_9 = SorterConfig(
    "rescue_9_9",
    {"Th_universal": 9, "Th_learned": 9},
)
RESCUE_9_8 = SorterConfig(
    "rescue_9_8",
    {"Th_universal": 9, "Th_learned": 8},
)
NAMED_CONFIGS = {
    c.label: c
    for c in (
        RESCUE,
        LEGACY_STYLE,
        RESCUE_RIGID,
        NONRIGID,
        RESCUE_10_9,
        RESCUE_9_9,
        RESCUE_9_8,
    )
}

# What each config must resolve to once KS4 has applied it.
EXPECTED_EFFECTIVE = {
    "rescue": {"effective_nblocks": 0, "do_CAR": True,
               "Th_universal": 12, "Th_learned": 9},
    "rescue_rigid": {"effective_nblocks": 1, "do_CAR": True,
                     "Th_universal": 12, "Th_learned": 9},
    "legacy_style": {"effective_nblocks": 1, "do_CAR": True,
                     "Th_universal": 9, "Th_learned": 8},
    "nonrigid": {"effective_nblocks": 6, "do_CAR": True,
                 "Th_universal": 12, "Th_learned": 9},
    "rescue_10_9": {"effective_nblocks": 0, "do_CAR": True,
                    "Th_universal": 10, "Th_learned": 9},
    "rescue_9_9": {"effective_nblocks": 0, "do_CAR": True,
                   "Th_universal": 9, "Th_learned": 9},
    "rescue_9_8": {"effective_nblocks": 0, "do_CAR": True,
                   "Th_universal": 9, "Th_learned": 8},
}


def effective_settings(sort_manifest: dict) -> dict:
    """Canonical applied settings, from either sort path's manifest shape.

    The two paths record different things. `pipeline.sorting.run_kilosort4` (the
    `rescue` arm) writes `summary.critical_saved_settings` -- `do_correction`,
    `effective_nblocks`, `do_CAR`, `artifact_threshold` -- and does not save the
    detection thresholds. `run_sorter_config` (every comparator arm) writes flat
    `effective_nblocks` / `applied_*` keys read back out of `ops.npy`.

    `effective_nblocks` is the one KS4 rewrites (`do_correction=False` forces it
    to 0 whatever was requested), so it must come from the saved ops and this
    raises if it is absent. KS4 does not alter the detection thresholds, so when
    a path does not save them they are taken from the request -- recorded in
    `_sources` so a reader can see which is which.
    """
    summary = dict(sort_manifest.get("summary", {}))
    merged = {**dict(summary.get("critical_saved_settings", {})), **summary}
    requested = dict(sort_manifest.get("sorter_params", {}))
    sources: dict[str, str] = {}

    def applied(canonical: str, *names):
        for name in names:
            if name in merged and merged[name] is not None:
                sources[canonical] = f"applied:{name}"
                return merged[name]
        return None

    nblocks = applied("effective_nblocks", "effective_nblocks")
    if nblocks is None:
        raise RuntimeError(
            "sort manifest records no effective nblocks; correction cannot be "
            "validated from the requested parameters alone"
        )
    values = {
        "effective_nblocks": int(nblocks),
        "do_CAR": applied("do_CAR", "applied_do_CAR", "do_CAR"),
        "Th_universal": applied("Th_universal", "applied_Th_universal"),
        "Th_learned": applied("Th_learned", "applied_Th_learned"),
    }
    for canonical, request_key in (("Th_universal", "Th_universal"),
                                   ("Th_learned", "Th_learned")):
        if values[canonical] is None and request_key in requested:
            values[canonical] = requested[request_key]
            sources[canonical] = f"requested:{request_key}"
    missing = [k for k, v in values.items() if v is None]
    if missing:
        raise RuntimeError(f"sort manifest records no value for {missing}")
    values["_sources"] = sources
    return values


def check_effective_settings(label: str, sort_manifest: dict) -> dict:
    """Fail closed unless KS4 actually applied what the config asked for."""
    observed = effective_settings(sort_manifest)
    expected = EXPECTED_EFFECTIVE[label]
    mismatch = {
        k: {"expected": v, "observed": observed[k]}
        for k, v in expected.items()
        if observed[k] != v
    }
    if mismatch:
        raise RuntimeError(f"{label} resolved to unexpected settings: {mismatch}")
    return observed


def _json_safe(params: dict) -> dict:
    return {
        k: ("Infinity" if isinstance(v, float) and math.isinf(v) else v)
        for k, v in params.items()
    }


def _summary(sorter_output: Path) -> dict:
    ops = np.load(sorter_output / "ops.npy", allow_pickle=True).item()
    applied = dict(ops.get("settings", {}))
    applied.update(ops)
    clusters = np.load(sorter_output / "spike_clusters.npy", mmap_mode="r").reshape(-1)
    label_path = sorter_output / "cluster_KSLabel.tsv"
    good = None
    if label_path.exists():
        import csv

        with label_path.open(newline="") as fh:
            good = sum(
                any(str(v).strip().lower() == "good" for v in row.values())
                for row in csv.DictReader(fh, delimiter="\t")
            )
    return {
        "final_spike_count": int(clusters.size),
        "unit_count": int(np.unique(clusters).size),
        "kilosort_good_unit_count": good,
        "effective_nblocks": int(applied.get("nblocks", -1)),
        "applied_do_CAR": bool(applied.get("do_CAR", True)),
        "applied_Th_universal": applied.get("Th_universal"),
        "applied_Th_learned": applied.get("Th_learned"),
    }


def run_sorter_config(
    recording_dir: Path | str,
    output_dir: Path | str,
    config: SorterConfig = RESCUE,
) -> dict:
    """Run KS4 under `config` into `output_dir`, cached and atomic.

    `output_dir` layout matches `pipeline.sorting.run_kilosort4` (a
    `sorter_output/` child and a `rescue_sort_manifest.json`), so
    `pipeline.downstream` consumes it unchanged.
    """
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/spikeglx-rescue-numba-cache")
    recording_dir, output_dir = Path(recording_dir), Path(output_dir)
    if str(output_dir).startswith("/mnt/"):
        raise ValueError("refusing to write a ladder sort under /mnt")

    recording_manifest = json.loads((recording_dir / MANIFEST_NAME).read_text())
    params = config.params()
    request = {
        "pipeline_version": PIPELINE_VERSION,
        "kind": "ladder_sort",
        "recording_request_digest": recording_manifest.get("request_digest"),
        "sorter": "kilosort4",
        "config_label": config.label,
        "config_digest": config.digest,
        "sorter_params": _json_safe(params),
    }
    request_digest = fingerprint(request)
    manifest_path = output_dir / LADDER_SORT_MANIFEST

    if output_dir.exists():
        if not manifest_path.exists():
            raise RuntimeError(f"existing sort lacks {LADDER_SORT_MANIFEST}: {output_dir}")
        existing = json.loads(manifest_path.read_text())
        if existing.get("request_digest") != request_digest:
            raise RuntimeError("existing sort belongs to another config/recording")
        return existing

    ensure_kilosort_compatibility()
    from spikeinterface.core import load
    from spikeinterface.sorters import run_sorter

    partial = output_dir.with_name(output_dir.name + ".partial")
    if partial.exists():
        _archive(partial)
    partial.parent.mkdir(parents=True, exist_ok=True)

    recording = load(recording_dir)
    run_sorter(
        "kilosort4",
        recording,
        folder=str(partial),
        verbose=True,
        remove_existing_folder=False,
        **params,
    )
    sorter_output = partial / "sorter_output"
    if not (sorter_output / "spike_times.npy").exists():
        raise RuntimeError(f"Kilosort ended without spike_times.npy in {sorter_output}")

    manifest = {
        **request,
        "request_digest": request_digest,
        "summary": _summary(sorter_output),
        "complete": True,
    }
    (partial / LADDER_SORT_MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(partial, output_dir)
    return manifest


def _archive(partial: Path) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    os.replace(partial, partial.with_name(partial.name + f".superseded-{stamp}"))
