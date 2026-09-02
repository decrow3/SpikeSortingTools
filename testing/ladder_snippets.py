"""Frozen snippet panel — the evaluation ladder's second core asset (plan §4).

A snippet is a **time window × depth strip** cut from the accepted rescue
recording, saved as a SpikeInterface binary folder plus a sealed manifest, and
never rebuilt. The panel is split into a **development** half and a **held-out**
half; the held-out half is opened once per promotion attempt (§4 rules).

Discipline this module enforces (plan §4):

* Panel selection uses **input-side and estimator-side signatures only** — a
  `SnippetSpec` must carry a non-empty `selection_basis` and its `axes` may not
  name sorter output. The time windows come from
  `testing/luke_motion_regime_windows.py`, which already never reads labels.
* Every snippet is content-hashed. `verify_snippet` recomputes the hash, so a
  silently mutated panel cannot pass. A panel change invalidates every cached
  score by construction (the digest is part of the L1 cache key).
* Nothing is written under `/mnt`. Snippets live on local disk — default
  `/media/huklab/Data/ladder_snippets`, override with `$LADDER_SNIPPET_ROOT` or
  `configs/ladder.toml`.

Injection contract (Phase C): `Snippet.raw_domain_float32()` returns the
`gain_uv_per_count`-scaled float32 voltage view. Injection happens there, never
into the stored int16 (matches `luke_injected_ground_truth_benchmark.py`).
"""

from __future__ import annotations

import hashlib
import json
import os
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from pipeline.config import PIPELINE_VERSION, fingerprint
from pipeline.bakeoff import resolve_bakeoff_window
from pipeline.preprocess import (
    MANIFEST_NAME,
    RECORDING_MANIFEST_SCHEMA,
    recording_binary_receipt,
)

SNIPPET_SCHEMA = "luke-ladder-snippet-v1"
PANEL_SCHEMA = "luke-ladder-panel-v1"

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SNIPPET_ROOT = Path("/media/huklab/Data/ladder_snippets")

SPLITS = ("development", "held_out")
AXES = ("motion_regime", "snr", "depth_strip", "artifact_proximity")
# Substrings that betray a label-derived selection; a spec naming one is rejected.
FORBIDDEN_BASIS_TOKENS = ("ks_good", " kslabel", "sorter label", "curated", "unit count")


def snippet_root() -> Path:
    """Resolve the local snippet store. Never `/mnt`."""
    env = os.environ.get("LADDER_SNIPPET_ROOT")
    if env:
        return Path(env)
    cfg = REPO_ROOT / "configs" / "ladder.toml"
    if cfg.is_file():
        data = tomllib.loads(cfg.read_text())
        if "snippet_root" in data:
            return Path(data["snippet_root"])
    return DEFAULT_SNIPPET_ROOT


# --------------------------------------------------------------------------- #
# spec
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SnippetSpec:
    """One panel entry. Selection is input-/estimator-side only (plan §4)."""

    name: str
    start_s: float
    duration_s: float
    channel_start: int
    channel_count: int
    split: str
    selection_basis: str
    axes: dict[str, str] = field(default_factory=dict)
    window_name: str = "snippet"

    def __post_init__(self) -> None:
        if self.split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {self.split!r}")
        if not self.selection_basis.strip():
            raise ValueError(
                f"snippet {self.name!r} has no selection_basis; panel selection "
                "must cite an input-side or estimator-side signature (plan §4)"
            )
        haystack = (
            self.selection_basis + " " + " ".join(map(str, self.axes.values()))
        ).lower()
        hit = next((t for t in FORBIDDEN_BASIS_TOKENS if t.strip() in haystack), None)
        if hit is not None:
            raise ValueError(
                f"snippet {self.name!r} selection references {hit!r}: panel "
                "selection must not use sorter labels or challenger results"
            )
        if self.channel_count <= 0 or self.channel_start < 0:
            raise ValueError("channel_start >= 0 and channel_count > 0 required")
        if self.duration_s <= 0 or self.start_s < 0:
            raise ValueError("start_s >= 0 and duration_s > 0 required")

    def as_dict(self) -> dict[str, Any]:
        return {"snippet_schema": SNIPPET_SCHEMA, **asdict(self)}

    @property
    def digest(self) -> str:
        return fingerprint(self.as_dict())

    @property
    def directory_name(self) -> str:
        return f"{self.name}-{self.digest[:10]}"


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
_RECORDING_MANIFEST_KEYS = (
    "request_digest",
    "sampling_frequency_hz",
    "selected_start_frame",
    "selected_end_frame",
    "gain_uv_per_count",
)


def _read_recording_manifest(recording_dir: Path) -> dict:
    """Accept either the v2 schema or the older v1 manifest (imec1).

    The v1 manifest predates `schema_version` and the content receipt, but it
    still carries everything the snippet builder needs; the snippet computes its
    own content hash from the bytes it writes.
    """
    path = recording_dir / "rescue_recording_manifest.json"
    manifest = json.loads(path.read_text())
    schema_ok = str(manifest.get("schema_version", "")).startswith(
        "rescue-recording-manifest"
    )
    fields_ok = all(k in manifest for k in _RECORDING_MANIFEST_KEYS)
    pipeline_ok = str(manifest.get("pipeline_version", "")).startswith(
        "spikeglx-ext-ref-rescue"
    )
    if not (schema_ok or (fields_ok and pipeline_ok)):
        raise ValueError(f"{path} is not a recognisable rescue recording manifest")
    return manifest


def _sha256_file(path: Path, _buf: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_buf), b""):
            h.update(chunk)
    return h.hexdigest()


def _content_hash(snippet_dir: Path, spec: SnippetSpec) -> str:
    """Seal the traces, the probe, and the spec together."""
    h = hashlib.sha256()
    h.update(spec.digest.encode())
    for name in ("traces_cached_seg0.raw", "probe.json"):
        p = snippet_dir / name
        h.update(name.encode())
        h.update(_sha256_file(p).encode())
    return h.hexdigest()


def build_snippet(
    spec: SnippetSpec,
    recording_dir: Path | str,
    out_root: Path | str | None = None,
    *,
    n_jobs: int = 8,
    chunk_duration: str = "10s",
    overwrite: bool = False,
) -> dict:
    """Cut, freeze and seal one snippet. Returns its manifest."""
    from spikeinterface.core import load

    recording_dir = Path(recording_dir)
    out_root = Path(out_root) if out_root is not None else snippet_root()
    if str(out_root).startswith("/mnt/"):
        raise ValueError("refusing to write a snippet under /mnt (plan §4)")

    recording_manifest = _read_recording_manifest(recording_dir)
    window = resolve_bakeoff_window(
        recording_manifest,
        name=spec.window_name,
        start_s=spec.start_s,
        duration_s=spec.duration_s,
    )

    recording = load(recording_dir)
    full = int(
        recording_manifest["selected_end_frame"]
        - recording_manifest["selected_start_frame"]
    )
    if recording.get_num_samples() != full:
        raise RuntimeError("accepted recording length differs from its manifest")

    sliced = recording.frame_slice(
        start_frame=window.start_frame, end_frame=window.end_frame
    )
    ch_stop = spec.channel_start + spec.channel_count
    if ch_stop > sliced.get_num_channels():
        raise ValueError(
            f"channel window [{spec.channel_start}, {ch_stop}) exceeds "
            f"{sliced.get_num_channels()} channels"
        )
    ch_ids = sliced.channel_ids[spec.channel_start : ch_stop]
    sliced = sliced.select_channels(channel_ids=ch_ids)

    out_dir = out_root / spec.directory_name
    if out_dir.exists():
        if not overwrite:
            existing = json.loads((out_dir / "snippet_manifest.json").read_text())
            if (
                existing.get("content_sha256")
                and (out_dir / MANIFEST_NAME).exists()
                and verify_snippet(out_dir)
            ):
                return existing
        _rmtree(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    sliced.save(
        folder=out_dir,
        dtype="int16",
        n_jobs=n_jobs,
        chunk_duration=chunk_duration,
        progress_bar=False,
    )
    locations = np.asarray(sliced.get_channel_locations(), dtype=np.float64)
    np.save(out_dir / "channel_positions.npy", locations)
    np.save(
        out_dir / "source_channel_indices.npy",
        np.arange(spec.channel_start, ch_stop, dtype=np.int64),
    )

    # A snippet is also a standalone accepted recording, so pipeline.sorting and
    # pipeline.downstream run over it unchanged (the L1 runner). The conditioning
    # graph was already applied to the source; the snippet inherits it.
    _write_recording_manifest(out_dir, spec, window, recording_manifest, sliced)

    manifest = {
        "snippet_schema": SNIPPET_SCHEMA,
        "pipeline_version": PIPELINE_VERSION,
        "name": spec.name,
        "spec": spec.as_dict(),
        "spec_digest": spec.digest,
        "split": spec.split,
        "axes": spec.axes,
        "selection_basis": spec.selection_basis,
        "recording_request_digest": recording_manifest["request_digest"],
        "source_recording_dir": str(recording_dir.resolve()),
        "window": {
            "name": window.name,
            "start_frame": window.start_frame,
            "end_frame": window.end_frame,
            "source_start_frame": window.source_start_frame,
            "source_end_frame": window.source_end_frame,
            "start_s": window.start_s,
            "duration_s": window.duration_s,
            "request_digest": window.request_digest,
        },
        "channel_start": spec.channel_start,
        "channel_count": spec.channel_count,
        "n_samples": window.frame_count,
        "sampling_frequency_hz": float(recording_manifest["sampling_frequency_hz"]),
        "gain_uv_per_count": float(recording_manifest["gain_uv_per_count"]),
        "depth_um_range": [float(locations[:, 1].min()), float(locations[:, 1].max())],
    }
    manifest["content_sha256"] = _content_hash(out_dir, spec)
    manifest["manifest_digest"] = fingerprint(manifest)
    (out_dir / "snippet_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest


def _write_recording_manifest(
    out_dir: Path, spec: SnippetSpec, window, source_manifest: dict, sliced
) -> None:
    receipt = recording_binary_receipt(out_dir)
    request = {
        "pipeline_version": PIPELINE_VERSION,
        "kind": "ladder_snippet",
        "parent_recording_request_digest": source_manifest["request_digest"],
        "spec_digest": spec.digest,
    }
    manifest = {
        "schema_version": RECORDING_MANIFEST_SCHEMA,
        "pipeline_version": PIPELINE_VERSION,
        "kind": "ladder_snippet",
        "complete": True,
        "request_digest": fingerprint(request),
        "parent_recording_request_digest": source_manifest["request_digest"],
        "spec_digest": spec.digest,
        "source_folder": source_manifest.get("source_folder"),
        "stream_id": source_manifest.get("stream_id"),
        "num_samples": int(window.frame_count),
        "num_channels": int(sliced.get_num_channels()),
        "sampling_frequency_hz": float(source_manifest["sampling_frequency_hz"]),
        "dtype": "int16",
        "selected_start_frame": 0,
        "selected_end_frame": int(window.frame_count),
        "gain_uv_per_count": float(source_manifest["gain_uv_per_count"]),
        "graph": source_manifest.get("graph"),
        "bad_channel_ids": source_manifest.get("bad_channel_ids"),
        "expected_binary_bytes": receipt["actual_binary_bytes"],
        "recording_content_sha256": receipt["recording_content_sha256"],
        "recording_binary_files": receipt["recording_binary_files"],
    }
    (out_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")


def _rmtree(path: Path) -> None:
    import shutil

    shutil.rmtree(path, ignore_errors=True)


# --------------------------------------------------------------------------- #
# load / verify
# --------------------------------------------------------------------------- #
@dataclass
class Snippet:
    dir: Path
    manifest: dict

    @property
    def fs(self) -> float:
        return float(self.manifest["sampling_frequency_hz"])

    @property
    def duration_s(self) -> float:
        return float(self.manifest["window"]["duration_s"])

    @property
    def n_channels(self) -> int:
        return int(self.manifest["channel_count"])

    @property
    def gain_uv_per_count(self) -> float:
        return float(self.manifest["gain_uv_per_count"])

    @property
    def channel_positions(self) -> np.ndarray:
        return np.load(self.dir / "channel_positions.npy")

    def recording(self):
        from spikeinterface.core import load

        return load(self.dir)

    def traces_int16(self) -> np.ndarray:
        """Read-only memmap of the stored int16 voltage, shape (n_samp, n_chan)."""
        rec = self.recording()
        return rec.get_traces()

    def raw_domain_float32(self) -> np.ndarray:
        """float32 µV view for injection (Phase C). Never mutate the int16 store."""
        return self.traces_int16().astype(np.float32) * np.float32(
            self.gain_uv_per_count
        )


def load_snippet(snippet_dir: Path | str) -> Snippet:
    snippet_dir = Path(snippet_dir)
    manifest = json.loads((snippet_dir / "snippet_manifest.json").read_text())
    if manifest.get("snippet_schema") != SNIPPET_SCHEMA:
        raise ValueError(f"{snippet_dir} is not a {SNIPPET_SCHEMA} snippet")
    return Snippet(dir=snippet_dir, manifest=manifest)


def verify_snippet(snippet_dir: Path | str) -> bool:
    """True iff the stored content still hashes to the sealed value."""
    snippet_dir = Path(snippet_dir)
    manifest = json.loads((snippet_dir / "snippet_manifest.json").read_text())
    spec = SnippetSpec(
        **{k: v for k, v in manifest["spec"].items() if k != "snippet_schema"}
    )
    return _content_hash(snippet_dir, spec) == manifest.get("content_sha256")


# --------------------------------------------------------------------------- #
# panel
# --------------------------------------------------------------------------- #
def freeze_panel(
    specs: list[SnippetSpec],
    recording_dir: Path | str,
    out_root: Path | str | None = None,
    *,
    require_balanced: bool = True,
    **build_kwargs,
) -> dict:
    """Build every snippet and seal the panel. 8 development + 8 held out (§4)."""
    names = [s.name for s in specs]
    if len(names) != len(set(names)):
        raise ValueError("snippet names must be unique")
    by_split = {s: [x for x in specs if x.split == s] for s in SPLITS}
    if require_balanced and not (
        len(by_split["development"]) == 8 and len(by_split["held_out"]) == 8
    ):
        raise ValueError(
            "panel must be 8 development + 8 held out; got "
            f"{len(by_split['development'])} + {len(by_split['held_out'])}"
        )

    out_root = Path(out_root) if out_root is not None else snippet_root()
    built = [
        build_snippet(spec, recording_dir, out_root, **build_kwargs) for spec in specs
    ]
    panel = {
        "panel_schema": PANEL_SCHEMA,
        "pipeline_version": PIPELINE_VERSION,
        "n_development": len(by_split["development"]),
        "n_held_out": len(by_split["held_out"]),
        "snippets": [
            {
                "name": m["name"],
                "split": m["split"],
                "directory": (out_root / f"{m['name']}-{m['spec_digest'][:10]}").name,
                "spec_digest": m["spec_digest"],
                "content_sha256": m["content_sha256"],
                "axes": m["axes"],
            }
            for m in built
        ],
    }
    panel["panel_digest"] = fingerprint(panel)
    (out_root / "panel_manifest.json").write_text(json.dumps(panel, indent=2) + "\n")
    return panel
