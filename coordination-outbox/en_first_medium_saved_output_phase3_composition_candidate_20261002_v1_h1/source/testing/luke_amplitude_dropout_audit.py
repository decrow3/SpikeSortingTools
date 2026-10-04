"""Amplitude-completeness dropout audit -- loader, replay and case selection.

Prescription: ``docs/amplitude_completeness_next_step_prescription.md``. This
module implements, in the prescription's own "Implementation order", layer 1
(loader and window-table normalization), layer 2 (historical/exact replay),
layer 3 (deterministic case selection) and the *numeric* half of the
``inspect`` stage: the provenance gate, ``case_windows.csv``, the cached-value
reproduction check and the exact-1,000 eligibility sensitivity flag.
``inventory``, ``select`` and ``inspect`` run. Still NOT implemented: the
evidence panel figures, voltage review, ``case_evidence.csv``, ``decision.md``
and the one-experiment contract.

Selection reads only the cached historical QC rows normalized into
``windows.csv``. No candidate, waveform, voltage or intervention outcome is an
input to it, and every triage constant is supplied by the JSON config (see
:class:`SelectionConstants`) rather than defaulted in code or overridden on the
command line -- the prescription requires them frozen *before* rankings are
read, so there is deliberately no CLI flag that can move one.

Selection is also bounded in TIME by the first-pipeline-candidate delivery
contract (``configs/first_pipeline_candidate.v1.json``), which partitions this
recording into a sealed held-out panel plus buffer, reserved healthy
evaluation intervals, and the development windows that remain. Every case span
must lie inside a single development window; a run outside them is excluded
before ranking (``outside_development_window``), never merely outranked, so a
stronger prohibited case cannot displace an eligible one. The contract's
intervals are re-derived and re-validated against this recording at runtime
(see :func:`read_permitted_intervals`) rather than trusted as shipped, and its
identity is recorded in ``selection.json``. The audit's own ``control`` cases
are DIAGNOSTIC controls, gated the same way and distinct from the contract's
reserved evaluation intervals.

Two verified facts from the prescription are load-bearing and must not be
"fixed" by this module:

* Production QC amplitudes are ``full_st[kept_spikes][:, 2]`` (sorter-native
  units), never ``amplitudes.npy``. :func:`load_curated_arrays` reproduces
  exactly the array ``pipeline.qc.run_qc`` feeds to
  ``pipeline.truncation.analyze_amplitude_truncation`` via
  ``pipeline.kilosort_results.KilosortResults.st``.
* ``pipeline.truncation.construct_windows`` records inclusive ``[i0, i1]``
  endpoints, but ``analyze_amplitude_truncation`` slices ``amps[i0:i1]``: a
  nominal 1,000-spike window historically fits 999 values. This module makes
  both counts explicit (:func:`historical_exact_counts`,
  :func:`historical_exact_fit`) without changing production's own behaviour.

``inventory`` reads recording metadata and cached arrays only -- it never
fits, extracts voltage, or sorts. ``windows.csv`` rows carry the *cached*
missing-percentage and fit parameters verbatim, classified by their stored
values; nothing is refit at this layer.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pipeline.truncation import (
    SATURATION_PCT, fit_amp_cdf, is_saturated, truncated_sigmoid,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "luke-amplitude-dropout-audit-v1"
DEFAULT_CONFIG = REPO_ROOT / "testing/configs/luke_amplitude_dropout_audit_v1.json"

STATUS_INVALID_INPUT = "invalid_input"
STATUS_NO_FIT = "no_fit"
STATUS_NONFINITE_FIT = "nonfinite_fit"
STATUS_BOUNDARY_PINNED = "boundary_pinned"
STATUS_FINITE_INTERIOR = "finite_interior"

#: Production's gap rule (``pipeline.truncation.analyze_amplitude_truncation``
#: default ``max_isi``): a block is split when two adjacent spikes of one
#: cluster are more than this many seconds apart. Recorded, never re-tuned.
PRODUCTION_MAX_ISI_S = 10.0

#: ``windows.csv``'s columns, in order. Named so that a table with no rows at
#: all still carries the header: an all-empty inventory is a legitimate
#: (indeed the most complete) insufficient-data result, and it has to survive
#: the round trip to `select` as an empty *table*, not as an unparseable file.
WINDOWS_COLUMNS = (
    "sort_id", "cluster_id", "source_row", "i0", "i1",
    "first_sample", "last_sample", "start_s", "end_s",
    "historical_count", "nominal_count", "missing_pct",
    "fit_x0", "fit_k", "fit_A", "status", "invalid_reason",
)

#: Columns read back byte-for-byte, with CSV type inference and NA conversion
#: both disabled. ``sort_id`` is the immutable identifier the config supplies;
#: ``status`` is a fixed vocabulary. ``cluster_id`` is deliberately NOT here --
#: it is a genuine integer whose value the exact-integer validators enforce.
_WINDOWS_TEXT_COLUMNS = ("sort_id", "status")

#: Free text that is legitimately empty; blanks become ``None``, not NaN.
_WINDOWS_FREE_TEXT_COLUMNS = ("invalid_reason",)

SELECTION_FLOAT_KEYS = (
    "reference_max_missing_pct",
    "failing_min_missing_pct",
    "min_median_difference_pp",
    "max_span_s",
    "control_max_missing_pct",
    "control_max_range_pp",
)
SELECTION_INT_KEYS = (
    "windows_per_case",
    "required_nominal_count",
    "max_failure_cases_per_sort",
    "max_control_cases_per_sort",
)
SELECTION_CONSTANT_KEYS = SELECTION_FLOAT_KEYS + SELECTION_INT_KEYS
SELECTION_BLOCK_KEYS = SELECTION_CONSTANT_KEYS + ("units",)

_CURATED_HASHED_FILES = ("spike_times.npy", "spike_clusters.npy", "full_st.npy", "kept_spikes.npy")


# --------------------------------------------------------------------------- #
# small shared helpers
# --------------------------------------------------------------------------- #
def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def sha256_file(path: Path, _buf: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(_buf), b""):
            h.update(chunk)
    return h.hexdigest()


_INT64_MIN = int(np.iinfo(np.int64).min)
_INT64_MAX = int(np.iinfo(np.int64).max)


def _exact_int_array(values: np.ndarray, label: str) -> np.ndarray:
    """Cast to int64 iff every value is real, finite, in range, and *exactly*
    integer-valued.

    ``np.allclose`` is deliberately not used here: its relative tolerance
    grows with magnitude, so e.g. ``100000.25`` reads as "close enough" to
    ``100000`` for large sample counts. The prescription requires rejecting
    invalid arrays rather than silently rounding them -- which also rules out
    a bare ``.astype(np.int64)``: that silently drops the imaginary part of a
    complex array and silently wraps/overflows a ``uint64`` or an
    out-of-int64-range float instead of raising.
    """
    values = np.asarray(values)
    if np.iscomplexobj(values):
        raise ValueError(f"{label} must be real-valued, got a complex dtype")
    if not np.isfinite(values).all():
        raise ValueError(f"{label} contains non-finite values")
    if np.issubdtype(values.dtype, np.integer):
        if values.size and (int(values.min()) < _INT64_MIN or int(values.max()) > _INT64_MAX):
            raise ValueError(f"{label} contains values outside the int64 range")
        return values.astype(np.int64, copy=False)
    rounded = np.round(values)
    if not np.array_equal(values, rounded):
        raise ValueError(f"{label} has non-integer values")
    if rounded.size and (float(rounded.min()) < _INT64_MIN or float(rounded.max()) > _INT64_MAX):
        raise ValueError(f"{label} contains values outside the int64 range")
    return rounded.astype(np.int64)


def _exact_int_scalar(value: Any, label: str) -> int:
    """Exact-integer validation for a single scalar, without a lossy float
    round-trip: a plain Python ``int`` (as JSON integers parse to) is returned
    as-is, so an int64-representable value like ``9007199254740993`` is never
    corrupted by a ``float(value)`` comparison along the way.
    """
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a number, got {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError(f"{label} must be a finite integer, got {value!r}")
        if value != int(value):
            raise ValueError(f"{label} must be an integer, got {value!r}")
        return int(value)
    raise ValueError(f"{label} must be a number, got {value!r}")


def _reject_unsafe_out_root(out_root: Path, input_paths: list[Path]) -> Path:
    """Refuse an output root under /mnt or under any configured input directory."""
    resolved = Path(out_root).resolve()
    if resolved == Path("/mnt") or str(resolved).startswith("/mnt/") or str(resolved) == "/mnt":
        raise ValueError(f"refusing to write audit output under /mnt: {resolved}")
    for p in input_paths:
        try:
            rp = Path(p).resolve()
        except OSError:
            continue
        if resolved == rp or resolved.is_relative_to(rp) or rp.is_relative_to(resolved):
            raise ValueError(
                f"refusing to write audit output under/over an input directory: {rp}"
            )
    return resolved


def _atomic_write_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    os.replace(tmp, path)


def _atomic_write_csv(path: Path, df: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SortConfig:
    sort_id: str
    curated: Path
    qc_dir: Path
    source_recording: Path
    sampling_frequency_hz: float
    selected_start_sample: int
    duration_s: float
    channel_geometry: Path | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sort_id or "/" in self.sort_id or os.sep in self.sort_id:
            raise ValueError(f"invalid sort_id {self.sort_id!r}")
        # `<= 0` alone would let a NaN duration/frequency (e.g. from a
        # "NaN"/null config field) silently pass -- every comparison against
        # NaN is False, so `nan <= 0` does not raise. isfinite() closes that.
        if not np.isfinite(self.sampling_frequency_hz) or self.sampling_frequency_hz <= 0:
            raise ValueError(f"{self.sort_id}: sampling_frequency_hz must be finite and positive")
        if self.selected_start_sample < 0:
            raise ValueError(f"{self.sort_id}: selected_start_sample must be >= 0")
        if not np.isfinite(self.duration_s) or self.duration_s <= 0:
            raise ValueError(f"{self.sort_id}: duration_s must be finite and positive")


@dataclass(frozen=True)
class SelectionConstants:
    """The frozen layer-3 triage constants, supplied only by CONFIG.

    The prescription requires these frozen "before reading rankings", and calls
    them "proposed triage constants, not calibrated biological acceptance
    gates". So they live in the JSON config with explicit units, there is no
    default for any of them (a config without a ``selection`` block is refused
    by :func:`load_config`), and no CLI flag can override one -- a run whose
    thresholds moved after a ranking was seen would not be the prescribed
    experiment.

    ``windows_per_case`` supplies both the failure-case run length ("four
    consecutive windows") and the control's minimum ("at least four consecutive
    valid windows"); controls are frozen at exactly that length because every
    case in ``selection.json`` carries exactly that many source rows.
    """

    reference_max_missing_pct: float
    failing_min_missing_pct: float
    min_median_difference_pp: float
    max_span_s: float
    control_max_missing_pct: float
    control_max_range_pp: float
    windows_per_case: int
    required_nominal_count: int
    max_failure_cases_per_sort: int
    max_control_cases_per_sort: int
    units: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for key in SELECTION_FLOAT_KEYS:
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"selection.{key} must be a number, got {value!r}")
            if not np.isfinite(value):
                raise ValueError(f"selection.{key} must be finite, got {value!r}")
        for key in SELECTION_INT_KEYS:
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"selection.{key} must be an integer, got {value!r}")
        for key in (
            "reference_max_missing_pct", "failing_min_missing_pct", "control_max_missing_pct",
        ):
            value = float(getattr(self, key))
            if not (0.0 <= value <= 100.0):
                raise ValueError(f"selection.{key} must lie in [0, 100], got {value}")
        if self.min_median_difference_pp < 0.0:
            raise ValueError("selection.min_median_difference_pp must be >= 0")
        if self.control_max_range_pp < 0.0:
            raise ValueError("selection.control_max_range_pp must be >= 0")
        if self.max_span_s <= 0.0:
            raise ValueError("selection.max_span_s must be > 0")
        if self.failing_min_missing_pct < self.reference_max_missing_pct:
            raise ValueError(
                "selection.failing_min_missing_pct must be >= reference_max_missing_pct "
                f"({self.failing_min_missing_pct} < {self.reference_max_missing_pct})"
            )
        if self.windows_per_case < 2 or self.windows_per_case % 2:
            raise ValueError(
                "selection.windows_per_case must be an even integer >= 2 (the first half is "
                f"the reference, the second half the failing side), got {self.windows_per_case}"
            )
        if self.required_nominal_count < 2:
            raise ValueError("selection.required_nominal_count must be >= 2")
        if self.max_failure_cases_per_sort < 0 or self.max_control_cases_per_sort < 0:
            raise ValueError("selection case caps must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key in SELECTION_FLOAT_KEYS:
            payload[key] = float(getattr(self, key))
        for key in SELECTION_INT_KEYS:
            payload[key] = int(getattr(self, key))
        payload["units"] = {str(k): str(v) for k, v in self.units.items()}
        return payload


def parse_selection_constants(payload: Any) -> SelectionConstants:
    """Parse CONFIG's ``selection`` block, requiring every constant and unit.

    Missing keys are refused rather than defaulted, and unknown keys are
    refused too: a typo'd constant name that silently fell back to a built-in
    default would defeat the whole point of freezing them.
    """
    if not isinstance(payload, dict):
        raise ValueError("config: 'selection' must be an object of frozen selection constants")
    missing = [k for k in SELECTION_BLOCK_KEYS if k not in payload]
    if missing:
        raise ValueError(
            f"config: 'selection' block is missing required key(s) {missing}; selection "
            "constants have no defaults and must be frozen in CONFIG before any ranking is read"
        )
    unknown = [k for k in payload if k not in SELECTION_BLOCK_KEYS]
    if unknown:
        raise ValueError(f"config: 'selection' block has unknown key(s) {sorted(unknown)}")
    units = payload["units"]
    if not isinstance(units, dict):
        raise ValueError("config: 'selection.units' must be an object naming each constant's unit")
    missing_units = [k for k in SELECTION_CONSTANT_KEYS if not str(units.get(k, "")).strip()]
    if missing_units:
        raise ValueError(f"config: 'selection.units' is missing a unit for {missing_units}")

    kwargs: dict[str, Any] = {}
    for key in SELECTION_FLOAT_KEYS:
        value = payload[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"selection.{key} must be a number, got {value!r}")
        kwargs[key] = float(value)
    for key in SELECTION_INT_KEYS:
        kwargs[key] = _exact_int_scalar(payload[key], f"selection.{key}")
    return SelectionConstants(units={str(k): str(v) for k, v in units.items()}, **kwargs)


@dataclass(frozen=True)
class AuditConfig:
    schema: str
    sorts: tuple[SortConfig, ...]
    selection: SelectionConstants
    #: Frozen thresholds for the evidence layer. Optional at parse time so
    #: `inventory` and `select` (which read no evidence) still run without one;
    #: `run_inspect` requires it before it classifies anything.
    evidence: "EvidenceConstants | None" = None
    #: Which delivery contract reserves this recording's sealed panel and
    #: healthy evaluation intervals. Optional at parse time so `inventory`
    #: (which ranks nothing) still runs without one; `run_select` requires it.
    interval_contract: IntervalContractRef | None = None

    def __post_init__(self) -> None:
        if self.schema != SCHEMA:
            raise ValueError(f"config schema {self.schema!r} != {SCHEMA!r}")
        if len(self.sorts) < 1:
            raise ValueError("config must declare at least one sort")
        ids = [s.sort_id for s in self.sorts]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate sort_id in config: {ids}")
        if not isinstance(self.selection, SelectionConstants):
            raise ValueError("config must supply frozen selection constants")
        if self.interval_contract is not None and not isinstance(
            self.interval_contract, IntervalContractRef
        ):
            raise ValueError("interval_contract must be an IntervalContractRef")
        if self.evidence is not None and not isinstance(self.evidence, EvidenceConstants):
            raise ValueError("evidence must be EvidenceConstants")

    def require_evidence_constants(self, config_path: Path | str) -> "EvidenceConstants":
        if self.evidence is None:
            raise ValueError(
                f"{Path(config_path)}: config has no 'evidence' block; the evidence layer's "
                "thresholds must be frozen in CONFIG (with units) before any observation is "
                "read, exactly as the selection constants are -- there are no defaults"
            )
        return self.evidence

    def require_interval_contract(self, config_path: Path | str) -> IntervalContractRef:
        if self.interval_contract is None:
            raise ValueError(
                f"{Path(config_path)}: config has no 'interval_contract' block; selection may "
                "only rank inside the delivery contract's development windows, which reserve the "
                "sealed held-out panel, its buffer and the healthy evaluation intervals -- there "
                "is no unrestricted ranking"
            )
        return self.interval_contract

    def by_id(self, sort_id: str) -> SortConfig:
        for s in self.sorts:
            if s.sort_id == sort_id:
                return s
        raise KeyError(sort_id)


def load_config(path: Path) -> AuditConfig:
    """Parse CONFIG from disk. Prefer :func:`read_config_once` where the file's
    hash is also recorded, so the bytes hashed are the bytes parsed."""
    cfg, _ = read_config_once(path)
    return cfg


def read_config_once(path: Path) -> tuple[AuditConfig, str]:
    """Read CONFIG exactly once, returning the parsed config and its digest.

    Hashing the path in one place and parsing it in another is two reads of a
    file that can change in between, so the recorded ``config_sha256`` could
    attest bytes no stage ever parsed. Same rule as
    :func:`read_attested_windows` and :func:`read_curated_arrays`.
    """
    data = Path(path).read_bytes()
    return load_config_from_bytes(data, path), hashlib.sha256(data).hexdigest()


def load_config_from_bytes(data: bytes, path: Path | str) -> AuditConfig:
    """Parse CONFIG from bytes already in hand; ``path`` is used only in messages."""
    payload = json.loads(data)
    sorts = []
    for entry in payload.get("sorts", []):
        sort_id = entry["sort_id"]
        sorts.append(SortConfig(
            sort_id=sort_id,
            curated=Path(entry["curated"]),
            qc_dir=Path(entry["qc_dir"]),
            source_recording=Path(entry["source_recording"]),
            sampling_frequency_hz=float(entry["sampling_frequency_hz"]),
            selected_start_sample=_exact_int_scalar(
                entry["selected_start_sample"], f"{sort_id}: selected_start_sample"
            ),
            duration_s=float(entry["duration_s"]),
            channel_geometry=(
                Path(entry["channel_geometry"]) if entry.get("channel_geometry") else None
            ),
            provenance=entry.get("provenance", {}),
        ))
    if "selection" not in payload:
        raise ValueError(
            f"{Path(path)}: config has no 'selection' block; every selection constant must be "
            "frozen in CONFIG (with units) before any ranking is read -- there are no defaults"
        )
    selection = parse_selection_constants(payload["selection"])
    interval_contract = (
        parse_interval_contract_ref(payload["interval_contract"], path)
        if "interval_contract" in payload else None
    )
    evidence = (
        parse_evidence_constants(payload["evidence"]) if "evidence" in payload else None
    )
    return AuditConfig(
        schema=payload.get("schema", ""), sorts=tuple(sorts), selection=selection,
        interval_contract=interval_contract, evidence=evidence,
    )


def _recording_metadata(source_recording: Path) -> dict[str, Any] | None:
    """Best-effort authoritative recording metadata.

    Returns ``None`` when no recognisable manifest exists. When a
    ``rescue_recording_manifest.json`` is present, its ``num_samples`` /
    ``selected_start_frame`` / ``selected_end_frame`` make duration and
    selected-start authoritative (``duration_authoritative=True``). A bare
    SpikeInterface ``binary.json`` (the legacy pipeline's recording folder)
    only carries sampling frequency -- its raw traces file may no longer be on
    disk, so duration/selected-start stay declared-only.
    """
    manifest = Path(source_recording) / "rescue_recording_manifest.json"
    if manifest.exists():
        m = json.loads(manifest.read_text())
        fs = float(m["sampling_frequency_hz"])
        if not np.isfinite(fs) or fs <= 0:
            raise ValueError(f"{manifest}: sampling_frequency_hz must be finite and positive")
        start = _exact_int_scalar(m["selected_start_frame"], f"{manifest}: selected_start_frame")
        end = _exact_int_scalar(m["selected_end_frame"], f"{manifest}: selected_end_frame")
        return {
            "sampling_frequency_hz": fs,
            "selected_start_sample": start,
            "duration_s": (end - start) / fs,
            "duration_authoritative": True,
            "source": str(manifest),
        }
    binary = Path(source_recording) / "binary.json"
    if binary.exists():
        kwargs = json.loads(binary.read_text())["kwargs"]
        return {
            "sampling_frequency_hz": float(kwargs["sampling_frequency"]),
            "duration_authoritative": False,
            "source": str(binary),
        }
    return None


def validate_recording_metadata(
    cfg: SortConfig, *, rtol: float = 1e-9, duration_atol_s: float = 1e-3,
) -> dict[str, Any]:
    """Check the config's declared frequency/duration/start against a manifest.

    Duration and selected-start are validated whenever a manifest carries
    authoritative bounds (the rescue recording); otherwise they stay
    declared-only, and this returns why (prescription: "missing data produces
    a blocked-stage receipt, not automatic reconstruction").
    """
    meta = _recording_metadata(cfg.source_recording)
    if meta is None:
        return {
            "sampling_frequency_validated": False,
            "duration_validated": False,
            "reason": f"no recognisable recording manifest under {cfg.source_recording}",
        }
    if not np.isclose(meta["sampling_frequency_hz"], cfg.sampling_frequency_hz, rtol=rtol):
        raise ValueError(
            f"{cfg.sort_id}: configured sampling_frequency_hz={cfg.sampling_frequency_hz} "
            f"!= recording metadata {meta['sampling_frequency_hz']} ({meta['source']})"
        )
    if not meta["duration_authoritative"]:
        return {
            "sampling_frequency_validated": True,
            "duration_validated": False,
            "reason": (
                f"{meta['source']} has no authoritative num_samples/selected_start "
                "(e.g. the raw recording file is no longer on disk); duration_s and "
                "selected_start_sample are declared, not independently re-derived"
            ),
        }
    if cfg.selected_start_sample != meta["selected_start_sample"]:
        raise ValueError(
            f"{cfg.sort_id}: configured selected_start_sample={cfg.selected_start_sample} "
            f"!= recording manifest {meta['selected_start_sample']} ({meta['source']})"
        )
    if abs(cfg.duration_s - meta["duration_s"]) > duration_atol_s:
        raise ValueError(
            f"{cfg.sort_id}: configured duration_s={cfg.duration_s} "
            f"!= recording manifest duration {meta['duration_s']} ({meta['source']})"
        )
    return {
        "sampling_frequency_validated": True,
        "duration_validated": True,
        "reason": f"validated against {meta['source']}",
    }


# --------------------------------------------------------------------------- #
# layer 1 -- loader and window-table normalization
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CuratedArrays:
    """The stable-time-sorted time/cluster/amplitude triplet production QC used.

    ``amplitudes`` is ``full_st[kept_spikes][:, 2]`` -- sorter-native units,
    never ``amplitudes.npy`` (prescription clause 1). ``row_id`` is the
    original 0-based row index into ``spike_times.npy``, carried through the
    stable sort so lineage survives. ``was_time_ordered`` records whether the
    array was already non-decreasing *before* this loader's defensive sort;
    cached per-cluster window indices from production can only be safely
    replayed when this is True (see :func:`build_windows_table`).
    """

    sort_id: str
    times: np.ndarray
    clusters: np.ndarray
    amplitudes: np.ndarray
    row_id: np.ndarray
    was_time_ordered: bool
    #: The attested full-table arrays this curated view was derived from, kept
    #: so the evidence layer's retained-row lineage does not have to re-read
    #: (and re-hash) files the replay already consumed. Optional: a caller that
    #: constructs a view directly need not supply them, and the lineage
    #: observation then degrades to `unavailable`.
    full_st: "np.ndarray | None" = None
    kept: "np.ndarray | None" = None


def _validate_kept_spikes(kept: np.ndarray, n_full: int) -> np.ndarray:
    kept = np.asarray(kept)
    if kept.ndim != 1:
        raise ValueError("kept_spikes must be one-dimensional")
    if kept.dtype == np.bool_:
        if kept.shape[0] != n_full:
            raise ValueError(
                f"boolean kept_spikes length {kept.shape[0]} != full table length {n_full}"
            )
    elif np.issubdtype(kept.dtype, np.integer):
        if kept.size and (int(kept.min()) < 0 or int(kept.max()) >= n_full):
            raise ValueError("integer kept_spikes indices out of range for full table")
    else:
        raise ValueError(f"kept_spikes must be boolean or integer, got dtype {kept.dtype}")
    return kept


def read_curated_arrays(curated: Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Read each curated array ONCE, hashing and parsing the very same bytes.

    Hashing a path in one place and ``np.load``-ing that path in another is
    two reads of a file that can change in between: the digest would attest
    bytes the replay never consumed. The sharpest form costs nothing to hide
    -- swapping only the amplitude at a window's inclusive endpoint leaves
    every historical fit reproducing its cache while moving the exact-1,000
    fit -- so the two must not be separable. This is the same rule
    :func:`read_attested_windows` enforces for ``windows.csv``.

    Each file's bytes are released as soon as that file is parsed, so peak
    memory stays at one array plus one buffer rather than the whole set.
    """
    curated = Path(curated)
    arrays: dict[str, np.ndarray] = {}
    digests: dict[str, str] = {}
    for name in _CURATED_HASHED_FILES:
        data = (curated / name).read_bytes()
        digests[name] = hashlib.sha256(data).hexdigest()
        arrays[name] = np.load(io.BytesIO(data), allow_pickle=False)
        del data
    return arrays, digests


def curated_arrays_from_raw(sort_id: str, raw: dict[str, np.ndarray]) -> CuratedArrays:
    """Validate and normalize already-read curated arrays.

    Split out of :func:`load_curated_arrays` so the attested path can parse
    the bytes it hashed instead of re-reading the files.
    """
    times = np.asarray(raw["spike_times.npy"]).reshape(-1)
    clusters = np.asarray(raw["spike_clusters.npy"]).reshape(-1)
    full_st = raw["full_st.npy"]
    kept = raw["kept_spikes.npy"]

    times = _exact_int_array(times, f"{sort_id}: spike_times.npy")
    if times.size and times.min() < 0:
        raise ValueError(f"{sort_id}: spike_times.npy contains negative samples")
    clusters = _exact_int_array(clusters, f"{sort_id}: spike_clusters.npy")

    kept = _validate_kept_spikes(kept, full_st.shape[0])
    st = full_st[kept]

    if not (times.size == clusters.size == st.shape[0]):
        raise ValueError(
            f"{sort_id}: unaligned curated arrays "
            f"(times={times.size}, clusters={clusters.size}, kept_st={st.shape[0]})"
        )
    st_times = _exact_int_array(st[:, 0], f"{sort_id}: full_st[kept_spikes][:, 0]")
    if not np.array_equal(times, st_times):
        raise ValueError(f"{sort_id}: full_st[kept_spikes] times do not match spike_times.npy")

    amplitudes = st[:, 2].astype(np.float64, copy=False)
    row_id = np.arange(times.size, dtype=np.int64)

    was_ordered = bool(times.size < 2 or bool(np.all(np.diff(times) >= 0)))
    order = np.argsort(times, kind="stable")

    return CuratedArrays(
        sort_id=sort_id,
        times=times[order],
        clusters=clusters[order],
        amplitudes=amplitudes[order],
        row_id=row_id[order],
        was_time_ordered=was_ordered,
        # the FULL table, not st (= full_st[kept]): the lineage observation
        # counts rows the kept mask excluded, which st no longer contains
        full_st=full_st,
        kept=kept,
    )


def load_curated_arrays(sort_id: str, curated: Path) -> CuratedArrays:
    """Load and validate the exact array triplet ``pipeline.qc.run_qc`` consumes.

    Reproduces ``KilosortResults.st`` (``full_st[kept_spikes]``) directly from
    disk rather than importing the class, so this loader can validate
    ``kept_spikes`` semantics explicitly instead of trusting them silently.
    """
    raw, _ = read_curated_arrays(curated)
    return curated_arrays_from_raw(sort_id, raw)


def read_attested_curated_arrays(
    sort_id: str, curated: Path, recorded_hashes: dict[str, Any],
) -> tuple[CuratedArrays, dict[str, str]]:
    """Load the curated arrays and refuse unless they are the attested bytes.

    One read per file: the digest compared here covers exactly the bytes that
    become the returned arrays, so nothing can be substituted between the
    check and the replay.
    """
    raw, digests = read_curated_arrays(curated)
    moved = [
        f"{sort_id}/{name}: inventory {recorded_hashes.get(name)!r} != current "
        f"{digests[name]!r}"
        for name in _CURATED_HASHED_FILES
        if recorded_hashes.get(name) != digests[name]
    ]
    if moved:
        raise RuntimeError(
            "refusing to inspect: curated array(s) moved since the inventory hashed them -- "
            + "; ".join(moved)
        )
    return curated_arrays_from_raw(sort_id, raw), digests


@dataclass(frozen=True)
class CachedTruncationQC:
    """Normalized, validated contents of production's ``truncation_qc.npz``."""

    sort_id: str
    cid: np.ndarray
    window_blocks: np.ndarray
    popts: np.ndarray
    mpcts: np.ndarray


def load_cached_truncation_qc(sort_id: str, qc_dir: Path) -> CachedTruncationQC:
    cached, _ = read_cached_truncation_qc(sort_id, qc_dir)
    return cached


def read_cached_truncation_qc(sort_id: str, qc_dir: Path) -> tuple[CachedTruncationQC, str]:
    """Read ``truncation_qc.npz`` once, returning its contents and its digest.

    The digest covers the very bytes parsed here, so the provenance the
    manifest records is the provenance of the rows that reached
    ``windows.csv`` -- not of a second read of the path.
    """
    path = Path(qc_dir) / "amp_truncation" / "truncation_qc.npz"
    if not path.exists():
        raise FileNotFoundError(f"{sort_id}: no cached truncation QC at {path}")
    blob = path.read_bytes()
    digest = hashlib.sha256(blob).hexdigest()
    data = np.load(io.BytesIO(blob), allow_pickle=False)
    for key in ("cid", "window_blocks", "popts", "mpcts"):
        if key not in data.files:
            raise ValueError(f"{sort_id}: {path} missing array {key!r}")

    cid = np.asarray(data["cid"]).reshape(-1)
    window_blocks = np.asarray(data["window_blocks"])
    popts = np.asarray(data["popts"])
    mpcts = np.asarray(data["mpcts"]).reshape(-1)
    n = cid.size

    if window_blocks.ndim != 2 or window_blocks.shape != (n, 2):
        raise ValueError(f"{sort_id}: window_blocks shape {window_blocks.shape} != ({n}, 2)")
    if popts.ndim != 2 or popts.shape != (n, 3):
        raise ValueError(f"{sort_id}: popts shape {popts.shape} != ({n}, 3)")
    if mpcts.shape != (n,):
        raise ValueError(f"{sort_id}: mpcts shape {mpcts.shape} != ({n},)")

    # Structural validation only (finite, integer-valued): a row with
    # semantically invalid bounds (negative start, i1 <= i0, out of range for
    # its cluster) is not rejected here -- it is retained and classified
    # `invalid_input` by build_windows_table, per the prescription's "keep all
    # rows and reasons".
    cid = _exact_int_array(cid, f"{sort_id}: cached cid")
    window_blocks = _exact_int_array(window_blocks, f"{sort_id}: cached window_blocks")

    return CachedTruncationQC(
        sort_id=sort_id, cid=cid, window_blocks=window_blocks, popts=popts, mpcts=mpcts,
    ), digest


def _classify_status(popt: np.ndarray, mpct: float) -> str:
    if not np.isfinite(mpct) or not np.isfinite(popt).all():
        return STATUS_NONFINITE_FIT
    if bool(is_saturated(np.array([mpct]))[0]):
        return STATUS_BOUNDARY_PINNED
    return STATUS_FINITE_INTERIOR


def _empty_window_row(sort_id: str, cluster_id: int, source_row: int, i0, i1, popt, mpct,
                      status: str, invalid_reason: str | None = None) -> dict[str, Any]:
    return {
        "sort_id": sort_id, "cluster_id": cluster_id, "source_row": source_row,
        "i0": i0, "i1": i1, "first_sample": None, "last_sample": None,
        "start_s": None, "end_s": None,
        "historical_count": (i1 - i0) if (i0 is not None and i1 is not None and i1 > i0) else None,
        "nominal_count": (i1 - i0 + 1) if (i0 is not None and i1 is not None and i1 > i0) else None,
        "missing_pct": mpct, "fit_x0": float(popt[0]), "fit_k": float(popt[1]), "fit_A": float(popt[2]),
        "status": status, "invalid_reason": invalid_reason,
    }


def build_windows_table(
    curated: CuratedArrays, cached: CachedTruncationQC, fs: float, *,
    max_isi_s: float = PRODUCTION_MAX_ISI_S,
) -> pd.DataFrame:
    """One row per stored cached window, plus one ``no_fit`` row per cluster
    with zero cached windows. Reads cached values only -- see module
    docstring; nothing here is refit.

    Status precedence per row: ``invalid_input``, ``nonfinite_fit``,
    ``boundary_pinned``, ``finite_interior`` (plus ``no_fit`` for clusters
    absent from the cache entirely, per the prescription's array contract).
    A cached row whose ``cluster_id`` does not appear in the curated arrays
    at all, or whose bounds are out of range, or whose span crosses a
    production gap (>``max_isi_s`` seconds between consecutive spikes) is
    kept as ``invalid_input`` with an explicit ``invalid_reason`` rather than
    silently dropped or fit across the gap.
    """
    if curated.sort_id != cached.sort_id:
        raise ValueError(f"sort_id mismatch: curated={curated.sort_id!r} cached={cached.sort_id!r}")
    if not curated.was_time_ordered:
        raise ValueError(
            f"{curated.sort_id}: spike_times.npy was not already time-ordered; cached "
            "window_blocks indices cannot be safely replayed against a re-sorted array"
        )

    all_cids = np.unique(curated.clusters)
    curated_cid_set = {int(c) for c in all_cids}
    cluster_positions = {int(c): np.flatnonzero(curated.clusters == c) for c in all_cids}

    cached_by_cid: dict[int, list[int]] = {}
    for row_idx, c in enumerate(cached.cid):
        cached_by_cid.setdefault(int(c), []).append(row_idx)

    gap_samples = float(max_isi_s) * float(fs)
    rows: list[dict[str, Any]] = []

    for c in all_cids:
        c = int(c)
        positions = cluster_positions[c]
        cluster_len = int(positions.size)
        row_indices = cached_by_cid.get(c, [])

        if not row_indices:
            rows.append(_empty_window_row(
                curated.sort_id, c, -1, None, None,
                (np.nan, np.nan, np.nan), float("nan"), STATUS_NO_FIT,
            ))
            continue

        for r in row_indices:
            i0, i1 = int(cached.window_blocks[r, 0]), int(cached.window_blocks[r, 1])
            popt = cached.popts[r]
            mpct = float(cached.mpcts[r])

            if i0 < 0 or i1 <= i0 or i1 >= cluster_len:
                rows.append(_empty_window_row(
                    curated.sort_id, c, r, i0, i1, popt, mpct, STATUS_INVALID_INPUT,
                    invalid_reason=f"window bounds [{i0}, {i1}] out of range for cluster length {cluster_len}",
                ))
                continue

            window_times = curated.times[positions[i0:i1 + 1]]
            if window_times.size >= 2 and float(np.diff(window_times).max()) > gap_samples:
                rows.append(_empty_window_row(
                    curated.sort_id, c, r, i0, i1, popt, mpct, STATUS_INVALID_INPUT,
                    invalid_reason=f"window spans a gap > {max_isi_s}s between consecutive spikes",
                ))
                continue

            first_sample = int(window_times[0])
            last_sample = int(window_times[-1])
            rows.append({
                "sort_id": curated.sort_id, "cluster_id": c, "source_row": r,
                "i0": i0, "i1": i1,
                "first_sample": first_sample, "last_sample": last_sample,
                "start_s": first_sample / fs, "end_s": last_sample / fs,
                "historical_count": i1 - i0, "nominal_count": i1 - i0 + 1,
                "missing_pct": mpct, "fit_x0": float(popt[0]), "fit_k": float(popt[1]),
                "fit_A": float(popt[2]),
                "status": _classify_status(popt, mpct), "invalid_reason": None,
            })

    # Cached rows whose cluster_id never appears in the curated arrays at all
    # must not silently vanish just because the outer loop only visits
    # curated cluster ids.
    for c, row_indices in cached_by_cid.items():
        if c in curated_cid_set:
            continue
        for r in row_indices:
            i0, i1 = int(cached.window_blocks[r, 0]), int(cached.window_blocks[r, 1])
            popt = cached.popts[r]
            mpct = float(cached.mpcts[r])
            rows.append(_empty_window_row(
                curated.sort_id, c, r, i0, i1, popt, mpct, STATUS_INVALID_INPUT,
                invalid_reason="cluster_id absent from curated arrays",
            ))

    return pd.DataFrame(rows, columns=list(WINDOWS_COLUMNS))


# --------------------------------------------------------------------------- #
# layer 2 -- historical / exact replay (consumed by the `inspect` stage below,
# on the frozen case selection)
# --------------------------------------------------------------------------- #
def historical_exact_counts(i0: int, i1: int) -> tuple[int, int]:
    """Historical (exclusive-stop) vs exact (inclusive-stop) sample counts.

    Production's ``construct_windows`` stores inclusive ``[i0, i1]``, but
    ``analyze_amplitude_truncation`` slices ``amps[i0:i1]``. A nominal
    1,000-spike window (``i1 - i0 + 1 == 1000``) therefore historically fits
    999 values; the exact count is 1,000.
    """
    if i0 < 0 or i1 <= i0:
        raise ValueError(f"invalid window bounds i0={i0}, i1={i1}")
    return i1 - i0, i1 - i0 + 1


def historical_exact_fit(cluster_amplitudes: np.ndarray, i0: int, i1: int) -> dict[str, Any]:
    """Refit the historical ``[i0:i1]`` slice and the exact ``[i0:i1+1]`` slice
    of one cluster's time-ordered amplitude sequence.

    Raises rather than silently truncating if the exact slice would run past
    the array end (prescription: "reject invalid arrays rather than silently
    rounding").
    """
    if i0 < 0 or i1 <= i0:
        raise ValueError(f"invalid window bounds i0={i0}, i1={i1}")
    cluster_amplitudes = np.asarray(cluster_amplitudes)
    if i1 >= cluster_amplitudes.size:
        raise ValueError(
            f"exact window [{i0}:{i1 + 1}) exceeds cluster length {cluster_amplitudes.size}"
        )

    hist_amps = cluster_amplitudes[i0:i1]
    exact_amps = cluster_amplitudes[i0:i1 + 1]
    assert exact_amps.size == hist_amps.size + 1  # guaranteed by the two slice bounds

    hist_popt, hist_mpct = fit_amp_cdf(hist_amps)
    exact_popt, exact_mpct = fit_amp_cdf(exact_amps)
    return {
        "historical_count": int(hist_amps.size),
        "exact_count": int(exact_amps.size),
        "historical_missing_pct": float(hist_mpct),
        "exact_missing_pct": float(exact_mpct),
        "historical_saturated": bool(is_saturated(np.array([hist_mpct]))[0]),
        "exact_saturated": bool(is_saturated(np.array([exact_mpct]))[0]),
        "historical_popt": [float(v) for v in hist_popt],
        "exact_popt": [float(v) for v in exact_popt],
    }


# --------------------------------------------------------------------------- #
# the delivery contract's interval gate
#
# The audit ranks cases out of one recording that the first-pipeline-candidate
# delivery contract has already partitioned: a sealed held-out panel (plus an
# exclusion buffer), reserved healthy evaluation intervals, and the development
# windows that remain. Only the development windows may be developed, tuned or
# inspected on, so a case span outside them must never reach a ranking -- not
# even to be beaten by an eligible one. The contract is the single source of
# these intervals (``configs/first_pipeline_candidate.v1.json``); this module
# re-validates them at runtime rather than trusting the shipped list.
# --------------------------------------------------------------------------- #
#: Tolerance for deciding that a RECOMPUTED interval bound equals an AUTHORED
#: one, in seconds. This is the only place a tolerance is defensible: both
#: sides describe the same instant and may differ by float representation.
#:
#: It is deliberately not used for containment, disjointness, merging or
#: subtraction. A tolerance there is not a representation correction, it is a
#: band in which a case may overhang a development window, or a contract may
#: overlap a region it reserved, and still pass. Those comparisons are exact,
#: which fails closed: a span that misses containment by a nanosecond is
#: excluded (costing at most one case), where the reverse would admit a case
#: from the sealed panel or a reserved evaluation interval.
INTERVAL_MATCH_EPSILON_S = 1e-9

#: Audit cases with role ``control`` are DIAGNOSTIC controls: quiet spans of a
#: non-failure cluster, selected inside development windows for comparison
#: within this audit. They are not the contract's reserved
#: ``intervals.healthy_control_intervals``, which are evaluation-only, never
#: tuned on, and excluded from every development window by construction.
DIAGNOSTIC_CONTROL_KIND = "diagnostic"


@dataclass(frozen=True)
class IntervalContractRef:
    """CONFIG's declaration of which delivery contract governs the intervals."""

    path: Path
    contract_id: str

    def __post_init__(self) -> None:
        if not str(self.path):
            raise ValueError("interval_contract.path must be a non-empty path")
        if not self.contract_id:
            raise ValueError("interval_contract.contract_id must be a non-empty string")


def parse_interval_contract_ref(payload: Any, config_path: Path | str) -> IntervalContractRef:
    if not isinstance(payload, dict):
        raise ValueError(
            f"{Path(config_path)}: 'interval_contract' must be an object declaring the delivery "
            "contract that reserves the sealed panel and the healthy evaluation intervals"
        )
    unknown = sorted(set(payload) - {"path", "contract_id", "purpose"})
    if unknown:
        raise ValueError(f"config: 'interval_contract' has unknown key(s) {unknown}")
    for key in ("path", "contract_id"):
        if not isinstance(payload.get(key), str) or not payload[key]:
            raise ValueError(f"config: 'interval_contract.{key}' must be a non-empty string")
    path = Path(payload["path"])
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return IntervalContractRef(path=path, contract_id=payload["contract_id"])


@dataclass(frozen=True)
class PermittedIntervals:
    """The contract's development windows, re-validated against this recording.

    ``development_windows_s`` is the only region selection may rank in.
    ``sealed_windows_s`` (expanded by ``sealed_exclusion_buffer_s``) and
    ``reserved_evaluation_windows_s`` are carried so provenance records what was
    excluded, and so the derivation can be recomputed rather than trusted.
    """

    contract_path: str
    contract_sha256: str
    contract_id: str
    contract_schema: str
    clock: str
    recording_duration_s: float
    development_windows_s: tuple[tuple[float, float], ...]
    sealed_windows_s: tuple[tuple[float, float], ...]
    sealed_exclusion_buffer_s: float
    reserved_evaluation_windows_s: tuple[tuple[float, float], ...]
    reserved_evaluation_names: tuple[str, ...]

    def containing_index(self, start_s: float, end_s: float) -> int | None:
        """Index of the single development window containing ``[start_s, end_s]``.

        ``None`` when the span straddles a boundary, falls in an excluded
        region, or is not a finite interval at all. Containment is required in
        ONE window: a span covering an excluded region between two windows is
        not permitted just because both of its ends are.
        """
        if start_s is None or end_s is None:
            return None
        if not (math.isfinite(start_s) and math.isfinite(end_s)) or end_s < start_s:
            return None
        # exact: a span that overhangs by any amount is not contained
        for index, (w_start, w_stop) in enumerate(self.development_windows_s):
            if start_s >= w_start and end_s <= w_stop:
                return index
        return None

    def to_provenance(self) -> dict[str, Any]:
        return {
            "contract_path": self.contract_path,
            "contract_sha256": self.contract_sha256,
            "contract_id": self.contract_id,
            "contract_schema": self.contract_schema,
            "clock": self.clock,
            "recording_duration_s": self.recording_duration_s,
            "development_windows_s": [list(w) for w in self.development_windows_s],
            "development_windows_total_s": round(
                sum(stop - start for start, stop in self.development_windows_s), 9
            ),
            "sealed_windows_s": [list(w) for w in self.sealed_windows_s],
            "sealed_exclusion_buffer_s": self.sealed_exclusion_buffer_s,
            "reserved_evaluation_windows_s": [
                list(w) for w in self.reserved_evaluation_windows_s
            ],
            "reserved_evaluation_names": list(self.reserved_evaluation_names),
            "rule": (
                "every case span (first window start_s to last window end_s) must lie inside a "
                "single development window; audit controls are diagnostic and are gated the same "
                "way, and are distinct from the contract's reserved healthy evaluation intervals"
            ),
        }


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Union of overlapping or exactly abutting intervals, sorted by start.

    Exact: two cuts separated by a real, positive gap stay separate, so a
    permitted sliver between two reserved regions is preserved rather than
    quietly absorbed into one of them.
    """
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged = [ordered[0]]
    for start, stop in ordered[1:]:
        last_start, last_stop = merged[-1]
        if start <= last_stop:
            merged[-1] = (last_start, max(last_stop, stop))
        else:
            merged.append((start, stop))
    return merged


def _subtract_intervals(
    base: list[tuple[float, float]], cuts: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """``base`` minus ``cuts``, exactly; only empty remnants are dropped."""
    remaining = list(base)
    for cut_start, cut_stop in _merge_intervals(cuts):
        nxt: list[tuple[float, float]] = []
        for start, stop in remaining:
            if cut_stop <= start or cut_start >= stop:
                nxt.append((start, stop))
                continue
            if cut_start > start:
                nxt.append((start, cut_start))
            if cut_stop < stop:
                nxt.append((cut_stop, stop))
        remaining = nxt
    return [(start, stop) for start, stop in remaining if stop > start]


def _intervals_equal(
    left: list[tuple[float, float]], right: list[tuple[float, float]],
    eps: float = INTERVAL_MATCH_EPSILON_S,
) -> bool:
    if len(left) != len(right):
        return False
    return all(
        abs(a_start - b_start) <= eps and abs(a_stop - b_stop) <= eps
        for (a_start, a_stop), (b_start, b_stop) in zip(left, right)
    )


def read_permitted_intervals(
    ref: IntervalContractRef, *, recording_duration_s: float,
) -> PermittedIntervals:
    """Read the delivery contract ONCE and re-derive its development windows.

    The contract's own ``derivation`` field says the shipped window list does
    not need to be trusted, so this recomputes it (full stream, minus every
    sealed window expanded by the exclusion buffer, minus the reserved healthy
    evaluation intervals) and refuses a mismatch. Bounds, recording limits and
    disjointness from every reserved period are checked here at runtime, not
    only in a test of the shipped JSON.

    The bytes hashed are the bytes parsed: this file becomes a consumed input
    of ``select``, and its recorded identity must attest what was actually read.
    """
    from testing.first_pipeline_candidate_contract import (
        SCHEMA as CONTRACT_SCHEMA,
        development_windows as contract_development_windows,
        _interval_list as contract_interval_list,
    )

    path = Path(ref.path)
    data = path.read_bytes()
    contract_sha256 = hashlib.sha256(data).hexdigest()
    payload = json.loads(data)
    if not isinstance(payload, dict):
        raise ValueError(f"interval contract {path} must be a JSON object")
    schema = payload.get("schema")
    if schema != CONTRACT_SCHEMA:
        raise ValueError(f"interval contract schema {schema!r} != {CONTRACT_SCHEMA!r}")
    contract_id = payload.get("contract_id")
    if contract_id != ref.contract_id:
        raise ValueError(
            f"interval contract at {path} declares contract_id {contract_id!r}, but CONFIG's "
            f"interval_contract.contract_id is {ref.contract_id!r}"
        )

    # the interval clock is "seconds from the start of the selected stream", so
    # the contract and CONFIG must be describing the same recording
    recording = payload.get("recording")
    if not isinstance(recording, dict):
        raise ValueError(f"interval contract {path} has no 'recording' block")
    contract_duration = recording.get("duration_s")
    if not isinstance(contract_duration, (int, float)) or isinstance(contract_duration, bool):
        raise ValueError(f"interval contract {path}: recording.duration_s must be a number")
    contract_duration = float(contract_duration)
    if not math.isfinite(contract_duration) or contract_duration <= 0:
        raise ValueError(
            f"interval contract {path}: recording.duration_s must be finite and positive"
        )
    if abs(contract_duration - float(recording_duration_s)) > INTERVAL_MATCH_EPSILON_S:
        raise ValueError(
            f"interval contract {path} governs a recording of {contract_duration} s, but CONFIG's "
            f"sorts declare {recording_duration_s} s; the audit and the contract are not "
            "describing the same recording"
        )

    intervals = payload.get("intervals")
    if not isinstance(intervals, dict):
        raise ValueError(f"interval contract {path} has no 'intervals' block")
    sealed = intervals.get("sealed_panel")
    healthy = intervals.get("healthy_control_intervals")
    if not isinstance(sealed, dict) or not isinstance(healthy, dict):
        raise ValueError(
            f"interval contract {path} must declare intervals.sealed_panel and "
            "intervals.healthy_control_intervals"
        )

    # parsed by the contract module itself, so both tracks read these the same way
    development = [tuple(w) for w in contract_development_windows(payload)]
    sealed_windows = [
        tuple(w) for w in contract_interval_list(
            sealed.get("windows_s"), "intervals.sealed_panel.windows_s"
        )
    ]
    buffer_s = sealed.get("exclusion_buffer_s")
    if not isinstance(buffer_s, (int, float)) or isinstance(buffer_s, bool):
        raise ValueError("intervals.sealed_panel.exclusion_buffer_s must be a number")
    buffer_s = float(buffer_s)
    if not math.isfinite(buffer_s) or buffer_s < 0:
        raise ValueError("intervals.sealed_panel.exclusion_buffer_s must be finite and >= 0")

    # A list, possibly empty: what actually protects the reserved regions is
    # the derivation check below (development == stream minus everything
    # reserved), not a count of how many regions the contract chose to reserve.
    reserved_nodes = healthy.get("windows")
    if not isinstance(reserved_nodes, list):
        raise ValueError("intervals.healthy_control_intervals.windows must be a list")
    reserved: list[tuple[float, float]] = []
    reserved_names: list[str] = []
    for node in reserved_nodes:
        if not isinstance(node, dict):
            raise ValueError("each healthy control interval must be an object")
        (start, stop), = contract_interval_list(
            [[node.get("start_s"), node.get("stop_s")]],
            "intervals.healthy_control_intervals.windows",
        )
        reserved.append((start, stop))
        reserved_names.append(str(node.get("name", "")))

    if not math.isfinite(recording_duration_s) or recording_duration_s <= 0:
        raise ValueError("recording_duration_s must be finite and positive")
    if not development:
        raise ValueError(f"interval contract {path} declares no development windows")

    # bounds and recording limits
    for label, windows in (
        ("development", development), ("sealed", sealed_windows), ("reserved evaluation", reserved),
    ):
        for start, stop in windows:
            if not (math.isfinite(start) and math.isfinite(stop)):
                raise ValueError(f"{label} interval [{start}, {stop}] is not finite")
            if stop <= start:
                raise ValueError(f"{label} interval [{start}, {stop}] must have stop > start")
            if start < 0.0 or stop > recording_duration_s:
                raise ValueError(
                    f"{label} interval [{start}, {stop}] falls outside the recording "
                    f"[0, {recording_duration_s}]"
                )

    # development windows must not overlap each other
    ordered = sorted(development)
    for (a_start, a_stop), (b_start, b_stop) in zip(ordered, ordered[1:]):
        if b_start < a_stop:
            raise ValueError(
                f"development windows [{a_start}, {a_stop}] and [{b_start}, {b_stop}] overlap"
            )

    # ... nor intersect anything the contract reserves
    sealed_expanded = [
        (max(0.0, start - buffer_s), min(recording_duration_s, stop + buffer_s))
        for start, stop in sealed_windows
    ]
    for label, cuts in (
        ("a sealed window expanded by its exclusion buffer", sealed_expanded),
        ("a reserved healthy evaluation interval", reserved),
    ):
        for d_start, d_stop in development:
            for c_start, c_stop in cuts:
                if d_start < c_stop and c_start < d_stop:
                    raise ValueError(
                        f"development window [{d_start}, {d_stop}] intersects {label} "
                        f"[{c_start}, {c_stop}]"
                    )

    # the shipped list must be exactly the documented derivation
    recomputed = _subtract_intervals(
        [(0.0, float(recording_duration_s))], sealed_expanded + reserved
    )
    if not _intervals_equal(ordered, recomputed):
        raise ValueError(
            f"interval contract {path}: intervals.development_windows.windows_s is not the "
            "documented derivation (full stream minus sealed windows expanded by "
            f"exclusion_buffer_s minus healthy control intervals); shipped {ordered}, "
            f"recomputed {recomputed}"
        )

    return PermittedIntervals(
        contract_path=str(path),
        contract_sha256=contract_sha256,
        contract_id=str(contract_id),
        contract_schema=str(schema),
        clock=str(intervals.get("clock", "")),
        recording_duration_s=float(recording_duration_s),
        development_windows_s=tuple(ordered),
        sealed_windows_s=tuple(sealed_windows),
        sealed_exclusion_buffer_s=buffer_s,
        reserved_evaluation_windows_s=tuple(reserved),
        reserved_evaluation_names=tuple(reserved_names),
    )


# --------------------------------------------------------------------------- #
# layer 3 -- deterministic case selection
# --------------------------------------------------------------------------- #
#: Why a candidate run of ``windows_per_case`` consecutive stored windows was
#: not eligible. Exactly one reason is attributed per examined run, in this
#: precedence order, so the counts partition every run the enumerator saw and a
#: slow unit dropped by the span cap stays visible in the report.
#:
#: ``outside_development_window`` is the delivery contract's gate: a run that
#: satisfies every threshold but whose span leaves the permitted development
#: windows is re-attributed here and never enters a ranking, so a stronger
#: prohibited case can never displace a weaker eligible one.
FAILURE_REJECTION_REASONS = (
    "non_contiguous_index",
    "status_not_finite_interior",
    "nominal_count_not_required",
    "missing_time_bounds",
    "gap_between_windows",
    "nonfinite_missing_pct",
    "reference_above_max_missing_pct",
    "failing_below_min_missing_pct",
    "median_difference_below_min_pp",
    "span_over_max_s",
    "outside_development_window",
    "qualified",
)
CONTROL_REJECTION_REASONS = (
    "cluster_selected_as_failure",
    "non_contiguous_index",
    "status_not_finite_interior",
    "nominal_count_not_required",
    "missing_time_bounds",
    "gap_between_windows",
    "nonfinite_missing_pct",
    "above_max_missing_pct",
    "range_above_max_pp",
    "nonpositive_span",
    "span_over_max_s",
    "outside_development_window",
    "qualified",
)

_WINDOW_FIELDS = (
    "source_row", "i0", "i1", "first_sample", "last_sample",
    "start_s", "end_s", "historical_count", "nominal_count", "missing_pct", "status",
)


def _opt_int(value: Any, label: str = "windows.csv integer field") -> int | None:
    """Exact-integer read of a CSV cell, or ``None`` when the cell is empty.

    numpy scalars are unwrapped first: ``_exact_int_scalar`` deliberately
    rejects anything that is not a Python ``int``/``float``, and pandas hands
    back ``np.int64``/``np.float64``.
    """
    if value is None:
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is None or (not isinstance(value, (str, bytes)) and pd.isna(value)):
        return None
    return _exact_int_scalar(value, label)


def _opt_float(value: Any) -> float:
    if value is None or pd.isna(value):
        return float("nan")
    return float(value)


def window_records(windows: pd.DataFrame) -> dict[tuple[str, int], list[dict[str, Any]]]:
    """Group ``windows.csv`` rows by ``(sort_id, cluster_id)``, ordered by ``i0``.

    Rows without stored endpoints (the per-cluster ``no_fit`` placeholders) are
    dropped -- they describe a cluster, not an interval. Every row that *does*
    carry ``[i0, i1]`` is kept, including ``invalid_input``, ``nonfinite_fit``
    and ``boundary_pinned`` ones: they occupy their interval, so they must be
    able to interrupt an otherwise-contiguous run rather than being filtered
    out and letting the two windows around them look adjacent.
    """
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for raw in windows.to_dict("records"):
        i0 = _opt_int(raw.get("i0"))
        i1 = _opt_int(raw.get("i1"))
        if i0 is None or i1 is None:
            continue
        cluster_id = _opt_int(raw.get("cluster_id"), "windows.csv cluster_id")
        if cluster_id is None:
            raise ValueError(f"windows.csv row {raw!r} has no cluster_id")
        key = (str(raw["sort_id"]), cluster_id)
        grouped.setdefault(key, []).append({
            "sort_id": key[0],
            "cluster_id": cluster_id,
            "source_row": _opt_int(raw.get("source_row")),
            "i0": i0,
            "i1": i1,
            "first_sample": _opt_int(raw.get("first_sample")),
            "last_sample": _opt_int(raw.get("last_sample")),
            "start_s": _opt_float(raw.get("start_s")),
            "end_s": _opt_float(raw.get("end_s")),
            "historical_count": _opt_int(raw.get("historical_count")),
            "nominal_count": _opt_int(raw.get("nominal_count")),
            "missing_pct": _opt_float(raw.get("missing_pct")),
            "status": str(raw.get("status")),
        })
    for rows in grouped.values():
        rows.sort(key=lambda r: (r["i0"], r["i1"], r["source_row"] if r["source_row"] is not None else -1))
    return grouped


def _structural_reason(run: list[dict[str, Any]], c: SelectionConstants) -> str | None:
    """Reject a run that is not ``windows_per_case`` comparable windows tiling
    one gap-free block, or ``None`` if it is.

    Contiguity is index-based, as production builds it: ``construct_windows``
    tiles a block as ``(iB, iB + 999)`` stepping by 1000, so consecutive
    windows of one cluster satisfy ``i0_next == i1_prev + 1``
    (``pipeline/truncation.py``). Adjacency is never inferred from time.

    The additional inter-window check is *not* an adjacency inference: index
    contiguity has already established that window ``k``'s last spike and
    window ``k+1``'s first spike are adjacent in that cluster's own sequence,
    and the production gap rule then says a separation above
    ``PRODUCTION_MAX_ISI_S`` splits the block there. Without it, two blocks
    whose lengths happen to be exact multiples of the window size can tile back
    to back in index space and a "run" would straddle the gap between them --
    which the prescription forbids ("four consecutive windows inside one
    gap-free block", "do not construct a fit across that gap").
    """
    for prev, nxt in zip(run, run[1:]):
        if nxt["i0"] != prev["i1"] + 1:
            return "non_contiguous_index"
    if any(r["status"] != STATUS_FINITE_INTERIOR for r in run):
        return "status_not_finite_interior"
    if any(r["nominal_count"] != c.required_nominal_count for r in run):
        return "nominal_count_not_required"
    if not all(np.isfinite(r["start_s"]) and np.isfinite(r["end_s"]) for r in run):
        return "missing_time_bounds"
    for prev, nxt in zip(run, run[1:]):
        if nxt["start_s"] - prev["end_s"] > PRODUCTION_MAX_ISI_S:
            return "gap_between_windows"
    return None


def _run_span_s(run: list[dict[str, Any]]) -> float:
    return float(run[-1]["end_s"] - run[0]["start_s"])


def classify_failure_run(
    run: list[dict[str, Any]], c: SelectionConstants,
) -> tuple[str, dict[str, Any] | None]:
    """Prescribed failure-transition test on one candidate run.

    All thresholds are inclusive: reference ``<=``, failing ``>=``, median
    difference ``>=``, span ``<=``.
    """
    structural = _structural_reason(run, c)
    if structural is not None:
        return structural, None
    half = c.windows_per_case // 2
    reference = [r["missing_pct"] for r in run[:half]]
    failing = [r["missing_pct"] for r in run[half:]]
    if not all(np.isfinite(v) for v in reference + failing):
        return "nonfinite_missing_pct", None
    if not all(v <= c.reference_max_missing_pct for v in reference):
        return "reference_above_max_missing_pct", None
    if not all(v >= c.failing_min_missing_pct for v in failing):
        return "failing_below_min_missing_pct", None
    reference_median = float(np.median(reference))
    failing_median = float(np.median(failing))
    difference_pp = failing_median - reference_median
    if difference_pp < c.min_median_difference_pp:
        return "median_difference_below_min_pp", None
    span_s = _run_span_s(run)
    metrics = {
        "reference_median_missing_pct": reference_median,
        "failing_median_missing_pct": failing_median,
        "difference_pp": difference_pp,
        "span_s": span_s,
    }
    if span_s > c.max_span_s:
        return "span_over_max_s", metrics
    return "qualified", metrics


def classify_control_run(
    run: list[dict[str, Any]], c: SelectionConstants,
) -> tuple[str, dict[str, Any] | None]:
    """Prescribed stable-control test on one candidate run."""
    structural = _structural_reason(run, c)
    if structural is not None:
        return structural, None
    mpcts = [r["missing_pct"] for r in run]
    if not all(np.isfinite(v) for v in mpcts):
        return "nonfinite_missing_pct", None
    if not all(v <= c.control_max_missing_pct for v in mpcts):
        return "above_max_missing_pct", None
    range_pp = float(max(mpcts) - min(mpcts))
    if range_pp > c.control_max_range_pp:
        return "range_above_max_pp", None
    span_s = _run_span_s(run)
    metrics = {
        "max_missing_pct": float(max(mpcts)),
        "min_missing_pct": float(min(mpcts)),
        "range_pp": range_pp,
        "span_s": span_s,
    }
    if span_s <= 0.0:
        return "nonpositive_span", metrics
    if span_s > c.max_span_s:
        return "span_over_max_s", metrics
    return "qualified", metrics


def _case_windows(run: list[dict[str, Any]], roles: list[str]) -> list[dict[str, Any]]:
    out = []
    for position, (r, role) in enumerate(zip(run, roles)):
        entry = {"position": position, "window_role": role}
        entry.update({k: r[k] for k in _WINDOW_FIELDS})
        out.append(entry)
    return out


def _case_id(sort_id: str, cluster_id: int, role: str, rank: int) -> str:
    return f"{sort_id}__c{cluster_id}__{role}{rank}"


def _reported_sort_ids(
    windows: pd.DataFrame, grouped: dict[tuple[str, int], list[dict[str, Any]]],
    configured_sort_ids: list[str] | None,
) -> list[str]:
    """Every sort that must appear in the report, not just the ones with runs.

    ``window_records`` drops the per-cluster ``no_fit`` placeholders, so a sort
    whose whole inventory is ``no_fit`` has no entry there at all. Deriving the
    report's sort list from it would make that sort vanish instead of showing
    up with a valid zero-case result -- which is precisely the insufficient
    -data condition this audit exists to surface ("counts below the caps are
    valid"). So the list is the union of the sorts named in the raw window
    table and the configured ones, including a configured sort that produced no
    inventory rows whatsoever.
    """
    ids = {key[0] for key in grouped}
    if "sort_id" in getattr(windows, "columns", []):
        ids |= {str(v) for v in windows["sort_id"].dropna().unique()}
    if configured_sort_ids:
        ids |= {str(v) for v in configured_sort_ids}
    return sorted(ids)


def _run_window_index(run: list[dict[str, Any]], permitted: PermittedIntervals) -> int | None:
    """Development window containing this run's whole span, or ``None``.

    The span is the case's own span: the first window's ``start_s`` to the last
    window's ``end_s``. Every window in between is inside it by construction,
    so containment of the span is containment of the case.
    """
    return permitted.containing_index(run[0]["start_s"], run[-1]["end_s"])


def select_cases(
    windows: pd.DataFrame, c: SelectionConstants, *,
    permitted: PermittedIntervals,
    configured_sort_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Freeze the audit's cases from the cached historical QC inventory alone.

    No waveform, voltage, candidate or intervention outcome is read here, and
    nothing is refit; KS-good/MUA labels are descriptive elsewhere and are not
    an eligibility filter. Counts below the caps are a valid result -- no
    threshold is relaxed and no case is backfilled when cases are scarce, and
    every sort keeps a ``per_sort`` entry even when it yields nothing.

    ``permitted`` is required, with no default: the delivery contract reserves
    a sealed panel and healthy evaluation intervals out of the same recording,
    so there is no such thing as ranking "the whole recording". A run outside
    the development windows is excluded BEFORE ranking, not merely outranked.
    """
    grouped = window_records(windows)
    k = c.windows_per_case
    sort_ids = _reported_sort_ids(windows, grouped, configured_sort_ids)

    cases: list[dict[str, Any]] = []
    failure_exclusions: dict[str, int] = {r: 0 for r in FAILURE_REJECTION_REASONS}
    control_exclusions: dict[str, int] = {r: 0 for r in CONTROL_REJECTION_REASONS}
    per_sort: dict[str, Any] = {}

    for sort_id in sort_ids:
        cluster_ids = sorted(key[1] for key in grouped if key[0] == sort_id)

        best_by_cluster: dict[int, dict[str, Any]] = {}
        for cluster_id in cluster_ids:
            rows = grouped[(sort_id, cluster_id)]
            for j in range(len(rows) - k + 1):
                run = rows[j:j + k]
                reason, metrics = classify_failure_run(run, c)
                window_index = None
                if reason == "qualified":
                    # the contract's gate, applied before this run can become a
                    # candidate: a prohibited run is never ranked at all
                    window_index = _run_window_index(run, permitted)
                    if window_index is None:
                        reason = "outside_development_window"
                failure_exclusions[reason] += 1
                if reason != "qualified":
                    continue
                candidate = {
                    "sort_id": sort_id, "cluster_id": cluster_id, "run": run, **metrics,
                    "start_s": run[0]["start_s"],
                    "development_window_index": window_index,
                }
                current = best_by_cluster.get(cluster_id)
                if current is None or (
                    (-candidate["difference_pp"], candidate["start_s"])
                    < (-current["difference_pp"], current["start_s"])
                ):
                    best_by_cluster[cluster_id] = candidate

        ranked = sorted(
            best_by_cluster.values(),
            key=lambda x: (-x["difference_pp"], x["start_s"], x["cluster_id"]),
        )
        selected_failures = ranked[:c.max_failure_cases_per_sort]
        failure_cluster_ids = {x["cluster_id"] for x in selected_failures}

        half = k // 2
        roles = ["reference"] * half + ["failing"] * (k - half)
        for rank, candidate in enumerate(selected_failures, start=1):
            cases.append({
                "case_id": _case_id(sort_id, candidate["cluster_id"], "failure", rank),
                "sort_id": sort_id,
                "cluster_id": candidate["cluster_id"],
                "role": "failure",
                "rank": rank,
                "span_s": candidate["span_s"],
                "difference_pp": candidate["difference_pp"],
                "reference_median_missing_pct": candidate["reference_median_missing_pct"],
                "failing_median_missing_pct": candidate["failing_median_missing_pct"],
                "development_window_index": candidate["development_window_index"],
                "development_window_s": list(
                    permitted.development_windows_s[candidate["development_window_index"]]
                ),
                "windows": _case_windows(candidate["run"], roles),
                "reason": (
                    f"{k} consecutive stored windows of cluster {candidate['cluster_id']} tiling one "
                    f"gap-free block (i0_next == i1_prev + 1), all finite_interior with nominal "
                    f"count {c.required_nominal_count}; reference median "
                    f"{candidate['reference_median_missing_pct']:.6g}% <= "
                    f"{c.reference_max_missing_pct:.6g}% with both reference windows at or below "
                    f"that bound; failing median {candidate['failing_median_missing_pct']:.6g}% >= "
                    f"{c.failing_min_missing_pct:.6g}% with both failing windows at or above it; "
                    f"median difference {candidate['difference_pp']:.6g} pp >= "
                    f"{c.min_median_difference_pp:.6g} pp; span {candidate['span_s']:.6g} s <= "
                    f"{c.max_span_s:.6g} s; largest difference for this cluster, rank {rank} of "
                    f"{len(selected_failures)} kept for sort {sort_id} "
                    f"(cap {c.max_failure_cases_per_sort}) by difference desc, then start_s, then "
                    f"numeric cluster ID"
                ),
            })

        control_candidates: list[dict[str, Any]] = []
        for cluster_id in cluster_ids:
            rows = grouped[(sort_id, cluster_id)]
            for j in range(len(rows) - k + 1):
                run = rows[j:j + k]
                if cluster_id in failure_cluster_ids:
                    control_exclusions["cluster_selected_as_failure"] += 1
                    continue
                reason, metrics = classify_control_run(run, c)
                window_index = None
                if reason == "qualified":
                    # diagnostic controls are gated exactly like failure cases;
                    # the contract's own reserved healthy evaluation intervals
                    # are excluded from every development window by construction
                    window_index = _run_window_index(run, permitted)
                    if window_index is None:
                        reason = "outside_development_window"
                control_exclusions[reason] += 1
                if reason != "qualified":
                    continue
                control_candidates.append({
                    "sort_id": sort_id, "cluster_id": cluster_id, "run": run, **metrics,
                    "start_s": run[0]["start_s"],
                    "development_window_index": window_index,
                })

        reference_span_s = selected_failures[0]["span_s"] if selected_failures else None
        n_eligible_controls = len(control_candidates)
        if reference_span_s is not None:
            control_candidates.sort(key=lambda x: (
                abs(math.log(x["span_s"] / reference_span_s)), x["start_s"], x["cluster_id"],
            ))
        else:
            control_candidates.sort(key=lambda x: (x["start_s"], x["cluster_id"]))
        selected_controls = control_candidates[:c.max_control_cases_per_sort]

        for rank, candidate in enumerate(selected_controls, start=1):
            if reference_span_s is not None:
                log_ratio = abs(math.log(candidate["span_s"] / reference_span_s))
                why = (
                    f"minimal |log(control span {candidate['span_s']:.6g} s / first failure span "
                    f"{reference_span_s:.6g} s)| = {log_ratio:.6g} among {n_eligible_controls} "
                    f"eligible control runs (ties by start_s then numeric cluster ID)"
                )
            else:
                log_ratio = None
                why = (
                    f"sort {sort_id} has no selected failure case, so the earliest eligible control "
                    f"was taken among {n_eligible_controls} (ties by numeric cluster ID)"
                )
            cases.append({
                "case_id": _case_id(sort_id, candidate["cluster_id"], "control", rank),
                "sort_id": sort_id,
                "cluster_id": candidate["cluster_id"],
                "role": "control",
                # a diagnostic control of this audit, NOT one of the contract's
                # reserved healthy evaluation intervals
                "control_kind": DIAGNOSTIC_CONTROL_KIND,
                "rank": rank,
                "span_s": candidate["span_s"],
                "difference_pp": None,
                "range_pp": candidate["range_pp"],
                "max_missing_pct": candidate["max_missing_pct"],
                "min_missing_pct": candidate["min_missing_pct"],
                "abs_log_span_ratio": log_ratio,
                "development_window_index": candidate["development_window_index"],
                "development_window_s": list(
                    permitted.development_windows_s[candidate["development_window_index"]]
                ),
                "windows": _case_windows(candidate["run"], ["control"] * k),
                "reason": (
                    f"{k} consecutive stored windows of cluster {candidate['cluster_id']} tiling one "
                    f"gap-free block, all finite_interior with nominal count "
                    f"{c.required_nominal_count} and missing_pct <= "
                    f"{c.control_max_missing_pct:.6g}% (min {candidate['min_missing_pct']:.6g}%, max "
                    f"{candidate['max_missing_pct']:.6g}%, range {candidate['range_pp']:.6g} pp <= "
                    f"{c.control_max_range_pp:.6g} pp); span {candidate['span_s']:.6g} s <= "
                    f"{c.max_span_s:.6g} s; cluster is not one of this sort's selected failure "
                    f"clusters; {why}"
                ),
            })

        per_sort[sort_id] = {
            "n_clusters_with_stored_windows": len(cluster_ids),
            "n_failure_eligible_clusters": len(best_by_cluster),
            "n_failure_cases_selected": len(selected_failures),
            "failure_cap": c.max_failure_cases_per_sort,
            "n_eligible_control_runs": n_eligible_controls,
            "n_control_cases_selected": len(selected_controls),
            "control_cap": c.max_control_cases_per_sort,
            "first_failure_span_s": reference_span_s,
        }

    return {
        "cases": cases,
        "per_sort": per_sort,
        "exclusion_counts": {
            "failure_runs_by_reason": failure_exclusions,
            "control_runs_by_reason": control_exclusions,
            "failure_runs_excluded_by_span_cap": failure_exclusions["span_over_max_s"],
            "control_runs_excluded_by_span_cap": control_exclusions["span_over_max_s"],
            "failure_runs_excluded_outside_development_windows":
                failure_exclusions["outside_development_window"],
            "control_runs_excluded_outside_development_windows":
                control_exclusions["outside_development_window"],
        },
        "interval_contract": permitted.to_provenance(),
    }


def canonical_selection_digest(payload: dict[str, Any]) -> str:
    """sha256 over the frozen content of ``selection.json``.

    Everything except the ``selection_sha256`` field itself is hashed, in a
    canonical (key-sorted, whitespace-free) encoding, so W2's verification does
    not depend on how the file happened to be pretty-printed.
    """
    frozen = {k: v for k, v in payload.items() if k != "selection_sha256"}
    encoded = json.dumps(frozen, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# CLI -- `inventory`, `select` and `inspect`
# --------------------------------------------------------------------------- #
def audit_source_files() -> dict[str, Path]:
    """Working-tree source files whose content can change an audit answer.

    The prescription requires the Git commit *and* hashes of the relevant
    working-tree sources, "since the workspace may be dirty" -- so `inspect`
    compares these hashes, not the commit, when deciding whether the code that
    froze a selection is still the code replaying it.
    """
    return {
        "module": Path(__file__),
        "pipeline.truncation": REPO_ROOT / "pipeline/truncation.py",
        # parses and validates the delivery contract's intervals, so its
        # content can change which spans are eligible to be ranked
        "testing.first_pipeline_candidate_contract":
            REPO_ROOT / "testing/first_pipeline_candidate_contract.py",
    }


def _configured_recording_duration_s(cfg: AuditConfig) -> float:
    """The one recording duration the interval clock is measured against.

    The delivery contract's intervals are "seconds from the start of the
    selected imec0 stream", and every configured sort is a sort OF that one
    stream, so their declared durations must agree; a disagreement means the
    audit and the contract are not describing the same recording, which is
    exactly the W1/W6 coupling this gate exists to close.
    """
    durations = {s.sort_id: float(s.duration_s) for s in cfg.sorts}
    distinct = sorted(set(durations.values()))
    if len(distinct) != 1:
        raise ValueError(
            "configured sorts declare different recording durations, so the delivery "
            f"contract's interval clock is ambiguous: {durations}"
        )
    return distinct[0]


def _config_input_paths(cfg: AuditConfig) -> list[Path]:
    paths: list[Path] = []
    for s in cfg.sorts:
        paths += [s.curated, s.qc_dir, s.source_recording]
    return paths


def _require_stage_manifest(
    manifest_path: Path, config_path: Path, *, expected_stage: str, requester: str,
    config_sha256: str,
) -> dict[str, Any]:
    """Refuse anything but a *completed* ``expected_stage`` manifest of this config.

    Shared by `select` (which requires a completed ``inventory``) and `inspect`
    (which requires a completed ``select``). A mismatch is refused with a
    reason naming what disagreed; the earlier stage is never regenerated,
    repaired or rediscovered from a sibling directory.
    """
    if not manifest_path.exists():
        raise RuntimeError(
            f"{requester} requires a completed {expected_stage} in this out-root; no "
            f"manifest.json at {manifest_path}. Run `{expected_stage}` first; {requester} "
            "never generates one."
        )
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{manifest_path} is not readable JSON: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise RuntimeError(
            f"{manifest_path} has schema {manifest.get('schema') if isinstance(manifest, dict) else None!r}"
            f", expected {SCHEMA!r}"
        )
    if manifest.get("stage") != expected_stage:
        raise RuntimeError(
            f"{manifest_path} is at stage {manifest.get('stage')!r}, expected "
            f"{expected_stage!r}; {requester} runs exactly once on a freshly completed "
            f"{expected_stage}"
        )
    if manifest.get("status") != "complete":
        raise RuntimeError(
            f"{manifest_path} reports {expected_stage} status {manifest.get('status')!r}, "
            f"expected 'complete'; refusing to run {requester} from an incomplete "
            f"{expected_stage}"
        )
    # Required, not re-derived: the digest compared here must cover the very
    # bytes the caller parsed into its AuditConfig. Re-hashing config_path
    # would be a second read of a file that can change in between, which is
    # exactly the hole `read_config_once` exists to close.
    expected = config_sha256
    if manifest.get("config_sha256") != expected:
        raise RuntimeError(
            f"config_sha256 mismatch: {manifest_path} recorded "
            f"{manifest.get('config_sha256')!r} but {Path(config_path)} hashes to {expected!r}; "
            f"refusing to run {requester} against a different config than the {expected_stage} "
            "was built with"
        )
    return manifest


def _prepare_out_root(
    stage: str, out_root: Path, config_path: Path, input_paths: list[Path], *,
    config_sha256: str, selection_path: Path | None = None,
) -> tuple[Path, dict[str, Any] | None]:
    """Resolve and gate the single local output root, per stage.

    ``inventory`` refuses ANY non-empty root, not just one carrying our own
    manifest.json: an orphaned windows.csv from a partial/incompatible prior
    run must not be silently overwritten either. ``select`` must instead write
    into the very root ``inventory`` populated, so it requires that completed
    inventory and refuses an existing selection.json. ``inspect`` in turn
    requires the completed ``select`` in that same root and refuses an existing
    case_windows.csv. The ``/mnt``-and-input-directory rejection applies to all
    three (prescription: "Existing incompatible outputs must be refused. Do not
    implement automatic cache repair or recursive 'latest run' discovery").
    """
    resolved = _reject_unsafe_out_root(Path(out_root), input_paths)
    if stage == "inventory":
        if resolved.exists() and any(resolved.iterdir()):
            raise RuntimeError(
                f"refusing to write into non-empty output root {resolved}; "
                "pick a fresh --out-root or remove it yourself"
            )
        return resolved, None
    if stage == "select":
        # Checked before the manifest, so a second `select` is refused for the
        # true reason -- the cases are already frozen -- rather than for the
        # stage marker the first run left behind.
        if (resolved / "selection.json").exists():
            raise RuntimeError(
                f"refusing to overwrite an existing frozen selection at "
                f"{resolved / 'selection.json'}; case IDs are frozen once"
            )
        manifest = _require_stage_manifest(
            resolved / "manifest.json", config_path,
            expected_stage="inventory", requester="select", config_sha256=config_sha256,
        )
        _require_attested_windows_csv(resolved, manifest, requester="select")
        # The attested hash is deliberately NOT compared here. Hashing the path
        # in one place and parsing the path in another is two reads of a file
        # that can change in between, so the comparison lives in
        # `read_attested_windows`, which hashes and parses the SAME bytes.
        return resolved, manifest
    if stage == "inspect":
        if selection_path is None:
            raise ValueError("inspect requires the frozen selection path")
        # Checked before the manifest, so a second `inspect` is refused for the
        # true reason -- this root already carries a replay -- rather than for
        # the stage marker the first run left behind.
        if (resolved / "case_windows.csv").exists():
            raise RuntimeError(
                f"refusing to overwrite an existing replay at {resolved / 'case_windows.csv'}; "
                "pick a fresh out-root and re-run `inventory`/`select` there rather than "
                "repairing this one"
            )
        selection_path = Path(selection_path).resolve()
        if selection_path.parent != resolved:
            raise RuntimeError(
                f"--selection {selection_path} does not live in --out-root {resolved}; inspect "
                "verifies and extends one output root's own manifest, and never discovers a run "
                "in a sibling directory"
            )
        manifest = _require_stage_manifest(
            resolved / "manifest.json", config_path,
            expected_stage="select", requester="inspect", config_sha256=config_sha256,
        )
        _require_attested_windows_csv(resolved, manifest, requester="inspect")
        select_block = manifest.get("select")
        if not isinstance(select_block, dict) or not select_block.get("selection_sha256"):
            raise RuntimeError(
                f"{resolved / 'manifest.json'} records no select.selection_sha256; refusing to "
                "replay cases whose freeze this out-root never attested"
            )
        recorded_selection_path = select_block.get("selection_path")
        if not recorded_selection_path or Path(recorded_selection_path).resolve() != selection_path:
            raise RuntimeError(
                f"--selection {selection_path} is not the selection {resolved / 'manifest.json'} "
                f"attested ({recorded_selection_path!r}); refusing to replay a selection this "
                "out-root did not freeze"
            )
        return resolved, manifest
    raise ValueError(f"unknown stage {stage!r}")


def _require_attested_windows_csv(
    resolved: Path, manifest: dict[str, Any], *, requester: str,
) -> Path:
    """The inventory's windows.csv must exist AND have been attested by it."""
    windows_path = resolved / "windows.csv"
    if not windows_path.exists():
        raise RuntimeError(
            f"{requester} requires the inventory's windows.csv; none found in {resolved}"
        )
    if not manifest.get("windows_csv_sha256"):
        raise RuntimeError(
            f"{resolved / 'manifest.json'} records no windows_csv_sha256; refusing to use rows "
            "whose provenance the inventory never attested. Re-run `inventory` into a fresh "
            "out-root."
        )
    return windows_path


def run_inventory(config_path: Path, out_root: Path) -> dict[str, Any]:
    cfg, config_sha256 = read_config_once(config_path)
    out_root, _ = _prepare_out_root(
        "inventory", Path(out_root), Path(config_path), _config_input_paths(cfg),
        config_sha256=config_sha256,
    )
    manifest_path = out_root / "manifest.json"

    source_files = audit_source_files()
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "stage": "inventory",
        "status": "running",
        "git_commit": git_commit(),
        "source_sha256": {name: sha256_file(p) for name, p in source_files.items()},
        "config_path": str(Path(config_path).resolve()),
        "config_sha256": config_sha256,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sorts": {},
    }
    _atomic_write_json(manifest_path, manifest)

    try:
        tables = []
        for s in cfg.sorts:
            recording_check = validate_recording_metadata(s)
            # Each input is hashed from the very bytes that are parsed, so the
            # digests recorded below attest the arrays this table was built
            # from -- which is what `inspect` later checks the replay against.
            raw_curated, curated_hashes = read_curated_arrays(s.curated)
            curated = curated_arrays_from_raw(s.sort_id, raw_curated)
            cached, cached_qc_sha256 = read_cached_truncation_qc(s.sort_id, s.qc_dir)
            table = build_windows_table(curated, cached, fs=s.sampling_frequency_hz)
            tables.append(table)
            manifest["sorts"][s.sort_id] = {
                "curated": str(s.curated),
                "qc_dir": str(s.qc_dir),
                "source_recording": str(s.source_recording),
                "n_spikes": int(curated.times.size),
                "n_clusters": int(np.unique(curated.clusters).size),
                "n_cached_windows": int(cached.cid.size),
                "was_time_ordered": curated.was_time_ordered,
                "recording_metadata_check": recording_check,
                "curated_file_hashes": curated_hashes,
                "cached_qc_sha256": cached_qc_sha256,
            }

        windows = (
            pd.concat(tables, ignore_index=True) if tables
            else pd.DataFrame(columns=list(WINDOWS_COLUMNS))
        )
        _atomic_write_csv(out_root / "windows.csv", windows)

        manifest["status"] = "complete"
        # Recorded at inventory time so `select` can prove it is ranking the
        # rows this inventory actually produced. Without it, a windows.csv
        # replaced between the two stages would simply be re-hashed by select,
        # and selection.json's own hash would authenticate the replacement.
        manifest["windows_csv_sha256"] = sha256_file(out_root / "windows.csv")
        manifest["row_counts"] = {
            "windows_total": int(len(windows)),
            "status_counts": (
                {str(k): int(v) for k, v in windows["status"].value_counts().items()}
                if len(windows) else {}
            ),
        }
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["failure_reason"] = f"{type(exc).__name__}: {exc}"
        _atomic_write_json(manifest_path, manifest)
        raise

    _atomic_write_json(manifest_path, manifest)
    return manifest


def _parse_windows_bytes(data: bytes) -> pd.DataFrame:
    """Parse ``windows.csv`` from bytes already in hand, without letting CSV
    type inference rewrite an identity.

    ``pd.read_csv``'s inference silently mangles a sort ID on the way back in:
    a column of ``"001"``/``"002"`` returns as ``1``/``2``, ``"1e3"`` as
    ``1000.0``, and ``"NA"`` as ``nan`` even in a mixed column that otherwise
    keeps its strings. The configured sort ID would then no longer match the
    one keying the rows -- wrong case IDs, plus a phantom zero-case ``per_sort``
    entry for the same sort. The prescription makes this load-bearing: the sort
    ID is an immutable identifier supplied by config, and "IDs are values".

    So the file is read as raw text with NA conversion disabled, and only then
    are the *numeric* columns converted back:

    * identity text (``_WINDOWS_TEXT_COLUMNS``: ``sort_id``, ``status``) is
      taken byte-for-byte, so ``"001"``, ``"1e3"`` and ``"NA"`` all survive;
    * ``invalid_reason`` is free text, and an empty cell becomes ``None``;
    * every other column is a nullable numeric -- an empty cell becomes NaN,
      which is what ``_opt_int``/``_opt_float`` and the ``no_fit`` placeholder
      rows (blank ``i0``/``i1``/times) expect. ``keep_default_na=False`` is
      deliberately NOT applied to these; blanks must stay missing.

    A numeric column carrying junk is left as text rather than coerced to NaN,
    so the exact-integer validators reject it instead of silently reading a
    corrupt row as a missing one.

    An empty or header-less file still yields an empty table with the expected
    columns: an inventory that found nothing is a valid insufficient-data
    result and must reach `select` so every configured sort is still reported.
    """
    try:
        raw = pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False, na_filter=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=list(WINDOWS_COLUMNS))

    parsed = pd.DataFrame(index=raw.index)
    for column in raw.columns:
        values = raw[column]
        if column in _WINDOWS_TEXT_COLUMNS:
            parsed[column] = values
            continue
        blanked = values.where(values != "", other=None)
        if column in _WINDOWS_FREE_TEXT_COLUMNS:
            parsed[column] = blanked
            continue
        try:
            parsed[column] = pd.to_numeric(blanked)
        except (TypeError, ValueError):
            parsed[column] = values
    return parsed


def read_attested_windows(windows_path: Path, recorded_sha256: str) -> tuple[pd.DataFrame, str]:
    """Read ``windows.csv`` exactly once, then hash and parse those same bytes.

    Hashing the path and separately parsing the path would leave a window in
    which the file can be replaced: the substituted rows would be ranked even
    though the inventory never attested them. Reading once closes it -- the
    bytes that are hashed are the bytes that are parsed, so a mismatch is
    refused and nothing else can have been ranked.
    """
    data = Path(windows_path).read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != recorded_sha256:
        raise RuntimeError(
            f"windows.csv sha256 mismatch: the inventory recorded {recorded_sha256!r} but the "
            f"bytes read from {windows_path} hash to {actual!r}; it was modified, corrupted or "
            "replaced since the inventory completed. Refusing to select from it -- re-run "
            "`inventory` into a fresh out-root rather than repairing this one."
        )
    return _parse_windows_bytes(data), actual


def run_select(config_path: Path, out_root: Path) -> dict[str, Any]:
    """Freeze the audit's cases from the completed inventory in ``out_root``.

    Reads ``windows.csv`` only. Selection constants come from CONFIG and are
    recorded verbatim in ``selection.json`` alongside the hashes of every input
    that could change the answer, so the freeze is auditable after the fact.
    """
    cfg, config_sha256 = read_config_once(config_path)
    out_root, manifest = _prepare_out_root(
        "select", Path(out_root), Path(config_path), _config_input_paths(cfg),
        config_sha256=config_sha256,
    )
    manifest_path = out_root / "manifest.json"
    windows_path = out_root / "windows.csv"
    selection_path = out_root / "selection.json"

    # One read, before the manifest is touched: the table that gets ranked and
    # the hash that gets recorded both come from these bytes, so a refusal here
    # leaves the completed inventory exactly as it was.
    windows, windows_sha256 = read_attested_windows(
        windows_path, manifest["windows_csv_sha256"]
    )

    manifest["stage"] = "select"
    manifest["status"] = "running"
    manifest["select_started_at"] = datetime.now(timezone.utc).isoformat()
    manifest.pop("failure_reason", None)
    _atomic_write_json(manifest_path, manifest)

    try:
        ref = cfg.require_interval_contract(config_path)
        permitted = read_permitted_intervals(
            ref, recording_duration_s=_configured_recording_duration_s(cfg),
        )
        result = select_cases(
            windows, cfg.selection, permitted=permitted,
            configured_sort_ids=[s.sort_id for s in cfg.sorts],
        )
        payload: dict[str, Any] = {
            "schema": SCHEMA,
            "stage": "select",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit(),
            "source_sha256": {
                name: sha256_file(path) for name, path in audit_source_files().items()
            },
            "config_path": str(Path(config_path).resolve()),
            "config_sha256": config_sha256,
            "windows_csv_path": str(windows_path),
            # the hash of the bytes actually parsed, already proven equal to
            # the hash the inventory attested
            "windows_csv_sha256": windows_sha256,
            "selection_constants": cfg.selection.to_dict(),
            "production_constants": {
                "max_isi_s": PRODUCTION_MAX_ISI_S,
                "max_isi_s_units": "seconds",
                "contiguity_rule": "i0_next == i1_prev + 1 (pipeline/truncation.py construct_windows)",
            },
            "interval_contract": result["interval_contract"],
            "cases": result["cases"],
            "per_sort": result["per_sort"],
            "exclusion_counts": result["exclusion_counts"],
            "notes": [
                "Selection reads cached historical QC rows only; no candidate, waveform, "
                "voltage or intervention outcome is an input.",
                "Counts below the caps are a valid result; no threshold was relaxed and no "
                "case was backfilled.",
                "KS-good/MUA status is descriptive elsewhere and does not filter eligibility.",
                "Every case span lies inside one delivery-contract development window; runs "
                "outside them were excluded before ranking, never outranked.",
                "Audit controls are DIAGNOSTIC controls of this audit. The contract's "
                "healthy_control_intervals are reserved evaluation intervals, are excluded from "
                "every development window, and are never selected here.",
            ],
        }
        payload["selection_sha256"] = canonical_selection_digest(payload)
        _atomic_write_json(selection_path, payload)

        manifest["status"] = "complete"
        manifest["select"] = {
            "selection_path": str(selection_path),
            "selection_sha256": payload["selection_sha256"],
            "windows_csv_sha256": payload["windows_csv_sha256"],
            "n_cases": len(payload["cases"]),
            "case_ids": [c["case_id"] for c in payload["cases"]],
            "per_sort": payload["per_sort"],
            "exclusion_counts": payload["exclusion_counts"],
        }
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["failure_reason"] = f"{type(exc).__name__}: {exc}"
        _atomic_write_json(manifest_path, manifest)
        raise

    _atomic_write_json(manifest_path, manifest)
    return payload


# --------------------------------------------------------------------------- #
# layer 4a -- `inspect`: gate, then historical/exact replay on the frozen cases
#
# This is the *numeric* half of the prescription's `inspect` stage only:
# case_windows.csv, its reproduction check and the exact-1,000 sensitivity
# flag. Figures, voltage review, case_evidence.csv and decision.md belong to
# later layers and are deliberately absent.
# --------------------------------------------------------------------------- #
#: Acceptance-test 5's reproduction tolerance, in percentage points. Frozen as
#: module constants so no caller can widen them: a replayed historical estimate
#: outside this band of the cached one is recorded as an input/runtime mismatch
#: and its case is marked unstable ("record an input/runtime mismatch and stop
#: interpretation"), never accepted by relaxing the tolerance.
REPRODUCTION_RTOL = 1e-6
REPRODUCTION_ATOL_PP = 1e-6

#: Per-window replay outcome.
CASE_WINDOW_REPRODUCED = "reproduced"
CASE_WINDOW_REPRODUCTION_MISMATCH = "reproduction_mismatch"

#: Withheld eligibility verdict. A case whose historical replay did not
#: reproduce its cached value gets this instead of an exact-indexing verdict:
#: the prescription's "record an input/runtime mismatch and stop
#: interpretation" makes the verdict itself interpretation, and it could not be
#: told apart from the mismatch that was just detected.
EXACT_ELIGIBILITY_NOT_INTERPRETED = "not_interpreted_reproduction_mismatch"

#: Per-case outcome. ``unstable`` is set by a reproduction mismatch, by the
#: exact-1,000 eligibility sensitivity, or by both; the two flag columns say
#: which -- ``unstable_under_exact_indexing`` is blank when the verdict was
#: withheld. It never removes, replaces or re-ranks the case.
CASE_STATUS_STABLE = "stable"
CASE_STATUS_UNSTABLE = "unstable"

#: ``case_windows.csv``'s columns, in order. Both sample counts are carried on
#: every row (``historical_count`` 999 vs ``exact_count`` 1,000 for a nominal
#: 1,000-spike window), as are both fits' parameters and both saturation flags.
CASE_WINDOWS_COLUMNS = (
    "case_id", "sort_id", "cluster_id", "case_role", "window_role", "window_ordinal",
    "source_row", "i0", "i1", "first_sample", "last_sample", "start_s", "end_s",
    "historical_count", "exact_count",
    "cached_missing_pct", "replayed_historical_missing_pct", "exact_missing_pct",
    "reproduction_abs_diff_pp", "exact_minus_historical_pp",
    "historical_saturated", "exact_saturated",
    "historical_fit_x0", "historical_fit_k", "historical_fit_A",
    "exact_fit_x0", "exact_fit_k", "exact_fit_A",
    "status", "status_reason",
    "case_status", "unstable_reproduction_mismatch", "unstable_under_exact_indexing",
    "exact_eligibility_reason",
)


def load_attested_selection(selection_path: Path) -> dict[str, Any]:
    """Read ``selection.json`` and refuse it unless its own digest still holds.

    ``canonical_selection_digest`` hashes everything except the recorded digest
    itself, so an edited case, window, threshold or hash inside the frozen file
    is caught here before any array is read.
    """
    selection_path = Path(selection_path)
    if not selection_path.exists():
        raise RuntimeError(f"no frozen selection at {selection_path}")
    try:
        payload = json.loads(selection_path.read_text())
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{selection_path} is not readable JSON: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise RuntimeError(
            f"{selection_path} has schema "
            f"{payload.get('schema') if isinstance(payload, dict) else None!r}, "
            f"expected {SCHEMA!r}"
        )
    if payload.get("stage") != "select":
        raise RuntimeError(
            f"{selection_path} is at stage {payload.get('stage')!r}, expected 'select'"
        )
    recorded = payload.get("selection_sha256")
    actual = canonical_selection_digest(payload)
    if recorded != actual:
        raise RuntimeError(
            f"selection_sha256 mismatch: {selection_path} records {recorded!r} but its frozen "
            f"content hashes to {actual!r}; it was edited since `select` wrote it. Refusing to "
            "inspect a selection that is no longer the one that was frozen."
        )
    return payload


def verify_recorded_inputs(
    payload: dict[str, Any], config_path: Path,
) -> tuple[AuditConfig, dict[str, Any]]:
    """Refuse if any input the freeze recorded has moved, naming WHICH one.

    Working-tree source hashes are compared, not the Git commit: the
    prescription expects a dirty workspace, so an unchanged commit is no
    evidence that the fitter or this module is the one that produced the
    frozen numbers.

    CONFIG is read exactly once here and parsed from those same bytes, so the
    returned :class:`AuditConfig` is provably the attested one.
    """
    config_path = Path(config_path)
    try:
        # ONE read: the bytes hashed here are the bytes parsed below, so a
        # config swapped after the check can never reach the replay.
        config_bytes = config_path.read_bytes()
    except OSError as exc:
        raise RuntimeError(
            f"the config the selection was frozen against is unreadable: {config_path} ({exc})"
        ) from exc
    moved: list[str] = []
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    if payload.get("config_sha256") != config_sha256:
        moved.append(
            f"config ({config_path}): frozen {payload.get('config_sha256')!r} != current "
            f"{config_sha256!r}"
        )
    recorded_sources = payload.get("source_sha256") or {}
    sources: dict[str, str] = {}
    for name, path in audit_source_files().items():
        sources[name] = sha256_file(path)
        if recorded_sources.get(name) != sources[name]:
            moved.append(
                f"{name} ({path}): frozen {recorded_sources.get(name)!r} != current "
                f"{sources[name]!r}"
            )
    if moved:
        raise RuntimeError(
            "refusing to inspect: the following recorded input(s) moved since the selection was "
            "frozen -- " + "; ".join(moved) + ". Re-run `inventory` and `select` into a fresh "
            "out-root; nothing here is repaired in place."
        )
    return load_config_from_bytes(config_bytes, config_path), {
        "config": config_sha256, "sources": sources,
    }


def _same_frozen_value(frozen: Any, attested: Any) -> bool:
    if frozen is None or attested is None:
        return frozen is None and attested is None
    if isinstance(frozen, float) or isinstance(attested, float):
        f, a = float(frozen), float(attested)
        if math.isnan(f) or math.isnan(a):
            return math.isnan(f) and math.isnan(a)
        return f == a
    return frozen == attested


def verify_frozen_windows_against_inventory(
    payload: dict[str, Any], windows: pd.DataFrame,
) -> None:
    """Every frozen case window must still be exactly the attested inventory row.

    ``selection.json`` carries its own digest, so a hand-written one can be
    made self-consistent. The inventory's ``windows.csv`` cannot: the manifest
    attested its bytes, and ``read_attested_windows`` proved them. So the
    windows table -- not the selection file -- is the authority for every
    cached value this stage replays, and a disagreement is refused rather than
    replayed.
    """
    lookup: dict[tuple[str, int, int | None], dict[str, Any]] = {}
    for (sort_id, cluster_id), rows in window_records(windows).items():
        for row in rows:
            lookup[(sort_id, cluster_id, row["source_row"])] = row
    for case in payload.get("cases", []):
        for frozen in case["windows"]:
            key = (str(case["sort_id"]), int(case["cluster_id"]), frozen["source_row"])
            attested = lookup.get(key)
            if attested is None:
                raise RuntimeError(
                    f"case {case['case_id']} window {frozen['position']} (sort {key[0]}, cluster "
                    f"{key[1]}, source_row {key[2]}) is not in the attested windows.csv; refusing "
                    "to replay a case the inventory does not contain"
                )
            for field in _WINDOW_FIELDS:
                if not _same_frozen_value(frozen.get(field), attested.get(field)):
                    raise RuntimeError(
                        f"case {case['case_id']} window {frozen['position']} disagrees with the "
                        f"attested windows.csv on {field!r}: frozen {frozen.get(field)!r} != "
                        f"inventory {attested.get(field)!r}"
                    )


def read_attested_curated_inputs(
    payload: dict[str, Any], cfg: AuditConfig, manifest: dict[str, Any],
) -> tuple[dict[str, CuratedArrays], dict[str, dict[str, str]]]:
    """Load every curated array the replay will consume, attested as it is read.

    The replay refits from ``spike_times``/``spike_clusters``/``full_st``/
    ``kept_spikes``; the inventory hashed exactly those files. The check and
    the load are one operation (:func:`read_attested_curated_arrays`), so the
    arrays handed to the replay are provably the attested bytes -- which is
    what makes a reproduction mismatch reported below a mismatch of *runtime*
    rather than of silently substituted input.
    """
    sort_ids = sorted({str(case["sort_id"]) for case in payload.get("cases", [])})
    curated_by_sort: dict[str, CuratedArrays] = {}
    hashes: dict[str, dict[str, str]] = {}
    for sort_id in sort_ids:
        recorded = (manifest.get("sorts") or {}).get(sort_id)
        if not isinstance(recorded, dict) or not recorded.get("curated_file_hashes"):
            raise RuntimeError(
                f"the inventory manifest records no curated_file_hashes for sort {sort_id!r}; "
                "refusing to replay arrays whose provenance it never attested"
            )
        curated_by_sort[sort_id], hashes[sort_id] = read_attested_curated_arrays(
            sort_id, Path(cfg.by_id(sort_id).curated), recorded["curated_file_hashes"],
        )
    return curated_by_sort, hashes


def cluster_amplitude_sequence(
    curated: CuratedArrays, cluster_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """One cluster's time-ordered sample/amplitude sequence, as production QC built it.

    ``pipeline.qc.truncation_qc`` slices ``spike_amplitudes[spike_clusters ==
    cid]`` out of the already-time-ordered curated table, so cached
    ``window_blocks`` index into exactly this sequence -- not global rows,
    samples or seconds. A source that was not already time ordered cannot be
    replayed at all: re-sorting it would silently reinterpret those indices,
    so this refuses instead (same rule ``build_windows_table`` enforces).
    """
    if not curated.was_time_ordered:
        raise ValueError(
            f"{curated.sort_id}: spike_times.npy was not already time-ordered; cached "
            "window_blocks indices cannot be safely replayed against a re-sorted array"
        )
    positions = np.flatnonzero(curated.clusters == int(cluster_id))
    if positions.size == 0:
        raise ValueError(
            f"{curated.sort_id}: cluster {cluster_id} has no spikes in the curated arrays"
        )
    return curated.times[positions], curated.amplitudes[positions]


def exact_indexing_eligibility(
    case: dict[str, Any], replays: list[dict[str, Any]], c: SelectionConstants,
) -> str:
    """Re-test *this frozen case's own* eligibility with the exact-1,000 estimates.

    This is a sensitivity report, not a re-selection: it re-runs the very same
    classifier on the very same four windows with only ``missing_pct`` (and the
    status that follows from the exact fit) replaced. No other run, cluster or
    case is examined, nothing is re-ranked, and the answer never removes a
    case -- the prescription forbids replacing a case after results are seen.
    """
    run = []
    for frozen, replay in zip(case["windows"], replays):
        row = {field: frozen[field] for field in _WINDOW_FIELDS}
        row["missing_pct"] = replay["exact_missing_pct"]
        row["status"] = _classify_status(
            np.asarray(replay["exact_popt"], dtype=float), replay["exact_missing_pct"]
        )
        run.append(row)
    role = case.get("role")
    if role == "failure":
        reason, _ = classify_failure_run(run, c)
    elif role == "control":
        reason, _ = classify_control_run(run, c)
    else:
        raise ValueError(f"{case.get('case_id')!r}: unknown case role {role!r}")
    return reason


def replay_case_windows(
    payload: dict[str, Any], cfg: AuditConfig,
    curated_by_sort: dict[str, CuratedArrays],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Historical and exact replay of every frozen case window, in frozen order.

    ``curated_by_sort`` carries the arrays already attested by
    :func:`read_attested_curated_inputs`; nothing is re-read from disk here.
    Cases are emitted in ``selection.json``'s own order and none is dropped,
    added or re-ranked, whatever the replay finds.
    """
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for case in payload.get("cases", []):
        sort_id = str(case["sort_id"])
        curated = curated_by_sort[sort_id]
        times, amplitudes = cluster_amplitude_sequence(curated, int(case["cluster_id"]))

        replays: list[dict[str, Any]] = []
        case_rows: list[dict[str, Any]] = []
        for frozen in case["windows"]:
            i0, i1 = int(frozen["i0"]), int(frozen["i1"])
            fit = historical_exact_fit(amplitudes, i0, i1)

            # Invariants, not measurements: the window bounds, both counts and
            # the two endpoint samples are fixed by the attested inventory row,
            # so a disagreement here means the replay is not looking at the
            # sequence the inventory described.
            if (fit["historical_count"] != frozen["historical_count"]
                    or fit["exact_count"] != frozen["nominal_count"]):
                raise ValueError(
                    f"{case['case_id']} window {frozen['position']}: replayed counts "
                    f"({fit['historical_count']}, {fit['exact_count']}) != frozen "
                    f"({frozen['historical_count']}, {frozen['nominal_count']})"
                )
            if (int(times[i0]) != int(frozen["first_sample"])
                    or int(times[i1]) != int(frozen["last_sample"])):
                raise ValueError(
                    f"{case['case_id']} window {frozen['position']}: replayed endpoint samples "
                    f"({int(times[i0])}, {int(times[i1])}) != frozen "
                    f"({frozen['first_sample']}, {frozen['last_sample']})"
                )

            cached = float(frozen["missing_pct"])
            replayed = fit["historical_missing_pct"]
            reproduces = bool(np.isclose(
                replayed, cached,
                rtol=REPRODUCTION_RTOL, atol=REPRODUCTION_ATOL_PP, equal_nan=False,
            ))
            diff_pp = float(abs(replayed - cached))
            status_reason = None if reproduces else (
                f"replayed historical missing_pct {replayed!r} does not reproduce the cached "
                f"{cached!r} within rtol={REPRODUCTION_RTOL}, atol={REPRODUCTION_ATOL_PP} "
                f"percentage points (|diff| = {diff_pp!r} pp); input/runtime mismatch, "
                "interpretation of this case stops here"
            )
            replays.append(dict(fit, reproduces=reproduces, diff_pp=diff_pp))
            case_rows.append({
                "case_id": case["case_id"],
                "sort_id": sort_id,
                "cluster_id": int(case["cluster_id"]),
                "case_role": case.get("role"),
                "window_role": frozen.get("window_role"),
                "window_ordinal": int(frozen["position"]),
                "source_row": frozen["source_row"],
                "i0": i0, "i1": i1,
                "first_sample": frozen["first_sample"],
                "last_sample": frozen["last_sample"],
                "start_s": frozen["start_s"], "end_s": frozen["end_s"],
                "historical_count": fit["historical_count"],
                "exact_count": fit["exact_count"],
                "cached_missing_pct": cached,
                "replayed_historical_missing_pct": replayed,
                "exact_missing_pct": fit["exact_missing_pct"],
                "reproduction_abs_diff_pp": diff_pp,
                "exact_minus_historical_pp": fit["exact_missing_pct"] - replayed,
                "historical_saturated": fit["historical_saturated"],
                "exact_saturated": fit["exact_saturated"],
                "historical_fit_x0": fit["historical_popt"][0],
                "historical_fit_k": fit["historical_popt"][1],
                "historical_fit_A": fit["historical_popt"][2],
                "exact_fit_x0": fit["exact_popt"][0],
                "exact_fit_k": fit["exact_popt"][1],
                "exact_fit_A": fit["exact_popt"][2],
                "status": (
                    CASE_WINDOW_REPRODUCED if reproduces
                    else CASE_WINDOW_REPRODUCTION_MISMATCH
                ),
                "status_reason": status_reason,
            })

        unstable_repro = any(not r["reproduces"] for r in replays)
        if unstable_repro:
            # A detected input/runtime mismatch stops interpretation of this
            # case. The exact-1,000 fits come from the same arrays whose
            # historical fit just failed to reproduce its cache, so a verdict
            # drawn from them could not be told apart from the mismatch itself
            # -- and reporting one would read as an indexing finding. The
            # exact numbers are still emitted as data; only the verdict is
            # withheld, and the reason column says why.
            exact_reason = EXACT_ELIGIBILITY_NOT_INTERPRETED
            unstable_exact = None
        else:
            exact_reason = exact_indexing_eligibility(case, replays, cfg.selection)
            unstable_exact = exact_reason != "qualified"
        case_status = (
            CASE_STATUS_UNSTABLE if (unstable_repro or unstable_exact) else CASE_STATUS_STABLE
        )
        for row in case_rows:
            row["case_status"] = case_status
            row["unstable_reproduction_mismatch"] = unstable_repro
            row["unstable_under_exact_indexing"] = unstable_exact
            row["exact_eligibility_reason"] = exact_reason
        rows.extend(case_rows)

        summaries.append({
            "case_id": case["case_id"],
            "sort_id": sort_id,
            "cluster_id": int(case["cluster_id"]),
            "role": case.get("role"),
            # the freeze's own ordering, carried so the nomination can prefer
            # it without recomputing anything
            "rank": case.get("rank"),
            "n_windows": len(case_rows),
            "case_status": case_status,
            "unstable_reproduction_mismatch": unstable_repro,
            "unstable_under_exact_indexing": unstable_exact,
            "exact_eligibility_reason": exact_reason,
            "max_reproduction_abs_diff_pp": (
                max(r["diff_pp"] for r in replays) if replays else None
            ),
            "mismatched_window_ordinals": [
                row["window_ordinal"] for row in case_rows
                if row["status"] == CASE_WINDOW_REPRODUCTION_MISMATCH
            ],
        })

    return pd.DataFrame(rows, columns=list(CASE_WINDOWS_COLUMNS)), summaries


# --------------------------------------------------------------------------- #
# layer 4b -- evidence classification, panels and the decision
#
# The prescription's five-row evidence table, verbatim in intent: each case is
# classified into every category, several categories may be supported at once,
# and disagreement resolves to `ambiguous` -- never by a majority vote across
# metrics. A category whose prerequisite is missing is `unavailable` or
# `not_attempted` with the reason recorded; neither is evidence of absence.
# --------------------------------------------------------------------------- #
#: Category -> (permitted conclusion, next action). These are the prescription's
#: own words: the audit may not upgrade a conclusion beyond the one its evidence
#: row licenses.
EVIDENCE_TABLE = {
    "curation_exclusion": (
        "Curation contributes to exclusion",
        "Replay one specific curation rule on retained arrays",
    ),
    "identity_redistribution": (
        "Identity redistribution is supported",
        "Test one motion-aware identity rule",
    ),
    "motion_amplitude_change": (
        "Motion/amplitude change is a candidate explanation",
        "Choose a bounded existing registration or identity experiment; "
        "do not claim causality yet",
    ),
    "voltage_integrity": (
        "Local voltage integrity is suspect",
        "Replay one implicated processing operation on the same voltage",
    ),
    "unresolved": (
        "Stage is unresolved",
        "Stop with the missing observation identified",
    ),
}
EVIDENCE_CATEGORIES = tuple(EVIDENCE_TABLE)

#: Per-category verdicts. `unavailable` (a prerequisite observation does not
#: exist) and `not_attempted` (a protocol this checkpoint does not have) are
#: distinct from `unsupported` (looked, and the observation did not support it).
VERDICT_SUPPORTED = "supported"
VERDICT_UNSUPPORTED = "unsupported"
VERDICT_UNAVAILABLE = "unavailable"
VERDICT_NOT_ATTEMPTED = "not_attempted"
VERDICT_UNRESOLVED = "unresolved"

#: Case-level readings. `ambiguous` is what disagreement resolves to.
CASE_EVIDENCE_AMBIGUOUS = "ambiguous"
CASE_EVIDENCE_UNRESOLVED = "unresolved"
CASE_EVIDENCE_STOPPED = "interpretation_stopped"

#: decision.md's two permitted endings.
DECISION_INSUFFICIENT = "insufficient_evidence"

CASE_EVIDENCE_COLUMNS = (
    "case_id", "sort_id", "cluster_id", "case_role", "category", "verdict",
    "permitted_conclusion", "next_action", "observation", "limitations",
    "case_status", "case_evidence_reading", "figure_path", "case_windows_rows",
)


@dataclass(frozen=True)
class EvidenceConstants:
    """Frozen thresholds for the one comparison each evidence category makes.

    Same discipline as :class:`SelectionConstants`: supplied only by CONFIG,
    with units, no default in code and no CLI override, so they cannot move
    after an observation has been seen.
    """

    depth_shift_um_material: float
    amplitude_drop_frac_material: float
    min_events_per_window_for_shift: int
    voltage_saturation_uv: float
    voltage_saturated_frac_material: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.voltage_saturation_uv) or self.voltage_saturation_uv <= 0:
            raise ValueError("evidence.voltage_saturation_uv must be finite and > 0")
        if not (0.0 <= self.voltage_saturated_frac_material <= 1.0):
            raise ValueError("evidence.voltage_saturated_frac_material must lie in [0, 1]")
        if not math.isfinite(self.depth_shift_um_material) or self.depth_shift_um_material < 0:
            raise ValueError("evidence.depth_shift_um_material must be finite and >= 0")
        if not (0.0 <= self.amplitude_drop_frac_material <= 1.0):
            raise ValueError("evidence.amplitude_drop_frac_material must lie in [0, 1]")
        if self.min_events_per_window_for_shift < 1:
            raise ValueError("evidence.min_events_per_window_for_shift must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "depth_shift_um_material": self.depth_shift_um_material,
            "amplitude_drop_frac_material": self.amplitude_drop_frac_material,
            "min_events_per_window_for_shift": self.min_events_per_window_for_shift,
            "voltage_saturation_uv": self.voltage_saturation_uv,
            "voltage_saturated_frac_material": self.voltage_saturated_frac_material,
        }


_EVIDENCE_KEYS = ("depth_shift_um_material", "amplitude_drop_frac_material",
                  "min_events_per_window_for_shift", "voltage_saturation_uv",
                  "voltage_saturated_frac_material")


def parse_evidence_constants(payload: Any) -> EvidenceConstants:
    """Parse CONFIG's ``evidence`` block, requiring every constant and its unit."""
    if not isinstance(payload, dict):
        raise ValueError("config: 'evidence' must be an object of frozen evidence constants")
    missing = sorted(set(_EVIDENCE_KEYS) - set(payload))
    if missing:
        raise ValueError(f"config: 'evidence' block is missing required key(s) {missing}")
    unknown = sorted(set(payload) - set(_EVIDENCE_KEYS) - {"units", "note"})
    if unknown:
        raise ValueError(f"config: 'evidence' block has unknown key(s) {unknown}")
    units = payload.get("units")
    if not isinstance(units, dict) or sorted(units) != sorted(_EVIDENCE_KEYS):
        raise ValueError(
            "config: 'evidence.units' must name a unit for exactly the evidence constants"
        )
    for key in _EVIDENCE_KEYS:
        value = payload[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"evidence.{key} must be a number, got {value!r}")
        if not math.isfinite(float(value)):
            raise ValueError(f"evidence.{key} must be finite, got {value!r}")
    return EvidenceConstants(
        depth_shift_um_material=float(payload["depth_shift_um_material"]),
        amplitude_drop_frac_material=float(payload["amplitude_drop_frac_material"]),
        min_events_per_window_for_shift=_exact_int_scalar(
            payload["min_events_per_window_for_shift"],
            "evidence.min_events_per_window_for_shift",
        ),
        voltage_saturation_uv=float(payload["voltage_saturation_uv"]),
        voltage_saturated_frac_material=float(payload["voltage_saturated_frac_material"]),
    )


def read_attested_spike_positions(
    sort_id: str, curated: Path,
) -> tuple[np.ndarray | None, str | None, str | None]:
    """Read ``spike_positions.npy`` once, hashing the bytes that were parsed.

    Returns ``(positions, sha256, reason_unavailable)``. A missing file is a
    permitted outcome -- depth evidence becomes ``unavailable`` with the reason
    recorded -- not an exception and never a substitution from another sort.

    This is deliberately NOT folded into the inventory's attested set: that set
    is the replay's input contract, and widening it would silently invalidate
    every manifest written against the old one.
    """
    path = Path(curated) / "spike_positions.npy"
    if not path.exists():
        return None, None, f"{path} does not exist; per-spike depth was never exported"
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    positions = np.load(io.BytesIO(data), allow_pickle=False)
    del data
    positions = np.asarray(positions)
    if positions.ndim != 2 or positions.shape[1] < 2:
        return None, digest, (
            f"{path} has shape {positions.shape}; expected (n_spikes, >=2) with depth in "
            "column 1"
        )
    return positions, digest, None


def read_attested_full_labels(
    sort_id: str, curated: Path,
) -> tuple[np.ndarray | None, str | None, str | None]:
    """Read ``full_clu.npy`` once, hashing the bytes that were parsed.

    Returns ``(labels, sha256, reason_unavailable)``. Like
    :func:`read_attested_spike_positions` this is kept out of the inventory's
    attested set -- that set is the replay's input contract -- and a missing
    file degrades the lineage observation to ``unavailable`` rather than raising.
    """
    path = Path(curated) / "full_clu.npy"
    if not path.exists():
        return None, None, f"{path} does not exist; full-table labels were never exported"
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    labels = np.load(io.BytesIO(data), allow_pickle=False)
    del data
    return np.asarray(labels).reshape(-1), digest, None


def read_attested_geometry(
    sort_cfg: SortConfig,
) -> tuple[np.ndarray | None, str | None, str | None]:
    """Read the probe geometry once, hashing the bytes that were parsed."""
    if sort_cfg.channel_geometry is None:
        return None, None, "config declares no channel_geometry for this sort"
    path = Path(sort_cfg.channel_geometry)
    if not path.exists():
        return None, None, f"{path} does not exist"
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    geometry = np.load(io.BytesIO(data), allow_pickle=False)
    del data
    geometry = np.asarray(geometry)
    if geometry.ndim != 2 or geometry.shape[1] < 2:
        return None, digest, f"{path} has shape {geometry.shape}; expected (n_channels, >=2)"
    return geometry, digest, None


def _window_event_slices(
    case: dict[str, Any], times: np.ndarray,
) -> dict[str, list[tuple[int, int]]]:
    """Per-window ``[i0, i1]`` index ranges of the case, split by window role."""
    by_role: dict[str, list[tuple[int, int]]] = {}
    for window in case.get("windows", []):
        role = str(window.get("window_role"))
        by_role.setdefault(role, []).append((int(window["i0"]), int(window["i1"])))
    return by_role


def _role_values(values: np.ndarray, spans: list[tuple[int, int]]) -> np.ndarray:
    """Concatenate one role's per-window values, inclusive of each endpoint."""
    if not spans:
        return np.empty(0, dtype=float)
    return np.concatenate([np.asarray(values[i0:i1 + 1], dtype=float) for i0, i1 in spans])


def measure_case_shift(
    case: dict[str, Any], curated: CuratedArrays,
    positions: np.ndarray | None,
) -> dict[str, Any]:
    """Median amplitude and depth of the reference vs the failing windows.

    Amplitude is the sorter-native QC amplitude (``full_st[kept][:, 2]``); depth
    is column 1 of ``spike_positions.npy``. Both are read over the SAME frozen
    event spans the case was selected from, so nothing here re-selects anything.

    The amplitude comparison is a median of the raw values, not a re-reading of
    the truncation fit: the fitted missingness is what defined the case, so
    scoring the case on it again would be circular.
    """
    cluster_positions = np.flatnonzero(curated.clusters == int(case["cluster_id"]))
    amplitudes = curated.amplitudes[cluster_positions]
    by_role = _window_event_slices(case, curated.times[cluster_positions])
    reference_spans = by_role.get("reference", [])
    failing_spans = by_role.get("failing", [])

    out: dict[str, Any] = {
        "n_reference_events": int(sum(i1 - i0 + 1 for i0, i1 in reference_spans)),
        "n_failing_events": int(sum(i1 - i0 + 1 for i0, i1 in failing_spans)),
        "reference_median_amplitude": None, "failing_median_amplitude": None,
        "amplitude_drop_frac": None,
        "reference_median_depth_um": None, "failing_median_depth_um": None,
        "depth_shift_um": None, "depth_reason_unavailable": None,
    }
    if not reference_spans or not failing_spans:
        out["depth_reason_unavailable"] = (
            "this case has no reference/failing window split (control cases are "
            "single-role by construction)"
        )
        return out

    ref_amp = _role_values(amplitudes, reference_spans)
    fail_amp = _role_values(amplitudes, failing_spans)
    ref_median = float(np.median(ref_amp))
    fail_median = float(np.median(fail_amp))
    out["reference_median_amplitude"] = ref_median
    out["failing_median_amplitude"] = fail_median
    out["amplitude_drop_frac"] = (
        float((ref_median - fail_median) / ref_median) if ref_median > 0 else None
    )

    if positions is None:
        out["depth_reason_unavailable"] = "spike_positions.npy was not available"
        return out
    if positions.shape[0] != curated.times.size:
        out["depth_reason_unavailable"] = (
            f"spike_positions.npy has {positions.shape[0]} rows but the curated table has "
            f"{curated.times.size}; the arrays are not aligned, so depth is not read"
        )
        return out
    depths = np.asarray(positions[cluster_positions, 1], dtype=float)
    ref_depth = _role_values(depths, reference_spans)
    fail_depth = _role_values(depths, failing_spans)
    if not (np.isfinite(ref_depth).all() and np.isfinite(fail_depth).all()):
        out["depth_reason_unavailable"] = "spike_positions.npy carries non-finite depths here"
        return out
    out["reference_median_depth_um"] = float(np.median(ref_depth))
    out["failing_median_depth_um"] = float(np.median(fail_depth))
    out["depth_shift_um"] = float(
        out["failing_median_depth_um"] - out["reference_median_depth_um"]
    )
    return out


def measure_retained_row_lineage(
    case: dict[str, Any], curated: CuratedArrays, raw: dict[str, np.ndarray],
    full_labels: np.ndarray | None = None,
    labels_unavailable: str | None = None,
) -> dict[str, Any]:
    """Rows of the full table inside the case's failing span that were not kept.

    This is retained-row LINEAGE, not timestamp matching: the case's kept rows
    are located in the full table through ``kept_spikes`` itself, their
    ``full_clu`` label is read off there, and only rows carrying that same label
    are counted. Full-sort and curated cluster IDs need not be equal, so the
    label is never assumed.

    It reports an observation and nothing more. Calling a dropped row "removed
    by curation" -- or an absent one "never detected" -- would be a claim about
    retained-array semantics, which the prescription requires establishing
    against the installed KS4 source first. Until that exists, the category
    stays unresolved no matter what this returns.
    """
    out: dict[str, Any] = {
        "n_full_rows_in_span": None, "n_dropped_rows_in_span": None,
        "full_label": None, "reason_unavailable": None,
    }
    full_st = raw.get("full_st.npy")
    kept = raw.get("kept_spikes.npy")
    if full_st is None or kept is None:
        out["reason_unavailable"] = "the attested full-table arrays were not provided"
        return out
    if full_labels is None:
        out["reason_unavailable"] = labels_unavailable or "full_clu.npy was not available"
        return out

    by_role = _window_event_slices(case, curated.times)
    failing_spans = by_role.get("failing", [])
    if not failing_spans:
        out["reason_unavailable"] = "this case has no failing windows"
        return out

    cluster_positions = np.flatnonzero(curated.clusters == int(case["cluster_id"]))
    times = curated.times[cluster_positions]
    span_start = int(times[failing_spans[0][0]])
    span_stop = int(times[failing_spans[-1][1]])

    kept_arr = np.asarray(kept)
    if kept_arr.dtype == np.bool_:
        kept_rows = np.flatnonzero(kept_arr)
    else:
        kept_rows = np.asarray(kept_arr, dtype=np.int64)
    if kept_rows.size != curated.times.size:
        out["reason_unavailable"] = (
            "kept_spikes does not resolve to one full-table row per curated spike"
        )
        return out

    # the curated table was stable-sorted by time; row_id maps back to the
    # pre-sort order, which is the order kept_spikes indexes
    case_full_rows = kept_rows[curated.row_id[cluster_positions]]
    full_clu = np.asarray(full_labels)
    if full_clu.size != np.asarray(full_st).shape[0]:
        out["reason_unavailable"] = (
            f"full_clu.npy has {full_clu.size} rows but full_st.npy has "
            f"{np.asarray(full_st).shape[0]}; the full-table arrays are not aligned"
        )
        return out
    labels, counts = np.unique(full_clu[case_full_rows], return_counts=True)
    if labels.size == 0:
        out["reason_unavailable"] = "no full-table label could be read for this cluster"
        return out
    label = labels[int(np.argmax(counts))]
    out["full_label"] = int(label)

    full_times = np.asarray(full_st)[:, 0].astype(np.int64)
    in_span = (full_times >= span_start) & (full_times <= span_stop)
    same_label = full_clu == label
    kept_mask = np.zeros(full_times.size, dtype=bool)
    kept_mask[kept_rows] = True
    out["n_full_rows_in_span"] = int(np.count_nonzero(in_span & same_label))
    out["n_dropped_rows_in_span"] = int(np.count_nonzero(in_span & same_label & ~kept_mask))
    return out


def classify_case_evidence(
    case: dict[str, Any], summary: dict[str, Any], shift: dict[str, Any],
    lineage: dict[str, Any], c: EvidenceConstants,
    voltage: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """One row per evidence category for one case, plus the case's reading.

    Multiple supported categories are allowed; when more than one mechanism is
    supported the reading is ``ambiguous`` -- the disagreement is reported, not
    voted on. A case whose historical replay failed to reproduce its cache is
    ``interpretation_stopped`` before any category is judged: the prescription
    stops interpretation there, and every observation below is drawn from the
    same arrays whose fit just failed to reproduce.
    """
    stopped = bool(summary.get("unstable_reproduction_mismatch"))
    rows: list[dict[str, Any]] = []

    def emit(category: str, verdict: str, observation: str, limitations: str) -> None:
        conclusion, next_action = EVIDENCE_TABLE[category]
        rows.append({
            "category": category,
            "verdict": verdict,
            "permitted_conclusion": conclusion if verdict == VERDICT_SUPPORTED else "",
            "next_action": next_action if verdict == VERDICT_SUPPORTED else "",
            "observation": observation,
            "limitations": limitations,
        })

    if stopped:
        halted = (
            "the historical replay of this case did not reproduce its cached missing_pct, "
            "so interpretation stopped before any category was judged"
        )
        for category in EVIDENCE_CATEGORIES:
            emit(category, VERDICT_UNRESOLVED, halted,
                 "no observation from these arrays is interpretable while the mismatch stands")
        return rows, CASE_EVIDENCE_STOPPED

    # 1. curation exclusion -- observation only; the stage claim needs KS4 semantics
    if lineage.get("reason_unavailable"):
        emit("curation_exclusion", VERDICT_UNAVAILABLE,
             f"retained-row lineage not read: {lineage['reason_unavailable']}",
             "no lineage observation exists for this case")
    else:
        emit(
            "curation_exclusion", VERDICT_UNRESOLVED,
            f"{lineage['n_dropped_rows_in_span']} of {lineage['n_full_rows_in_span']} full-table "
            f"rows carrying full-sort label {lineage['full_label']} inside the failing span are "
            "absent from kept_spikes",
            "a row absent from kept_spikes is NOT thereby 'removed by curation' or 'never "
            "detected': the semantics of the retained arrays have not been established against "
            "the installed KS4 source, so this observation licenses no stage claim",
        )

    # 2. identity redistribution -- needs a frozen shift-null protocol
    emit("identity_redistribution", VERDICT_NOT_ATTEMPTED,
         "no spatially restricted exclusive event matching was run",
         "this checkpoint has no frozen shift-null protocol, and the prescription forbids an "
         "identity claim from time coincidence alone; not attempted is not evidence of absence")

    # 3. motion / amplitude change
    depth_shift = shift.get("depth_shift_um")
    drop = shift.get("amplitude_drop_frac")
    n_min = min(shift.get("n_reference_events") or 0, shift.get("n_failing_events") or 0)
    underpowered = n_min < c.min_events_per_window_for_shift
    observations = []
    if depth_shift is not None:
        observations.append(f"median depth shift {depth_shift:+.2f} um")
    if drop is not None:
        observations.append(f"median amplitude drop {100 * drop:+.1f}% of the reference median")
    observed = "; ".join(observations) if observations else "no shift could be measured"
    depth_material = depth_shift is not None and abs(depth_shift) >= c.depth_shift_um_material
    amp_material = drop is not None and drop >= c.amplitude_drop_frac_material
    if depth_shift is None and drop is None:
        emit("motion_amplitude_change", VERDICT_UNAVAILABLE, observed,
             shift.get("depth_reason_unavailable") or "no amplitude or depth observation")
    elif underpowered:
        emit("motion_amplitude_change", VERDICT_UNRESOLVED, observed,
             f"only {n_min} events in the smaller role (< "
             f"{c.min_events_per_window_for_shift}); the shift is reported but its verdict is "
             "withheld as underpowered")
    elif depth_material or amp_material:
        limits = [
            "no assignment explanation was established for this case, which is what this row "
            "requires; identity redistribution was not attempted",
            "a shift accompanying missingness is not a cause of it -- this is a candidate "
            "explanation, not a causal claim",
        ]
        if depth_shift is None:
            limits.append("depth was unavailable; this rests on amplitude alone")
        emit("motion_amplitude_change", VERDICT_SUPPORTED, observed, " | ".join(limits))
    else:
        emit("motion_amplitude_change", VERDICT_UNSUPPORTED, observed,
             f"below the frozen thresholds (depth >= {c.depth_shift_um_material} um or "
             f"amplitude drop >= {c.amplitude_drop_frac_material:.2f} of the reference median)")

    # 4. voltage integrity -- filled by the bounded voltage review, when it ran
    if voltage is None:
        emit("voltage_integrity", VERDICT_UNAVAILABLE,
             "no voltage review was performed for this case",
             "voltage evidence is unavailable, which lowers this case's conclusion strength; "
             "it is never reconstructed or substituted from another sort")
    else:
        emit("voltage_integrity", voltage["verdict"], voltage["observation"],
             voltage["limitations"])

    supported = [r["category"] for r in rows if r["verdict"] == VERDICT_SUPPORTED]

    # 5. unresolved -- the prescription's own fallback row
    if supported:
        emit("unresolved", VERDICT_UNSUPPORTED,
             f"supported categories: {', '.join(supported)}",
             "the supported rows carry their own limitations")
        reading = supported[0] if len(supported) == 1 else CASE_EVIDENCE_AMBIGUOUS
    else:
        missing = [r["category"] for r in rows
                   if r["verdict"] in (VERDICT_UNAVAILABLE, VERDICT_NOT_ATTEMPTED,
                                       VERDICT_UNRESOLVED)]
        emit("unresolved", VERDICT_SUPPORTED,
             "no category is supported: only missingness/amplitude changes were observed, or "
             f"the required intermediates are absent ({', '.join(missing)})",
             "the missing observations are named per category above")
        reading = CASE_EVIDENCE_UNRESOLVED
    return rows, reading


# --------------------------------------------------------------------------- #
# layer 4c -- the bounded voltage review
#
# Hard caps, from the prescription: at most the two highest-ranked failure
# cases and their corresponding controls; at most 100 evenly spaced assigned
# events per window; at most 16 channels nearest the reference peak channel,
# with the channel set FROZEN from the reference windows and reused unchanged
# for the failing windows. Reads are bounded chunks -- a full-session voltage
# array is never materialized.
#
# The legacy sort has no raw voltage on disk. That is a permitted outcome: its
# cases degrade to `unavailable` with the reason recorded. The rescue
# recording is never substituted for a legacy case.
# --------------------------------------------------------------------------- #
VOLTAGE_MAX_REVIEW_FAILURE_CASES = 2
VOLTAGE_MAX_EVENTS_PER_WINDOW = 100
VOLTAGE_MAX_CHANNELS = 16
VOLTAGE_EXCERPT_MS = 50.0
VOLTAGE_WAVEFORM_HALF_MS = 1.3


@dataclass(frozen=True)
class VoltageMeta:
    """What the voltage view is, recorded so a reader knows what was read."""

    path: str
    dtype: str
    n_samples: int
    n_channels: int
    gain_uv_per_count: float
    selected_start_sample: int
    view: str


class BoundedVoltageReader:
    """Bounded reads out of a raw interleaved int16 recording.

    Backed by a memmap and sliced per request, so only the requested frames and
    channels are ever brought into memory. Every request is checked against the
    caps and against the file's own bounds: an out-of-range request is clipped
    and the clipped margin is REPORTED rather than silently satisfied.
    """

    def __init__(self, path: Path, meta: VoltageMeta):
        self.path = Path(path)
        self.meta = meta
        self._map = np.memmap(
            self.path, dtype=np.dtype(meta.dtype), mode="r",
            shape=(int(meta.n_samples), int(meta.n_channels)),
        )

    @property
    def n_samples(self) -> int:
        return int(self.meta.n_samples)

    @property
    def n_channels(self) -> int:
        return int(self.meta.n_channels)

    def read(self, start_frame: int, end_frame: int, channels) -> tuple[np.ndarray, dict]:
        channels = np.asarray(list(channels), dtype=int)
        if channels.size > VOLTAGE_MAX_CHANNELS:
            raise ValueError(
                f"voltage read requested {channels.size} channels, over the cap of "
                f"{VOLTAGE_MAX_CHANNELS}"
            )
        if channels.size and (channels.min() < 0 or channels.max() >= self.n_channels):
            raise ValueError("voltage read requested a channel outside the recording")
        clipped_start = max(0, int(start_frame))
        clipped_end = min(self.n_samples, int(end_frame))
        clip = {
            "clipped_start_frames": int(clipped_start - int(start_frame)),
            "clipped_end_frames": int(int(end_frame) - clipped_end),
        }
        if clipped_end <= clipped_start:
            return np.empty((0, channels.size), dtype=np.float32), clip
        block = np.asarray(self._map[clipped_start:clipped_end, :][:, channels], dtype=np.float32)
        return block * float(self.meta.gain_uv_per_count), clip


def open_voltage_source(
    sort_cfg: SortConfig,
) -> tuple[BoundedVoltageReader | None, VoltageMeta | None, str | None]:
    """Resolve one sort's OWN raw voltage, or say why there is none.

    A sort's voltage comes from its own recording folder or not at all: the
    prescription's `unavailable` is a permitted, conclusion-weakening outcome,
    and substituting another sort's recording would fabricate the evidence.
    The source offset comes from the verified config field, never from a
    filename.
    """
    recording = Path(sort_cfg.source_recording)
    manifest = recording / "rescue_recording_manifest.json"
    binary = recording / "binary.json"
    if manifest.exists():
        payload = json.loads(manifest.read_text())
        dtype = str(payload.get("dtype", "int16"))
        n_samples = _exact_int_scalar(payload["num_samples"], "recording num_samples")
        n_channels = _exact_int_scalar(payload["num_channels"], "recording num_channels")
        gain = float(payload.get("gain_uv_per_count", 1.0))
        view = "rescue recording (conditioned voltage the sorter consumed)"
    elif binary.exists():
        payload = json.loads(binary.read_text())["kwargs"]
        dtype = str(payload.get("dtype", "int16")).lstrip("<>|")
        n_channels = _exact_int_scalar(payload["num_channels"], "recording num_channels")
        gain_values = payload.get("gain_to_uV")
        gain = float(np.asarray(gain_values).reshape(-1)[0]) if gain_values is not None else 1.0
        n_samples = 0
        view = "spikeinterface binary recording folder"
    else:
        return None, None, (
            f"{recording} carries no recognisable recording manifest, so this sort has no "
            "voltage view; it is never substituted from another sort"
        )

    candidates = sorted(recording.glob("*.raw")) + sorted(recording.glob("*.dat"))
    if not candidates:
        return None, None, (
            f"no raw voltage file remains in {recording} (the sorter input was deleted after "
            "sorting); voltage evidence is unavailable for this sort and is never reconstructed"
        )
    path = candidates[0]
    itemsize = np.dtype(dtype).itemsize
    if not n_samples:
        n_samples = int(path.stat().st_size // (itemsize * n_channels))
    meta = VoltageMeta(
        path=str(path), dtype=dtype, n_samples=int(n_samples), n_channels=int(n_channels),
        gain_uv_per_count=gain, selected_start_sample=int(sort_cfg.selected_start_sample),
        view=view,
    )
    return BoundedVoltageReader(path, meta), meta, None


def voltage_available_by_sort(cfg: AuditConfig, sort_ids) -> dict[str, bool]:
    """Whether each sort's OWN raw voltage still exists on disk.

    A filesystem question, asked without opening a reader: the answer is a
    property of the inputs, not of any ranking. It is what lets a nomination
    prefer a case whose evidence can actually be completed -- the legacy sort's
    traces file was deleted after sorting, so its voltage limb is uncollectable
    in principle rather than merely uncollected.
    """
    available: dict[str, bool] = {}
    for sort_id in sort_ids:
        recording = Path(cfg.by_id(str(sort_id)).source_recording)
        available[str(sort_id)] = bool(
            sorted(recording.glob("*.raw")) or sorted(recording.glob("*.dat"))
        )
    return available


def select_voltage_review_cases(payload: dict[str, Any]) -> list[str]:
    """The at-most-two highest-ranked failure cases and their controls.

    Rank is the freeze's own rank; nothing is re-ranked here. A control
    "corresponds" to a failure case by being its sort's control, which is how
    selection chose it (its span is matched to that sort's first failure span).
    """
    failures = [c for c in payload.get("cases", []) if c.get("role") == "failure"]
    failures = sorted(failures, key=lambda c: (int(c.get("rank", 0)), str(c["case_id"])))
    chosen = failures[:VOLTAGE_MAX_REVIEW_FAILURE_CASES]
    sorts = {str(c["sort_id"]) for c in chosen}
    controls = [
        c for c in payload.get("cases", [])
        if c.get("role") == "control" and str(c["sort_id"]) in sorts
    ]
    return [str(c["case_id"]) for c in chosen + controls]


def _evenly_spaced(indices: np.ndarray, cap: int) -> np.ndarray:
    """At most ``cap`` evenly spaced picks, deterministically."""
    indices = np.asarray(indices)
    if indices.size <= cap:
        return indices
    picks = np.linspace(0, indices.size - 1, cap).round().astype(int)
    return indices[np.unique(picks)]


def freeze_review_channels(
    case: dict[str, Any], curated: CuratedArrays, positions: np.ndarray | None,
    geometry: np.ndarray | None, n_channels: int,
) -> tuple[np.ndarray | None, int | None, str | None]:
    """The frozen channel set: the peak channel of the REFERENCE windows, plus
    its nearest neighbours by probe geometry, capped at 16.

    Frozen from the reference windows and reused unchanged for the failing
    windows -- recomputing it there would follow the unit and hide exactly the
    displacement this review is looking for.
    """
    if positions is None or geometry is None:
        return None, None, "per-spike depth or probe geometry was unavailable"
    if positions.shape[0] != curated.times.size:
        return None, None, "spike_positions.npy is not aligned with the curated table"
    cluster_positions = np.flatnonzero(curated.clusters == int(case["cluster_id"]))
    by_role = _window_event_slices(case, curated.times[cluster_positions])
    reference_spans = by_role.get("reference") or by_role.get("control") or []
    if not reference_spans:
        return None, None, "this case has no reference windows to freeze a channel set from"
    depths = np.asarray(positions[cluster_positions, 1], dtype=float)
    reference_depth = float(np.median(_role_values(depths, reference_spans)))
    if not math.isfinite(reference_depth):
        return None, None, "the reference windows carry no finite depth"
    peak = int(np.argmin(np.abs(np.asarray(geometry[:, 1], dtype=float) - reference_depth)))
    order = np.argsort(np.abs(np.asarray(geometry[:, 1], dtype=float) - geometry[peak, 1]),
                       kind="stable")
    keep = np.sort(order[:min(VOLTAGE_MAX_CHANNELS, n_channels)])
    return keep, peak, None


def review_case_voltage(
    case: dict[str, Any], curated: CuratedArrays, reader, meta: VoltageMeta,
    positions: np.ndarray | None, geometry: np.ndarray | None, fs: float,
    c: EvidenceConstants,
) -> dict[str, Any]:
    """Bounded voltage review for one case.

    Per window: at most 100 evenly spaced ASSIGNED events, read on the frozen
    channel set only, plus one fixed 50 ms continuous excerpt centred in the
    window. Waveforms are NOT recentred -- recentring would conceal the
    displacement this review exists to show. Every read is a bounded chunk and
    every clipped margin is reported.

    The verdict rests on one declared, frozen indicator (the fraction of read
    samples at or beyond the saturation threshold). "Visibly alters the
    waveform" is a human judgement this audit does not make; the panel is
    rendered so a person can make it.
    """
    channels, peak, reason = freeze_review_channels(
        case, curated, positions, geometry, reader.n_channels,
    )
    if channels is None:
        return {
            "verdict": VERDICT_UNAVAILABLE,
            "observation": "no voltage was read",
            "limitations": f"channel set could not be frozen: {reason}",
            "extraction": {"reason_unavailable": reason},
        }

    cluster_positions = np.flatnonzero(curated.clusters == int(case["cluster_id"]))
    times = curated.times[cluster_positions]
    offset = int(meta.selected_start_sample)
    half = max(1, int(round(VOLTAGE_WAVEFORM_HALF_MS * 1e-3 * fs)))
    excerpt_half = max(1, int(round(VOLTAGE_EXCERPT_MS * 1e-3 * fs / 2)))

    per_window: list[dict[str, Any]] = []
    clipped_total = {"clipped_start_frames": 0, "clipped_end_frames": 0}
    saturated_samples = 0
    total_samples = 0

    for window in case.get("windows", []):
        i0, i1 = int(window["i0"]), int(window["i1"])
        event_indices = _evenly_spaced(np.arange(i0, i1 + 1), VOLTAGE_MAX_EVENTS_PER_WINDOW)
        waveforms = []
        for index in event_indices:
            centre = int(times[index]) + offset
            block, clip = reader.read(centre - half, centre + half, channels)
            for key in clipped_total:
                clipped_total[key] += clip[key]
            if block.shape[0] == 2 * half:
                waveforms.append(block)
            total_samples += int(block.size)
            saturated_samples += int(np.count_nonzero(
                np.abs(block) >= c.voltage_saturation_uv
            ))
        stack = np.stack(waveforms) if waveforms else np.empty((0, 2 * half, channels.size))

        centre_frame = int(times[(i0 + i1) // 2]) + offset
        excerpt_start = centre_frame - excerpt_half
        excerpt_stop = centre_frame + excerpt_half
        excerpt, excerpt_clip = reader.read(excerpt_start, excerpt_stop, channels)
        for key in clipped_total:
            clipped_total[key] += excerpt_clip[key]
        # assigned events inside the excerpt, as offsets from its first frame,
        # so the excerpt can be marked rather than merely shown
        window_samples = np.asarray(times[i0:i1 + 1], dtype=np.int64) + offset
        inside = window_samples[
            (window_samples >= excerpt_start) & (window_samples < excerpt_stop)
        ]
        excerpt_event_offsets = (inside - excerpt_start).astype(int)

        per_window.append({
            "window_role": window.get("window_role"),
            "window_ordinal": int(window.get("position", len(per_window))),
            "n_events_read": int(len(waveforms)),
            "n_events_requested": int(event_indices.size),
            # peak-to-peak per event on the frozen channels, NOT recentred
            "median_peak_to_peak_uv": (
                float(np.median(stack.max(axis=1) - stack.min(axis=1))) if stack.size else None
            ),
            "median_waveform": stack.mean(axis=0) if stack.size else None,
            "excerpt": excerpt,
            "excerpt_centre_frame": centre_frame,
            "excerpt_event_offsets": excerpt_event_offsets,
            "n_excerpt_events": int(excerpt_event_offsets.size),
        })

    saturated_fraction = (saturated_samples / total_samples) if total_samples else 0.0
    reference = [w for w in per_window if w["window_role"] in ("reference", "control")]
    failing = [w for w in per_window if w["window_role"] == "failing"]

    def _median_pp(rows):
        values = [r["median_peak_to_peak_uv"] for r in rows if r["median_peak_to_peak_uv"]]
        return float(np.median(values)) if values else None

    ref_pp, fail_pp = _median_pp(reference), _median_pp(failing)
    observation = (
        f"{sum(w['n_events_read'] for w in per_window)} events read on {channels.size} frozen "
        f"channels around peak channel {peak}; "
        f"{100 * saturated_fraction:.3f}% of read samples at or beyond "
        f"{c.voltage_saturation_uv:.0f} uV"
    )
    if ref_pp is not None and fail_pp is not None:
        observation += (
            f"; median peak-to-peak {ref_pp:.1f} uV reference vs {fail_pp:.1f} uV failing"
        )

    supported = saturated_fraction >= c.voltage_saturated_frac_material
    limitations = (
        "excerpts and waveform distributions illustrate evidence; they do not estimate recall "
        "or a false-positive rate. Waveforms are not recentred. 'Artifacts or voltage "
        "processing visibly alter the waveform' is a human judgement this audit does not "
        "automate: only the frozen saturation indicator is scored"
    )
    return {
        "verdict": VERDICT_SUPPORTED if supported else VERDICT_UNSUPPORTED,
        "observation": observation,
        "limitations": limitations,
        "per_window": per_window,
        "channels": channels,
        "extraction": {
            "voltage_view": meta.view,
            "voltage_path": meta.path,
            "dtype": meta.dtype,
            "gain_uv_per_count": meta.gain_uv_per_count,
            "filter_margins": "none applied; the stored view is read as the sorter consumed it",
            "selected_start_sample_offset": offset,
            "frozen_channels": [int(x) for x in channels],
            "peak_channel": int(peak),
            "channel_freeze_rule": (
                "peak channel of the REFERENCE windows by median depth, plus nearest "
                "neighbours by probe geometry, reused unchanged for the failing windows"
            ),
            "max_events_per_window": VOLTAGE_MAX_EVENTS_PER_WINDOW,
            "max_channels": VOLTAGE_MAX_CHANNELS,
            "waveform_half_ms": VOLTAGE_WAVEFORM_HALF_MS,
            "excerpt_ms": VOLTAGE_EXCERPT_MS,
            "events_recentred": False,
            "clipped_margins_frames": clipped_total,
            "saturation_threshold_uv": c.voltage_saturation_uv,
            "saturated_sample_fraction": saturated_fraction,
            "excerpt_centre_frames": [w["excerpt_centre_frame"] for w in per_window],
            "excerpt_marked_events": [w["n_excerpt_events"] for w in per_window],
        },
    }


def render_voltage_panel(case_id: str, review: dict[str, Any], fs: float, out_path: Path) -> Path:
    """Before/during waveforms on the frozen channels, plus one 50 ms excerpt."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    windows = review.get("per_window") or []
    fig, axes = plt.subplots(2, max(len(windows), 1), figsize=(3.0 * max(len(windows), 1), 6.4),
                             squeeze=False)
    for column, window in enumerate(windows):
        ax_wave = axes[0][column]
        waveform = window.get("median_waveform")
        if waveform is not None:
            time_ms = (np.arange(waveform.shape[0]) - waveform.shape[0] / 2) / fs * 1e3
            ax_wave.plot(time_ms, waveform, lw=0.7)
        ax_wave.set_title(
            f"w{window['window_ordinal']} {window['window_role']}\n"
            f"{window['n_events_read']} events, not recentred", fontsize=7)
        ax_wave.set_xlabel("ms", fontsize=7)
        if column == 0:
            ax_wave.set_ylabel("uV (frozen channels)", fontsize=7)
        ax_wave.tick_params(labelsize=6)

        ax_excerpt = axes[1][column]
        excerpt = window.get("excerpt")
        if excerpt is not None and getattr(excerpt, "size", 0):
            span_ms = np.arange(excerpt.shape[0]) / fs * 1e3
            offsets = np.arange(excerpt.shape[1]) * (np.ptp(excerpt) or 1.0)
            ax_excerpt.plot(span_ms, excerpt + offsets, lw=0.4, color="0.25")
            for sample in np.asarray(window.get("excerpt_event_offsets", []), dtype=int):
                if 0 <= sample < excerpt.shape[0]:
                    ax_excerpt.axvline(sample / fs * 1e3, color="#c0392b", lw=0.6,
                                       alpha=0.75, zorder=3)
        ax_excerpt.set_title(
            f"{VOLTAGE_EXCERPT_MS:.0f} ms excerpt\n"
            f"{window.get('n_excerpt_events', 0)} assigned events marked", fontsize=7)
        ax_excerpt.set_xlabel("ms", fontsize=7)
        ax_excerpt.tick_params(labelsize=6)
        ax_excerpt.set_yticks([])

    fig.suptitle(
        f"{case_id} -- bounded voltage review\n"
        "excerpts illustrate evidence; they do not estimate recall or a false-positive rate",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# layer 4b -- the static evidence panel
# --------------------------------------------------------------------------- #
#: Context shown either side of the frozen interval, as a fraction of its span.
#: The frozen four-window interval itself is never cropped.
PANEL_CONTEXT_FRACTION = 0.5


def _fitted_cdf(popt: tuple[float, float, float], x: np.ndarray) -> np.ndarray:
    x0, k, amplitude = (float(v) for v in popt)
    return truncated_sigmoid(x, x0, k, amplitude, float(np.min(x)) if x.size else 0.0)


def render_case_panel(
    case: dict[str, Any], rows: list[dict[str, Any]], curated: CuratedArrays,
    positions: np.ndarray | None, out_path: Path, *,
    shift: dict[str, Any] | None = None,
) -> Path:
    """One aligned static panel for one case.

    Time-domain rows share one x axis and the whole frozen interval stays
    visible. Missingness is drawn at each window's ACTUAL width, in percent;
    boundary-pinned fits are drawn distinctly and excluded from the drawn
    change score; no-fit time is shaded separately from boundary-pinned fits,
    and both are in the legend. The CDF row is per window, in amplitude, which
    is not a time axis and is labelled as such.
    """
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    cluster_positions = np.flatnonzero(curated.clusters == int(case["cluster_id"]))
    times = curated.times[cluster_positions]
    amps = curated.amplitudes[cluster_positions]
    fs = float(case.get("_fs") or 1.0)
    seconds = times / fs

    ordered = sorted(rows, key=lambda r: int(r["window_ordinal"]))
    starts = [float(r["start_s"]) for r in ordered]
    ends = [float(r["end_s"]) for r in ordered]
    frozen_lo, frozen_hi = min(starts), max(ends)
    pad = max((frozen_hi - frozen_lo) * PANEL_CONTEXT_FRACTION, 1e-6)
    lo, hi = frozen_lo - pad, frozen_hi + pad
    keep = (seconds >= lo) & (seconds <= hi)

    fig = plt.figure(figsize=(11.0, 12.0))
    grid = fig.add_gridspec(5, len(ordered), height_ratios=[3, 2, 2, 2, 3], hspace=0.55)
    ax_amp = fig.add_subplot(grid[0, :])
    ax_missing = fig.add_subplot(grid[1, :], sharex=ax_amp)
    ax_depth = fig.add_subplot(grid[2, :], sharex=ax_amp)
    ax_rate = fig.add_subplot(grid[3, :], sharex=ax_amp)

    # --- amplitude vs time -------------------------------------------------- #
    if np.count_nonzero(keep) > 1:
        ax_amp.hist2d(seconds[keep], amps[keep], bins=(80, 40), cmap="Blues")
    ax_amp.set_ylabel("amplitude\n(sorter-native)")
    ax_amp.set_title(
        f"{case['case_id']}  --  sort {case['sort_id']}, cluster {case['cluster_id']}, "
        f"role {case.get('role')}\n"
        "amplitudes are sorter-native QC units (full_st[kept][:, 2]), never microvolts",
        fontsize=10,
    )

    # --- missingness at actual window widths -------------------------------- #
    pinned_any = False
    for row in ordered:
        start, end = float(row["start_s"]), float(row["end_s"])
        width = max(end - start, 1e-9)
        for key, colour, offset, label in (
            ("replayed_historical_missing_pct", "#3b6ea5", 0.0, "historical (999)"),
            ("exact_missing_pct", "#c46a1f", 0.5, "exact (1,000)"),
        ):
            value = row.get(key)
            if value is None or not np.isfinite(float(value)):
                continue
            saturated = bool(row.get(
                "historical_saturated" if "historical" in key else "exact_saturated"
            ))
            pinned_any = pinned_any or saturated
            ax_missing.bar(
                start + offset * width / 2.0, float(value), width=width / 2.0, align="edge",
                color="none" if saturated else colour,
                edgecolor=colour, hatch="///" if saturated else None,
                linewidth=1.2, label=label,
            )
    ax_missing.set_ylabel("estimated\nmissing (%)")
    ax_missing.set_ylim(0, 55)
    ax_missing.axhline(SATURATION_PCT, color="0.4", lw=0.8, ls=":")
    ax_missing.text(
        lo, SATURATION_PCT + 0.6,
        "50% = censoring bound: a hatched bar is boundary-pinned, not a measurement",
        fontsize=7, color="0.3", va="bottom",
    )

    # --- depth vs time (waveform/depth summary) ----------------------------- #
    if positions is not None and positions.shape[0] == curated.times.size:
        depths = np.asarray(positions[cluster_positions, 1], dtype=float)
        ax_depth.plot(seconds[keep], depths[keep], ".", ms=2, color="#4b7f52", alpha=0.5)
        ax_depth.set_ylabel("depth (um)")
        if shift and shift.get("depth_shift_um") is not None:
            ax_depth.set_title(
                f"median depth shift, failing minus reference: "
                f"{shift['depth_shift_um']:+.2f} um", fontsize=8, loc="left",
            )
    else:
        ax_depth.set_ylabel("depth (um)")
        ax_depth.text(0.5, 0.5, "per-spike depth unavailable", ha="center", va="center",
                      transform=ax_depth.transAxes, fontsize=9, color="0.35")
        ax_depth.set_yticks([])

    # --- rate as context ---------------------------------------------------- #
    if np.count_nonzero(keep) > 1:
        edges = np.linspace(lo, hi, 60)
        counts, _ = np.histogram(seconds[keep], bins=edges)
        widths = np.diff(edges)
        ax_rate.step(edges[:-1], counts / widths, where="post", color="0.35", lw=1.0)
    ax_rate.set_ylabel("rate (Hz)\ncontext only")
    ax_rate.set_xlabel("time (s, selected-recording relative)")
    ax_rate.set_xlim(lo, hi)

    # --- frozen interval, no-fit shading ------------------------------------ #
    covered = [(float(r["start_s"]), float(r["end_s"])) for r in ordered]
    for axis in (ax_amp, ax_missing, ax_depth, ax_rate):
        axis.axvspan(frozen_lo, frozen_hi, facecolor="#f2c14e", alpha=0.10, zorder=0)
        cursor = lo
        for start, end in sorted(covered):
            if start > cursor:
                axis.axvspan(cursor, start, facecolor="0.75", alpha=0.30, hatch="xx",
                             edgecolor="0.5", lw=0.0, zorder=0)
            cursor = max(cursor, end)
        if cursor < hi:
            axis.axvspan(cursor, hi, facecolor="0.75", alpha=0.30, hatch="xx",
                         edgecolor="0.5", lw=0.0, zorder=0)
        axis.grid(alpha=0.15)

    handles = [
        Patch(facecolor="#3b6ea5", edgecolor="#3b6ea5", label="historical fit (999 samples)"),
        Patch(facecolor="#c46a1f", edgecolor="#c46a1f", label="exact fit (1,000 samples)"),
        Patch(facecolor="none", edgecolor="0.2", hatch="///",
              label="boundary-pinned (censored at 50%, excluded from change scores)"),
        Patch(facecolor="0.75", edgecolor="0.5", hatch="xx",
              label="no fit in this time (outside the frozen windows)"),
        Patch(facecolor="#f2c14e", alpha=0.3, label="frozen four-window interval"),
    ]
    ax_missing.legend(handles=handles, fontsize=7, loc="upper left", framealpha=0.9)

    # --- per-window CDFs ---------------------------------------------------- #
    for column, row in enumerate(ordered):
        axis = fig.add_subplot(grid[4, column])
        i0, i1 = int(row["i0"]), int(row["i1"])
        window_amps = np.sort(np.asarray(amps[i0:i1 + 1], dtype=float))
        if window_amps.size:
            empirical = np.arange(window_amps.size) / window_amps.size
            axis.step(window_amps, empirical, where="post", color="0.25", lw=1.0,
                      label="empirical")
            popt = (row.get("exact_fit_x0"), row.get("exact_fit_k"), row.get("exact_fit_A"))
            if all(v is not None and np.isfinite(float(v)) for v in popt):
                axis.plot(window_amps, _fitted_cdf(popt, window_amps), color="#c46a1f",
                          lw=1.0, label="fitted")
        saturated = bool(row.get("exact_saturated"))
        axis.set_title(
            f"w{row['window_ordinal']} {row.get('window_role')}\n"
            + ("boundary-pinned" if saturated
               else f"{float(row['exact_missing_pct']):.2f}% missing"),
            fontsize=7,
        )
        axis.set_xlabel("amplitude", fontsize=7)
        if column == 0:
            axis.set_ylabel("CDF", fontsize=7)
            axis.legend(fontsize=6)
        axis.tick_params(labelsize=6)

    fig.suptitle(
        "CDF row is amplitude-domain, not time. Missingness in %, changes in percentage points.",
        y=0.055, fontsize=7, color="0.35",
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# layer 4b -- decision.md
# --------------------------------------------------------------------------- #
def _nomination_effect(row: dict[str, Any]) -> float:
    """Magnitude on whichever limb carries the supported verdict.

    Parsed from the observation the category itself wrote, so this cannot
    reach for a quantity the verdict did not rest on. Unreadable means zero:
    it is only a tie-break, and it never promotes a case past criteria 1-2.
    """
    text = str(row.get("observation") or "")
    best = 0.0
    for match in re.finditer(r"([+-]?\d+(?:\.\d+)?)\s*(um|%)", text):
        best = max(best, abs(float(match.group(1))))
    return best


def _nominate(
    evidence_rows: list[dict[str, Any]], summaries: list[dict[str, Any]],
    *, voltage_available: dict[str, bool] | None = None,
) -> dict[str, Any] | None:
    """The single nominated case, or ``None`` for insufficient_evidence.

    Eligibility is unchanged and conservative: a failure case whose replay
    reproduced, with exactly one supported mechanism. An ``ambiguous`` reading
    is never a nomination -- it is a disagreement, and the prescription forbids
    resolving it by majority.

    ORDER among eligible cases (prespec:
    docs/luke_amplitude_dropout_audit_nomination_rule_prespec.md):

    1. executable evidence -- a case whose sort has raw voltage available,
       because a case whose voltage limb is uncollectable IN PRINCIPLE cannot
       have its nominated experiment's evidence completed at any cost;
    2. the frozen rank, computed before any evidence was read;
    3. the larger effect on the supported limb;
    4. sort_id then numeric cluster_id, for determinism.

    Criterion 1 rests on a constraint that predates every ranking (the legacy
    sort's raw voltage was deleted, recorded 2026-09-05 14:14 and 14:21, while
    the first ranking was written at 19:35). The previous rule -- first
    eligible in frozen order, i.e. alphabetical by sort ID -- was arbitrary
    with respect to evidence; see the prespec for why this is a correction and
    not a retrofit.
    """
    voltage_available = voltage_available or {}
    by_case = {s["case_id"]: s for s in summaries}
    eligible: list[dict[str, Any]] = []
    for row in evidence_rows:
        summary = by_case.get(row["case_id"])
        if summary is None or row["case_role"] != "failure":
            continue
        if summary.get("case_status") != CASE_STATUS_STABLE:
            continue
        if row["case_evidence_reading"] in (
            CASE_EVIDENCE_AMBIGUOUS, CASE_EVIDENCE_UNRESOLVED, CASE_EVIDENCE_STOPPED,
        ):
            continue
        if row["verdict"] != VERDICT_SUPPORTED or row["category"] == "unresolved":
            continue
        eligible.append(row)
    if not eligible:
        return None

    def order(row: dict[str, Any]):
        summary = by_case[row["case_id"]]
        return (
            0 if voltage_available.get(str(row["sort_id"]), False) else 1,
            int(summary.get("rank") or 0) or 10**6,
            -_nomination_effect(row),
            str(row["sort_id"]),
            int(row["cluster_id"]),
        )

    return sorted(eligible, key=order)[0]


def write_decision_md(
    path: Path, *, evidence_rows: list[dict[str, Any]], summaries: list[dict[str, Any]],
    selection: dict[str, Any], evidence_constants: EvidenceConstants,
    voltage_available: dict[str, bool] | None = None,
) -> str:
    """Write ``decision.md``; return the decision (a case_id or the sentinel).

    Ends with exactly one of: a nominated case + intervention + expected
    observable + why it is preferred, or ``insufficient_evidence`` naming which
    observation was unavailable. A collection of figures without one of those
    is not a completed deliverable, so this refuses to write anything else.
    """
    nomination = _nominate(evidence_rows, summaries, voltage_available=voltage_available)
    lines = [
        "# Amplitude-completeness dropout audit -- decision",
        "",
        f"Schema: `{SCHEMA}`  ",
        f"Selection: `{selection.get('selection_sha256', '')}`  ",
        f"Cases examined: {len(summaries)}",
        "",
        "## What this audit is",
        "",
        "A local candidate screen on cached amplitude-truncation QC from one recording.",
        "It is not proof of true recall, not a measure of production superiority, and not",
        "a causal claim: a change that accompanies missingness is not thereby its cause.",
        "Selected extremes can regress toward typical behaviour. Any positive result",
        "advances only through the existing independent-window/context, held-out and",
        "session-replication gates.",
        "",
        "## Cases",
        "",
        "| case | sort | cluster | role | status | reading |",
        "|---|---|---|---|---|---|",
    ]
    readings = {r["case_id"]: r["case_evidence_reading"] for r in evidence_rows}
    for summary in summaries:
        lines.append(
            f"| `{summary['case_id']}` | {summary['sort_id']} | {summary['cluster_id']} | "
            f"{summary.get('role')} | {summary.get('case_status')} | "
            f"{readings.get(summary['case_id'], CASE_EVIDENCE_UNRESOLVED)} |"
        )

    lines += ["", "## Evidence that was unavailable or not attempted", ""]
    any_missing = False
    for summary in summaries:
        case_id = summary["case_id"]
        missing_rows = [
            r for r in evidence_rows
            if r["case_id"] == case_id
            and r["verdict"] in (VERDICT_UNAVAILABLE, VERDICT_NOT_ATTEMPTED)
        ]
        if not missing_rows:
            continue
        any_missing = True
        lines.append(f"- `{case_id}`:")
        for row in missing_rows:
            lines.append(
                f"  - **{row['category']}** ({row['verdict']}): {row['limitations']}"
            )
    if not any_missing:
        lines.append("- none")

    lines += ["", "## Decision", ""]
    if nomination is None:
        stopped = [s["case_id"] for s in summaries
                   if s.get("case_status") != CASE_STATUS_STABLE]
        lines += [
            f"**`{DECISION_INSUFFICIENT}`**",
            "",
            "No case reached a single supported mechanism with a reproducing replay, so no",
            "intervention is nominated. Precisely which observation was unavailable, per case",
            "(a category that was measured and came out below its frozen threshold is listed",
            "as `unsupported`, which is a result, not a missing observation):",
            "",
        ]
        for summary in summaries:
            case_id = summary["case_id"]
            case_rows = [r for r in evidence_rows if r["case_id"] == case_id
                         and r["category"] != "unresolved"]
            lines.append(f"- `{case_id}`:")
            for row in case_rows:
                lines.append(
                    f"  - `{row['category']}` -- {row['verdict']}: "
                    + (row["limitations"] if row["verdict"] in (
                        VERDICT_UNAVAILABLE, VERDICT_NOT_ATTEMPTED, VERDICT_UNRESOLVED)
                       else f"measured, {row['observation']}")
                )
        if stopped:
            lines.append(
                f"- replay did not reproduce the cached estimate for: {', '.join(stopped)}; "
                "interpretation of those cases stopped there"
            )
        lines += [
            "",
            "Per the prescription this closes the checkpoint. No threshold is relaxed, no",
            "case is backfilled, and no further variant is launched to reach a result.",
        ]
        decision = DECISION_INSUFFICIENT
    else:
        conclusion, next_action = EVIDENCE_TABLE[nomination["category"]]
        others = sorted({
            f"`{r['category']}` ({r['verdict']})" for r in evidence_rows
            if r["case_id"] == nomination["case_id"] and r["category"] != nomination["category"]
            and r["category"] != "unresolved"
        })
        lines += [
            f"**Nominated case:** `{nomination['case_id']}` "
            f"(sort {nomination['sort_id']}, cluster {nomination['cluster_id']})",
            "",
            f"**Supported category:** `{nomination['category']}` -- {conclusion}.",
            "",
            f"**Observation:** {nomination['observation']}",
            "",
            f"**Nominated intervention:** {next_action}.",
            "",
            "**Expected observable:** lower estimated missingness in this case's failing",
            "windows under the candidate, in the same physical-time interval, with added-event",
            "waveform support and no breached contamination, refractory or healthy-control",
            "margin. The comparison is inconclusive unless enough fits fall inside the frozen",
            "interval under both configurations.",
            "",
            "**Why this rather than the other interventions:** the remaining categories did not",
            f"reach a supported verdict for this case ({', '.join(others) or 'none recorded'}),",
            "so their interventions have nothing to act on here.",
            "",
            f"**Limitations carried into the experiment:** {nomination['limitations']}",
            "",
            "Before execution, an experiment JSON must fix both applied-setting maps, input and",
            "target identities, correspondence criteria, intervals, amplitude semantics,",
            "contamination/refractory endpoints, runtime cap, improvement and regression",
            "margins, and the decision rule. Those margins need this case's baseline evidence",
            "and must not be chosen after candidate results are seen.",
        ]
        decision = nomination["case_id"]

    lines += [
        "",
        "## Frozen constants",
        "",
        f"- evidence: `{json.dumps(evidence_constants.to_dict(), sort_keys=True)}`",
        f"- selection: `{json.dumps(selection.get('selection_constants', {}), sort_keys=True)}`",
        "",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(path).with_suffix(".md.tmp")
    tmp.write_text("\n".join(lines))
    os.replace(tmp, path)
    return decision


def build_case_evidence(
    payload: dict[str, Any], cfg: AuditConfig, curated_by_sort: dict[str, CuratedArrays],
    case_windows: pd.DataFrame, summaries: list[dict[str, Any]], out_root: Path,
    *, evidence_constants: EvidenceConstants, render_figures: bool = True,
    voltage_by_case: dict[str, dict[str, Any]] | None = None,
    review_voltage: bool = True,
    voltage_reader_factory=None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Classify every frozen case and render its panel.

    Reads only the frozen cases -- nothing here re-selects, re-ranks or drops a
    case, and no threshold moves. Extra arrays this layer needs
    (``spike_positions.npy``, ``full_clu.npy``) are attested on read and are
    permitted to be absent: the affected category degrades to ``unavailable``
    with the reason recorded.
    """
    voltage_by_case = dict(voltage_by_case or {})
    by_case_id = {s["case_id"]: s for s in summaries}
    review_ids = set(select_voltage_review_cases(payload)) if review_voltage else set()
    rows: list[dict[str, Any]] = []
    extras: dict[str, Any] = {"spike_positions_sha256": {}, "full_clu_sha256": {},
                              "unavailable": {}}
    figures_dir = Path(out_root) / "figures"

    per_sort_extra: dict[str, tuple] = {}
    per_sort_geometry: dict[str, np.ndarray | None] = {}
    voltage_sources: dict[str, tuple] = {}
    all_sort_ids = sorted({str(c["sort_id"]) for c in payload.get("cases", [])})
    extras["voltage"] = {"reviewed_case_ids": sorted(review_ids), "sources": {},
                         "extraction": {},
                         "raw_voltage_available_by_sort": voltage_available_by_sort(
                             cfg, all_sort_ids)}
    for sort_id in sorted({str(c["sort_id"]) for c in payload.get("cases", [])}):
        curated_dir = Path(cfg.by_id(sort_id).curated)
        positions, pos_sha, pos_reason = read_attested_spike_positions(sort_id, curated_dir)
        labels, lab_sha, lab_reason = read_attested_full_labels(sort_id, curated_dir)
        per_sort_extra[sort_id] = (positions, labels, lab_reason)
        if pos_sha:
            extras["spike_positions_sha256"][sort_id] = pos_sha
        if lab_sha:
            extras["full_clu_sha256"][sort_id] = lab_sha
        geometry, geo_sha, geo_reason = read_attested_geometry(cfg.by_id(sort_id))
        per_sort_geometry[sort_id] = geometry
        if geo_sha:
            extras.setdefault("channel_geometry_sha256", {})[sort_id] = geo_sha
        if pos_reason or lab_reason or geo_reason:
            extras["unavailable"][sort_id] = {
                k: v for k, v in (("spike_positions.npy", pos_reason),
                                  ("full_clu.npy", lab_reason),
                                  ("channel_positions.npy", geo_reason)) if v
            }

    if review_ids:
        for sort_id in sorted({str(c["sort_id"]) for c in payload.get("cases", [])
                               if str(c["case_id"]) in review_ids}):
            factory = voltage_reader_factory or open_voltage_source
            reader, meta, reason = factory(cfg.by_id(sort_id))
            voltage_sources[sort_id] = (reader, meta, reason)
            extras["voltage"]["sources"][sort_id] = (
                {"unavailable": reason} if reader is None
                else {"path": meta.path, "view": meta.view, "dtype": meta.dtype,
                      "gain_uv_per_count": meta.gain_uv_per_count,
                      "selected_start_sample": meta.selected_start_sample}
            )

    for case in payload.get("cases", []):
        case_id = str(case["case_id"])
        sort_id = str(case["sort_id"])
        curated = curated_by_sort[sort_id]
        positions, labels, labels_reason = per_sort_extra[sort_id]
        summary = by_case_id.get(case_id, {})
        window_rows = case_windows[case_windows["case_id"] == case_id].to_dict("records")

        shift = measure_case_shift(case, curated, positions)
        lineage = measure_retained_row_lineage(
            case, curated,
            {"full_st.npy": curated.full_st, "kept_spikes.npy": curated.kept},
            full_labels=labels, labels_unavailable=labels_reason,
        )
        if case_id in review_ids and case_id not in voltage_by_case:
            reader, meta, reason = voltage_sources.get(sort_id, (None, None, None))
            if reader is None:
                voltage_by_case[case_id] = {
                    "verdict": VERDICT_UNAVAILABLE,
                    "observation": "no raw voltage exists for this sort",
                    "limitations": (
                        f"{reason}; this lowers the case's conclusion strength and is a "
                        "permitted outcome -- voltage is never reconstructed, and another "
                        "sort's recording is never substituted"
                    ),
                    "extraction": {"reason_unavailable": reason},
                }
            else:
                review = review_case_voltage(
                    case, curated, reader, meta, positions,
                    per_sort_geometry.get(sort_id),
                    float(cfg.by_id(sort_id).sampling_frequency_hz), evidence_constants,
                )
                voltage_by_case[case_id] = review
                if render_figures and review.get("per_window"):
                    render_voltage_panel(
                        case_id, review, float(cfg.by_id(sort_id).sampling_frequency_hz),
                        figures_dir / f"{case_id}_voltage.png",
                    )
            extras["voltage"]["extraction"][case_id] = voltage_by_case[case_id].get(
                "extraction", {})

        category_rows, reading = classify_case_evidence(
            case, summary, shift, lineage, evidence_constants,
            voltage=voltage_by_case.get(case_id),
        )

        figure_path = ""
        if render_figures and window_rows:
            case_for_panel = dict(case, _fs=cfg.by_id(sort_id).sampling_frequency_hz)
            rendered = render_case_panel(
                case_for_panel, window_rows, curated, positions,
                figures_dir / f"{case_id}.png", shift=shift,
            )
            figure_path = str(Path("figures") / rendered.name)

        source_rows = ",".join(str(r.get("source_row")) for r in window_rows)
        for row in category_rows:
            rows.append({
                "case_id": case_id, "sort_id": sort_id,
                "cluster_id": int(case["cluster_id"]), "case_role": case.get("role"),
                **row,
                "case_status": summary.get("case_status"),
                "case_evidence_reading": reading,
                "figure_path": figure_path,
                "case_windows_rows": source_rows,
            })

    return pd.DataFrame(rows, columns=list(CASE_EVIDENCE_COLUMNS)), extras


def run_inspect(selection_path: Path, out_root: Path) -> dict[str, Any]:
    """Verify the frozen selection, then replay only its cases.

    Every gate runs before the manifest is touched, so a refusal leaves the
    completed `select` stage exactly as it was.
    """
    selection_path = Path(selection_path)
    payload = load_attested_selection(selection_path)
    config_path = Path(payload["config_path"])
    cfg, source_hashes = verify_recorded_inputs(payload, config_path)

    out_root, manifest = _prepare_out_root(
        "inspect", Path(out_root), config_path, _config_input_paths(cfg),
        selection_path=selection_path, config_sha256=source_hashes["config"],
    )
    manifest_path = out_root / "manifest.json"
    case_windows_path = out_root / "case_windows.csv"
    case_evidence_path = out_root / "case_evidence.csv"
    decision_path = out_root / "decision.md"
    evidence_constants = cfg.require_evidence_constants(config_path)

    if manifest["select"]["selection_sha256"] != payload["selection_sha256"]:
        raise RuntimeError(
            f"selection_sha256 mismatch: {manifest_path} attested "
            f"{manifest['select']['selection_sha256']!r} but {selection_path} carries "
            f"{payload['selection_sha256']!r}"
        )
    if manifest["windows_csv_sha256"] != payload.get("windows_csv_sha256"):
        raise RuntimeError(
            f"windows.csv hash moved between the inventory and the freeze: {manifest_path} "
            f"attested {manifest['windows_csv_sha256']!r}, {selection_path} recorded "
            f"{payload.get('windows_csv_sha256')!r}"
        )

    # One read: the bytes hashed are the bytes parsed (see read_attested_windows).
    windows, windows_sha256 = read_attested_windows(
        out_root / "windows.csv", payload["windows_csv_sha256"]
    )
    verify_frozen_windows_against_inventory(payload, windows)
    curated_by_sort, curated_hashes = read_attested_curated_inputs(payload, cfg, manifest)

    manifest["stage"] = "inspect"
    manifest["status"] = "running"
    manifest["inspect_started_at"] = datetime.now(timezone.utc).isoformat()
    manifest.pop("failure_reason", None)
    _atomic_write_json(manifest_path, manifest)

    try:
        case_windows, summaries = replay_case_windows(payload, cfg, curated_by_sort)
        _atomic_write_csv(case_windows_path, case_windows)

        case_evidence, evidence_extras = build_case_evidence(
            payload, cfg, curated_by_sort, case_windows, summaries, out_root,
            evidence_constants=evidence_constants,
        )
        _atomic_write_csv(case_evidence_path, case_evidence)
        decision = write_decision_md(
            decision_path, evidence_rows=case_evidence.to_dict("records"),
            summaries=summaries, selection=payload,
            evidence_constants=evidence_constants,
            voltage_available=evidence_extras.get("voltage", {}).get(
                "raw_voltage_available_by_sort", {}),
        )

        manifest["status"] = "complete"
        manifest["inspect"] = {
            "selection_path": str(selection_path),
            "selection_sha256": payload["selection_sha256"],
            "windows_csv_sha256": windows_sha256,
            "config_sha256": source_hashes["config"],
            "source_sha256": source_hashes["sources"],
            "git_commit": git_commit(),
            "curated_file_hashes": curated_hashes,
            "case_windows_path": str(case_windows_path),
            "case_windows_sha256": sha256_file(case_windows_path),
            "n_cases": len(summaries),
            "n_case_windows": int(len(case_windows)),
            "reproduction_tolerance": {
                "rtol": REPRODUCTION_RTOL,
                "atol_pp": REPRODUCTION_ATOL_PP,
            },
            "cases": summaries,
            "case_evidence_path": str(case_evidence_path),
            "case_evidence_sha256": sha256_file(case_evidence_path),
            "decision_path": str(decision_path),
            "decision_sha256": sha256_file(decision_path),
            "decision": decision,
            "figures_dir": str(out_root / "figures"),
            "evidence_constants": evidence_constants.to_dict(),
            "evidence_extra_inputs": evidence_extras,
            "voltage_review": {
                "caps": {
                    "max_failure_cases": VOLTAGE_MAX_REVIEW_FAILURE_CASES,
                    "max_events_per_window": VOLTAGE_MAX_EVENTS_PER_WINDOW,
                    "max_channels": VOLTAGE_MAX_CHANNELS,
                    "excerpt_ms": VOLTAGE_EXCERPT_MS,
                    "waveform_half_ms": VOLTAGE_WAVEFORM_HALF_MS,
                },
                **evidence_extras.get("voltage", {}),
            },
            "case_evidence_readings": {
                row["case_id"]: row["case_evidence_reading"]
                for row in case_evidence.to_dict("records")
            },
            "unstable_under_exact_indexing": [
                s["case_id"] for s in summaries if s["unstable_under_exact_indexing"]
            ],
            "exact_sensitivity_not_interpreted": [
                s["case_id"] for s in summaries
                if s["exact_eligibility_reason"] == EXACT_ELIGIBILITY_NOT_INTERPRETED
            ],
            "reproduction_mismatch_cases": [
                s["case_id"] for s in summaries if s["unstable_reproduction_mismatch"]
            ],
            "notes": [
                "Historical replay fits amps[i0:i1] (999 values for a nominal 1,000-spike "
                "window); exact replay fits amps[i0:i1+1] (1,000). Both counts are kept.",
                "A case flagged unstable_under_exact_indexing is reported, never re-selected, "
                "dropped or re-ranked; selection stays frozen.",
                "A reproduction mismatch is an input/runtime mismatch: it is recorded and "
                "interpretation of that case stops, and the tolerance is never widened. No "
                "exact-indexing eligibility verdict is produced for such a case: its "
                f"exact_eligibility_reason is {EXACT_ELIGIBILITY_NOT_INTERPRETED!r} and its "
                "unstable_under_exact_indexing is left undetermined.",
                "Evidence categories are classified per case; several may be supported at "
                "once, and disagreement is reported as `ambiguous`, never resolved by a "
                "majority across metrics. `unavailable` and `not_attempted` record a missing "
                "prerequisite and are not evidence of absence.",
                "decision.md ends with exactly one nomination or `insufficient_evidence`.",
                "The bounded voltage review covers at most the two highest-ranked failure "
                "cases and their corresponding controls, reads at most 100 evenly spaced "
                "assigned events per window on at most 16 channels frozen from the reference "
                "windows, and never materializes a full-session voltage array.",
                "A sort with no raw voltage on disk yields `unavailable` with the reason "
                "recorded; voltage is never reconstructed and another sort's recording is "
                "never substituted.",
            ],
        }
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["failure_reason"] = f"{type(exc).__name__}: {exc}"
        _atomic_write_json(manifest_path, manifest)
        raise

    _atomic_write_json(manifest_path, manifest)
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    sub = ap.add_subparsers(dest="command", required=True)

    inv = sub.add_parser(
        "inventory",
        help="read recording metadata and cached QC arrays only; no fitting, no voltage",
    )
    inv.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    inv.add_argument("--out-root", type=Path, required=True)

    sel = sub.add_parser(
        "select",
        help=(
            "freeze case IDs from the completed inventory in --out-root; every selection "
            "constant comes from CONFIG and is deliberately NOT overridable here"
        ),
    )
    sel.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sel.add_argument("--out-root", type=Path, required=True)

    insp = sub.add_parser(
        "inspect",
        help=(
            "verify the frozen selection and replay only its cases: historical vs exact-1,000 "
            "fits into case_windows.csv. No figures, voltage or classification at this layer"
        ),
    )
    insp.add_argument("--selection", type=Path, required=True)
    insp.add_argument("--out-root", type=Path, required=True)

    args = ap.parse_args(argv)
    if args.command == "inventory":
        manifest = run_inventory(args.config, args.out_root)
        print(json.dumps({k: v for k, v in manifest.items() if k != "sorts"}, indent=2, default=str))
        return 0 if manifest["status"] == "complete" else 1
    if args.command == "select":
        payload = run_select(args.config, args.out_root)
        summary = {
            "selection_sha256": payload["selection_sha256"],
            "case_ids": [c["case_id"] for c in payload["cases"]],
            "per_sort": payload["per_sort"],
            "exclusion_counts": payload["exclusion_counts"],
        }
        print(json.dumps(summary, indent=2, default=str))
        return 0
    if args.command == "inspect":
        manifest = run_inspect(args.selection, args.out_root)
        print(json.dumps(manifest["inspect"], indent=2, default=str))
        return 0 if manifest["status"] == "complete" else 1

    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
