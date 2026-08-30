"""Plan a sealed, session-wide validation cohort for the Luke recordings.

This module only plans windows and writes/prints a JSON manifest.  It does not
open raw recordings, extract events, run motion estimation, or invoke a sorter.
Motion summaries are supplied as small JSON/CSV tables (or generated explicitly
with ``--mock-motion-summary`` for smoke tests).

The default design selects one 30 s window from every
probe x time-quartile x motion-stratum cell: 2 x 4 x 3 = 24 windows.  Known
discovery regions are hard exclusions.  Event quotas are preregistered in the
manifest but event extraction is deliberately left for a later stage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROBES = ("imec0", "imec1")
MOTION_STRATA = ("quiet", "middle", "high")
DISCOVERY_EXCLUSIONS_S = (
    (3891.0, 4011.0, "3951 s discovery window +/- 60 s"),
    (7095.0, 7335.0, "shared/template discovery interval"),
    (8160.0, 8280.0, "registration-outlier/pathological discovery interval"),
)


@dataclass(frozen=True)
class MotionRow:
    probe: str
    time_s: float
    motion_score: float
    motion_stratum: str | None = None
    source_row: int | None = None


@dataclass(frozen=True)
class Candidate:
    probe: str
    start_s: float
    stop_s: float
    center_s: float
    time_quartile: int
    motion_stratum: str
    motion_score: float
    source_row: int | None


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_motion_rows(rows: Iterable[Mapping[str, Any]]) -> list[MotionRow]:
    """Normalize a small motion-summary table without accessing raw voltage."""
    normalized = []
    for index, row in enumerate(rows):
        probe = str(row["probe"])
        if probe not in PROBES:
            raise ValueError(f"unsupported probe {probe!r}; expected one of {PROBES}")
        if "time_s" in row and row["time_s"] not in (None, ""):
            time_s = float(row["time_s"])
        elif "start_s" in row and "stop_s" in row:
            time_s = (float(row["start_s"]) + float(row["stop_s"])) / 2.0
        else:
            raise ValueError("each motion row needs time_s or start_s and stop_s")
        score = float(row["motion_score"])
        if not math.isfinite(time_s) or not math.isfinite(score):
            raise ValueError("motion times and scores must be finite")
        stratum = row.get("motion_stratum")
        if stratum in (None, ""):
            stratum = None
        else:
            stratum = str(stratum).lower()
            if stratum not in MOTION_STRATA:
                raise ValueError(f"invalid motion_stratum {stratum!r}")
        source_row = row.get("source_row", index)
        normalized.append(MotionRow(probe, time_s, score, stratum, int(source_row)))
    return normalized


def load_motion_summary(path: Path) -> tuple[list[MotionRow], str]:
    """Read a compact JSON/CSV summary and return rows plus its byte hash."""
    payload = path.read_bytes()
    suffix = path.suffix.lower()
    if suffix == ".json":
        value = json.loads(payload)
        rows = value["rows"] if isinstance(value, dict) else value
    elif suffix == ".csv":
        rows = list(csv.DictReader(payload.decode("utf-8").splitlines()))
    else:
        raise ValueError("motion summary must be .json or .csv")
    if not isinstance(rows, list):
        raise ValueError("motion summary JSON must be a list or an object with a rows list")
    return normalize_motion_rows(rows), sha256_bytes(payload)


def make_mock_motion_summary(
    durations_s: Mapping[str, float], points_per_cell: int = 8
) -> list[MotionRow]:
    """Create deterministic planner-only data; never represent it as measured motion."""
    rows = []
    source_row = 0
    for probe_index, probe in enumerate(PROBES):
        duration = float(durations_s[probe])
        for quartile in range(4):
            lo = quartile * duration / 4.0
            hi = (quartile + 1) * duration / 4.0
            for j in range(points_per_cell * 3):
                fraction = (j + 1) / (points_per_cell * 3 + 1)
                rows.append(
                    MotionRow(
                        probe=probe,
                        time_s=lo + fraction * (hi - lo),
                        motion_score=float(j + 0.01 * probe_index),
                        source_row=source_row,
                    )
                )
                source_row += 1
    return rows


def _assign_strata(rows: Sequence[MotionRow], durations_s: Mapping[str, float]) -> list[tuple[MotionRow, int, str]]:
    grouped: dict[tuple[str, int], list[MotionRow]] = {}
    for row in rows:
        duration = float(durations_s[row.probe])
        if not 0 <= row.time_s <= duration:
            continue
        quartile = min(3, int(4 * row.time_s / duration))
        grouped.setdefault((row.probe, quartile), []).append(row)

    result = []
    for key, group in grouped.items():
        explicit = [row.motion_stratum is not None for row in group]
        if any(explicit) and not all(explicit):
            raise ValueError(f"motion strata are only partially supplied in cell {key}")
        if all(explicit):
            for row in group:
                result.append((row, key[1], str(row.motion_stratum)))
            continue
        ordered = sorted(group, key=lambda row: (row.motion_score, row.time_s, row.source_row or -1))
        n_rows = len(ordered)
        for rank, row in enumerate(ordered):
            stratum_index = min(2, (3 * rank) // n_rows)
            result.append((row, key[1], MOTION_STRATA[stratum_index]))
    return result


def _overlaps(start_s: float, stop_s: float, interval: Sequence[Any]) -> bool:
    return start_s < float(interval[1]) and stop_s > float(interval[0])


def build_candidates(
    rows: Sequence[MotionRow],
    durations_s: Mapping[str, float],
    window_s: float = 30.0,
    edge_margin_s: float = 60.0,
    exclusions_s: Sequence[Sequence[Any]] = DISCOVERY_EXCLUSIONS_S,
) -> list[Candidate]:
    candidates = []
    for row, quartile, stratum in _assign_strata(rows, durations_s):
        start = row.time_s - window_s / 2.0
        stop = start + window_s
        duration = float(durations_s[row.probe])
        if start < edge_margin_s or stop > duration - edge_margin_s:
            continue
        if any(_overlaps(start, stop, interval) for interval in exclusions_s):
            continue
        candidates.append(
            Candidate(
                probe=row.probe,
                start_s=round(start, 6),
                stop_s=round(stop, 6),
                center_s=round(row.time_s, 6),
                time_quartile=quartile + 1,
                motion_stratum=stratum,
                motion_score=row.motion_score,
                source_row=row.source_row,
            )
        )
    return candidates


def select_windows(
    candidates: Sequence[Candidate], seed: int, min_spacing_s: float = 60.0
) -> list[Candidate]:
    """Select all 24 cells using deterministic randomized backtracking."""
    cells = [(probe, quartile, stratum) for probe in PROBES for quartile in range(1, 5) for stratum in MOTION_STRATA]
    by_cell: dict[tuple[str, int, str], list[Candidate]] = {cell: [] for cell in cells}
    for candidate in candidates:
        by_cell[(candidate.probe, candidate.time_quartile, candidate.motion_stratum)].append(candidate)
    missing = [cell for cell, values in by_cell.items() if not values]
    if missing:
        raise ValueError(f"no eligible candidates for cells: {missing}")

    rng = random.Random(seed)
    for values in by_cell.values():
        values.sort(key=lambda item: (item.center_s, item.motion_score, item.source_row or -1))
        rng.shuffle(values)
    # Hardest cells first improves the chance of satisfying global spacing.
    search_cells = sorted(cells, key=lambda cell: (len(by_cell[cell]), cells.index(cell)))
    chosen: list[Candidate] = []

    def search(index: int) -> bool:
        if index == len(search_cells):
            return True
        cell = search_cells[index]
        for candidate in by_cell[cell]:
            if all(abs(candidate.center_s - old.center_s) >= min_spacing_s for old in chosen):
                chosen.append(candidate)
                if search(index + 1):
                    return True
                chosen.pop()
        return False

    if not search(0):
        raise ValueError(f"unable to satisfy {min_spacing_s:g} s global spacing across all cells")
    return sorted(chosen, key=lambda item: (item.probe, item.time_quartile, MOTION_STRATA.index(item.motion_stratum)))


def event_quota_scaffold(n_windows: int) -> dict[str, Any]:
    return {
        "status": "preregistered_not_extracted",
        "extraction_performed": False,
        "crossed_selection_dimensions": {
            "polarity": ["negative", "positive"],
            "depth_quartile": [1, 2, 3, 4],
        },
        "target_per_polarity_depth_cell": 2,
        "marginal_balance_dimension": {
            "amplitude_band_sigma": ["4_to_6", "6_to_8", "greater_than_or_equal_to_8"],
            "target_counts_per_window": [5, 5, 6],
            "sparse_cell_policy": "record_as_sparse_do_not_silently_substitute",
        },
        "target_per_window": 16,
        "target_total": 16 * n_windows,
        "note": "Quotas define later event sampling only; this planner reads no voltage.",
    }


def plan_manifest(
    rows: Sequence[MotionRow] | Sequence[Mapping[str, Any]],
    durations_s: Mapping[str, float],
    *,
    seed: int = 20250804,
    window_s: float = 30.0,
    min_spacing_s: float = 60.0,
    edge_margin_s: float = 60.0,
    motion_summary_sha256: str | None = None,
    motion_summary_source: str = "supplied_in_memory",
    input_provenance: Mapping[str, str] | None = None,
    exclusions_s: Sequence[Sequence[Any]] = DISCOVERY_EXCLUSIONS_S,
) -> dict[str, Any]:
    if set(durations_s) != set(PROBES):
        raise ValueError(f"durations_s must contain exactly {PROBES}")
    if window_s <= 0 or min_spacing_s < window_s or edge_margin_s < 0:
        raise ValueError("require window_s > 0, min_spacing_s >= window_s, and edge_margin_s >= 0")
    normalized = list(rows) if all(isinstance(row, MotionRow) for row in rows) else normalize_motion_rows(rows)  # type: ignore[arg-type]
    durations = {probe: float(durations_s[probe]) for probe in PROBES}
    candidates = build_candidates(normalized, durations, window_s, edge_margin_s, exclusions_s)
    selected = select_windows(candidates, seed, min_spacing_s)
    config = {
        "seed": seed,
        "window_s": window_s,
        "min_spacing_s": min_spacing_s,
        "edge_margin_s": edge_margin_s,
        "durations_s": durations,
        "probes": list(PROBES),
        "time_quartiles": 4,
        "motion_strata": list(MOTION_STRATA),
        "discovery_exclusions_s": [list(interval) for interval in exclusions_s],
    }
    if motion_summary_sha256 is None:
        motion_summary_sha256 = sha256_bytes(_canonical_json([asdict(row) for row in normalized]))
    script_path = Path(__file__).resolve()
    manifest = {
        "schema_version": "luke-sealed-holdout-v1",
        "manifest_only": True,
        "sealed": True,
        "design": config,
        "windows": [dict(window_id=f"LH{index:02d}", **asdict(candidate)) for index, candidate in enumerate(selected, 1)],
        "event_quotas": event_quota_scaffold(len(selected)),
        "provenance": {
            "motion_summary_source": motion_summary_source,
            "motion_summary_sha256": motion_summary_sha256,
            "planner_script": str(script_path),
            "planner_script_sha256": sha256_file(script_path),
            "configuration_sha256": sha256_bytes(_canonical_json(config)),
            "inputs": dict(input_provenance or {}),
            "raw_files_opened": False,
        },
    }
    manifest["manifest_content_sha256"] = sha256_bytes(_canonical_json(manifest))
    return manifest


def parse_key_values(values: Sequence[str], cast=str) -> dict[str, Any]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected KEY=VALUE, got {value!r}")
        key, item = value.split("=", 1)
        result[key] = cast(item)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--motion-summary", type=Path, help="Small JSON/CSV motion summary")
    source.add_argument("--mock-motion-summary", action="store_true", help="Use synthetic planner-only motion rows")
    parser.add_argument("--duration", action="append", required=True, metavar="PROBE=SECONDS")
    parser.add_argument("--seed", type=int, default=20250804)
    parser.add_argument("--window-s", type=float, default=30.0)
    parser.add_argument("--min-spacing-s", type=float, default=60.0)
    parser.add_argument("--edge-margin-s", type=float, default=60.0)
    parser.add_argument("--provenance", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--output", type=Path, help="Write manifest here; default is stdout only")
    args = parser.parse_args(argv)

    durations = parse_key_values(args.duration, float)
    if args.mock_motion_summary:
        rows = make_mock_motion_summary(durations)
        source_name = "explicit_mock_motion_summary"
        source_hash = sha256_bytes(_canonical_json([asdict(row) for row in rows]))
    else:
        rows, source_hash = load_motion_summary(args.motion_summary)
        source_name = str(args.motion_summary.resolve())
    manifest = plan_manifest(
        rows,
        durations,
        seed=args.seed,
        window_s=args.window_s,
        min_spacing_s=args.min_spacing_s,
        edge_margin_s=args.edge_margin_s,
        motion_summary_sha256=source_hash,
        motion_summary_source=source_name,
        input_provenance=parse_key_values(args.provenance),
    )
    rendered = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
