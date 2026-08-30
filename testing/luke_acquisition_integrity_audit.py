"""Metadata-first SpikeGLX acquisition integrity audit.

The default path reads only ``.meta`` files, binary file statistics, and optional
small CSV summaries.  Full binary hashing is deliberately isolated behind the
explicit ``--verify-full-bin-sha1`` flag.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


CONSISTENCY_FIELDS = (
    "imAiRangeMin",
    "imAiRangeMax",
    "imMaxInt",
    "imDatPrb_type",
    "imChan0apGain",
    "imChan0lfGain",
    "imroTbl",
    "snsGeomMap",
)


def parse_meta(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw_line in path.read_text(errors="strict").splitlines():
        if "=" not in raw_line:
            continue
        key, value = raw_line.split("=", 1)
        result[key.lstrip("~")] = value.strip()
    return result


def stream_kind(path: Path) -> str:
    name = path.name.lower()
    for kind in ("ap", "lf", "nidq"):
        if f".{kind}.meta" in name or name.endswith(f"{kind}.meta"):
            return kind
    return "unknown"


def probe_id(meta: dict[str, str], path: Path) -> str:
    if "imDatPrb_sn" in meta:
        return meta["imDatPrb_sn"]
    match = re.search(r"imec\d+", path.name)
    return match.group(0) if match else path.stem


def binary_path(meta_path: Path) -> Path:
    return meta_path.with_suffix(".bin")


def disconnected_sites(geom_map: str | None) -> list[dict[str, int]]:
    """Return zero-enabled geometry entries, indexed in acquisition order."""
    if not geom_map:
        return []
    entries = re.findall(r"\((\d+):(\d+):(\d+):([01])\)", geom_map)
    return [
        {"channel": index, "shank": int(shank), "x_um": int(x), "y_um": int(y)}
        for index, (shank, x, y, enabled) in enumerate(entries)
        if enabled == "0"
    ]


def imro_reference_ids(imro: str | None) -> list[int]:
    """Extract NP1 reference selections without inferring an ADC-bank mapping."""
    if not imro:
        return []
    rows = re.findall(r"\((\d+)\s+(\d+)\s+(\d+)(?:\s+[^)]*)\)", imro)
    return sorted({int(reference) for _, _, reference in rows})


def _float(meta: dict[str, str], key: str) -> float | None:
    try:
        return float(meta[key])
    except (KeyError, ValueError):
        return None


def _int(meta: dict[str, str], key: str) -> int | None:
    value = _float(meta, key)
    return int(value) if value is not None and value.is_integer() else None


def _sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def audit_acquisition(
    meta_paths: list[Path],
    *,
    alignment_tolerance_s: float = 0.002,
    verify_full_bin_sha1: bool = False,
) -> dict:
    records: list[dict] = []
    parsed: list[tuple[Path, dict[str, str]]] = []
    for meta_path in meta_paths:
        meta_path = Path(meta_path)
        meta = parse_meta(meta_path)
        parsed.append((meta_path, meta))
        bin_path = binary_path(meta_path)
        declared_size = _int(meta, "fileSizeBytes")
        channels = _int(meta, "nSavedChans")
        actual_size = bin_path.stat().st_size if bin_path.exists() else None
        frame_bytes = 2 * channels if channels is not None else None
        sample_count = (
            actual_size // frame_bytes
            if actual_size is not None and frame_bytes and actual_size % frame_bytes == 0
            else None
        )
        rate = _float(meta, "imSampRate") or _float(meta, "niSampRate")
        duration = sample_count / rate if sample_count is not None and rate else None
        first_sample = _int(meta, "firstSample")
        start_s = first_sample / rate if first_sample is not None and rate else None
        declared_sha1 = meta.get("fileSHA1")
        sha1 = _sha1(bin_path) if verify_full_bin_sha1 and bin_path.exists() else None
        checks = {
            "binary_exists": bin_path.exists(),
            "size_matches_fileSizeBytes": actual_size == declared_size
            if actual_size is not None and declared_size is not None
            else None,
            "size_divisible_by_int16_frame": actual_size % frame_bytes == 0
            if actual_size is not None and frame_bytes
            else None,
            "duration_matches_fileTimeSecs": abs(duration - float(meta["fileTimeSecs"]))
            <= max(1.0 / rate, 1e-6)
            if duration is not None and rate and "fileTimeSecs" in meta
            else None,
        }
        if verify_full_bin_sha1:
            checks["sha1_matches"] = sha1 == declared_sha1.upper() if sha1 and declared_sha1 else None
        records.append(
            {
                "meta": str(meta_path),
                "binary": str(bin_path),
                "probe": probe_id(meta, meta_path),
                "stream": stream_kind(meta_path),
                "actual_bytes": actual_size,
                "declared_bytes": declared_size,
                "n_saved_channels": channels,
                "sample_count": sample_count,
                "sample_rate_hz": rate,
                "duration_s": duration,
                "start_s_from_first_sample": start_s,
                "disconnected_sites": disconnected_sites(meta.get("snsGeomMap")),
                "imro_reference_ids": imro_reference_ids(meta.get("imroTbl")),
                "full_binary_sha1_read": verify_full_bin_sha1,
                "sha1": sha1,
                "checks": checks,
            }
        )

    by_probe: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_probe[record["probe"]].append(record)
    ap_lf: list[dict] = []
    for probe, probe_records in sorted(by_probe.items()):
        kinds = {record["stream"]: record for record in probe_records}
        if "ap" not in kinds or "lf" not in kinds:
            continue
        ap, lf = kinds["ap"], kinds["lf"]
        duration_delta = (
            abs(ap["duration_s"] - lf["duration_s"])
            if ap["duration_s"] is not None and lf["duration_s"] is not None
            else None
        )
        start_delta = (
            abs(ap["start_s_from_first_sample"] - lf["start_s_from_first_sample"])
            if ap["start_s_from_first_sample"] is not None
            and lf["start_s_from_first_sample"] is not None
            else None
        )
        ap_lf.append(
            {
                "probe": probe,
                "duration_delta_s": duration_delta,
                "start_delta_s": start_delta,
                "duration_aligned": duration_delta <= alignment_tolerance_s
                if duration_delta is not None
                else None,
                "start_aligned": start_delta <= alignment_tolerance_s
                if start_delta is not None
                else None,
            }
        )

    consistency: dict[str, dict] = {}
    for field in CONSISTENCY_FIELDS:
        values = defaultdict(list)
        for path, meta in parsed:
            if field in meta:
                values[meta[field]].append(str(path))
        consistency[field] = {
            "available": bool(values),
            "consistent": len(values) <= 1 if values else None,
            "distinct_value_count": len(values),
        }
    # AP and LF channel maps intentionally have different labels/ranges. Validate
    # exact channel-map agreement only within stream type.
    channel_map: dict[str, dict] = {}
    for kind in ("ap", "lf"):
        values = {meta["snsChanMap"] for path, meta in parsed if stream_kind(path) == kind and "snsChanMap" in meta}
        channel_map[kind] = {"available": bool(values), "consistent": len(values) <= 1 if values else None}
    consistency["snsChanMap_by_stream"] = channel_map
    reference_sets = {
        tuple(imro_reference_ids(meta.get("imroTbl")))
        for _, meta in parsed
        if "imroTbl" in meta
    }
    consistency["imro_reference_selection"] = {
        "available": bool(reference_sets),
        "consistent": len(reference_sets) <= 1 if reference_sets else None,
        "distinct_value_count": len(reference_sets),
    }

    cross_probe: list[dict] = []
    for kind in ("ap", "lf"):
        stream_records = [record for record in records if record["stream"] == kind]
        starts = [record["start_s_from_first_sample"] for record in stream_records]
        durations = [record["duration_s"] for record in stream_records]
        if len(stream_records) < 2 or any(value is None for value in starts + durations):
            continue
        start_span = float(max(starts) - min(starts))
        duration_span = float(max(durations) - min(durations))
        cross_probe.append(
            {
                "stream": kind,
                "probe_count": len(stream_records),
                "start_span_s": start_span,
                "duration_span_s": duration_span,
                "start_aligned": start_span <= alignment_tolerance_s,
                "duration_aligned": duration_span <= alignment_tolerance_s,
            }
        )

    boolean_checks = [
        value
        for record in records
        for value in record["checks"].values()
        if value is not None
    ] + [
        value
        for pair in ap_lf
        for key, value in pair.items()
        if key.endswith("aligned") and value is not None
    ]
    boolean_checks += [
        value
        for group in cross_probe
        for key, value in group.items()
        if key.endswith("aligned") and value is not None
    ]
    boolean_checks += [
        entry["consistent"]
        for entry in consistency.values()
        if isinstance(entry, dict) and isinstance(entry.get("consistent"), bool)
    ]
    return {
        "schema_version": 1,
        "status": "pass" if boolean_checks and all(boolean_checks) else "fail",
        "safety": {"full_binary_sha1_read": verify_full_bin_sha1},
        "files": records,
        "ap_lf_alignment": ap_lf,
        "cross_probe_alignment": cross_probe,
        "metadata_consistency": consistency,
    }


def polarity_bank_analysis(
    event_csv: Path,
    mapping_csv: Path | None,
    *,
    depth_degree: int = 3,
    permutations: int = 999,
    seed: int = 20250804,
) -> dict:
    if mapping_csv is None:
        return {"status": "unavailable", "reason": "explicit_electrical_mapping_csv_required"}
    events = pd.read_csv(event_csv)
    mapping = pd.read_csv(mapping_csv)
    required_events = {"channel", "y_um", "polarity", "event_count"}
    required_mapping = {"channel", "electrical_bank"}
    if not required_events <= set(events):
        raise ValueError(f"Event CSV missing columns: {sorted(required_events - set(events))}")
    if not required_mapping <= set(mapping):
        raise ValueError(f"Mapping CSV missing columns: {sorted(required_mapping - set(mapping))}")
    grouped = events.groupby(["channel", "y_um", "polarity"], as_index=False)["event_count"].sum()
    wide = grouped.pivot_table(index=["channel", "y_um"], columns="polarity", values="event_count", fill_value=0).reset_index()
    if not {"positive", "negative"} <= set(wide):
        raise ValueError("Event CSV must contain positive and negative polarity rows")
    frame = wide.merge(mapping[["channel", "electrical_bank"]], on="channel", how="inner", validate="one_to_one")
    if len(frame) < depth_degree + 3 or frame["electrical_bank"].nunique() < 2:
        return {"status": "unavailable", "reason": "insufficient_mapped_channels_or_banks"}
    response = np.log((frame["positive"].to_numpy() + 0.5) / (frame["negative"].to_numpy() + 0.5))
    depth = frame["y_um"].to_numpy(dtype=float)
    depth = (depth - depth.mean()) / (depth.std() or 1.0)
    base = np.column_stack([np.ones(len(frame))] + [depth**power for power in range(1, depth_degree + 1)])

    def rss(design: np.ndarray) -> float:
        residual = response - design @ np.linalg.lstsq(design, response, rcond=None)[0]
        return float(residual @ residual)

    def bank_design(labels: np.ndarray) -> np.ndarray:
        levels = sorted(pd.unique(labels).tolist(), key=str)
        dummies = np.column_stack([labels == level for level in levels[1:]]).astype(float)
        return np.column_stack([base, dummies])

    labels = frame["electrical_bank"].to_numpy()
    base_rss = rss(base)
    observed_rss = rss(bank_design(labels))
    partial_r2 = (base_rss - observed_rss) / base_rss if base_rss else 0.0
    rng = np.random.default_rng(seed)
    null = np.array([(base_rss - rss(bank_design(rng.permutation(labels)))) / base_rss for _ in range(permutations)]) if permutations else np.array([])
    return {
        "status": "available",
        "mapping_source": str(mapping_csv),
        "n_channels": int(len(frame)),
        "n_banks": int(frame["electrical_bank"].nunique()),
        "response": "log((positive_count+0.5)/(negative_count+0.5))",
        "depth_control_polynomial_degree": depth_degree,
        "bank_partial_r2": float(partial_r2),
        "permutation_count": permutations,
        "permutation_p_value": float((1 + np.sum(null >= partial_r2)) / (1 + permutations)) if permutations else None,
        "interpretation": "association_only_not_an_adc_causal_test",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta", type=Path, nargs="+", required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--alignment-tolerance-s", type=float, default=0.002)
    parser.add_argument("--event-csv", type=Path)
    parser.add_argument("--electrical-mapping-csv", type=Path)
    parser.add_argument("--polarity-permutations", type=int, default=999)
    parser.add_argument(
        "--verify-full-bin-sha1",
        action="store_true",
        help="EXPLICIT SLOW OPT-IN: sequentially read and SHA-1 every full binary",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    receipt = audit_acquisition(
        args.meta,
        alignment_tolerance_s=args.alignment_tolerance_s,
        verify_full_bin_sha1=args.verify_full_bin_sha1,
    )
    receipt["polarity_bank_analysis"] = (
        polarity_bank_analysis(
            args.event_csv,
            args.electrical_mapping_csv,
            permutations=args.polarity_permutations,
        )
        if args.event_csv
        else {"status": "not_requested"}
    )
    text = json.dumps(receipt, indent=2) + "\n"
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
