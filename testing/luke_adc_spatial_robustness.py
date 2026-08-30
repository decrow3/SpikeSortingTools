"""Spatially constrained robustness audit for Luke imec1 polarity/ADC associations.

This consumes channel-level event counts and small, source-traceable mapping
tables only.  It never opens a recording.  ADC identity and ADC sampling phase
are tested separately because they are different acquisition properties on NP1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


STRATUM = ["dataset", "window_kind", "stage"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rss(response: np.ndarray, design: np.ndarray) -> float:
    residual = response - design @ np.linalg.lstsq(design, response, rcond=None)[0]
    return float(residual @ residual)


def _dummies(values: np.ndarray) -> np.ndarray:
    levels = sorted(pd.unique(values).tolist(), key=str)
    if len(levels) < 2:
        return np.empty((len(values), 0))
    return np.column_stack([values == level for level in levels[1:]]).astype(float)


def _spatial_base(frame: pd.DataFrame) -> np.ndarray:
    """x fixed effects plus a piecewise-linear, smooth depth trend."""
    y = frame["y_um"].to_numpy(float)
    z = (y - y.mean()) / (y.std() or 1.0)
    # Knots are deliberately much broader than the 20/40 um site spacing and
    # do not key off ADC boundaries.
    knots = np.quantile(z, [0.2, 0.4, 0.6, 0.8])
    spline = np.column_stack([z] + [np.maximum(z - knot, 0.0) for knot in knots])
    return np.column_stack([np.ones(len(frame)), _dummies(frame["x_um"].to_numpy()), spline])


def _partial_r2(response: np.ndarray, base: np.ndarray, effect: np.ndarray) -> float:
    base_rss = _rss(response, base)
    if base_rss <= np.finfo(float).eps:
        return 0.0
    return max(0.0, (base_rss - _rss(response, np.column_stack([base, effect]))) / base_rss)


def _cyclic_null(
    response: np.ndarray,
    frame: pd.DataFrame,
    base: np.ndarray,
    effect: np.ndarray,
) -> np.ndarray:
    """Shift responses by full 40-um site rows, preserving x and autocorrelation.

    NP1 has four channels per 40-um repeating lateral geometry.  A common
    circular shift by multiples of four channels preserves the x sequence and
    the complete local response profile, while moving it relative to ADC/phase
    labels.  All non-zero shifts are enumerated, so this is an exact finite null
    rather than an unrestricted random label permutation.
    """
    order = np.argsort(frame["channel"].to_numpy())
    if not np.array_equal(order, np.arange(len(frame))) or len(frame) % 4:
        raise ValueError("Cyclic null requires complete channel-sorted NP1 quartets")
    return np.asarray(
        [_partial_r2(np.roll(response, shift), base, effect) for shift in range(4, len(frame), 4)],
        dtype=float,
    )


def _within_block_cyclic_null(
    response: np.ndarray,
    frame: pd.DataFrame,
    base: np.ndarray,
    effect: np.ndarray,
    *,
    permutations: int = 999,
    seed: int = 20250804,
) -> np.ndarray:
    """Independently rotate complete geometry rows inside each ADC block."""
    if len(frame) % 24:
        raise ValueError("Block cyclic null requires complete 24-channel ADC blocks")
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(permutations):
        shifted = response.copy()
        for start in range(0, len(frame), 24):
            # Multiples of four retain each channel's lateral x position.
            shift = int(rng.integers(0, 6)) * 4
            shifted[start : start + 24] = np.roll(response[start : start + 24], shift)
        null.append(_partial_r2(shifted, base, effect))
    return np.asarray(null)


def _one_analysis(frame: pd.DataFrame) -> dict:
    frame = frame.sort_values("channel").reset_index(drop=True)
    response = np.log((frame["positive"].to_numpy() + 0.5) / (frame["negative"].to_numpy() + 0.5))
    base = _spatial_base(frame)

    adc_effect = _dummies(frame["adc_index"].to_numpy())
    adc_r2 = _partial_r2(response, base, adc_effect)
    adc_null = _cyclic_null(response, frame, base, adc_effect)

    # Sampling phase is periodic within each 24-channel/two-ADC block.  Block
    # fixed effects remove broad local depth offsets before asking whether the
    # recurring 12-step phase pattern explains additional variation.
    phase_base = np.column_stack([base, _dummies(frame["adc_block"].to_numpy())])
    phase_effect = _dummies(frame["sampling_phase_index"].to_numpy())
    phase_r2 = _partial_r2(response, phase_base, phase_effect)
    phase_null = _within_block_cyclic_null(response, frame, phase_base, phase_effect)

    def pvalue(observed: float, null: np.ndarray) -> float:
        return float((1 + np.sum(null >= observed)) / (1 + len(null)))

    return {
        "n_channels": int(len(frame)),
        "adc_identity_partial_r2": adc_r2,
        "adc_identity_cyclic_p": pvalue(adc_r2, adc_null),
        "adc_identity_null_shifts": int(len(adc_null)),
        "sampling_phase_partial_r2": phase_r2,
        "sampling_phase_cyclic_p": pvalue(phase_r2, phase_null),
        "sampling_phase_null_permutations": int(len(phase_null)),
    }


def run_audit(
    event_csv: Path,
    mapping_csv: Path,
    geometry_csv: Path,
    mapping_source_json: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    events = pd.read_csv(event_csv)
    mapping = pd.read_csv(mapping_csv)
    geometry = pd.read_csv(geometry_csv)
    required_map = {"channel", "adc_index", "mapping_kind"}
    if not required_map <= set(mapping):
        raise ValueError(f"Mapping CSV missing {sorted(required_map - set(mapping))}")
    if mapping["mapping_kind"].nunique() != 1 or mapping["mapping_kind"].iloc[0] != "NP1_ADC_identity":
        raise ValueError("Mapping must explicitly identify NP1 ADC identity")
    geom = geometry[["row", "raw_x_um", "raw_y_um"]].rename(
        columns={"row": "channel", "raw_x_um": "x_um", "raw_y_um": "geometry_y_um"}
    )
    channel_map = mapping[["channel", "adc_index"]].merge(geom, on="channel", validate="one_to_one")
    channel_map["adc_block"] = channel_map["channel"] // 24
    channel_map["sampling_phase_index"] = (channel_map["channel"] % 24) // 2

    grouped = events.groupby(STRATUM + ["channel", "y_um", "polarity"], as_index=False)["event_count"].sum()

    def make_frame(part: pd.DataFrame) -> pd.DataFrame:
        wide = part.pivot_table(
            index=["channel", "y_um"], columns="polarity", values="event_count", fill_value=0
        ).reset_index()
        if not {"positive", "negative"} <= set(wide):
            raise ValueError("Both polarity rows are required")
        merged = wide.merge(channel_map, on="channel", validate="one_to_one")
        if not np.allclose(merged["y_um"], merged["geometry_y_um"]):
            raise ValueError("Event and source-traceable geometry depths disagree")
        if len(merged) != 384:
            raise ValueError("Each analysis requires all 384 NP1 neural channels")
        return merged

    rows = [{**{key: "ALL" for key in STRATUM}, "scope": "aggregate_descriptive", **_one_analysis(make_frame(grouped))}]
    for keys, part in grouped.groupby(STRATUM, sort=True):
        rows.append({**dict(zip(STRATUM, keys)), "scope": "stratum", **_one_analysis(make_frame(part))})
    result = pd.DataFrame(rows)
    receipt = {
        "schema_version": 1,
        "status": "pass",
        "safety": {"raw_recording_read": False, "gpu_used": False},
        "inputs": {
            "event_csv": {"path": str(event_csv), "sha256": _sha256(event_csv)},
            "mapping_csv": {"path": str(mapping_csv), "sha256": _sha256(mapping_csv)},
            "geometry_csv": {"path": str(geometry_csv), "sha256": _sha256(geometry_csv)},
            "mapping_source_json": (
                {"path": str(mapping_source_json), "sha256": _sha256(mapping_source_json)}
                if mapping_source_json is not None
                else None
            ),
        },
        "mapping_semantics": {
            "adc_identity": "adc_index from explicit NP1_ADC_identity mapping",
            "sampling_phase": "(channel mod 24)//2; equivalent to the 12 sequential values assigned per ADC by get_neuropixels_sample_shifts(..., 12, 13)",
            "software_source_lines": "spikeinterface.extractors.neuropixels_utils:get_neuropixels_sample_shifts lines 49-56; package/module hash recorded in mapping_source_json",
            "warning": "ADC identity is structurally tied to a 240-um depth block and channel parity; association is not causal identification.",
        },
        "controls": {
            "geometry": "lateral x fixed effects plus piecewise-linear depth trend at depth quintiles",
            "sampling_phase_extra": "240-um ADC-block fixed effects",
            "adc_identity_null": "all 95 nonzero global cyclic shifts by complete four-channel/40-um NP1 geometry rows",
            "sampling_phase_null": "999 independent within-ADC-block cyclic shifts by complete four-channel/40-um geometry rows",
        },
        "aggregate_note": "Aggregate sums reused stage/window counts and is descriptive, not an independent replication.",
        "row_count": int(len(result)),
        "stratum_count": int((result["scope"] == "stratum").sum()),
        "results": result.to_dict(orient="records"),
    }
    return result, receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-csv", type=Path, required=True)
    parser.add_argument("--mapping-csv", type=Path, required=True)
    parser.add_argument("--geometry-csv", type=Path, required=True)
    parser.add_argument("--mapping-source-json", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result, receipt = run_audit(
        args.event_csv, args.mapping_csv, args.geometry_csv, args.mapping_source_json
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_dir / "polarity_adc_spatial_robustness.csv", index=False)
    (args.output_dir / "polarity_adc_spatial_robustness.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": "pass", "rows": len(result), "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
