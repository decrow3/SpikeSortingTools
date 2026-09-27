#!/usr/bin/env python3
"""Freeze BC spatial domains from existing AZ placement outcomes, without voltage."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

try:
    from testing.luke_au_cpu_preparation import exact_same_column_map
except ModuleNotFoundError:
    from luke_au_cpu_preparation import exact_same_column_map


ROOT = Path(__file__).resolve().parents[1]
QUAL = ROOT / "testing/outputs/luke_au_cpu_preparation/donor_qualification_v1"
EXTRACT = ROOT / "testing/outputs/luke_au_cpu_preparation/full_probe_extraction_v1/ba_attempt4"
OUT = ROOT / "testing/outputs/luke_au_cpu_preparation/bc_spatial_reliability_v1"
STATES = np.asarray([0.0, -40.0, -80.0, -120.0, -160.0, -200.0, -240.0])


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def array_digest(values: np.ndarray) -> str:
    values = np.ascontiguousarray(values)
    return hashlib.sha256(f"{values.dtype.str}|{values.shape}".encode() + values.tobytes()).hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    with np.load(EXTRACT / "full_probe_final_label_templates.npz", allow_pickle=False) as z:
        unit_ids = np.asarray(z["unit_ids"], dtype=np.int64)
        geom = np.asarray(z["geometry_um"], dtype=np.float64)
        channel_ids = np.asarray(z["channel_ids"])
    with (QUAL / "placement_attempts.csv").open(newline="") as stream:
        attempts = list(csv.DictReader(stream))
    by_unit: dict[int, list[dict]] = {}
    for row in attempts:
        by_unit.setdefault(int(row["unit_id"]), []).append(row)
    target_mask = np.zeros(geom.shape[0], dtype=bool)
    target_mask[202:] = True
    bases = []
    state_domains = np.zeros((unit_ids.size, STATES.size, geom.shape[0]), dtype=bool)
    rows = []
    for i, unit_id in enumerate(unit_ids):
        choices = by_unit[int(unit_id)]
        best = max(
            choices,
            key=lambda row: (
                float(row["minimum_retained_energy_fraction"]),
                -int(row["base_rank"]),
            ),
        )
        base = float(best["base_shift_um"])
        bases.append(base)
        for j, state in enumerate(STATES):
            mapping = exact_same_column_map(geom, base + state)
            valid = mapping >= 0
            keep = np.zeros(geom.shape[0], dtype=bool)
            keep[valid] = target_mask[mapping[valid]]
            state_domains[i, j] = keep
        common = np.all(state_domains[i], axis=0)
        rows.append(
            {
                "unit_id": int(unit_id),
                "frozen_best_base_shift_um": base,
                "source_channels_in_all_state_domain": int(common.sum()),
                "source_depth_min_um": float(geom[common, 1].min()),
                "source_depth_max_um": float(geom[common, 1].max()),
                "previous_best_minimum_retention": float(best["minimum_retained_energy_fraction"]),
                "previous_base_rank": int(best["base_rank"]),
            }
        )
    bases = np.asarray(bases, dtype=np.float64)
    common_domains = np.all(state_domains, axis=1)
    np.savez_compressed(
        OUT / "frozen_domains.npz",
        unit_ids=unit_ids,
        best_base_shift_um=bases,
        occupied_states_um=STATES,
        state_source_domain=state_domains,
        all_state_source_domain=common_domains,
        channel_ids=channel_ids,
        geometry_um=geom,
        target_channel_indices=np.arange(202, 384, dtype=np.int64),
    )
    with (OUT / "frozen_domains.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    static_td = Path("/mnt/NPX/Luke/DARTsort_motion_experiments/av_aw_handoff_20260926_v1/static_w2/template_data.npz")
    with np.load(static_td, allow_pickle=False) as z:
        components = np.asarray(z["tsvd_components"])
    receipt = {
        "schema": "luke-bc-domain-freeze-v1",
        "status": "frozen_before_repeat_voltage_read",
        "unit_count": int(unit_ids.size),
        "selection": "maximum previously measured minimum all-state retention; ties use lower prior base rank",
        "domain": "source channels whose exact same-column relocation lands on AP202-AP383 in every occupied state at the frozen base",
        "states_um": STATES.tolist(),
        "representative_units": [407, 13, 435, 70],
        "placement_attempts_sha256": digest(QUAL / "placement_attempts.csv"),
        "prior_templates_sha256": digest(EXTRACT / "full_probe_final_label_templates.npz"),
        "frozen_domains_sha256": digest(OUT / "frozen_domains.npz"),
        "frozen_domains_csv_sha256": digest(OUT / "frozen_domains.csv"),
        "shared_temporal_basis": {
            "source_npz": str(static_td),
            "source_npz_sha256": digest(static_td),
            "components_shape": list(components.shape),
            "components_array_sha256": array_digest(components),
            "used_by_both_halves": True,
            "halves_statistically_independent": False,
            "reason": "The disjoint spike halves are projected through the same sealed learned temporal basis and share the preprocessing/common-reference operator.",
            "cross_product_policy": "descriptive only; correlated estimator error can bias cross-products upward or downward",
        },
        "scientific_gates_changed": False,
        "voltage_read": False,
    }
    (OUT / "domain_freeze_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
