#!/usr/bin/env python3
"""Read-only BI audit of saved hybrid donor/event/injection provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def record(path: Path) -> dict:
    return {
        "path": str(path), "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else None,
        "sha256": sha256(path) if path.is_file() else None,
    }


def proposed_schema() -> dict:
    return {
        "schema": "bi-hybrid-provenance-bundle-v1-proposed",
        "backward_compatibility": (
            "Existing truth_samples/donor_ids arrays remain derivable in event_row_id order; "
            "existing scorer indices remain valid, while optional immutable row IDs make recovery auditable."
        ),
        "tables": {
            "donors": {
                "primary_key": "donor_id",
                "required": [
                    "donor_id", "source_unit_id", "metadata_row_id", "template_index",
                    "template_sha256", "source_sorting_sha256", "qualification_row_id",
                    "placement_base_shift_um", "occupied_states_um", "selected",
                ],
            },
            "source_events": {
                "primary_key": "event_row_id",
                "foreign_keys": {"donor_id": "donors.donor_id"},
                "required": [
                    "event_row_id", "donor_id", "local_sample", "absolute_sample",
                    "window_start_sample", "sampling_frequency_hz", "generator_schema",
                    "generator_sha256", "seed", "available",
                ],
            },
            "injection_membership": {
                "primary_key": "injection_event_id",
                "foreign_keys": {
                    "source_event_row_id": "source_events.event_row_id",
                    "donor_id": "donors.donor_id",
                },
                "required": [
                    "injection_event_id", "source_event_row_id", "donor_id", "arm_id",
                    "background_id", "template_sha256", "state_um", "channel_map_sha256",
                    "sample_start", "sample_stop", "target_channel_ids_sha256",
                    "status", "collision_group",
                ],
                "status_values": ["injected", "unavailable", "excluded"],
            },
            "bundle_manifest": {
                "required": [
                    "schema", "window_id", "window_start_sample", "sampling_frequency_hz",
                    "donors_sha256", "source_events_sha256", "injection_membership_sha256",
                    "background_sha256", "generator_code_sha256", "scorer_code_sha256",
                ]
            },
        },
        "scorer_extension": {
            "optional_inputs": ["truth_event_row_ids", "output_event_row_ids"],
            "optional_match_columns": ["truth_event_row_id", "output_event_row_id"],
            "category_codes": {"unmatched": -1, "unavailable": -2},
            "note": "Optional additions preserve existing positional-index outputs and metrics.",
        },
    }


def fixture_bundle() -> dict:
    donors = [{
        "donor_id": 7, "source_unit_id": 407, "metadata_row_id": "unit:407",
        "template_index": 0, "template_sha256": "template-hash-7",
        "source_sorting_sha256": "sorting-hash", "qualification_row_id": "qual:407:360",
        "placement_base_shift_um": 360.0, "occupied_states_um": [0.0, -40.0], "selected": True,
    }]
    events = [
        {"event_row_id": "e7:0", "donor_id": 7, "local_sample": 100,
         "absolute_sample": 27100100, "window_start_sample": 27100000,
         "sampling_frequency_hz": 30000.0, "generator_schema": "seeded-independent-v1",
         "generator_sha256": "generator-hash", "seed": 20260933, "available": True},
        {"event_row_id": "e7:1", "donor_id": 7, "local_sample": 200,
         "absolute_sample": 27100200, "window_start_sample": 27100000,
         "sampling_frequency_hz": 30000.0, "generator_schema": "seeded-independent-v1",
         "generator_sha256": "generator-hash", "seed": 20260933, "available": False},
    ]
    membership = [
        {"injection_event_id": "inj:e7:0:S_h", "source_event_row_id": "e7:0", "donor_id": 7,
         "arm_id": "S_h", "background_id": "W2-background", "template_sha256": "template-hash-7",
         "state_um": 0.0, "channel_map_sha256": "map-hash-0", "sample_start": 58,
         "sample_stop": 179, "target_channel_ids_sha256": "channels-hash", "status": "injected",
         "collision_group": None},
        {"injection_event_id": "inj:e7:1:S_h", "source_event_row_id": "e7:1", "donor_id": 7,
         "arm_id": "S_h", "background_id": "W2-background", "template_sha256": "template-hash-7",
         "state_um": -40.0, "channel_map_sha256": "map-hash--40", "sample_start": 158,
         "sample_stop": 279, "target_channel_ids_sha256": "channels-hash", "status": "unavailable",
         "collision_group": None},
    ]
    return {"donors": donors, "source_events": events, "injection_membership": membership}


def validate_fixture(bundle: dict) -> dict:
    donors = {x["donor_id"]: x for x in bundle["donors"]}
    events = {x["event_row_id"]: x for x in bundle["source_events"]}
    memberships = bundle["injection_membership"]
    assert len(donors) == len(bundle["donors"])
    assert len(events) == len(bundle["source_events"])
    assert len({x["injection_event_id"] for x in memberships}) == len(memberships)
    for event in events.values():
        assert event["donor_id"] in donors
        assert event["absolute_sample"] == event["window_start_sample"] + event["local_sample"]
    for member in memberships:
        event = events[member["source_event_row_id"]]
        donor = donors[member["donor_id"]]
        assert event["donor_id"] == member["donor_id"]
        assert donor["template_sha256"] == member["template_sha256"]
        assert member["sample_start"] < member["sample_stop"]
        assert member["status"] in {"injected", "unavailable", "excluded"}
    return {
        "pass": True, "donors": len(donors), "source_events": len(events),
        "memberships": len(memberships), "foreign_keys_valid": True,
        "local_absolute_clock_valid": True, "template_hash_lineage_valid": True,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    outroot = ROOT / "testing/outputs/luke_au_cpu_preparation"
    sources = {
        "measured_template_bank": record(outroot / "full_probe_extraction_v1/bc_attempt5/full_probe_final_label_templates.npz"),
        "measured_unit_table": record(outroot / "full_probe_extraction_v1/bc_attempt5/full_probe_candidate_measurements.csv"),
        "qualification_summary": record(outroot / "donor_qualification_v1/qualification_summary.json"),
        "placement_attempts": record(outroot / "donor_qualification_v1/placement_attempts.csv"),
        "qualified_five": record(outroot / "donor_qualification_v1/qualified_relocated_donors.csv"),
        "selected_30_cohort": record(outroot / "donor_qualification_v1/selected_donor_cohort.csv"),
        "population_generator_source": record(ROOT / "testing/luke_au_cpu_preparation.py"),
        "scorer_source": record(ROOT / "testing/luke_az_hybrid_scorer.py"),
        "materialized_source_events": record(outroot / "worker_assets/source_events.npz"),
        "injection_membership": record(outroot / "worker_assets/injection_membership.csv"),
        "worker_bundle_manifest": record(outroot / "worker_assets/bundle_manifest.json"),
    }
    schema = proposed_schema()
    fixture = fixture_bundle()
    fixture_result = validate_fixture(fixture)
    audit = {
        "schema": "bi-h1-hybrid-provenance-audit-v1",
        "status": "preparation_only_hybrid_still_blocked",
        "sources": sources,
        "minimum_records": {
            "donor": "sealed metadata row + template index/hash + qualification placement/state record",
            "source_event": "immutable event row ID + donor FK + local/absolute sample + generator hash/seed",
            "injection_membership": "event FK + arm/background + template/map/channel hashes + half-open write bounds + status",
            "bundle": "hashes of all three tables, background, generator and scorer",
        },
        "missing_now": [name for name in ("selected_30_cohort", "materialized_source_events", "injection_membership", "worker_bundle_manifest") if not sources[name]["exists"]],
        "interpretation": (
            "The measured 90-unit bank and five-unit qualification are preserved, but no frozen 30-donor "
            "cohort, materialized population trains, injection membership, or worker bundle exists because "
            "AZ is scientifically blocked. Future authorized work should write these compact records before "
            "injection and score through immutable row IDs; this audit does not alter the donor gate."
        ),
        "fixture_result": fixture_result,
        "raw_reads": 0, "sorting": False, "gpu": False, "donor_gate_changed": False,
    }
    (args.output / "proposed_schema.json").write_text(json.dumps(schema, indent=2) + "\n")
    (args.output / "provenance_fixture.json").write_text(json.dumps(fixture, indent=2) + "\n")
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    readme = """# BI h1 hybrid provenance audit

This read-only audit identifies the minimum compact lineage needed before any
future authorized hybrid run: donor/template selection, immutable source-event
rows, event-to-injection membership, and a bundle manifest binding their hashes
to background/generator/scorer identities.

Present: the 90-unit measured bank, candidate table, complete placement audit,
the frozen five qualified donors, generator source and scorer source. Missing
by design: a selected 30-donor cohort, materialized population trains,
injection membership and worker manifest. Those absences reflect the existing
5/90 scientific block; they must not be repaired by changing donor thresholds.

The proposed schema is backward compatible: current `truth_samples` and
`donor_ids` are derived by stable event-row order, while optional immutable row
IDs are added to match outputs. The tiny fixture verifies foreign keys,
local/absolute clocks, template lineage and half-open injection bounds.
"""
    (args.output / "README.md").write_text(readme)
    files = [args.output / x for x in ("audit.json", "proposed_schema.json", "provenance_fixture.json", "README.md")]
    with (args.output / "SHA256SUMS").open("w") as f:
        for path in files:
            f.write(f"{sha256(path)}  {path.name}\n")
    print(json.dumps({"fixture_pass": fixture_result["pass"], "missing": audit["missing_now"]}))


if __name__ == "__main__":
    main()
