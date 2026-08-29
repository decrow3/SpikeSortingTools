"""Patch the complete Luke motion report artifact with the final detection audit.

This intentionally updates the existing full artifact in place so all previous
motion, geometry, identity, and interpolation evidence remains unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "docs" / "luke_20250804_motion_candidate_artifact.json"


def source(source_id: str, label: str, path: str, description: str) -> dict:
    return {
        "id": source_id,
        "label": label,
        "path": path,
        "query": {
            "description": description,
            "language": "python",
            "tables_used": [path],
        },
    }


def main() -> None:
    artifact = json.loads(ARTIFACT.read_text())
    manifest = artifact["manifest"]
    snapshot = artifact["snapshot"]

    manifest["generatedAt"] = "2026-08-28T23:45:00-07:00"
    snapshot["generatedAt"] = manifest["generatedAt"]
    manifest["description"] = (
        "Technical validation of Luke motion estimation/application, geometry, "
        "Kilosort-input event recovery, and conditioning alternatives relative to Yates."
    )

    new_sources = [
        source(
            "detection_source",
            "Matched Luke--Yates Kilosort-input audit",
            "testing/outputs/luke_motion_candidate_results/luke_yates_detection_stage_dataset_metrics.csv",
            "Physically normalize Kilosort-input extrema and sorter outputs by sampled probe depth.",
        ),
        source(
            "conditioning_source",
            "Luke upstream conditioning ablations",
            "testing/outputs/luke_motion_candidate_results/conditioning_replication_summary.csv",
            "Compare fixed-event and sorter guardrails under reference/CAR and single-pass conditioning changes.",
        ),
        source(
            "unit_structure_source",
            "Luke preprocessing unit-structure audit",
            "testing/outputs/luke_motion_candidate_results/luke_preprocessing_unit_structure_summary.csv",
            "Compare temporal stability and nearby-template similarity graphs for current and single-pass sorts.",
        ),
    ]
    existing_source_ids = {item["id"] for item in manifest["sources"]}
    manifest["sources"].extend(s for s in new_sources if s["id"] not in existing_source_ids)
    top_source_ids = {item["id"] for item in artifact["sources"]}
    artifact["sources"].extend(
        {"id": s["id"], "label": s["label"], "path": s["path"]}
        for s in new_sources
        if s["id"] not in top_source_ids
    )

    detection_rows = [
        {"condition": "Luke pathological", "contacts_per_mm": 100.52, "negative_events_per_mm_s": 126.766, "strong_event_recovery": 0.937393, "learned_per_mm_s": 1192.857, "final_per_mm_s": 1143.970, "units_per_mm": 63.35, "coincidence": 0.365293},
        {"condition": "Luke current", "contacts_per_mm": 100.52, "negative_events_per_mm_s": 118.847, "strong_event_recovery": 0.961432, "learned_per_mm_s": 1647.696, "final_per_mm_s": 1602.278, "units_per_mm": 76.44, "coincidence": 0.371589},
        {"condition": "Luke single-pass", "contacts_per_mm": 100.52, "negative_events_per_mm_s": 99.672, "strong_event_recovery": 0.932396, "learned_per_mm_s": 1456.795, "final_per_mm_s": 1434.871, "units_per_mm": 102.88, "coincidence": 0.351442},
        {"condition": "Yates known-good", "contacts_per_mm": 29.49, "negative_events_per_mm_s": 238.018, "strong_event_recovery": 0.969425, "learned_per_mm_s": 667.627, "final_per_mm_s": 666.129, "units_per_mm": 66.36, "coincidence": 0.359999},
    ]
    conditioning_rows = [
        {"condition": "Current no motion", "window": "120 s pathological", "recovered": "23/27", "learned": 562417, "ks_good": 95, "coincidence": 0.365294, "contamination_pct": 38.65},
        {"condition": "Bandpass, no reference", "window": "120 s pathological", "recovered": "23/27", "learned": 579298, "ks_good": 90, "coincidence": 0.364280, "contamination_pct": 44.10},
        {"condition": "Global reference", "window": "120 s pathological", "recovered": "22/27", "learned": 574693, "ks_good": 91, "coincidence": 0.363089, "contamination_pct": 47.40},
        {"condition": "Local reference, no KS CAR", "window": "120 s pathological", "recovered": "23/27", "learned": 561947, "ks_good": 96, "coincidence": 0.362319, "contamination_pct": 40.10},
        {"condition": "Single KS preprocessing", "window": "120 s pathological", "recovered": "22/27", "learned": 517187, "ks_good": 114, "coincidence": 0.350521, "contamination_pct": 38.40},
        {"condition": "Current conditioning", "window": "240 s shared", "recovered": "119/126", "learned": 1408807, "ks_good": 130, "coincidence": 0.370683, "contamination_pct": 26.40},
        {"condition": "Single KS preprocessing", "window": "240 s shared", "recovered": "120/126", "learned": 1255885, "ks_good": 168, "coincidence": 0.349972, "contamination_pct": 29.90},
    ]
    structure_rows = [
        {"condition": "Current", "unit_scope": "All", "units": 292, "similar_pairs": 62, "pairs_per_unit": 0.212, "graph_components": 240, "redundant_nodes": 52, "median_presence": 1.0, "median_rate_cv": 0.879},
        {"condition": "Single-pass", "unit_scope": "All", "units": 393, "similar_pairs": 170, "pairs_per_unit": 0.433, "graph_components": 273, "redundant_nodes": 120, "median_presence": 1.0, "median_rate_cv": 0.936},
        {"condition": "Current", "unit_scope": "KS-good", "units": 130, "similar_pairs": 17, "pairs_per_unit": 0.131, "graph_components": 113, "redundant_nodes": 17, "median_presence": 0.8542, "median_rate_cv": 1.338},
        {"condition": "Single-pass", "unit_scope": "KS-good", "units": 168, "similar_pairs": 35, "pairs_per_unit": 0.208, "graph_components": 139, "redundant_nodes": 29, "median_presence": 0.8333, "median_rate_cv": 1.322},
    ]
    snapshot["datasets"]["detection_audit"] = detection_rows
    snapshot["datasets"]["conditioning_ablation"] = conditioning_rows
    snapshot["datasets"]["unit_structure"] = structure_rows

    chart = {
        "id": "detection_density_chart",
        "title": "Strong negative Kilosort-input events by physical depth",
        "description": "Median physically deduplicated 6-sigma events per sampled millimetre and second; 60 two-second batches per condition.",
        "type": "bar",
        "dataset": "detection_audit",
        "encodings": {
            "x": {"field": "condition", "type": "nominal", "title": "Recording / conditioning"},
            "y": {"field": "negative_events_per_mm_s", "type": "quantitative", "title": "Events/mm/s"},
        },
        "options": {"orientation": "vertical", "grouping": "grouped", "legend": False},
        "sourceId": "detection_source",
        "source": {
            "id": "detection_source",
            "label": "Matched Luke--Yates Kilosort-input audit",
            "path": "testing/outputs/luke_motion_candidate_results/luke_yates_detection_stage_dataset_metrics.csv",
            "query": {
                "description": "Select the four reviewed condition summaries after physical-depth normalization.",
                "engine": "DuckDB",
                "language": "sql",
                "sql": "SELECT dataset AS condition, negative_6sigma_events_per_depth_mm_s AS negative_events_per_mm_s, negative_6sigma_final_recovery, learned_per_depth_mm_s, final_per_depth_mm_s, units_per_depth_mm, cross_unit_near_coincident_fraction FROM read_csv_auto('testing/outputs/luke_motion_candidate_results/luke_yates_detection_stage_dataset_metrics.csv') ORDER BY dataset",
                "tables_used": ["testing/outputs/luke_motion_candidate_results/luke_yates_detection_stage_dataset_metrics.csv"],
                "metric_definitions": ["Negative events are physically deduplicated local minima exceeding six batch-specific MAD units per sampled millimetre and second."],
            },
        },
    }
    table_detection = {
        "id": "detection_audit_table",
        "title": "Depth-normalized input and sorter comparison",
        "description": "Same event detector and physical-depth denominator across conditions.",
        "dataset": "detection_audit",
        "columns": [
            {"field": "condition", "label": "Condition", "type": "text"},
            {"field": "contacts_per_mm", "label": "Contacts/mm", "type": "number"},
            {"field": "negative_events_per_mm_s", "label": "Negative events/mm/s", "type": "number"},
            {"field": "strong_event_recovery", "label": "Final recovery", "type": "number"},
            {"field": "learned_per_mm_s", "label": "Learned/mm/s", "type": "number"},
            {"field": "final_per_mm_s", "label": "Final/mm/s", "type": "number"},
            {"field": "units_per_mm", "label": "Units/mm", "type": "number"},
            {"field": "coincidence", "label": "Coincidence", "type": "number"},
        ],
        "defaultSort": {"field": "condition", "direction": "asc"},
        "sourceId": "detection_source",
    }
    table_conditioning = {
        "id": "conditioning_ablation_table",
        "title": "Reference and preprocessing ablations",
        "description": "Fixed 120 s causal ablations plus independent 240 s single-pass replication.",
        "dataset": "conditioning_ablation",
        "columns": [
            {"field": "window", "label": "Window", "type": "text"},
            {"field": "condition", "label": "Condition", "type": "text"},
            {"field": "recovered", "label": "Recovered", "type": "text"},
            {"field": "learned", "label": "Learned", "type": "number"},
            {"field": "ks_good", "label": "KS-good", "type": "number"},
            {"field": "coincidence", "label": "Coincidence", "type": "number"},
            {"field": "contamination_pct", "label": "Contamination (%)", "type": "number"},
        ],
        "defaultSort": {"field": "window", "direction": "asc"},
        "sourceId": "conditioning_source",
    }
    table_structure = {
        "id": "unit_structure_table",
        "title": "Unit yield versus nearby-template redundancy",
        "description": "Similarity >=0.8 within 100 micrometres; graph components are a conservative effective-yield proxy.",
        "dataset": "unit_structure",
        "columns": [
            {"field": "unit_scope", "label": "Scope", "type": "text"},
            {"field": "condition", "label": "Condition", "type": "text"},
            {"field": "units", "label": "Units", "type": "number"},
            {"field": "similar_pairs", "label": "Similar pairs", "type": "number"},
            {"field": "graph_components", "label": "Components", "type": "number"},
            {"field": "redundant_nodes", "label": "Redundant nodes", "type": "number"},
            {"field": "median_presence", "label": "Median presence", "type": "number"},
            {"field": "median_rate_cv", "label": "Median rate CV", "type": "number"},
        ],
        "defaultSort": {"field": "unit_scope", "direction": "asc"},
        "sourceId": "unit_structure_source",
    }
    table_detection["source"] = chart["source"]
    table_conditioning["source"] = {
        "id": "conditioning_source",
        "label": "Luke upstream conditioning ablations",
        "path": "testing/outputs/luke_motion_candidate_results/conditioning_replication_summary.csv",
        "query": {
            "description": "Read the reviewed 120-second stage ablations and independent 240-second replication summary.",
            "engine": "DuckDB",
            "language": "sql",
            "sql": "SELECT * FROM read_csv_auto('testing/outputs/luke_motion_candidate_results/conditioning_replication_summary.csv') ORDER BY window, condition",
            "tables_used": ["testing/outputs/luke_motion_candidate_results/conditioning_replication_summary.csv"],
        },
    }
    table_structure["source"] = {
        "id": "unit_structure_source",
        "label": "Luke preprocessing unit-structure audit",
        "path": "testing/outputs/luke_motion_candidate_results/luke_preprocessing_unit_structure_summary.csv",
        "query": {
            "description": "Compare temporal stability and nearby-template similarity graph summaries for current and single-pass sorts.",
            "engine": "DuckDB",
            "language": "sql",
            "sql": "SELECT * FROM read_csv_auto('testing/outputs/luke_motion_candidate_results/luke_preprocessing_unit_structure_summary.csv') ORDER BY unit_scope, condition",
            "tables_used": ["testing/outputs/luke_motion_candidate_results/luke_preprocessing_unit_structure_summary.csv"],
            "metric_definitions": ["Similar pairs have template correlation at least 0.8 and peak positions within 100 micrometres."],
        },
    }
    for item in [chart]:
        by_id = {x["id"]: i for i, x in enumerate(manifest["charts"])}
        if item["id"] in by_id:
            manifest["charts"][by_id[item["id"]]] = item
        else:
            manifest["charts"].append(item)
    for item in [table_detection, table_conditioning, table_structure]:
        by_id = {x["id"]: i for i, x in enumerate(manifest["tables"])}
        if item["id"] in by_id:
            manifest["tables"][by_id[item["id"]]] = item
        else:
            manifest["tables"].append(item)

    new_blocks = [
        {"id": "detection_audit_intro", "type": "markdown", "sourceId": "detection_source", "body": "## The original per-channel deficit was denominator-confounded\n\nLuke has 3.41x more contacts per millimetre than Yates. Per physical depth, Luke has similar unit density and near-identical coincidence, while Kilosort recovers 94--96% of strong negative input events versus 97% in Yates. Luke still has about half as many such events per millimetre; that is an upstream density difference, not a large sorter miss rate."},
        {"id": "detection_density_chart_block", "type": "chart", "chartId": "detection_density_chart"},
        {"id": "detection_audit_table_block", "type": "table", "tableId": "detection_audit_table"},
        {"id": "conditioning_intro", "type": "markdown", "sourceId": "conditioning_source", "body": "## Reference changes are null; single-pass conditioning is promising\n\nRemoving local reference, substituting global reference, or disabling Kilosort CAR does not improve fixed-event recovery. In the independent 240 s window, a single Kilosort high-pass/CAR pass reduces learned detections by 10.9%, raises KS-good units from 130 to 168, and slightly improves reviewed recovery and coincidence."},
        {"id": "conditioning_ablation_table_block", "type": "table", "tableId": "conditioning_ablation_table"},
        {"id": "unit_structure_intro", "type": "markdown", "sourceId": "unit_structure_source", "body": "## Much of the single-pass unit gain is redundant\n\nThe 101-unit raw increase includes 68 additional redundant nodes in a nearby-template similarity graph. Effective all-unit graph components rise only 14%, while good-unit components rise 23%. The condition may improve separation, but it also over-splits and must be compared after identical merging."},
        {"id": "unit_structure_table_block", "type": "table", "tableId": "unit_structure_table"},
    ]
    existing_block_ids = {b["id"] for b in manifest["blocks"]}
    new_blocks = [b for b in new_blocks if b["id"] not in existing_block_ids]
    insert_at = next(i for i, b in enumerate(manifest["blocks"]) if b["id"] == "motion_comparison_intro")
    manifest["blocks"][insert_at:insert_at] = new_blocks

    blocks_by_id = {b["id"]: b for b in manifest["blocks"]}
    blocks_by_id["detection_audit_intro"]["sourceId"] = "detection_source"
    blocks_by_id["conditioning_intro"]["sourceId"] = "conditioning_source"
    blocks_by_id["unit_structure_intro"]["sourceId"] = "unit_structure_source"
    blocks_by_id["summary"]["body"] = (
        "## Technical summary\n\n"
        "Luke genuinely moves more than Yates, but full non-rigid voltage warps from both DREDGE and MEDiCINe are causal failures: they more than double learned detections and coincidence while reducing recovery. Zero-displacement interpolation is byte-identical to no correction, localizing the catastrophe to applied depth-dependent displacement rather than a generic export path.\n\n"
        "The original Luke--Yates per-channel deficit was also denominator-confounded because Luke has 3.41x more contacts per millimetre. Per physical depth, Luke's no-motion unit density and coincidence are similar to Yates and Kilosort recovers nearly all strong input events. Luke still has about half as many strong negative events per millimetre, an upstream density difference that remains unexplained.\n\n"
        "No external voltage correction remains the production baseline. Single-pass Kilosort conditioning is the leading preprocessing candidate, but its higher unit yield is partly nearby-template redundancy and requires a matched full-session merge/continuity audit."
    )
    blocks_by_id["limits"]["body"] = (
        "## Limitations and robustness\n\n"
        "The fixed-event cohorts are descriptive and Kilosort labels are not ground truth. Per-depth normalization removes the largest probe-density confound, but Luke and Yates still use different probe geometries and may sample different laminar populations. The exact legacy Yates preprocessing graph is unavailable. The template-similarity graph uses a declared 0.8 correlation and 100 micrometre neighborhood and does not replace matched merging."
    )
    blocks_by_id["next"]["body"] = (
        "## Recommended next steps\n\n"
        "1. Keep no external voltage correction as the Luke production baseline.\n"
        "2. Run a matched full-session current-versus-single-pass conditioning comparison with identical merging and unit-family continuity checks.\n"
        "3. Do not use full non-rigid DREDGE or MEDiCINe warps; both recreate overdecomposition.\n"
        "4. If interpolation is revisited, scale the kernel to probe geometry and require strict recovery, contamination, and collision safeguards.\n"
        "5. Replace per-channel Luke--Yates comparisons with physical-depth and matched-layer denominators; use a common raw detector to locate the remaining negative-event density difference.\n"
        "6. Prefer motion-aware coordinate/template tracking that avoids resampling raw voltage."
    )
    blocks_by_id["questions"]["body"] = (
        "## Further questions\n\n"
        "- Is Luke's lower strong-negative event density already present in matched raw AP voltage and comparable laminar bands?\n"
        "- Does single-pass conditioning retain its 23% increase in good similarity-graph components after identical merging over the full session?\n"
        "- What property of Luke's spike fingerprint makes Kilosort coarse registration switch between modes separated by 70--103 micrometres?"
    )

    ARTIFACT.write_text(json.dumps(artifact, indent=2) + "\n")


if __name__ == "__main__":
    main()
