#!/usr/bin/env python3
"""Publish the execution-disabled evaluator spatial-provenance review candidate."""

from __future__ import annotations

import difflib
import hashlib
import importlib
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path("/home/huklaban5/Documents/SpikeSortingTools/SpikeSortingTools")
SHARED = Path("/mnt/NPX/Luke/DARTsort_motion_experiments")
BASE = SHARED / "en_first_medium_four_arm_evaluator_repair_candidate_20261002_v3_h1"
POST_QC_REQUEST = SHARED / "en_first_medium_evaluator_post_qc_execution_request_20261002_v1_h1"
POST_QC_RESULT = SHARED / "en_first_medium_evaluator_post_qc_execution_result_20261002_v1_h5"
PREP = SHARED / "en_first_medium_trained_pair_preparation_readiness_20261002_v1_h5"
STAGE = Path("/tmp/en_spatial_provenance_candidate_20261002_v1")
DEST = SHARED / "en_first_medium_evaluator_spatial_provenance_candidate_20261002_v1_h5"
NATIVE = ROOT / "environments/rescue-production/.venv/lib/python3.12/site-packages/kilosort"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def bind(packet: Path, manifest: str, complete: str) -> dict:
    assert sha(packet / "MANIFEST.sha256") == manifest
    assert sha(packet / "COMPLETE.json") == complete
    return {"path": str(packet), "manifest_sha256": manifest, "complete_sha256": complete}


def main() -> None:
    assert not DEST.exists(), f"refusing to overwrite {DEST}"
    assert STAGE.is_dir()
    base_binding = bind(
        BASE, "7e0d4d7e16c5ade2ceb376cd058b3ad3e6f05ffb9cd17b4830450f14277bfd1e",
        "959a1e139c3289851d3905ed815bd543b32fa821884937eb1519cdbbac777ec6",
    )
    post_qc_binding = bind(
        POST_QC_RESULT, "e417eceb918f44110c4c43eaac58fe4e33ae4c87a4c4e520f9fc655b4640b8c2",
        "e77f50d7f4e00b39c9fa6a99bfad1cf00ecb92b9dca7823d4c06c18d29c0a15c",
    )
    prep_binding = bind(
        PREP, "d412d46d505873225a8f2066e413a347672493ca99b1a88f3b7a04b59048aed2",
        "b73a3132ac23e8c6eb251fb1641191f51bf919432f554a2bdea0e8740ae99499",
    )

    sys.path.insert(0, str(STAGE / "source"))
    from pipeline.config import fingerprint
    from testing.first_medium_evaluator import spatial_event_lineage_sha256
    from testing.luke_amplitude_dropout_audit import read_curated_arrays

    source_hashes = {
        str(path.relative_to(STAGE)): sha(path)
        for path in sorted((STAGE / "source").rglob("*.py"))
    }
    test_hashes = {
        str(path.relative_to(STAGE)): sha(path)
        for path in sorted((STAGE / "tests").rglob("*.py"))
    }
    config = json.loads((POST_QC_REQUEST / "config/en_first_medium_four_arm_evaluator.post_qc.execution_disabled.v1.json").read_text())
    original_region = json.loads(json.dumps(config["spatial_region"]))
    config["schema"] = "en-first-medium-four-arm-evaluator-request-v3-spatial-provenance"
    config["purpose"] = (
        "execution-disabled evaluator input contract with fail-closed physical spatial provenance; "
        "does not launch evaluator, sorting, training, detection, voltage, RF, or holdout work"
    )
    config["spatial_provenance_policy"] = {
        "schema": "evaluator-spatial-provenance-v1",
        "positions_file": "spike_positions.npy",
        "reference_frame": "Kilosort PC-feature-mass weighted probe x/y; these arms require nblocks=0 and dshift=None, hence unshifted physical probe coordinates",
        "units": "micrometers",
        "columns": {"0": "x", "1": "physical_probe_depth_y"},
        "row_lineage": "same native kept_spikes-filtered exported event order as spike_times/spike_clusters/amplitudes; any evaluator stable time sort is applied jointly through original row_id",
        "unit_cohort": "median bound event depth per cluster",
        "region_change": "none; processing/scoring/edge intervals are byte-for-byte retained from the accepted request",
    }

    input_receipts = {}
    for arm in config["arms"]:
        if arm["evidence_kind"] != "real_saved":
            arm["provenance"]["spatial_provenance_status"] = "REQUIRED_WHEN_ARM_BECOMES_AVAILABLE"
            continue
        arm_id = arm["arm_id"]
        curated = Path(arm["curated_output"])
        raw, hashes = read_curated_arrays(curated)
        position_path = curated / "spike_positions.npy"
        position_bytes = position_path.read_bytes()
        position_sha = hashlib.sha256(position_bytes).hexdigest()
        positions = np.load(__import__("io").BytesIO(position_bytes), allow_pickle=False)
        del position_bytes
        ops_path = curated / "ops.npy"
        ops_sha = sha(ops_path)
        ops = np.load(ops_path, allow_pickle=True).item()
        geometry = np.stack([
            np.asarray(ops["xc"], dtype="<f8"), np.asarray(ops["yc"], dtype="<f8")
        ], axis=1)
        geometry_sha = hashlib.sha256(np.ascontiguousarray(geometry).tobytes()).hexdigest()
        assert geometry_sha == config["common_geometry"]["geometry_float64_sha256"]
        assert int(ops["nblocks"]) == 0 and ops.get("dshift") is None
        assert np.array_equal(np.asarray(ops["chanMap"]), np.arange(384))
        kept = np.asarray(raw["kept_spikes.npy"])
        selected = np.asarray(raw["full_st.npy"])[kept]
        lineage = spatial_event_lineage_sha256(
            np.asarray(raw["spike_times.npy"]).reshape(-1),
            np.asarray(raw["spike_clusters.npy"]).reshape(-1),
            selected[:, 2], positions,
        )
        clock = arm["clock_normalization"]
        spatial = {
            "schema": "evaluator-spatial-provenance-v1",
            "path": str(position_path.resolve()), "sha256": position_sha,
            "ops_path": str(ops_path.resolve()), "ops_sha256": ops_sha,
            "row_count": int(len(positions)), "event_lineage_sha256": lineage,
            "reference_frame": "kilosort_probe_xy_feature_mass_unshifted_nblocks0",
            "units": "micrometers",
            "columns": [
                {"index": 0, "semantic": "x", "units": "micrometers"},
                {"index": 1, "semantic": "physical_probe_depth_y", "units": "micrometers"},
            ],
            "row_semantics": "exported_event_order_after_native_kept_spikes",
            "clock_binding": clock,
            "spatial_region": original_region,
            "geometry_float64_sha256": geometry_sha,
            "source_generation": {
                "function": "kilosort.postprocessing.compute_spike_positions",
                "io": {"path": str(NATIVE / "io.py"), "sha256": sha(NATIVE / "io.py")},
                "postprocessing": {"path": str(NATIVE / "postprocessing.py"),
                                   "sha256": sha(NATIVE / "postprocessing.py")},
            },
        }
        arm["spatial_provenance"] = spatial
        hashes["spike_positions.npy"] = position_sha
        labels = (curated / "cluster_KSLabel.tsv").read_bytes()
        new_identity = fingerprint({
            "curated": str(curated.resolve()), "files": hashes,
            "labels_sha256": hashlib.sha256(labels).hexdigest(),
        })
        prior_identity = arm["expected_identity_digest"]
        arm["expected_identity_digest"] = new_identity
        arm["provenance"]["spatial_identity"] = lineage
        input_receipts[arm_id] = {
            "prior_identity_without_position_binding": prior_identity,
            "effective_identity_with_position_binding": new_identity,
            "spike_positions_sha256": position_sha,
            "shape": list(positions.shape), "dtype": str(positions.dtype),
            "event_lineage_sha256": lineage, "ops_sha256": ops_sha,
            "geometry_float64_sha256": geometry_sha,
            "nblocks": 0, "dshift": None, "full_probe_identity": True,
            "clock_binding": clock, "spatial_region": original_region,
        }

    assert config["spatial_region"] == original_region
    waveform_plan = {
        "schema": "first-medium-physical-waveform-evidence-plan-v1",
        "status": "FROZEN_DESIGN_ONLY_NO_VOLTAGE_ACCESS",
        "decision": "technical physical-signal support screen; cannot establish identity, purity, or sorter efficacy",
        "comparisons": [
            "REF384_vs_existing_corrected_B384", "REF384_vs_repaired_B384", "REF384_vs_REF384_repeat",
        ],
        "epochs": {
            "rule": "partition the exact common global half-open 600-second crop by integer sample count into contiguous thirds [F0+floor(k*N/3), F0+floor((k+1)*N/3)); labels early/middle/late",
            "coverage": "all common-crop frames exactly once; no widening, clipping, or padding in denominators",
        },
        "classification": {
            "matching": "deterministic maximum-cardinality one-to-one global-frame matching within +/-1 sample; minimize absolute lag; exact nonunique ties are ambiguous",
            "retained": "unique matched detection", "gained": "candidate detection without admissible REF partner",
            "lost": "REF detection without admissible candidate partner",
            "ambiguous": "nonunique/tied match, unsupported physical anchor, >80-um competing anchors, or provenance conflict",
            "partition": "every event in the full eligible union belongs to exactly one category before waveform access",
        },
        "sampling": {
            "order": "SHA-256(recording identity, comparison, epoch, category, global frame, source label)",
            "maximum_per_category_epoch_comparison": 128,
            "maximum_records": 4608,
            "replacement": False, "outcome_driven_substitution": False,
            "report": ["population_N", "requested_n", "sampled_n", "technically_evaluable_n", "each_exclusion_n"],
        },
        "physical_extraction": {
            "recording": "original accepted 384-channel physical acquisition order only",
            "anchor": "pre-voltage exact physical-channel back-map per event using frozen field cells; unsupported=-1; no depth-nearness substitute",
            "channels": "union of predeclared anchors and exact physical channels within 80 um",
            "core_samples_half_open": [-60, 61], "filter_padding_each_side_samples": 512,
            "padded_samples": 1145,
            "preprocessing": "one frozen centering/CAR/high-pass chain on padded full physical data, then retain 121-sample core",
            "baseline_samples": "[0,30) union [91,121)", "signal_samples": "[40,81)",
            "noise": "1.4826*MAD per channel, floor 1 ADC count",
        },
        "technical_support": {
            "pass": "within frozen <=80-um union, leading-channel PTP SNR>=6, adjacent physical channel PTP SNR>=4, extremum center offset <=15 samples",
            "unevaluable": ["missing padding", "saturation/clipping", "unsupported anchor", "provenance failure"],
            "retained_correlation": "normalized common physical-channel/time waveform correlation; descriptive only",
        },
        "denominators_and_uncertainty": {
            "population": "full event union by comparison and epoch",
            "rates": "raw population category counts/core minute; waveform-supported rate is deterministic category expansion using sampled support fraction",
            "intervals": "Wilson 95% for sampled technical/support fractions; exact Poisson exposure intervals for full population rates; propagate support-fraction bounds to supported-rate bounds",
        },
        "quality_rules": {
            "PASS": "technically evaluable >=80% and waveform-support Wilson lower bound >=0.60",
            "CAUTION": "technically evaluable >=60% or waveform-support lower bound in [0.40,0.60)",
            "FAIL": "otherwise",
            "harm_margin": "repaired minus existing waveform-supported lost rate upper 95% bound <=0.5 events/core minute in every epoch",
            "benefit": "at least two epochs improve waveform-supported lost rate by >=0.5 events/core minute or ambiguous fraction by >=5 percentage points, without unsupported-gained harm >0.5 events/core minute",
            "repeat_gate": "REF repeat is not FAIL in any epoch; otherwise efficacy interpretation is inconclusive",
        },
        "resources": {
            "logical_read_bytes_max": 4052090880,
            "logical_read_gib_max": 3.774,
            "event_reads_max": 4608, "cpu_threads_max": 4, "ram_bytes_max": 2147483648,
            "wall_seconds_max": 2700, "gpu_count": 0, "local_transient_cache_bytes_max": 5368709120,
            "compact_persistent_bytes_max": 104857600,
        },
        "outputs": [
            "EVENT_LEDGER.csv", "SAMPLE_LEDGER.csv", "WAVEFORM_METRICS.csv", "DENOMINATORS.csv",
            "EPOCH_SCORECARD.csv", "resource/read counters", "failure receipt", "MANIFEST", "COMPLETE last",
        ],
        "publication": "no numeric voltage or transferable waveform arrays in shared packet",
        "exclusions": ["RF", "sealed holdout", "new sort/training/detection", "threshold search", "template refit", "identity/purity claim"],
    }

    open_gaps = {
        "trained_entry_global_vs_crop_local_validator_defect": "OPEN_UNCHANGED; outside this changed path",
        "no_mask_REF_wrapper_gap": "OPEN_UNCHANGED; outside this changed path",
        "evaluator_execution": "NOT_AUTHORIZED_BY_THIS_PACKET_AND_NOT_RUN",
        "data_sort_training_detection": "NOT_RUN",
        "waveform_voltage_access": "NOT_RUN; plan only",
        "RF_holdout": "SEALED_NOT_ACCESSED",
    }
    review = {
        "schema": "evaluator-spatial-provenance-review-request-v1",
        "status": "execution_disabled_pending_H1_independent_review",
        "requested_review": [
            "trace same-byte hash/parse of spike_positions.npy",
            "verify joint event-row lineage digest and joint stable sort",
            "verify ops/source/geometry/frame/units/columns/clock bindings",
            "verify unchanged physical processing/scoring/edge intervals and cohort construction",
            "rerun correct, byte mismatch, row permutation, row count mismatch and frame/region mismatch fixtures",
            "review waveform plan assumptions and frozen harm/benefit thresholds before any voltage access",
        ],
        "execution_authorization": "NONE_REVIEW_PACKET_ONLY",
    }

    with tempfile.TemporaryDirectory(prefix=".spatial_provenance_packet_", dir=DEST.parent) as td:
        packet = Path(td) / DEST.name
        shutil.copytree(STAGE / "source", packet / "source")
        shutil.copytree(STAGE / "tests", packet / "tests")
        (packet / "config").mkdir()
        write_json(packet / "config/en_first_medium_four_arm_evaluator.spatial_provenance.execution_disabled.v1.json", config)
        config_path = packet / "config/en_first_medium_four_arm_evaluator.spatial_provenance.execution_disabled.v1.json"
        write_json(packet / "SPATIAL_INPUT_RECEIPTS.json", input_receipts)
        write_json(packet / "PHYSICAL_WAVEFORM_EVIDENCE_PLAN.json", waveform_plan)
        write_json(packet / "OPEN_GAPS.json", open_gaps)
        write_json(packet / "REVIEW_REQUEST.json", review)
        write_json(packet / "EFFECTIVE_CONFIG_RECEIPT.json", {
            "path": str(config_path.relative_to(packet)), "sha256": sha(config_path),
            "execution_enabled": False, "schema": config["schema"],
            "spatial_region_unchanged": True, "spatial_region": original_region,
            "real_arms": input_receipts, "source_hashes": source_hashes,
        })
        delta_dir = packet / "delta"
        delta_dir.mkdir()
        for relative in ("testing/first_medium_evaluator.py", "testing/sort_comparison.py"):
            old = (BASE / "source" / relative).read_text().splitlines(keepends=True)
            new = (STAGE / "source" / relative).read_text().splitlines(keepends=True)
            diff = difflib.unified_diff(old, new, fromfile=f"accepted/{relative}", tofile=f"candidate/{relative}")
            (delta_dir / (relative.replace("/", "__") + ".diff")).write_text("".join(diff))
        (packet / "TEST_RECEIPT.txt").write_text(
            "PYTHONPATH=<packet>/source python -m pytest -q tests/test_first_medium_evaluator.py\n"
            "16 passed, 1 warning in 3.94s\n"
            "PYTHONPATH=<packet>/source python tests/run_packaged_cli_proof.py\n"
            "status=passed; scientific_rows=[REF384__existing_corrected_B384]; all five module paths under packet source\n"
            "No evaluator or real data/sort/training/detection/voltage/RF/holdout job ran.\n"
        )
        (packet / "README.md").write_text(
            "# Evaluator spatial-provenance candidate v1\n\n"
            "Status: **execution disabled; H1 independent review requested; no launch**.\n\n"
            "This narrow delta makes physical cohort eligibility fail closed. It hashes and parses the same `spike_positions.npy` bytes, binds exact Kilosort source/ops/geometry/frame/units/columns, binds position rows jointly to exported time/cluster/amplitude order, applies any stable time sort to positions through explicit original row IDs, and emits receipts in arm provenance, pair manifests, and the four-arm report.\n\n"
            "The accepted physical region is unchanged: processing `[1400,2380]` um, scoring/interior `[1600,2180]` um, minimum edge exclusion `80` um. No outside pair is admitted by this delta.\n\n"
            "Known-answer negatives reject stale bytes, a row permutation even when the position-file hash is updated, wrong row count, wrong coordinate frame, and widened spatial region. The correct binding passes.\n\n"
            "A separate frozen physical-waveform evidence plan covers retained/gained/lost/ambiguous activity across exact early/middle/late thirds with maximum 4,608 deterministic samples and 3.774 GiB logical reads. It remains plan-only.\n\n"
            "The trained-entry global/crop clock validator defect and missing no-mask REF wrapper remain explicitly open and untouched.\n\n"
            "## Implementation checks\n\n"
            "- Done: verified accepted source/result bindings and real REF384/B384 position, ops, geometry, clock and row-lineage identities.\n"
            "- Done: traced Kilosort position generation (`compute_spike_positions`) and native kept-row filtering; bound both source files by hash.\n"
            "- Done: 16 focused tests and packaged CLI proof passed from the staged source.\n"
            "- Done: unchanged physical spatial region asserted in config and tested against widening.\n"
            "- Not done: independent H1 review, evaluator execution, prospective-arm positions, waveform voltage access, sorting/training/detection, RF or holdout.\n"
            "- Can establish after review: that spatial eligibility and guardrails use hash-bound physical position rows with explicit semantics.\n"
            "- Cannot establish: biological identity, waveform support, repaired-arm efficacy, repeatability, or advancement.\n"
        )
        bindings = packet / "bindings"
        bindings.mkdir()
        for name, source in (("accepted_evaluator", BASE), ("post_qc_result", POST_QC_RESULT), ("trained_preparation", PREP)):
            shutil.copy2(source / "MANIFEST.sha256", bindings / f"{name}_MANIFEST.sha256")
            shutil.copy2(source / "COMPLETE.json", bindings / f"{name}_COMPLETE.json")
        members = [p for p in sorted(packet.rglob("*")) if p.is_file()]
        (packet / "MANIFEST.sha256").write_text("\n".join(f"{sha(p)}  {p.relative_to(packet)}" for p in members) + "\n")
        manifest = sha(packet / "MANIFEST.sha256")
        packet.rename(DEST)
        write_json(DEST / "COMPLETE.json", {
            "schema": "immutable-review-candidate-completion-v1", "status": "complete",
            "execution_enabled": False, "manifest_sha256": manifest,
            "effective_config_sha256": sha(DEST / "config/en_first_medium_four_arm_evaluator.spatial_provenance.execution_disabled.v1.json"),
            "completion_written_last": True, "completed_utc": datetime.now(timezone.utc).isoformat(),
        })
        print(json.dumps({
            "destination": str(DEST), "manifest_sha256": manifest,
            "complete_sha256": sha(DEST / "COMPLETE.json"),
            "config_sha256": sha(DEST / "config/en_first_medium_four_arm_evaluator.spatial_provenance.execution_disabled.v1.json"),
        }, indent=2))


if __name__ == "__main__":
    main()
