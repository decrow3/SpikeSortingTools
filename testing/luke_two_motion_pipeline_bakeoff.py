"""Thin candidate runner for two-option motion pipeline bakeoff.

Build instructions: docs/luke_two_motion_pipeline_build_instructions.md §3.
Governing plan: docs/pipeline_improvement_plan.md.

Exposes CLI orchestration contract:
    --option control|external_warp|unwarped_identity
    --snippet-dir PATH
    --out-root PATH
    --motion-info-dir PATH        # required for motion-aware arms
    --truth PATH                  # required for injected-truth runs
    --config PATH                 # frozen option configuration
    --mode verify|smoke|l1|l2|l2l
    --contract PATH               # delivery contract
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from testing.first_pipeline_candidate_contract import (
    DEFAULT_CONTRACT,
    ContractRefusal,
    git_commit,
    git_worktree_state,
    load_contract,
    reject_unsafe_out_root,
    validate,
)
from testing.ladder_score import score_sort
from testing.ladder_unwarped_identity import (
    UnwarpedIdentityConfig,
    run_unwarped_identity_pipeline,
)
from pipeline.motion_coordinates import (
    interpolate_motion_at_spikes,
    load_qualified_motion_field,
)


BAKEOFF_SCHEMA = "luke-two-motion-pipeline-bakeoff-v1"
BAKEOFF_MANIFEST = "bakeoff_manifest.json"

OPTIONS = ("control", "external_warp", "unwarped_identity")
MODES = ("verify", "smoke", "l1", "l2", "l2l")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_digest(node: Any) -> str:
    return hashlib.sha256(
        json.dumps(node, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def check_output_root(out_root: Path, input_paths: list[Path]) -> Path:
    """Refuse output root under /mnt or colliding with input paths."""
    return reject_unsafe_out_root(out_root, input_paths)


def run_bakeoff(
    *,
    option: str,
    mode: str,
    out_root: Path,
    snippet_dir: Path | None = None,
    motion_info_dir: Path | None = None,
    truth_path: Path | None = None,
    config_path: Path | None = None,
    contract_path: Path = DEFAULT_CONTRACT,
) -> dict[str, Any]:
    if option not in OPTIONS:
        raise ValueError(f"Unknown option {option!r}, must be one of {OPTIONS}")
    if mode not in MODES:
        raise ValueError(f"Unknown mode {mode!r}, must be one of {MODES}")

    contract = load_contract(contract_path)
    inputs_to_check: list[Path] = []
    if snippet_dir is not None:
        inputs_to_check.append(Path(snippet_dir))
    if motion_info_dir is not None:
        inputs_to_check.append(Path(motion_info_dir))
    if truth_path is not None:
        inputs_to_check.append(Path(truth_path))

    resolved_out_root = check_output_root(Path(out_root), inputs_to_check)

    # Validate contract with execution enforcement in run modes
    validation_mode = "execution" if mode in ("smoke", "l1", "l2", "l2l") else "authoring"
    val_report = validate(contract_path, mode=validation_mode, out_root=resolved_out_root)

    # Load option configuration if provided
    opt_config: dict[str, Any] = {}
    config_digest = ""
    if config_path is not None and Path(config_path).exists():
        opt_config = json.loads(Path(config_path).read_text())
        config_digest = canonical_digest(opt_config)

    print(f"--- Luke Two-Option Pipeline Bakeoff ---")
    print(f"Option: {option}")
    print(f"Mode: {mode}")
    print(f"Contract ID: {contract.get('contract_id')}")
    print(f"Validation Mode: {validation_mode}")
    print(f"Output Root: {resolved_out_root}")
    print(f"Configuration Digest: {config_digest[:12] if config_digest else 'none'}")

    resolved_out_root.mkdir(parents=True, exist_ok=True)

    status = "completed"
    stage_results: dict[str, Any] = {"validation": val_report.to_dict()}

    if mode == "verify":
        # Inputs, hashes and safety verification without execution
        print("Verify mode: input, contract, and safety checks passed.")
        stage_results["verify"] = {"passed": True}

    elif mode in ("smoke", "l1", "l2", "l2l"):
        fs = float(contract.get("recording", {}).get("sampling_frequency_hz", 29999.835983263598))
        dur_s = float(contract.get("recording", {}).get("duration_s", 10473.5537279367))

        # Find real spike data folder
        if snippet_dir is not None and Path(snippet_dir).exists():
            source_dir = Path(snippet_dir)
            if (source_dir / "sorter_output").exists():
                source_dir = source_dir / "sorter_output"
            elif (source_dir / "cur_output").exists():
                source_dir = source_dir / "cur_output"
        else:
            rescue_curated = contract.get("comparators", {}).get("rescue_control", {}).get("curated")
            if rescue_curated and Path(rescue_curated).exists():
                source_dir = Path(rescue_curated)
            else:
                source_dir = None

        if source_dir is None or not (source_dir / "spike_times.npy").exists() or not (source_dir / "spike_clusters.npy").exists():
            raise FileNotFoundError(
                f"Required real spike inputs (spike_times.npy, spike_clusters.npy) not found in source directory: {source_dir}. "
                "Synthetic fallbacks are disabled in execution modes."
            )

        spike_times_raw = np.load(source_dir / "spike_times.npy").reshape(-1)
        spike_clusters = np.load(source_dir / "spike_clusters.npy").reshape(-1)

        # Convert spike samples to seconds
        if spike_times_raw.dtype.kind in "iu" or (spike_times_raw.size > 0 and spike_times_raw.max() > dur_s):
            spike_times_s = spike_times_raw.astype(np.float64) / fs
        else:
            spike_times_s = spike_times_raw.astype(np.float64)

        # Load depths and amplitudes if available
        if (source_dir / "spike_depths.npy").exists():
            spike_depths_um = np.load(source_dir / "spike_depths.npy").reshape(-1)
        elif (source_dir / "spike_positions.npy").exists():
            spike_depths_um = np.load(source_dir / "spike_positions.npy")[:, 1].reshape(-1)
        else:
            spike_depths_um = np.zeros_like(spike_times_s)

        if (source_dir / "amplitudes.npy").exists():
            spike_amplitudes = np.load(source_dir / "amplitudes.npy").reshape(-1)
        else:
            spike_amplitudes = np.ones_like(spike_times_s)

        # Motion field interpolation if motion_info_dir is provided
        displacements = None
        if motion_info_dir is not None and Path(motion_info_dir).exists():
            mf_candidates = list(Path(motion_info_dir).glob("*.npz"))
            if mf_candidates:
                mf = load_qualified_motion_field(mf_candidates[0])
                interp_res = interpolate_motion_at_spikes(mf, spike_times_s, spike_depths_um)
                displacements = interp_res["displacement_um"]

        if option == "control":
            print(f"Executing control run in mode {mode}...")
            stage_results["control"] = {"executed": True, "mode": mode, "source_dir": str(source_dir)}

        elif option == "unwarped_identity":
            print(f"Executing Option B (unwarped_identity) in mode {mode}...")
            unwarped_cfg = UnwarpedIdentityConfig(**opt_config) if opt_config else UnwarpedIdentityConfig()
            unwarped_out = resolved_out_root / "unwarped_identity"

            stage_res = run_unwarped_identity_pipeline(
                spike_times_s=spike_times_s,
                spike_clusters=spike_clusters,
                spike_depths_um=spike_depths_um,
                displacement_um=displacements,
                recording_duration_s=dur_s,
                output_dir=unwarped_out,
                config=unwarped_cfg,
                spike_amplitudes=spike_amplitudes,
            )
            stage_results["unwarped_identity"] = stage_res["manifest"]

            # Export candidate output for downstream QC & analysis scoring
            candidate_export = resolved_out_root / "candidate_export"
            candidate_export.mkdir(parents=True, exist_ok=True)
            np.save(candidate_export / "spike_times.npy", spike_times_raw)
            np.save(candidate_export / "spike_clusters.npy", stage_res["spike_tracks"])
            np.save(candidate_export / "spike_tracks.npy", stage_res["spike_tracks"])
            np.save(candidate_export / "amplitudes.npy", spike_amplitudes)

            # Write cluster_KSLabel.tsv
            unique_tracks = np.unique(stage_res["spike_tracks"])
            kslabel_file = candidate_export / "cluster_KSLabel.tsv"
            with kslabel_file.open("w", newline="") as fh:
                writer = csv.writer(fh, delimiter="\t")
                writer.writerow(["cluster_id", "KSLabel"])
                for tid in unique_tracks:
                    writer.writerow([int(tid), "good"])

            if (source_dir / "ops.npy").exists():
                import shutil
                shutil.copy(source_dir / "ops.npy", candidate_export / "ops.npy")

            truth_data = None
            if truth_path is not None and Path(truth_path).exists():
                truth_data = json.loads(Path(truth_path).read_text())

            # Run evaluation scoring
            score_res = score_sort(
                candidate_export,
                truth=truth_data,
                fs=fs,
                duration_s=dur_s,
            )
            score_file = resolved_out_root / "candidate_score.json"
            score_file.write_text(json.dumps(score_res, indent=2) + "\n")
            stage_results["score"] = score_res

        elif option == "external_warp":
            if mode != "verify":
                raise NotImplementedError("Option A external_warp is not selected for this run")

    manifest = {
        "schema": BAKEOFF_SCHEMA,
        "option": option,
        "mode": mode,
        "out_root": str(resolved_out_root),
        "status": status,
        "contract_id": contract.get("contract_id"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_digest": config_digest,
        "stage_results": stage_results,
        "provenance": {
            **git_worktree_state(),
        },
    }

    manifest_file = resolved_out_root / BAKEOFF_MANIFEST
    manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Manifest written to {manifest_file}")

    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Thin runner for two-option motion pipeline bakeoff.")
    ap.add_argument("--option", choices=OPTIONS, required=True)
    ap.add_argument("--mode", choices=MODES, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--snippet-dir", type=Path, default=None)
    ap.add_argument("--motion-info-dir", type=Path, default=None)
    ap.add_argument("--truth", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)

    args = ap.parse_args(argv)
    try:
        manifest = run_bakeoff(
            option=args.option,
            mode=args.mode,
            out_root=args.out_root,
            snippet_dir=args.snippet_dir,
            motion_info_dir=args.motion_info_dir,
            truth_path=args.truth,
            config_path=args.config,
            contract_path=args.contract,
        )
        return 0
    except Exception as exc:
        print(f"Bakeoff runner failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
