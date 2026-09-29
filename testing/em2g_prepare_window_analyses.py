#!/usr/bin/env python3
"""Bind completed full-sort slices to frozen W2/W3 analysis rules."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def packet_values(
    master: dict[str, object],
    slice_result: dict[str, object],
    master_sha256: str,
    output: Path,
) -> dict[str, dict[str, object]]:
    packets = {}
    score = master["analyses"]["scorecard"]
    overlap = master["analyses"]["event_overlap"]
    for name, specification in master["windows"].items():
        candidate = slice_result["windows"][name]
        reference = specification["exact_reference"]
        scorecard = {
            "schema": f"em2g-{name.lower()}-full-training-scorecard-v1",
            "result_schema": f"em2g-{name.lower()}-full-training-scorecard-result-v1",
            "frozen_before_candidate_scoring": True,
            "master_contract_sha256": master_sha256,
            "scope": {
                "dataset": f"Luke0804 imec1 {name}",
                "sampling_frequency_hz": master["scope"]["sampling_frequency_hz"],
                "frames_half_open": specification["frames_half_open"],
                "outer_holdout_access": False,
                "rf_analysis": False,
            },
            "inputs": {
                f"{key}_sha256": value["sha256"]
                for key, value in master["domain_inputs"].items()
            },
            "eligibility": {
                "unit_scope": "arm-local assigned units; no cross-arm unit matching",
                "minimum_flat_events": score["eligibility_minimum_flat_events"],
                "bootstrap_policy": f"freeze each arm's full-{name} eligible unit set for every draw",
                "pseudocount": 0.5,
            },
            "bootstrap": {
                "draws": score["bootstrap_draws"],
                "seed": score["bootstrap_seed"],
                "block_seconds": score["block_seconds"],
                "minimum_valid_draws": 1900,
            },
            "comparisons": {
                "primary": ["full_session_rounded_kriging", "rounded_exact_lattice"],
                "secondary": [],
                "bridge": [],
                "delta_orientation": "first arm rho minus second arm rho",
            },
            "guardrails": {
                "yield": {
                    "directional_max_first_arm_relative_loss": score[
                        "equivalence_yield_relative_margin"
                    ],
                    "equivalence_max_absolute_relative_difference": score[
                        "equivalence_yield_relative_margin"
                    ],
                },
                "segment_safe_short_isi": {
                    "directional_max_first_arm_increase": score[
                        "equivalence_isi_absolute_margin"
                    ],
                    "equivalence_max_absolute_difference": score[
                        "equivalence_isi_absolute_margin"
                    ],
                    "apply_in_each_domain": True,
                },
            },
            "decisions": {
                "meaningful_first_arm_advantage": "delta_rho >= 0.05, 95% CI lower bound > 0, and directional yield/ISI guardrails pass",
                "practical_equivalence": "entire 95% delta-rho CI lies within [-0.05, 0.05] and symmetric yield/ISI guardrails pass",
                "otherwise": "mixed or inconclusive",
                "undefined": "report unresolved rather than substituting zero",
            },
            "interpretation": master["interpretation"],
        }
        scorecard_path = output / f"{name}_scorecard_contract.json"
        arms = {
            "schema": f"em2g-{name.lower()}-scorecard-arms-v1",
            "contract_path": str(scorecard_path.resolve()),
            "contract_sha256": None,
            **master["domain_inputs"],
            "arms": {
                "rounded_exact_lattice": reference,
                "full_session_rounded_kriging": {
                    "path": candidate["path"],
                    "sha256": candidate["sha256"],
                    "source_full_sorting_sha256": slice_result["source_sorting"]["sha256"],
                },
            },
        }
        event_config = {
            "schema": f"em2g-{name.lower()}-full-training-event-overlap-v1",
            "frozen_before_pair_scoring": True,
            "master_contract_sha256": master_sha256,
            "scope": {
                "dataset": f"Luke0804 imec1 {name}",
                "clock": "window-local samples",
                "rf": False,
                "voltage_read": False,
                "outer_holdout": False,
            },
            "reference_arm": {"name": "rounded_exact_lattice", **reference},
            "comparators": [
                {
                    "name": "full_session_rounded_kriging",
                    "role": "selected operator trained over the full session",
                    "path": candidate["path"],
                    "sha256": candidate["sha256"],
                }
            ],
            "matching": {
                "primary_tolerance_samples": overlap["primary_tolerance_samples"],
                "exact_sample_sensitivity": overlap["exact_sample_sensitivity"],
                "minimum_candidate_matches": overlap["minimum_candidate_matches"],
                "score": "F1 = 2 * one_to_one_matches / (reference_events + comparator_events)",
                "pairing": overlap["pairing"],
                "ambiguous": f"best F1 minus second-best F1 <= {overlap['ambiguity_margin_f1']}",
                "negative_labels": "exclude",
                "cross_arm_integer_ids": "never treated as identities",
            },
            "interpretation": master["interpretation"],
        }
        packets[name] = {
            "scorecard": scorecard,
            "scorecard_path": scorecard_path,
            "arms": arms,
            "event_overlap": event_config,
        }
    return packets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--slices", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    master = json.loads(args.contract.read_text())
    slices = json.loads(args.slices.read_text())
    contract_sha = sha256(args.contract)
    if slices.get("status") != "complete" or slices.get("contract_sha256") != contract_sha:
        raise RuntimeError("slice result does not bind the frozen master contract")
    for value in master["domain_inputs"].values():
        if sha256(Path(value["path"])) != value["sha256"]:
            raise RuntimeError("domain input hash differs")
    for name, specification in master["windows"].items():
        reference = specification["exact_reference"]
        if sha256(Path(reference["path"])) != reference["sha256"]:
            raise RuntimeError(f"{name} reference hash differs")
        candidate = slices["windows"][name]
        if sha256(Path(candidate["path"])) != candidate["sha256"]:
            raise RuntimeError(f"{name} candidate slice hash differs")

    args.output.mkdir(parents=True)
    packets = packet_values(master, slices, contract_sha, args.output)
    products = []
    for name, packet in packets.items():
        scorecard_path = packet["scorecard_path"]
        atomic_json(scorecard_path, packet["scorecard"])
        packet["arms"]["contract_sha256"] = sha256(scorecard_path)
        arms_path = args.output / f"{name}_scorecard_arms.json"
        event_path = args.output / f"{name}_event_overlap.json"
        atomic_json(arms_path, packet["arms"])
        atomic_json(event_path, packet["event_overlap"])
        for path in (scorecard_path, arms_path, event_path):
            products.append(
                {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            )
    manifest_path = args.output / "MANIFEST.json"
    atomic_json(manifest_path, {"products": products})
    atomic_json(
        args.output / "COMPLETE.json",
        {"status": "complete", "manifest_sha256": sha256(manifest_path)},
    )


if __name__ == "__main__":
    main()
