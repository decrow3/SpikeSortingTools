#!/usr/bin/env python3
"""Verify and preserve the cached W3 exact-lattice transfer result."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PACKET = Path(
    "/home/huklaban5/DARTsort_experiment_scratch/ef_w3_lattice_20260928/"
    "ej_v1/analysis"
)
BINDING = PACKET.parent / "SCORE_BINDING.json"
SOURCE = PACKET.parent / "EJ_SOURCE_RECEIPT.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_packet() -> tuple[dict[str, object], dict[str, object]]:
    manifest_path = PACKET / "MANIFEST.json"
    complete = json.loads((PACKET / "COMPLETE.json").read_text())
    if complete.get("status") != "complete" or complete.get("manifest_sha256") != sha256(
        manifest_path
    ):
        raise RuntimeError("cached W3 completion receipt differs")
    manifest = json.loads(manifest_path.read_text())
    for product in manifest["products"]:
        path = PACKET / product["path"]
        if path.stat().st_size != product["bytes"] or sha256(path) != product["sha256"]:
            raise RuntimeError(f"cached W3 product differs: {path}")
    binding = json.loads(BINDING.read_text())
    for name, item in binding["bindings"].items():
        path = Path(item["path"])
        if sha256(path) != item["sha256"]:
            raise RuntimeError(f"cached W3 sorting differs: {name}")
    return json.loads((PACKET / "DECISION.json").read_text()), binding


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-state-audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    decision, binding = validate_packet()
    state_audit = args.new_state_audit.resolve()
    state = json.loads(state_audit.read_text())
    if not state.get("new_state_inside_w2_envelope_all_metrics"):
        raise RuntimeError("new W3 rounded state is outside the W2 kernel envelope")
    if decision.get("verdict") != "ADVANCE" or decision.get("rf_used") is not False:
        raise RuntimeError("cached W3 decision is not the accepted no-RF transfer result")
    args.output.mkdir(parents=True)
    audit = {
        "schema": "em2c-cached-w3-lattice-transfer-audit-v1",
        "status": "complete",
        "cached_packet": {
            "path": str(PACKET),
            "manifest_sha256": sha256(PACKET / "MANIFEST.json"),
            "complete_sha256": sha256(PACKET / "COMPLETE.json"),
            "decision_sha256": sha256(PACKET / "DECISION.json"),
            "source_receipt_sha256": sha256(SOURCE),
            "score_binding_sha256": sha256(BINDING),
        },
        "sorting_bindings": binding["bindings"],
        "decision": decision,
        "new_state_kernel_audit": {
            "path": str(state_audit),
            "sha256": sha256(state_audit),
            "new_w3_states_um": state["new_w3_states_um"],
            "inside_w2_metric_envelope": state[
                "new_state_inside_w2_envelope_all_metrics"
            ],
        },
        "scope": {
            "new_voltage_read": False,
            "new_sort_launched": False,
            "rf_used": False,
            "window_status": decision["window_status"],
        },
        "interpretation": (
            "Cached W3 exact-lattice transfer supports the rounded-field choice. "
            "W2 practical equivalence and the no-voltage W3 state audit support, but do "
            "not directly demonstrate, rounded-kriging sorting equivalence on W3."
        ),
    }
    (args.output / "AUDIT.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    products = [
        {
            "path": "AUDIT.json",
            "bytes": (args.output / "AUDIT.json").stat().st_size,
            "sha256": sha256(args.output / "AUDIT.json"),
        }
    ]
    (args.output / "MANIFEST.json").write_text(
        json.dumps({"products": products}, indent=2, sort_keys=True) + "\n"
    )
    (args.output / "COMPLETE.json").write_text(
        json.dumps(
            {"status": "complete", "manifest_sha256": sha256(args.output / "MANIFEST.json")},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
