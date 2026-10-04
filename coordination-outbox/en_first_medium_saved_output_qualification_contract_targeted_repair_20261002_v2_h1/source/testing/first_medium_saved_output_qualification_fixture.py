"""Dedicated entry point with no caller-selectable input for packaged fixtures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile

from pipeline.config import fingerprint
from testing.luke_amplitude_dropout_audit import read_curated_arrays
from testing.qualification_managed_launcher import launch


def _resolve_fixture_request(packet: Path) -> dict:
    request = json.loads((packet / "fixtures" / "FIXTURE_REQUEST.template.json").read_text())
    arm = request["arms"][0]
    for key in ("curated_output", "qc_dir"):
        arm[key] = str((packet / arm[key]).resolve())
    spatial = arm["spatial_provenance"]
    for key in ("path", "ops_path"):
        spatial[key] = str((packet / spatial[key]).resolve())
    dependency = spatial["source_generation"]["dependency"]
    dependency["root"] = str((packet / dependency["root"]).resolve())
    curated = Path(arm["curated_output"])
    _, hashes = read_curated_arrays(curated)
    hashes["spike_positions.npy"] = hashlib.sha256((curated / "spike_positions.npy").read_bytes()).hexdigest()
    arm["expected_identity_digest"] = fingerprint({
        "curated": str(curated.resolve()), "files": hashes,
        "labels_sha256": hashlib.sha256((curated / "cluster_KSLabel.tsv").read_bytes()).hexdigest(),
    })
    return request


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packet = Path(__file__).resolve().parents[2]
    contract = packet / "configs" / "en_first_medium_saved_output_qualification.execution_disabled.v1.json"
    request = _resolve_fixture_request(packet)
    with tempfile.TemporaryDirectory(prefix="qualification-fixture-request-") as temporary:
        request_path = Path(temporary) / "request.json"
        request_path.write_text(json.dumps(request, indent=2, sort_keys=True) + "\n")
        result = launch(
            contract, request_path, args.output,
            "testing.first_medium_saved_output_qualification_fixture_worker",
        )
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
