import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from testing.em2g_prepare_window_analyses import main, packet_values


ROOT = Path(__file__).resolve().parents[1]


def test_packet_values_preserve_frozen_rules_and_training_context():
    master = json.loads((ROOT / "configs/em2g_full_slice_validation.v1.json").read_text())
    fake = {
        "source_sorting": {"sha256": "f" * 64},
        "windows": {
            name: {"path": f"/tmp/{name}.npz", "sha256": name.lower().ljust(64, "0")}
            for name in master["windows"]
        },
    }
    master_sha = hashlib.sha256(
        (ROOT / "configs/em2g_full_slice_validation.v1.json").read_bytes()
    ).hexdigest()
    packets = packet_values(master, fake, master_sha, Path("/tmp/packets"))
    assert set(packets) == {"W2", "W3"}
    for name, packet in packets.items():
        scorecard = packet["scorecard"]
        assert scorecard["frozen_before_candidate_scoring"] is True
        assert scorecard["master_contract_sha256"] == master_sha
        assert scorecard["comparisons"]["primary"] == [
            "full_session_rounded_kriging",
            "rounded_exact_lattice",
        ]
        assert scorecard["bootstrap"]["draws"] == 2000
        assert scorecard["guardrails"]["yield"][
            "equivalence_max_absolute_relative_difference"
        ] == 0.05
        overlap = packet["event_overlap"]
        assert overlap["matching"]["primary_tolerance_samples"] == 2
        assert overlap["scope"]["rf"] is False
        assert overlap["comparators"][0]["role"] == (
            "selected operator trained over the full session"
        )


def test_packet_builder_binds_real_frozen_inputs_and_synthetic_slices(tmp_path, monkeypatch):
    contract_path = ROOT / "configs/em2g_full_slice_validation.v1.json"
    contract_sha = hashlib.sha256(contract_path.read_bytes()).hexdigest()
    windows = {}
    for name in ("W2", "W3"):
        path = tmp_path / f"{name}.npz"
        np.savez(
            path,
            times_samples=np.asarray([0], dtype=np.int64),
            labels=np.asarray([0], dtype=np.int32),
            channels=np.asarray([0], dtype=np.int16),
            sampling_frequency=np.asarray(29999.759166666667),
            geom=np.asarray([[0.0, 0.0]]),
        )
        windows[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    slices = {
        "status": "complete",
        "contract_sha256": contract_sha,
        "source_sorting": {"sha256": "a" * 64},
        "windows": windows,
    }
    slices_path = tmp_path / "slices.json"
    slices_path.write_text(json.dumps(slices))
    output = tmp_path / "packets"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "em2g-packets",
            "--contract",
            str(contract_path),
            "--slices",
            str(slices_path),
            "--output",
            str(output),
        ],
    )
    main()
    assert json.loads((output / "COMPLETE.json").read_text())["status"] == "complete"
    for name in ("W2", "W3"):
        scorecard_path = output / f"{name}_scorecard_contract.json"
        arms = json.loads((output / f"{name}_scorecard_arms.json").read_text())
        events = json.loads((output / f"{name}_event_overlap.json").read_text())
        assert arms["contract_sha256"] == hashlib.sha256(scorecard_path.read_bytes()).hexdigest()
        assert arms["arms"]["full_session_rounded_kriging"]["sha256"] == windows[name][
            "sha256"
        ]
        assert events["master_contract_sha256"] == contract_sha
        assert events["comparators"][0]["sha256"] == windows[name]["sha256"]
