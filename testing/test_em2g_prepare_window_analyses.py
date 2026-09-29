import hashlib
import json
from pathlib import Path

from testing.em2g_prepare_window_analyses import packet_values


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
