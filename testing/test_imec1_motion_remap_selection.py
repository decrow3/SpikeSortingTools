import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SELECTION = ROOT / "configs/luke0804_imec1_motion_remap_selection.v1.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_selected_imec1_remap_is_bound_to_tested_artifacts_and_result():
    selected = json.loads(SELECTION.read_text())
    assert selected["status"] == "selected_on_w2_with_cached_w3_lattice_transfer_support"
    assert selected["scope"]["outer_holdout_accessed"] is False
    assert selected["scope"]["rf_evaluated"] is False

    for key in ("extractor_source", "tested_recording_descriptor", "tested_sort_config"):
        path = ROOT / selected["implementation"][key]
        assert sha256(path) == selected["implementation"][f"{key}_sha256"]
    for key in (
        "scorecard_contract",
        "arms_manifest",
        "scorecard_result",
        "measurement_audit",
        "cached_qc_result",
        "w3_new_state_kernel_audit",
        "w3_cached_lattice_transfer_audit",
    ):
        path = ROOT / selected["evidence"][key]
        assert sha256(path) == selected["evidence"][f"{key}_sha256"]

    result = json.loads((ROOT / selected["evidence"]["scorecard_result"]).read_text())
    decisions = {
        (row["first_arm"], row["second_arm"]): row["decision"]
        for row in result["comparisons"]
    }
    assert decisions[("rounded_exact_dd", "rounded_kriging_bridge")] == "practical_equivalence"
    assert decisions[("rounded_kriging_bridge", "unrounded_kriging")] == (
        "meaningful_first_arm_advantage"
    )
    assert selected["routing"]["em2_stage_2_trigger_met"] is False


def test_selected_graph_preserves_production_order_and_pinned_kernel():
    selected = json.loads(SELECTION.read_text())
    order = selected["selection"]["production_order"]
    assert order.index("interpolate imec1.ap#AP191") < order.index(
        "round rigid displacement to the accepted 40-um lattice"
    )
    assert order.index("round rigid displacement to the accepted 40-um lattice") < order.index(
        "apply SpikeInterface kriging remap"
    )
    assert order[-1] == "crop to the 182 target sites"
    assert selected["selection"]["kriging"] == {
        "method": "kriging",
        "sigma_um": 20.0,
        "p": 1.0,
        "num_closest": 4,
        "bad_channel_sigma_um": 20.0,
        "bad_channel_p": 1.3,
    }
