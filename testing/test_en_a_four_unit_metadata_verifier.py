import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).with_name("en_a_four_unit_metadata_verifier.py")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_end_to_end_metadata_only_verifier(tmp_path):
    states = [-200, -80, -160, -240, -120, 0]
    centers = np.arange(len(states), dtype=float) + 0.5
    field = tmp_path / "field.npz"
    np.savez(field, time_s=centers, displacement_um=np.asarray(states)[:, None])
    geometry = np.c_[np.zeros(4), np.arange(4) * 1000.0]
    channel_positions = tmp_path / "channel_positions.npy"
    np.save(channel_positions, geometry)

    targets = {
        278: (-200, -80), 271: (-160, -240),
        588: (-120, -160), 445: (-200, -120),
    }
    state_base = {state: i * 1000 for i, state in enumerate(states)}
    events = []
    for unit, pair_states in targets.items():
        for state in pair_states:
            base = state_base[state]
            events.extend((base + 10, unit, unit, 2000.0) for _ in range(1))
            events.extend((base + 20, unit, unit, 2000.0) for _ in range(1))
            events.extend((base + 500, unit, unit, 2000.0) for _ in range(1))
        events.append((state_base[0] + 500, unit, unit, 2000.0))
        events.append((state_base[0] + 900, unit, unit, 2000.0))
    events.sort()
    times = np.array([e[0] for e in events], dtype=np.int64)
    clusters = np.array([e[1] for e in events], dtype=np.int32)
    assigned = np.array([e[2] for e in events], dtype=np.int32)
    positions = np.c_[np.zeros(len(events)), [e[3] for e in events]].astype(np.float32)
    paths = {}
    for name, value in (("spike_times", times), ("spike_clusters", clusters),
                        ("spike_templates", assigned), ("spike_positions", positions)):
        paths[name] = tmp_path / f"{name}.npy"; np.save(paths[name], value)
    bank = np.zeros((589, 61, 4), dtype=np.float32)
    for unit in targets:
        bank[unit, 30, unit % 4] = float(unit)
    paths["templates"] = tmp_path / "templates.npy"; np.save(paths["templates"], bank)
    paths["ops"] = tmp_path / "ops.npy"
    np.save(paths["ops"], {"fs": 1000.0, "nt": 61, "chanMap": np.arange(4),
                            "xc": geometry[:, 0], "yc": geometry[:, 1]})
    paths["channel_positions"] = channel_positions
    paths["field"] = field

    contract = tmp_path / "selection.csv"
    rules = {
        "short_pair": "earliest original adjacent same-unit both-supported pair with lag 1..29 in this state and one contiguous segment",
        "ordinary_event": "earliest same-unit event in this state with both same-unit neighboring lags at least 300 samples and not a selected pair endpoint",
        "q0_identity_control": "earliest q0 same-unit event with both same-unit neighboring lags at least 300 samples",
    }
    rows = []
    for unit, pair_states in targets.items():
        rows.extend([
            {"slot_id": f"u{unit}_pair_primary", "unit_id": unit, "role": "short_pair", "state_um": pair_states[0], "selection_rule": rules["short_pair"]},
            {"slot_id": f"u{unit}_pair_secondary", "unit_id": unit, "role": "short_pair", "state_um": pair_states[1], "selection_rule": rules["short_pair"]},
            {"slot_id": f"u{unit}_ordinary_primary", "unit_id": unit, "role": "ordinary_event", "state_um": pair_states[0], "selection_rule": rules["ordinary_event"]},
            {"slot_id": f"u{unit}_ordinary_secondary", "unit_id": unit, "role": "ordinary_event", "state_um": pair_states[1], "selection_rule": rules["ordinary_event"]},
            {"slot_id": f"u{unit}_q0_control", "unit_id": unit, "role": "q0_identity_control", "state_um": 0, "selection_rule": rules["q0_identity_control"]},
        ])
    with contract.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    paths["selection_contract"] = contract

    command = [sys.executable, str(SCRIPT)]
    for name, path in paths.items():
        option = name.replace("_", "-")
        command.extend((f"--{option}", str(path), f"--{option}-sha256", digest(path)))
    output = tmp_path / "output"
    command.extend(("--expected-source-sha256", digest(SCRIPT),
                    "--sampling-frequency-hz", "1000", "--output", str(output)))
    subprocess.run(command, check=True)
    evidence = json.loads((output / "EVIDENCE.json").read_text())
    assert evidence["selection_rows"] == 20
    assert evidence["recording_binary_opened"] is False
    assert all(m["template_distribution"] == {str(m["unit_id"]): m["event_count"]}
               for m in evidence["unit_template_mappings"])
    selected = list(csv.DictReader((output / "SELECTION_EVIDENCE.csv").open()))
    assert len(selected) == 20
    assert all(row["status"] == "resolved" for row in selected)
    assert (output / "COMPLETE.json").is_file()

