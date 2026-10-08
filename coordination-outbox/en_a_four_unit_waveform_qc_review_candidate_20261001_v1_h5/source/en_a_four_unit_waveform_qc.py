#!/usr/bin/env python3
"""Review-gated QC for two already-retained four-unit waveform archives."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def exact_mapping(geometry: np.ndarray, shift_um: float) -> np.ndarray:
    """Output target -> exactly translated source; -1 means force-zero."""
    g = np.asarray(geometry, dtype=np.float64)
    if g.shape != (384, 2) or not np.isfinite(g).all():
        raise ValueError("geometry must be finite (384,2)")
    out = np.full(384, -1, dtype=np.int64)
    for target, (x, y) in enumerate(g):
        hit = np.flatnonzero(np.all(np.isclose(g, [x, y + shift_um], rtol=0, atol=1e-6), axis=1))
        if hit.size > 1:
            raise ValueError("ambiguous exact mapping")
        if hit.size == 1:
            out[target] = int(hit[0])
    if np.unique(out[out >= 0]).size != np.count_nonzero(out >= 0):
        raise ValueError("mapping is not injective")
    return out


def remap_array(original: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    out = np.zeros_like(original)
    valid = mapping >= 0
    out[valid] = original[mapping[valid]]
    return out


def stage_arrays(raw: np.ndarray) -> dict[str, np.ndarray]:
    x = np.asarray(raw, dtype=np.float64)
    centered = x - x.mean(axis=1, keepdims=True)
    car = centered - np.median(centered, axis=0, keepdims=True)
    return {"raw_counts": x, "temporal_centered": centered, "car_centered": car}


def frozen_windows(role: str, lag: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    expected = 121 + (lag if role == "short_pair" else 0)
    if width != expected:
        raise ValueError("member width differs from frozen rule")
    effective_lag = lag if role == "short_pair" else 0
    signal = np.arange(40, 101 + effective_lag)
    baseline = np.r_[np.arange(0, 30), np.arange(111 + effective_lag, 121 + effective_lag)]
    return signal, baseline


def per_channel_metrics(values: np.ndarray, signal: np.ndarray, baseline: np.ndarray) -> dict[str, np.ndarray]:
    b = values[:, baseline]
    s = values[:, signal]
    median = np.median(b, axis=1)
    sigma = 1.4826 * np.median(np.abs(b - median[:, None]), axis=1)
    denom = np.maximum(sigma, 1.0)
    ptp = np.ptp(s, axis=1)
    rms_z = np.sqrt(np.mean((s - median[:, None]) ** 2, axis=1)) / denom
    return {"baseline_median": median, "robust_noise_sigma": sigma,
            "signal_ptp": ptp, "ptp_snr": ptp / denom, "signal_rms_z": rms_z}


def local_channels(geometry: np.ndarray, anchor: int, radius_um: float) -> np.ndarray:
    return np.flatnonzero(np.linalg.norm(geometry - geometry[anchor], axis=1) <= radius_um + 1e-9)


def shape_correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    x = np.asarray(a, dtype=np.float64).ravel(); y = np.asarray(b, dtype=np.float64).ravel()
    x -= x.mean(); y -= y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return None if denom == 0 else float(np.dot(x, y) / denom)


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 20 or len({r["slot_id"] for r in rows}) != 20:
        raise ValueError("expected 20 unique frozen slots")
    return rows


def validate_static(config: dict[str, Any], config_path: Path, expected_hash: str) -> None:
    if sha256(config_path) != expected_hash:
        raise ValueError("config hash mismatch")
    if config.get("schema") != "en-a-four-unit-waveform-qc-v1":
        raise ValueError("wrong schema")
    if config["axes_and_scale"] != {
        **config["axes_and_scale"], "channels": 384, "dtype": "int16"
    }:
        raise ValueError("axis contract mismatch")
    if config["resources"]["recording_read_calls_max"] != 0:
        raise ValueError("recording reads must be zero")


def validate_metadata_only(config: dict[str, Any]) -> dict[str, Any]:
    """Verify selection/geometry/channel identities without touching outcomes."""
    for key in ("selection_receipt", "channel_positions"):
        item = config["inputs"][key]
        if sha256(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"metadata hash mismatch: {key}")
    rows = load_rows(Path(config["inputs"]["selection_receipt"]["path"]))
    geometry = np.load(config["inputs"]["channel_positions"]["path"], allow_pickle=False)
    if geometry.shape != (384, 2):
        raise ValueError("geometry shape mismatch")
    for unit_text, identity in config["unit_channel_identities"].items():
        anchor = int(identity["corrected_anchor_channel"])
        if not np.array_equal(geometry[anchor], np.asarray(identity["corrected_anchor_xy_um"])):
            raise ValueError(f"corrected anchor coordinate mismatch: {unit_text}")
        for state_text, expected_source in identity["states_to_original_anchor"].items():
            mapping = exact_mapping(geometry, float(state_text))
            if int(mapping[anchor]) != int(expected_source):
                raise ValueError(f"state/source anchor mismatch: {unit_text}/{state_text}")
    roles = [r["role"] for r in rows]
    if roles.count("short_pair") != 8 or roles.count("ordinary_event") != 8 or roles.count("q0_identity_control") != 4:
        raise ValueError("selection role counts mismatch")
    return {"selection_slots": 20, "geometry_shape": [384, 2], "unit_channel_identities": 4}


def execute(config: dict[str, Any], output: Path) -> None:
    gates = config["gates"]
    if gates["execution_enabled"] is not True or gates["outcome_archive_open_authorized"] is not True:
        raise PermissionError("review-pending contract is execution-disabled")
    if not isinstance(gates["independent_h1_review_manifest_sha256"], str):
        raise PermissionError("independent H1 review binding is absent")
    if output.exists():
        raise FileExistsError("fresh output root required")
    started = time.monotonic()
    for item in config["inputs"].values():
        if sha256(Path(item["path"])) != item["sha256"]:
            raise ValueError("input hash mismatch")
    rows = load_rows(Path(config["inputs"]["selection_receipt"]["path"]))
    geometry = np.load(config["inputs"]["channel_positions"]["path"], allow_pickle=False)
    archive_paths = [Path(config["inputs"][k]["path"]) for k in ("original_archive", "corrected_archive")]
    with np.load(archive_paths[0], allow_pickle=False) as left, np.load(archive_paths[1], allow_pickle=False) as right:
        if set(left.files) != {r["slot_id"] for r in rows} or set(right.files) != set(left.files):
            raise ValueError("archive keys differ from frozen slots")
        arrays = {"original": {k: left[k] for k in left.files}, "corrected_A": {k: right[k] for k in right.files}}
    numeric_bytes = sum(v.nbytes for rep in arrays.values() for v in rep.values())
    if numeric_bytes > config["resources"]["archive_numeric_bytes_max"]:
        raise RuntimeError("archive numeric-byte ceiling exceeded")
    output.mkdir(parents=False); (output / "figures").mkdir()
    channel_rows: list[dict[str, Any]] = []; slot_rows: list[dict[str, Any]] = []; comparisons: list[dict[str, Any]] = []
    visibility: dict[int, list[bool]] = {271: [], 278: [], 445: [], 588: []}
    for row in rows:
        slot = row["slot_id"]; unit = int(row["unit_id"]); state = int(row["state_um"]); lag = int(row["lag_samples"])
        width = 121 + (lag if row["role"] == "short_pair" else 0)
        for rep in arrays:
            a = arrays[rep][slot]
            if a.shape != (384, width) or a.dtype != np.int16:
                raise ValueError(f"unexpected member shape/dtype: {rep}/{slot}")
        mapping = exact_mapping(geometry, state)
        predicted = remap_array(arrays["original"][slot], mapping)
        exact = bool(np.array_equal(predicted, arrays["corrected_A"][slot]))
        comparisons.append({"slot_id": slot, "comparison": "exact_remap", "passed": exact,
                            "equality_fraction": float(np.mean(predicted == arrays["corrected_A"][slot]))})
        if state == 0:
            comparisons.append({"slot_id": slot, "comparison": "q0_identity",
                                "passed": bool(np.array_equal(arrays["original"][slot], arrays["corrected_A"][slot])),
                                "equality_fraction": float(np.mean(arrays["original"][slot] == arrays["corrected_A"][slot]))})
        if row["role"] == "ordinary_event" and state != 0:
            for name, trial in (("wrong_sign", remap_array(arrays["original"][slot], exact_mapping(geometry, -state))),
                                ("no_shift", arrays["original"][slot])):
                fraction = float(np.mean(trial == arrays["corrected_A"][slot]))
                comparisons.append({"slot_id": slot, "comparison": name, "passed": fraction < 0.99,
                                    "equality_fraction": fraction})
        signal, baseline = frozen_windows(row["role"], lag, width)
        for rep in ("original", "corrected_A"):
            anchor = (int(config["unit_channel_identities"][str(unit)]["states_to_original_anchor"][str(state)])
                      if rep == "original" else int(config["unit_channel_identities"][str(unit)]["corrected_anchor_channel"]))
            local = local_channels(geometry, anchor, config["channel_domains"]["local_radius_um"])
            for stage, values in stage_arrays(arrays[rep][slot]).items():
                metrics = per_channel_metrics(values, signal, baseline)
                peak = int(np.flatnonzero(metrics["ptp_snr"] == metrics["ptp_snr"].max())[0])
                for channel in range(384):
                    channel_rows.append({"slot_id": slot, "unit_id": unit, "role": row["role"], "state_um": state,
                                         "representation": rep, "stage": stage, "channel": channel,
                                         **{k: float(v[channel]) for k, v in metrics.items()}})
                summary = {"slot_id": slot, "unit_id": unit, "role": row["role"], "state_um": state,
                           "representation": rep, "stage": stage, "anchor_channel": anchor, "peak_channel": peak,
                           "peak_distance_um": float(np.linalg.norm(geometry[peak] - geometry[anchor])),
                           "local_max_ptp_snr": float(metrics["ptp_snr"][local].max()),
                           "local_median_ptp_snr": float(np.median(metrics["ptp_snr"][local]))}
                slot_rows.append(summary)
                if rep == "corrected_A" and stage == "car_centered" and row["role"] != "short_pair":
                    visibility[unit].append(summary["local_max_ptp_snr"] >= 6 and summary["peak_distance_um"] <= 80)
        valid = mapping >= 0
        corr = shape_correlation(stage_arrays(arrays["corrected_A"][slot])["car_centered"][valid][:, signal],
                                 stage_arrays(arrays["original"][slot])["car_centered"][mapping[valid]][:, signal])
        comparisons.append({"slot_id": slot, "comparison": "car_shape_correlation", "passed": corr is not None,
                            "equality_fraction": "", "correlation": corr})
    integrity = all(bool(x["passed"]) for x in comparisons if x["comparison"] != "car_shape_correlation")
    usable = {str(k): sum(v) >= 2 for k, v in visibility.items()}
    n_usable = sum(usable.values())
    verdict = "PASS" if integrity and n_usable >= 3 else ("CAUTION" if integrity and n_usable == 2 else "FAIL")
    def write_csv(name: str, records: list[dict[str, Any]]) -> None:
        fields = sorted({k for r in records for k in r})
        with (output / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(records)
    write_csv("CHANNEL_METRICS.csv", channel_rows); write_csv("SLOT_SUMMARIES.csv", slot_rows); write_csv("COMPARISONS.csv", comparisons)
    atomic_json(output / "VALIDATION.json", {"archives": 2, "members_per_archive": 20, "numeric_bytes": numeric_bytes,
                                               "recording_reads": 0, "elapsed_seconds": time.monotonic() - started})
    atomic_json(output / "DECISION.json", {"verdict": verdict, "integrity_controls_passed": integrity,
                                             "unit_usable": usable, "usable_unit_count": n_usable,
                                             "scope": config["decision_scope"], "prohibited_claims": config["prohibited_claims"]})
    # Direct single-event waveforms at the frozen representation-specific anchors.
    import matplotlib.pyplot as plt
    for unit in visibility:
        singles = [r for r in rows if int(r["unit_id"]) == unit and r["role"] != "short_pair"]
        fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
        for ax, row in zip(axes, singles):
            slot = row["slot_id"]; state = int(row["state_um"])
            corrected_anchor = int(config["unit_channel_identities"][str(unit)]["corrected_anchor_channel"])
            original_anchor = int(config["unit_channel_identities"][str(unit)]["states_to_original_anchor"][str(state)])
            original = stage_arrays(arrays["original"][slot])["car_centered"][original_anchor]
            corrected = stage_arrays(arrays["corrected_A"][slot])["car_centered"][corrected_anchor]
            t_ms = (np.arange(original.size) - 60) * 1000 / config["axes_and_scale"]["sampling_frequency_hz"]
            ax.plot(t_ms, original, label=f"original ch{original_anchor}", lw=1)
            ax.plot(t_ms, corrected, label=f"corrected ch{corrected_anchor}", lw=1, ls="--")
            ax.axvline(0, color="0.7", lw=0.6); ax.set_title(f"{row['role']} q={state}")
            ax.set_xlabel("ms from event"); ax.legend(fontsize=6)
        axes[0].set_ylabel("CAR-centered ADC counts")
        fig.suptitle(f"Unit {unit}: frozen anchor waveforms")
        fig.tight_layout(); fig.savefig(output / "figures" / f"unit_{unit}.png", dpi=150); plt.close(fig)
    if time.monotonic() - started > config["resources"]["wall_seconds_max"]:
        raise RuntimeError("wall-time ceiling exceeded")
    if sum(p.stat().st_size for p in output.rglob("*") if p.is_file()) > config["resources"]["persistent_output_bytes_max"]:
        raise RuntimeError("persistent-output ceiling exceeded")
    atomic_json(output / "COMPLETE.json", {"status": "complete", "verdict": verdict, "recording_reads": 0})


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--config", required=True, type=Path)
    p.add_argument("--expected-config-sha256", required=True); p.add_argument("--phase", choices=("preflight", "run"), required=True)
    a = p.parse_args(); config = json.loads(a.config.read_text()); validate_static(config, a.config, a.expected_config_sha256)
    if a.phase == "preflight":
        metadata = validate_metadata_only(config)
        print(json.dumps({"status": "review_pending_execution_disabled", "outcome_archives_opened": 0,
                          "metadata": metadata}, sort_keys=True)); return
    execute(config, Path(config["output_schema"]["output_root"]))


if __name__ == "__main__":
    main()
