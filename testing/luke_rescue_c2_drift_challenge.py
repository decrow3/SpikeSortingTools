"""Phase C2 — the paired static-vs-moving injected identity challenge.

`docs/pipeline_improvement_plan.md` C2. Phase A2 established that rescue's
re-partitioning is temporally-complementary, refractory-clean template flicker
that the *rigid* DREDGE estimate does not explain. A2 could not separate
*non-rigid / fast motion* from *KS4 template competition*. **C2 separates them**:

> Inject the same waveform and the same spike train twice — once at a fixed
> depth, once translated along a known Luke-like trajectory. If static KS4
> recovers the stationary neuron cleanly but breaks the moving version into
> A→B→C, the fragmentation is the cost of the no-motion strategy.

The decisive quantity is the **drift penalty** — Δaccuracy, Δoutput-identities,
Δlabel-switches — the change caused *solely* by motion.

**Status: diagnostic.** This reuses the discovery-cohort donor templates from
`luke_injected_ground_truth_pilot` (real reviewed imec1 neural events, qualified
against independent events). Per that scaffold's contract the results are
diagnostic, never confirmatory — which is exactly C2's role: it sets Phase D's
*direction*, it does not promote anything.

Confound control (plan C2): the static arm is drawn from a **quiet** window, so
the background's own tissue motion is minimal; the moving trajectory is imposed
on top and reported in µm and channels. Recorded here, not corrected for.

    python testing/luke_rescue_c2_drift_challenge.py --templates T01 T04 T06

Outputs to testing/outputs/luke_rescue_c2_drift_challenge/. Nothing under /mnt.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pipeline.config import fingerprint
from testing.ladder_inject import (
    channels_per_um,
    drift_penalty,
    paired_injection,
    rigid_oscillation,
    rigid_ramp,
    static_trajectory,
    write_injected_recording,
)
from testing.ladder_l1 import l1_run
from testing.ladder_score import score_sort
from testing.luke_injected_ground_truth_benchmark import validate_template

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "testing/outputs/luke_rescue_c2_drift_challenge"
LUKE_ROOT = Path("/mnt/NPX/Luke/20250804")
DONOR_TEMPLATES = (
    REPO_ROOT
    / "testing/outputs/luke_injected_ground_truth_pilot/donor_templates.npz"
)

PRESPEC = {
    "schema": "luke-rescue-c2-drift-challenge-v1",
    "frozen": "2026-09-02",
    "status": "diagnostic_discovery_cohort_reuse_not_confirmatory",
    "question": (
        "Does an injected neuron on a known Luke-like trajectory fragment where "
        "the identical static injection does not?"
    ),
    "probe": "imec1",
    "background": {
        "regime": "quiet",
        "start_s": 4080.0,
        "duration_s": 120.0,
        "channel_start": 136,
        "channel_count": 112,
    },
    "template_prep": {
        "source": "luke_injected_ground_truth_pilot donor_templates.npz",
        "time_samples": 61,
        "channel_radius": 16,
        "edge_guard_samples": 3,
        "baseline": "edge median",
    },
    "train": {"kind": "regular", "rate_hz": 6.0, "guard_s": 1.0},
    "amplitude_scale": 1.0,
    "trajectories": {
        "static": {"kind": "static"},
        "rigid_15um": {"kind": "rigid_ramp", "total_um": 15.0},
        "rigid_40um": {"kind": "rigid_ramp", "total_um": 40.0},
        "osc_20um_40s": {"kind": "rigid_oscillation", "amp_um": 20.0, "period_s": 40.0},
    },
    "drift_penalty": ["delta_accuracy", "delta_n_identities", "delta_label_switches"],
    "sanity": (
        "the static arm must recover the highest-SNR template at accuracy >= "
        "0.9; otherwise the benchmark is wrong, not the pipeline (plan C step 2)"
    ),
}

BG = PRESPEC["background"]
TP = PRESPEC["template_prep"]


def _freeze_prespec() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "prespec.json"
    if path.exists():
        if json.loads(path.read_text()) != PRESPEC:
            raise SystemExit(
                f"{path} differs from the frozen PRESPEC. C2 is run-once; "
                "delete the output tree to re-freeze."
            )
    else:
        path.write_text(json.dumps(PRESPEC, indent=2) + "\n")


def _recording_dir() -> Path:
    return LUKE_ROOT / (
        f"rescue_pipeline_results_Luke0804_V2V1_g0_{PRESPEC['probe']}/recording"
    )


def load_background():
    """Quiet-window depth strip of the accepted recording, as float32 µV."""
    from spikeinterface.core import load

    rec_dir = _recording_dir()
    manifest = json.loads((rec_dir / "rescue_recording_manifest.json").read_text())
    fs = float(manifest["sampling_frequency_hz"])
    gain = float(manifest["gain_uv_per_count"])
    rec = load(rec_dir)
    start = int(round(BG["start_s"] * fs))
    stop = start + int(round(BG["duration_s"] * fs))
    sliced = rec.frame_slice(start_frame=start, end_frame=stop)
    ch = sliced.channel_ids[BG["channel_start"] : BG["channel_start"] + BG["channel_count"]]
    sliced = sliced.select_channels(channel_ids=ch)
    traces = sliced.get_traces().astype(np.float32)  # int16 counts
    bg_uv = traces * np.float32(gain)
    geometry = np.asarray(sliced.get_channel_locations(), dtype=np.float64)
    return bg_uv, geometry, fs, gain, start


def prepare_template(full_384: np.ndarray) -> tuple[np.ndarray, int]:
    """Crop one donor template in time and channels, baseline-correct, taper."""
    a = np.asarray(full_384, dtype=np.float32)
    peak_t, peak_c = np.unravel_index(np.argmax(np.abs(a)), a.shape)
    half_t = TP["time_samples"] // 2
    t0 = max(0, peak_t - half_t)
    a = a[t0 : t0 + TP["time_samples"]]
    r = TP["channel_radius"]
    c0 = max(0, peak_c - r)
    a = a[:, c0 : peak_c + r + 1]
    peak_col = peak_c - c0

    guard = TP["edge_guard_samples"]
    edge = np.concatenate((a[:guard], a[-guard:]))
    a = a - np.median(edge, axis=0, keepdims=True)
    # Hard-zero the outer `guard` samples so the template passes the sealed
    # primitive's edge check, then a short raised-cosine ramp back to full
    # amplitude so the injected waveform has no step discontinuity.
    a[:guard] = 0.0
    a[-guard:] = 0.0
    ramp_n = 5
    w = (np.sin(np.linspace(0, np.pi / 2, ramp_n, dtype=np.float32)) ** 2)[:, None]
    a[guard : guard + ramp_n] *= w
    a[-guard - ramp_n : -guard] *= w[::-1]
    return validate_template(a.astype(np.float32), edge_guard_samples=guard), int(peak_col)


def _train(duration_s: float, fs: float) -> np.ndarray:
    guard = int(PRESPEC["train"]["guard_s"] * fs)
    step = int(round(fs / PRESPEC["train"]["rate_hz"]))
    return np.arange(guard, int(duration_s * fs) - guard, step, dtype=np.int64)


def _trajectory_fn(name: str, geometry: np.ndarray, duration_s: float):
    spec = PRESPEC["trajectories"][name]
    cpu = channels_per_um(geometry)
    if spec["kind"] == "static":
        return static_trajectory(), {"total_channels": 0.0}
    if spec["kind"] == "rigid_ramp":
        ch = spec["total_um"] * cpu
        return rigid_ramp(ch, duration_s), {"total_um": spec["total_um"], "total_channels": round(ch, 2)}
    if spec["kind"] == "rigid_oscillation":
        ch = spec["amp_um"] * cpu
        return rigid_oscillation(ch, spec["period_s"]), {
            "amp_um": spec["amp_um"], "period_s": spec["period_s"],
            "amp_channels": round(ch, 2),
        }
    raise ValueError(name)


def run(
    templates: list[str],
    out_root: Path | None = None,
    sorters: list[str] | None = None,
) -> dict:
    _freeze_prespec()
    from testing.ladder_sorter import NAMED_CONFIGS

    out_root = out_root or (OUTPUT / "runs")
    out_root.mkdir(parents=True, exist_ok=True)
    sorter_cfgs = [NAMED_CONFIGS[s] for s in (sorters or ["rescue"])]

    bg_uv, geometry, fs, gain, src_start = load_background()
    duration_s = bg_uv.shape[0] / fs
    train = _train(duration_s, fs)
    donors = np.load(DONOR_TEMPLATES)

    rows = []
    for tid in templates:
        template, peak_col = prepare_template(donors[tid])
        base_channel = BG["channel_count"] // 2 - peak_col

        # inject once per trajectory, write the recording, then sort under each config
        injected: dict[str, tuple[Path, dict]] = {}
        for traj_name in PRESPEC["trajectories"]:
            traj_fn, traj_meta = _trajectory_fn(traj_name, geometry, duration_s)
            static_uv, moving_uv, truth = paired_injection(
                bg_uv, template, train, fs=fs, base_channel=base_channel,
                moving_trajectory=traj_fn, amplitude_scale=PRESPEC["amplitude_scale"],
                unit_id="inj0", edge_guard_samples=TP["edge_guard_samples"],
            )
            arm_uv = static_uv if traj_name == "static" else moving_uv
            rec_dir = out_root / f"{tid}_{traj_name}"
            write_injected_recording(
                rec_dir, arm_uv, channel_positions=geometry, fs=fs,
                gain_uv_per_count=gain, source_snippet_dir=str(_recording_dir()),
                name=f"{tid}_{traj_name}",
            )
            injected[traj_name] = (rec_dir, {k: v for k, v in truth.items()}, traj_meta)

        for cfg in sorter_cfgs:
            scores: dict[str, dict] = {}
            for traj_name, (rec_dir, truth, traj_meta) in injected.items():
                sorter = None if cfg.label == "rescue" else cfg
                result = l1_run(
                    rec_dir, sorter=sorter, truth=truth, out_root=out_root / "_l1"
                )
                scores[traj_name] = result["score"]
                u = result["score"]["primary"]["units"][0]
                rows.append({
                    "template": tid, "sorter": cfg.label, "trajectory": traj_name,
                    **traj_meta,
                    "accuracy": u["accuracy"],
                    "n_output_units_capturing": u["n_output_units_capturing"],
                    "label_switches": u["label_switches"],
                    "min_bin_accuracy": u["min_bin_accuracy"],
                    "recovered": u["recovered"],
                    "tp": u["tp"], "fp": u["fp"], "fn": u["fn"],
                })
            for traj_name in PRESPEC["trajectories"]:
                if traj_name == "static":
                    continue
                pen = drift_penalty(scores["static"], scores[traj_name], "inj0")
                rows.append({
                    "template": tid, "sorter": cfg.label,
                    "trajectory": f"PENALTY:{traj_name}",
                    "accuracy": pen["delta_accuracy"],
                    "n_output_units_capturing": pen["delta_n_identities"],
                    "label_switches": pen["delta_label_switches"],
                    "recovered": pen["moving_recovered"],
                })

    import pandas as pd

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT / "drift_challenge.csv", index=False)

    is_pen = df.trajectory.str.startswith("PENALTY")
    arms = df[~is_pen]
    static_arms = arms[arms.trajectory == "static"]
    best_static = static_arms.sort_values("accuracy").iloc[-1]
    summary = {
        "probe": PRESPEC["probe"],
        "templates": templates,
        "sorters": sorted(df["sorter"].unique().tolist()),
        "n_conditions": int(len(arms)),
        "sanity_static_best_accuracy": round(float(best_static.accuracy), 3),
        "sanity_passed": bool(best_static.accuracy >= 0.9),
        "static_accuracy_by_sorter": {
            s: round(float(sub.accuracy.max()), 3)
            for s, sub in static_arms.groupby("sorter")
        },
        "penalties": {
            f"{r.sorter}:{r.template}:{r.trajectory.split(':')[1]}": {
                "delta_accuracy": round(float(r.accuracy), 3),
                "delta_n_identities": int(r.n_output_units_capturing),
                "delta_label_switches": int(r.label_switches),
            }
            for r in df[is_pen].itertuples()
        },
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--templates", nargs="+", default=["T01", "T04", "T06"])
    ap.add_argument("--sorters", nargs="+", default=["rescue"],
                    choices=["rescue", "legacy_style"])
    args = ap.parse_args()
    summary = run(args.templates, sorters=args.sorters)
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {OUTPUT}")


if __name__ == "__main__":
    main()
