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

**Status: diagnostic.** C2 v3 uses all 14 D2b-2 spatially compact donors. The
pilot T01/T04/T06 waveforms are explicitly forbidden: they are common-mode/LFP
plateaus (or noise-level), not localized neuron footprints. A donor contributes
to the primary drift comparison only when its static arm reaches accuracy >=
0.8 under both sorter configurations.

Confound control (plan C2): the static arm is drawn from a **quiet** window, so
the background's own tissue motion is minimal; the moving trajectory is imposed
on top and reported in µm and channels. Recorded here, not corrected for.

    python testing/luke_rescue_c2_drift_challenge.py

Outputs to testing/outputs/luke_rescue_c2_drift_challenge_v3/. Nothing under /mnt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from pipeline.config import fingerprint
from testing.ladder_inject import (
    channels_per_um,
    drift_penalty,
    rigid_oscillation,
    rigid_ramp,
    static_trajectory,
    write_injected_recording,
)
from testing.ladder_l1 import l1_run
from testing.ladder_motion import paired_geometry_motion_injection
from testing.ladder_score import score_sort
from testing.luke_injected_ground_truth_benchmark import validate_template

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "testing/outputs/luke_rescue_c2_drift_challenge_v3"
LUKE_ROOT = Path("/mnt/NPX/Luke/20250804")
DONOR_TEMPLATES = (
    REPO_ROOT
    / "testing/outputs/luke_d2b2_donor_cohort/donor_templates.npz"
)
DONOR_MANIFEST = DONOR_TEMPLATES.with_name("donor_manifest.csv")
DONOR_GEOMETRY = (
    LUKE_ROOT
    / "rescue_pipeline_results_Luke0804_V2V1_g0_imec0/cur/cur_output/channel_positions.npy"
)

PRESPEC = {
    "schema": "luke-rescue-c2-drift-challenge-v3",
    "frozen": "2026-09-03",
    "status": "compact_donor_geometry_aware_rerun_pending",
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
        "source": "luke_d2b2_donor_cohort/donor_templates.npz",
        "cohort": "all 14 spatially compact real donors",
        "forbidden_source": "luke_injected_ground_truth_pilot T01/T04/T06",
        "time_samples": 61,
        "channel_radius": 16,
        "edge_guard_samples": 3,
        "preparation": "use sealed D2b-2 template unchanged",
        "templates_sha256": "998e4dbd067cd3529fe0c18038173f62c57a79e57b2b2ef7b7ced5c348695d24",
        "manifest_sha256": "43da816a7c52f8c8995c0f608548bc93501cae75a46dbbb0320f0b5dc905d319",
        "source_geometry_sha256": "0469ca92fb739a0cfd2f1613262d3a2d75af1098385385d7462ee6e3fd038d75",
        "placement": (
            "translate each donor to the nearest strip-centre site whose full "
            "relative x/y geometry matches its original imec0 channel crop"
        ),
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
    "static_qualification": {
        "accuracy_min": 0.8,
        "required_sorters": ["rescue", "legacy_style"],
        "rule": (
            "a donor contributes to primary drift-penalty comparisons only if "
            "its static accuracy is >= 0.8 under both required sorters"
        ),
    },
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_donor_cohort() -> None:
    expected = PRESPEC["template_prep"]
    for path, key in (
        (DONOR_TEMPLATES, "templates_sha256"),
        (DONOR_MANIFEST, "manifest_sha256"),
        (DONOR_GEOMETRY, "source_geometry_sha256"),
    ):
        if not path.exists() or _sha256(path) != expected[key]:
            raise RuntimeError(f"C2 v3 donor cohort is missing or changed: {path}")


def _resolve_frozen_cohort(available, requested: list[str] | None) -> list[str]:
    """Enforce the all-donor v3 prespec; subsets belong in separate diagnostics."""
    frozen = sorted(str(template_id) for template_id in available)
    if len(frozen) != 14 or any(template_id.startswith("T") for template_id in frozen):
        raise RuntimeError("C2 v3 requires exactly the 14 compact D-donor cohort")
    if requested is not None and sorted(requested) != frozen:
        raise ValueError("C2 v3 is frozen to all 14 compact donors; subsets are not C2")
    return frozen


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


def prepare_template(compact: np.ndarray) -> tuple[np.ndarray, int]:
    """Validate a sealed D2b-2 donor without reshaping or re-tapering it."""
    a = np.asarray(compact, dtype=np.float32)
    if a.ndim != 2 or a.shape[0] != TP["time_samples"]:
        raise ValueError(
            f"C2 v3 requires a ({TP['time_samples']}, channels) compact donor; "
            f"got {a.shape}"
        )
    if a.shape[1] > 2 * TP["channel_radius"] + 1:
        raise ValueError(f"compact donor is too wide: {a.shape}")
    sealed = validate_template(a, edge_guard_samples=TP["edge_guard_samples"])
    peak_col = int(np.unravel_index(np.argmax(np.abs(sealed)), sealed.shape)[1])
    return sealed, peak_col


def donor_base_channel(
    template: np.ndarray,
    peak_col: int,
    source_peak_channel: int,
    source_geometry: np.ndarray,
    target_geometry: np.ndarray,
) -> tuple[int, int]:
    """Place a donor without changing its relative four-column geometry."""
    width = int(template.shape[1])
    source_start = int(source_peak_channel) - int(peak_col)
    source_stop = source_start + width
    if source_start < 0 or source_stop > len(source_geometry):
        raise ValueError("donor crop does not fit its recorded source geometry")
    source_relative = (
        np.asarray(source_geometry[source_start:source_stop], dtype=np.float64)
        - np.asarray(source_geometry[source_peak_channel], dtype=np.float64)
    )
    candidates = []
    for target_peak in range(peak_col, len(target_geometry) - (width - peak_col) + 1):
        target_start = target_peak - peak_col
        target_relative = (
            np.asarray(target_geometry[target_start:target_start + width], dtype=np.float64)
            - np.asarray(target_geometry[target_peak], dtype=np.float64)
        )
        if np.allclose(target_relative, source_relative, atol=1e-6, rtol=0.0):
            candidates.append((abs(target_peak - len(target_geometry) / 2), target_start, target_peak))
    if not candidates:
        raise ValueError("no target placement preserves the donor's relative probe geometry")
    _, target_start, target_peak = min(candidates)
    return int(target_start), int(target_peak)


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
    templates: list[str] | None = None,
    out_root: Path | None = None,
    sorters: list[str] | None = None,
) -> dict:
    _freeze_prespec()
    _verify_donor_cohort()
    from testing.ladder_sorter import NAMED_CONFIGS

    out_root = out_root or (OUTPUT / "runs")
    out_root.mkdir(parents=True, exist_ok=True)
    required_sorters = PRESPEC["static_qualification"]["required_sorters"]
    if sorters is not None and sorted(sorters) != sorted(required_sorters):
        raise ValueError("C2 v3 requires both rescue and legacy_style")
    sorter_names = required_sorters
    sorter_cfgs = [NAMED_CONFIGS[s] for s in sorter_names]

    bg_uv, geometry, fs, gain, src_start = load_background()
    duration_s = bg_uv.shape[0] / fs
    train = _train(duration_s, fs)
    donors = np.load(DONOR_TEMPLATES)
    templates = _resolve_frozen_cohort(donors.files, templates)
    import pandas as pd

    donor_meta = pd.read_csv(DONOR_MANIFEST).set_index("template_id").to_dict("index")
    source_geometry = np.load(DONOR_GEOMETRY)

    rows = []
    for tid in templates:
        template, peak_col = prepare_template(donors[tid])
        base_channel, target_peak = donor_base_channel(
            template,
            peak_col,
            int(donor_meta[tid]["peak_channel"]),
            source_geometry,
            geometry,
        )

        # inject once per trajectory, write the recording, then sort under each config
        injected: dict[str, tuple[Path, dict]] = {}
        for traj_name in PRESPEC["trajectories"]:
            traj_fn, traj_meta = _trajectory_fn(traj_name, geometry, duration_s)
            static_uv, moving_uv, truth = paired_geometry_motion_injection(
                bg_uv, template, train, fs=fs, base_channel=base_channel,
                moving_trajectory=traj_fn, amplitude_scale=PRESPEC["amplitude_scale"],
                unit_id="inj0", edge_guard_samples=TP["edge_guard_samples"],
                channel_positions=geometry,
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
                    "donor_peak_uv": donor_meta.get(tid, {}).get("peak_uv"),
                    "donor_polarity": donor_meta.get(tid, {}).get("polarity"),
                    "donor_amplitude_band": donor_meta.get(tid, {}).get("amplitude_band"),
                    "injection_base_channel": base_channel,
                    "injection_peak_channel": target_peak,
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
                    "donor_peak_uv": donor_meta.get(tid, {}).get("peak_uv"),
                    "donor_polarity": donor_meta.get(tid, {}).get("polarity"),
                    "donor_amplitude_band": donor_meta.get(tid, {}).get("amplitude_band"),
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
    static_min = static_arms.pivot(index="template", columns="sorter", values="accuracy")
    required = PRESPEC["static_qualification"]["required_sorters"]
    threshold = PRESPEC["static_qualification"]["accuracy_min"]
    qualified = sorted(
        tid for tid, row in static_min.iterrows()
        if all(sorter in row.index and row[sorter] >= threshold for sorter in required)
    )
    penalty_rows = df[is_pen].copy()
    qualified_penalty = penalty_rows[penalty_rows.template.isin(qualified)]
    summary = {
        "probe": PRESPEC["probe"],
        "templates": templates,
        "sorters": sorted(df["sorter"].unique().tolist()),
        "n_conditions": int(len(arms)),
        "static_qualification": PRESPEC["static_qualification"],
        "qualified_templates": qualified,
        "n_qualified_templates": len(qualified),
        "primary_comparison_available": bool(qualified),
        "all_donors_static_qualified": len(qualified) == len(templates),
        "static_accuracy_by_template_sorter": {
            str(tid): {
                str(sorter): round(float(value), 3)
                for sorter, value in row.items()
            }
            for tid, row in static_min.iterrows()
        },
        "penalties": {
            f"{r.sorter}:{r.template}:{r.trajectory.split(':')[1]}": {
                "delta_accuracy": round(float(r.accuracy), 3),
                "delta_n_identities": int(r.n_output_units_capturing),
                "delta_label_switches": int(r.label_switches),
            }
            for r in qualified_penalty.itertuples()
        },
        "unqualified_penalties_excluded": sorted(set(templates) - set(qualified)),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.parse_args()
    summary = run()
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {OUTPUT}")


if __name__ == "__main__":
    main()
