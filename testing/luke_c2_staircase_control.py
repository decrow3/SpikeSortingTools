"""C2 v4 machinery positive control: a lattice-commensurate staircase.

Role, and what it cannot do
---------------------------
`luke_c2_operator_calibration` showed that the injected-motion forward model
attenuates compact donors by 20-50 % at Luke-scale displacements, because
fractional-offset resampling of an under-sampled footprint is lossy. The one
family of displacements it reproduces exactly is the **lattice-commensurate**
one: the target strip has a 20 um row pitch with a two-column x-stagger, so a
40 um shift maps every site onto another site and the correct answer is a pure
channel roll, which the operator reproduces to ~1e-6.

A trajectory that *dwells only at multiples of 40 um* therefore delivers real
displacement and real template mismatch with the forward-model attenuation
essentially absent. Its question is narrow:

    Can this experiment and this sorter demonstrate a correction benefit at
    all, when interpolation error is essentially absent?

It is **not** an anchor for the Luke-calibrated arms. 40 um is roughly twice the
largest proposed Luke displacement and the motion is discontinuous rather than
Luke-like, so a positive result here **cannot** establish that correction helps
at 5-22 um. Results are reported separately from the dose-response.

Why per-bin reasoning is exact
------------------------------
`interpolate_motion_on_traces` builds one interpolation kernel per time bin from
that bin's centre and applies it to every frame in the bin -- displacement is
piecewise constant in time, with no interpolation between bins, and the
interpolation bins are the bins `sampled_displacement` constructs. So a spike is
resampled exactly iff every bin its template touches carries the same
commensurate displacement. That makes truth admission decidable rather than
approximate, which is safeguard 2 below.

Safeguards implemented here
---------------------------
1. long plateaus at exactly 0 and 40 um, with plateau and transition durations
   that are whole multiples of `bin_s` so plateau boundaries fall on bin edges;
2. injected truth spikes are admitted only when the *whole* template window
   lies in settled bins of one commensurate level, plus a guard of `guard_bins`;
3. boundary behaviour is verified on the real full-channel background with real
   noise -- both the channel edges, where a 40 um shift must draw on sites
   outside the recorded strip, and the first/last time bins -- not on the
   centred noise-free donor field the calibration used;
4. every output is written to its own namespace and labelled as a positive
   control.

Run: `python testing/luke_c2_staircase_control.py [--full]`
(`--full` warps the whole 120 s strip end to end; the default verifies on a
20 s slice, which is sufficient for the spatial question and much faster.)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from testing.ladder_motion import frame_bin_assignment, warp_array_with_known_motion
from testing.ladder_score import assert_paired_truth, build_truth_contract
from testing.luke_rescue_c2_drift_challenge import PRESPEC, _recording_dir

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "testing/outputs/luke_c2_staircase_control"

STAIRCASE = {
    "schema": "luke-c2-staircase-control-v1",
    "role": "machinery positive control",
    "question": (
        "can this experiment and this sorter demonstrate a correction benefit "
        "at all, when interpolation error is essentially absent?"
    ),
    "cannot_answer": (
        "whether correction helps at 5-22 um; 40 um is ~2x the largest proposed "
        "Luke displacement and the motion is discontinuous, not Luke-like"
    ),
    "report_separately_from": "the Luke-calibrated 5/11/22 um dose-response",
    "duration_s": 120.0,
    "bin_s": 0.5,
    # HARD, bin-aligned steps: 4 plateaus x 30 s, no transition segment at all.
    # A ramped transition leaves fractional-offset bins in the recording, and
    # those degraded waveforms reach KS4's detection and template formation
    # whether or not their truth spikes are scored. The only way to keep the
    # control interpolation-free is for no such bin to exist.
    "levels_um": [0.0, 40.0],
    "plateau_s": 30.0,
    "transition_s": 0.0,
    "n_plateaus": 4,
    # The sorter input is cropped from a wider warped strip so every channel it
    # sees has a defined source site. Forward (+40 um) pulls from `shift`
    # channels below, the exact inverse pulls from `shift` above, so the margin
    # must cover both.
    "spatial_margin_channels": 8,
    "geometry": {"row_pitch_um": 20.0, "sites_per_row": 2, "commensurate_shift_channels": 4},
    "truth_admission": {
        "template_pre_samples": 30,
        "template_post_samples": 30,
        "guard_bins": 1,
        "rule": (
            "admit a spike only if every time bin its template window touches, "
            "extended by guard_bins on each side, carries the same exactly "
            "commensurate displacement"
        ),
    },
    "exactness_max_rel_err": 1e-5,
    "verification_slice_s": 62.0,
    "arms": {
        "static": "unwarped crop of the wide strip",
        "staircase": "wide strip warped by the staircase, then cropped",
        "staircase_corrected": "the staircase arm warped by the exact inverse, then cropped",
    },
    "arms_note": (
        "all three arms are cropped identically from the same wide strip and "
        "carry the identical admitted truth train, so every paired score uses "
        "the same spatial support and the same denominator"
    ),
}


# --------------------------------------------------------------------------- #
# trajectory
# --------------------------------------------------------------------------- #
def segments() -> list[dict]:
    """The staircase as explicit (kind, level, start, stop) segments."""
    plateau_s = STAIRCASE["plateau_s"]
    transition_s = STAIRCASE["transition_s"]
    levels = STAIRCASE["levels_um"]
    out, t = [], 0.0
    for index in range(STAIRCASE["n_plateaus"]):
        level = levels[index % len(levels)]
        out.append({"kind": "plateau", "level_um": float(level),
                    "start_s": t, "stop_s": t + plateau_s})
        t += plateau_s
        if transition_s > 0 and index < STAIRCASE["n_plateaus"] - 1:
            nxt = levels[(index + 1) % len(levels)]
            out.append({"kind": "transition", "from_um": float(level),
                        "to_um": float(nxt), "start_s": t, "stop_s": t + transition_s})
            t += transition_s
    if abs(t - STAIRCASE["duration_s"]) > 1e-9:
        raise RuntimeError(f"staircase segments total {t} s, not {STAIRCASE['duration_s']}")
    return out


def staircase_um(t_s) -> np.ndarray:
    """Displacement in µm. Exactly a level on plateaus; linear on transitions."""
    t = np.asarray(t_s, dtype=float)
    out = np.empty_like(t)
    for seg in segments():
        inside = (t >= seg["start_s"]) & (t < seg["stop_s"])
        if seg["kind"] == "plateau":
            out[inside] = seg["level_um"]
        else:
            span = seg["stop_s"] - seg["start_s"]
            frac = (t[inside] - seg["start_s"]) / span
            out[inside] = seg["from_um"] + frac * (seg["to_um"] - seg["from_um"])
    last = segments()[-1]
    out[t >= last["stop_s"]] = last["level_um"]  # clipped like the operator does
    return out


def bin_table(duration_s: float | None = None, bin_s: float | None = None) -> dict:
    """Per-bin displacement, matching `sampled_displacement`'s bin construction."""
    duration_s = float(duration_s or STAIRCASE["duration_s"])
    bin_s = float(bin_s or STAIRCASE["bin_s"])
    n_bins = max(2, int(round(duration_s / bin_s)))
    width = duration_s / n_bins
    centers = (np.arange(n_bins) + 0.5) * width
    edges = np.arange(n_bins + 1) * width
    displacement = staircase_um(centers)
    settled = np.isin(displacement, np.asarray(STAIRCASE["levels_um"], dtype=float))
    return {
        "n_bins": n_bins, "width_s": width, "centers_s": centers, "edges_s": edges,
        "displacement_um": displacement, "settled": settled,
    }


def admissible_train(train_samples: np.ndarray, fs: float) -> dict:
    """Keep only spikes resampled at one exactly commensurate displacement.

    Safeguard 2. A spike is admitted iff every bin its template window touches --
    widened by `guard_bins` on each side -- is settled and carries the same
    level. Straddling a plateau/transition boundary splits the waveform across
    two different kernels, which is exactly what must never reach the scorer.
    """
    rule = STAIRCASE["truth_admission"]
    table = bin_table()
    edges, settled = table["edges_s"], table["settled"]
    displacement = table["displacement_um"]
    guard = int(rule["guard_bins"])

    train = np.asarray(train_samples, dtype=np.int64)
    start_s = (train - rule["template_pre_samples"]) / fs
    stop_s = (train + rule["template_post_samples"]) / fs
    first = np.clip(np.searchsorted(edges, start_s, side="right") - 1 - guard,
                    0, table["n_bins"] - 1)
    last = np.clip(np.searchsorted(edges, stop_s, side="right") - 1 + guard,
                   0, table["n_bins"] - 1)

    keep = np.zeros(train.size, dtype=bool)
    level = np.full(train.size, np.nan)
    for i, (lo, hi) in enumerate(zip(first, last)):
        window = slice(int(lo), int(hi) + 1)
        levels = displacement[window]
        if settled[window].all() and np.all(levels == levels[0]):
            keep[i] = True
            level[i] = levels[0]
    return {
        "keep": keep, "level_um": level,
        "n_total": int(train.size), "n_admitted": int(keep.sum()),
        "n_by_level": {
            str(lvl): int(np.sum(level[keep] == lvl)) for lvl in STAIRCASE["levels_um"]
        },
    }


# --------------------------------------------------------------------------- #
# boundary verification on the real background (safeguard 3)
# --------------------------------------------------------------------------- #
def expected_shift_channels(level_um: float) -> int:
    """Channel shift a commensurate displacement corresponds to, from geometry."""
    geo = STAIRCASE["geometry"]
    return int(round(level_um / geo["row_pitch_um"]) * geo["sites_per_row"])


def load_wide_background(margin: int | None = None):
    """The C2 background strip plus a margin on each side, for warp-then-crop.

    Warping the exact strip the sorter sees leaves its edge channels with no
    source site, so the operator extrapolates there and those channels reach
    KS4 regardless of where donors are placed. Warping a wider region and
    cropping afterwards gives every sorter channel a defined source.
    """
    from spikeinterface.core import load

    margin = int(STAIRCASE["spatial_margin_channels"] if margin is None else margin)
    rec_dir = _recording_dir()
    manifest = json.loads((rec_dir / "rescue_recording_manifest.json").read_text())
    fs = float(manifest["sampling_frequency_hz"])
    gain = float(manifest["gain_uv_per_count"])
    rec = load(rec_dir)
    bg = PRESPEC["background"]
    lo = int(bg["channel_start"]) - margin
    hi = int(bg["channel_start"]) + int(bg["channel_count"]) + margin
    if lo < 0 or hi > rec.get_num_channels():
        raise RuntimeError(
            f"a {margin}-channel margin does not fit: needs [{lo}, {hi}) of "
            f"{rec.get_num_channels()} channels"
        )
    start = int(round(bg["start_s"] * fs))
    stop = start + int(round(bg["duration_s"] * fs))
    sliced = rec.frame_slice(start_frame=start, end_frame=stop)
    sliced = sliced.select_channels(channel_ids=sliced.channel_ids[lo:hi])
    wide_uv = sliced.get_traces().astype(np.float32) * np.float32(gain)
    wide_geometry = np.asarray(sliced.get_channel_locations(), dtype=np.float64)
    wide_channel_ids = np.asarray(sliced.channel_ids)
    crop = slice(margin, margin + int(bg["channel_count"]))
    return wide_uv, wide_geometry, wide_channel_ids, fs, gain, crop, margin


def build_arms(wide_uv, wide_geometry, fs, *, duration_s=None, crop=None,
               margin=None, wide_channel_ids=None, trajectory_fn=None,
               labels=("static", "staircase", "staircase_corrected"),
               with_corrected: bool = True):
    """Static / moved / exactly-corrected arms, cropped identically.

    `trajectory_fn` returns µm of displacement against time and defaults to the
    staircase; C2 v4 passes its Luke-calibrated ramps through the same builder,
    so every condition gets the same identical-support guarantee.

    All three come from the same wide strip and the same crop, so they have
    byte-identical spatial support and channel geometry; the caller injects the
    same admitted truth train into each.
    """
    margin = int(STAIRCASE["spatial_margin_channels"] if margin is None else margin)
    crop = crop or slice(margin, margin + int(PRESPEC["background"]["channel_count"]))
    if duration_s is not None:
        wide_uv = np.ascontiguousarray(wide_uv[: int(round(duration_s * fs))])

    trajectory_fn = trajectory_fn or staircase_um
    static_label, moved_label, corrected_label = labels
    warp = dict(fs=fs, bin_s=STAIRCASE["bin_s"], trajectory_units="um")
    moved_wide = warp_array_with_known_motion(
        wide_uv, wide_geometry, trajectory_fn=trajectory_fn, sign=-1.0, **warp
    )
    # `with_corrected=False` skips the exact-inverse warp when a caller only
    # needs the static/moved pair -- it halves the CPU warping cost per donor.
    corrected_wide = warp_array_with_known_motion(
        moved_wide, wide_geometry, trajectory_fn=trajectory_fn, sign=+1.0, **warp
    ) if with_corrected else None
    arms = {
        static_label: np.ascontiguousarray(wide_uv[:, crop]),
        moved_label: np.ascontiguousarray(moved_wide[:, crop]),
        "geometry": np.ascontiguousarray(wide_geometry[crop]),
        "channel_ids": (
            np.asarray(wide_channel_ids)[crop] if wide_channel_ids is not None
            else np.arange(crop.start, crop.stop)
        ),
        "_wide": wide_uv,
        "crop": crop,
    }
    if corrected_wide is not None:
        arms[corrected_label] = np.ascontiguousarray(corrected_wide[:, crop])
    return arms


def verify_arms(arms, fs) -> dict:
    """Every sorter channel exact at every plateau, and correction restores.

    For a plateau at level L the cropped staircase arm must equal the wide strip
    shifted by `expected_shift_channels(L)` -- on **all** channels, because the
    margin supplies the sources the crop's edges need. The corrected arm must
    equal the static arm.
    """
    wide, crop = arms["_wide"], arms["crop"]
    static, staircase = arms["static"], arms["staircase"]
    corrected = arms["staircase_corrected"]
    n_samples, n_channels = static.shape
    duration_s = n_samples / fs
    table = bin_table(duration_s=duration_s)
    starts, stops = frame_bin_assignment(n_samples, fs, table["edges_s"], table["n_bins"])
    scale = float(np.sqrt((static.astype(np.float64) ** 2).mean()))

    rows = []
    for index in range(table["n_bins"]):
        lo, hi = int(starts[index]), int(stops[index])
        if hi <= lo:
            continue
        level = float(table["displacement_um"][index])
        shift = expected_shift_channels(level)
        source = slice(crop.start - shift, crop.start - shift + n_channels)
        if source.start < 0 or source.stop > wide.shape[1]:
            raise RuntimeError(f"margin too small for a {level} um step")
        reference = wide[lo:hi, source]
        rows.append({
            "bin": index,
            "t_start_s": float(table["edges_s"][index]),
            "displacement_um": level,
            "channel_shift": shift,
            "rel_err": float(
                np.sqrt(((staircase[lo:hi] - reference).astype(np.float64) ** 2).mean())
            ) / scale,
        })

    per_channel = np.sqrt(
        ((staircase - _reference_full(wide, crop, table, starts, stops, n_channels))
         .astype(np.float64) ** 2).mean(axis=0)
    ) / scale
    correction_err = float(
        np.sqrt(((corrected - static).astype(np.float64) ** 2).mean())
    ) / scale
    worst = max(rows, key=lambda r: r["rel_err"])
    tolerance = STAIRCASE["exactness_max_rel_err"]
    return {
        "n_bins": len(rows),
        "n_channels_verified": int(n_channels),
        "all_bins_exact": bool(worst["rel_err"] <= tolerance),
        "max_bin_rel_err": worst["rel_err"],
        "worst_bin": {k: worst[k] for k in ("bin", "t_start_s", "displacement_um")},
        "first_bin_rel_err": rows[0]["rel_err"],
        "last_bin_rel_err": rows[-1]["rel_err"],
        "max_channel_rel_err": float(per_channel.max()),
        "worst_channel": int(np.argmax(per_channel)),
        "n_channels_exact": int((per_channel <= tolerance).sum()),
        "levels_seen_um": sorted({r["displacement_um"] for r in rows}),
        "shifts_seen_channels": sorted({r["channel_shift"] for r in rows}),
        "correction_restores_static": bool(correction_err <= tolerance),
        "corrected_vs_static_rel_err": correction_err,
        "spatial_support_identical": bool(
            static.shape == staircase.shape == corrected.shape
        ),
    }


def _reference_full(wide, crop, table, starts, stops, n_channels):
    """The exact expected staircase arm, assembled bin by bin."""
    out = np.empty((int(stops[-1]), n_channels), dtype=wide.dtype)
    for index in range(table["n_bins"]):
        lo, hi = int(starts[index]), int(stops[index])
        if hi <= lo:
            continue
        shift = expected_shift_channels(float(table["displacement_um"][index]))
        out[lo:hi] = wide[lo:hi, crop.start - shift: crop.start - shift + n_channels]
    return out


def staircase_admitted_truth(train, fs, *, unit_id: str = "inj0"):
    """Filter the train **first**. The result is what must be injected.

    Returns `(truth, admission)`. Callers inject exactly `truth[unit_id]`; the
    contract is then built from the array that was actually injected, so an
    inject-then-filter ordering fails closed rather than certifying itself.
    """
    rule = STAIRCASE["truth_admission"]
    admission = admissible_train(np.asarray(train, dtype=np.int64), fs)
    admitted = np.asarray(train, dtype=np.int64)[admission["keep"]]
    return {unit_id: admitted}, {
        "schema": STAIRCASE["schema"],
        "rule": rule["rule"],
        "guard_bins": rule["guard_bins"],
        "template_pre_samples": rule["template_pre_samples"],
        "template_post_samples": rule["template_post_samples"],
        "bin_s": STAIRCASE["bin_s"],
        "levels_um": STAIRCASE["levels_um"],
        "plateau_s": STAIRCASE["plateau_s"],
        "transition_s": STAIRCASE["transition_s"],
        "n_total": admission["n_total"],
        "n_admitted": admission["n_admitted"],
        "counts_by_level_um": admission["n_by_level"],
    }


def staircase_truth_contract(truth, injected, arms, admission):
    """Bind the admitted train to the array injected and the cropped support."""
    return build_truth_contract(
        truth,
        injected=injected,
        admission=dict(admission),
        channel_ids=arms["channel_ids"],
        geometry=arms["geometry"],
        crop=(arms["crop"].start, arms["crop"].stop),
    )


# --------------------------------------------------------------------------- #
def run(full: bool = False) -> dict:
    if str(OUTPUT).startswith("/mnt/"):
        raise ValueError("refusing to write the staircase control under /mnt")
    OUTPUT.mkdir(parents=True, exist_ok=True)

    wide_uv, wide_geometry, wide_ids, fs, _gain, crop, margin = load_wide_background()
    duration_s = STAIRCASE["duration_s"] if full else STAIRCASE["verification_slice_s"]
    arms = build_arms(wide_uv, wide_geometry, fs, duration_s=duration_s,
                      crop=crop, margin=margin, wide_channel_ids=wide_ids)
    del wide_uv
    verification = verify_arms(arms, fs)

    train = np.arange(
        int(PRESPEC["train"]["guard_s"] * fs),
        int(STAIRCASE["duration_s"] * fs) - int(PRESPEC["train"]["guard_s"] * fs),
        int(round(fs / PRESPEC["train"]["rate_hz"])),
        dtype=np.int64,
    )
    truth, admission_record = staircase_admitted_truth(train, fs)
    # the control verifies voltage only, but it must model the real order:
    # filter first, then treat that array as the one injected
    contract = staircase_truth_contract(truth, truth, arms, admission_record)
    admission = admissible_train(train, fs)
    paired = assert_paired_truth([contract, contract, contract],
                                 labels=list(STAIRCASE["arms"]))
    table = bin_table()

    summary = {
        "staircase": STAIRCASE,
        "segments": segments(),
        "verified_on": {
            "background": PRESPEC["background"], "probe": PRESPEC["probe"],
            "duration_s": duration_s, "full_duration": bool(full),
            "real_noise": True,
            "wide_channels": int(PRESPEC["background"]["channel_count"] + 2 * margin),
            "cropped_channels": int(PRESPEC["background"]["channel_count"]),
            "margin_channels": margin,
        },
        "bins": {
            "n_bins": table["n_bins"],
            "n_settled": int(table["settled"].sum()),
            "n_fractional": int((~table["settled"]).sum()),
            "levels_present_um": sorted(set(np.round(table["displacement_um"], 6).tolist())),
        },
        "truth_admission": {
            "n_total": admission["n_total"], "n_admitted": admission["n_admitted"],
            "fraction_admitted": round(admission["n_admitted"] / admission["n_total"], 4),
            "n_by_level": admission["n_by_level"],
            "applied": "before injection, identically in every arm",
        },
        "arm_verification": verification,
        "truth_contract": contract,
        "paired_arms": paired,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--full", action="store_true",
                    help="verify on the whole 120 s recording, not a 20 s slice")
    args = ap.parse_args()
    summary = run(full=args.full)
    print(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote {OUTPUT}")


if __name__ == "__main__":
    main()
