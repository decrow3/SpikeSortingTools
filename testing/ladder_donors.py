"""Build a spatially-compact injected-donor cohort — D2b-2.

The C2 / D2b pilot donors (`luke_injected_ground_truth_pilot`) are 10 *single*
reviewed-event raw snippets drawn from **3** imec1 units, one common-mode
dominated. D2b-1 showed every one is a near-flat ±160 µm plateau — not a
single-neuron footprint — because a single-event snippet of an **unfiltered**
recording is dominated by LFP/common-mode, not the spike. So the
waveform-preservation guardrail cannot be frozen on them and the drift-penalty
magnitudes may not transfer to compact real neurons.

This module builds a better cohort from an existing imec0 sort: **de-whitened
Kilosort templates** — the spike-triggered averages KS already computed after
its internal high-pass + CAR + whitening — mapped back to sensor space with
`whitening_mat_inv.npy` and scaled to µV by each unit's bandpass
spike-triggered-average peak.
Real neurons, a real amplitude-decay footprint, on the probe the promotion
question is about, and no new manual review.

    from testing.ladder_donors import build_donor_cohort
    build_donor_cohort(SORT_DIR, OUT_DIR, n_donors=12)

`build_donor_cohort` writes `donor_templates.npz` (id -> (n_samp, n_chan) µV,
edge-tapered, `validate_template`-clean) + `donor_manifest.csv`, spanning
amplitude bands and both polarities. The templates drop straight into
`ladder_inject.paired_injection`. Nothing is written under /mnt.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.config import fingerprint
from testing.luke_injected_ground_truth_benchmark import validate_template

DONOR_SCHEMA = "luke-ladder-donor-v1"


@dataclass(frozen=True)
class DonorConfig:
    n_samples: int = 61            # matches the C2 template width
    radius_ch: int = 16            # ±16 channels -> 33-wide footprint
    edge_guard_samples: int = 3
    ramp_samples: int = 5
    min_spikes: int = 800
    max_spikes_unit: int = 150000  # skip hyper-active (likely merged) units
    max_contam_pct: float = 8.0
    min_peak_uv: float = 70.0      # true bandpass-STA µV floor for a usable donor
    sta_spikes: int = 400          # spikes averaged for the µV-scale STA
    # compactness gate on the de-whitened template shape
    min_energy_frac_pm3: float = 0.45   # >=45% of per-channel energy within ±3 ch of peak
    max_half_energy_width_ch: int = 16
    # true-µV amplitude bands for spreading the cohort
    amp_bands_uv: tuple = (120.0, 200.0)
    per_cell_cap: int = 3          # ≤ this many donors per (amplitude band × polarity)
    prefilter_top: int = 45       # candidates to STA after the cheap compactness pass
    seed: int = 2026

    @property
    def digest(self) -> str:
        return fingerprint({"stage": "donor", **asdict(self)})


def _load_sort(sort_dir: Path) -> dict:
    sort_dir = Path(sort_dir)
    st = np.load(sort_dir / "spike_times.npy").reshape(-1).astype(np.int64)
    cl = np.load(sort_dir / "spike_clusters.npy").reshape(-1).astype(np.int64)
    templates = np.load(sort_dir / "templates.npy")
    winv = np.load(sort_dir / "whitening_mat_inv.npy")
    positions = np.load(sort_dir / "channel_positions.npy")
    labels = pd.read_csv(sort_dir / "cluster_KSLabel.tsv", sep="\t")
    lcol = next(c for c in labels.columns if c != "cluster_id")
    good = set(
        labels.loc[labels[lcol].astype(str).str.strip().str.lower() == "good", "cluster_id"]
    )
    contam = pd.read_csv(sort_dir / "cluster_ContamPct.tsv", sep="\t")
    ccol = next(c for c in contam.columns if c != "cluster_id")
    amp = pd.read_csv(sort_dir / "cluster_Amplitude.tsv", sep="\t")
    acol = next(c for c in amp.columns if c != "cluster_id")
    return {
        "st": st, "cl": cl, "templates": templates, "winv": winv, "positions": positions,
        "good": good,
        "contam_pct": dict(zip(contam["cluster_id"], contam[ccol].astype(float))),
        "amplitude": dict(zip(amp["cluster_id"], amp[acol].astype(float))),
    }


def _compactness(template: np.ndarray) -> dict:
    """Per-channel energy profile of a (samples, channels) template."""
    e = np.sqrt((np.asarray(template, dtype=np.float64) ** 2).sum(axis=0))
    if e.sum() == 0:
        return {"peak_ch": 0, "energy_frac_pm3": 0.0, "half_energy_width_ch": len(e)}
    pk = int(np.argmax(e))
    near = float(e[max(0, pk - 3): pk + 4].sum() / e.sum())
    above = np.flatnonzero(e > 0.5 * e.max())
    width = int(above.max() - above.min() + 1) if above.size else len(e)
    return {"peak_ch": pk, "energy_frac_pm3": round(near, 3), "half_energy_width_ch": width}


def _taper(template: np.ndarray, config: DonorConfig) -> np.ndarray:
    g, r = config.edge_guard_samples, config.ramp_samples
    a = np.asarray(template, dtype=np.float32)
    edge = np.concatenate((a[:g], a[-g:]))
    a = a - np.median(edge, axis=0, keepdims=True)
    a[:g] = 0.0
    a[-g:] = 0.0
    w = (np.sin(np.linspace(0, np.pi / 2, r, dtype=np.float32)) ** 2)[:, None]
    a[g:g + r] *= w
    a[-g - r:-g] *= w[::-1]
    return a


def _dewhitened_shape(sort: dict, cid: int, config: DonorConfig) -> tuple[np.ndarray, int, str]:
    """De-whitened KS template, time-centred and cropped ±radius, unit-peak-normalised.

    Shape only — KS's CAR + whitening make this cleaner than a raw STA (which
    keeps residual imec0 common-mode). The µV scale comes from a bandpass STA
    (`_bandpass_peak_uv`); `cluster_Amplitude` is NOT µV (it runs ~4-7× small).
    """
    dw = np.asarray(sort["templates"][cid], dtype=np.float64) @ sort["winv"]
    peak_t = int(np.unravel_index(np.argmax(np.abs(dw)), dw.shape)[0])
    half = config.n_samples // 2
    t0 = min(max(0, peak_t - half), dw.shape[0] - config.n_samples)
    dw = dw[t0: t0 + config.n_samples]
    peak_c = int(np.argmax(np.max(np.abs(dw), axis=0)))
    lo = max(0, peak_c - config.radius_ch)
    hi = min(dw.shape[1], peak_c + config.radius_ch + 1)
    crop = dw[:, lo:hi]
    peak_amp = float(np.abs(crop).max())
    shape = (crop / peak_amp) if peak_amp else crop
    trough = shape[:, int(np.argmax(np.max(np.abs(shape), axis=0)))]
    polarity = "neg" if trough[np.argmax(np.abs(trough))] < 0 else "pos"
    return shape.astype(np.float32), peak_c, polarity


def _bandpass_peak_uv(
    recording, spike_samples: np.ndarray, peak_ch: int, gain: float, config: DonorConfig, rng
) -> float:
    """True µV peak of a unit from a bandpass spike-triggered average."""
    half = config.n_samples // 2
    total = recording.get_num_samples()
    s = spike_samples[(spike_samples - half >= 0) & (spike_samples + half + 1 <= total)]
    if s.size == 0:
        return 0.0
    if s.size > config.sta_spikes:
        s = rng.choice(s, config.sta_spikes, replace=False)
    lo = max(0, peak_ch - 4)
    hi = min(recording.get_num_channels(), peak_ch + 5)
    ids = recording.channel_ids[lo:hi]
    acc = np.zeros((config.n_samples, len(ids)), dtype=np.float64)
    for sample in np.sort(s):
        acc += recording.get_traces(
            start_frame=int(sample - half),
            end_frame=int(sample - half + config.n_samples),
            channel_ids=ids,
        ).astype(np.float64)
    return float(np.abs(acc / len(s) * gain).max())


def _amp_band(peak_uv: float, bands: tuple) -> str:
    lo, hi = bands
    return "low" if peak_uv < lo else "high" if peak_uv >= hi else "mid"


def _select(rows: list[dict], config: DonorConfig, n_donors: int) -> list[dict]:
    """Spread the picks across absolute amplitude bands × polarity."""
    if not rows:
        return []
    picked: list[dict] = []
    caps: dict[tuple, int] = {}
    ordered = sorted(rows, key=lambda r: r["energy_frac_pm3"], reverse=True)
    for r in ordered:
        band = _amp_band(r["peak_uv"], config.amp_bands_uv)
        cell = (band, r["polarity"])
        if caps.get(cell, 0) >= config.per_cell_cap:
            continue
        caps[cell] = caps.get(cell, 0) + 1
        picked.append({**r, "amplitude_band": band})
        if len(picked) >= n_donors:
            break
    return picked


def build_donor_cohort(
    sort_dir: Path | str,
    recording_dir: Path | str,
    out_dir: Path | str,
    *,
    n_donors: int = 12,
    config: DonorConfig | None = None,
) -> dict:
    config = config or DonorConfig()
    out_dir = Path(out_dir)
    if str(out_dir).startswith("/mnt/"):
        raise ValueError("refusing to write a donor cohort under /mnt")
    out_dir.mkdir(parents=True, exist_ok=True)

    from spikeinterface.core import load
    from spikeinterface.preprocessing import astype, bandpass_filter

    sort = _load_sort(sort_dir)
    counts = pd.Series(sort["cl"]).value_counts()
    rec_manifest = json.loads((Path(recording_dir) / "rescue_recording_manifest.json").read_text())
    gain = float(rec_manifest["gain_uv_per_count"])
    recording = bandpass_filter(
        astype(load(recording_dir), "float32"), freq_min=300.0, freq_max=6000.0
    )
    rng = np.random.default_rng(config.seed)

    # cheap pass: compact de-whitened shape + spike/contam gates
    prelim: list[dict] = []
    for cid in sorted(sort["good"]):
        n = int(counts.get(cid, 0))
        if not (config.min_spikes <= n <= config.max_spikes_unit):
            continue
        if sort["contam_pct"].get(cid, 100.0) > config.max_contam_pct:
            continue
        shape, peak_c, polarity = _dewhitened_shape(sort, cid, config)
        comp = _compactness(shape)
        if not (
            comp["energy_frac_pm3"] >= config.min_energy_frac_pm3
            and comp["half_energy_width_ch"] <= config.max_half_energy_width_ch
        ):
            continue
        prelim.append({
            "cluster_id": int(cid), "n_spikes": n,
            "contam_pct": round(sort["contam_pct"].get(cid, np.nan), 2),
            "peak_channel": peak_c, "polarity": polarity,
            "energy_frac_pm3": comp["energy_frac_pm3"],
            "half_energy_width_ch": comp["half_energy_width_ch"],
            "_shape": shape,
        })
    prelim.sort(key=lambda r: r["energy_frac_pm3"], reverse=True)

    # expensive pass: true µV scale from a bandpass STA
    rows: list[dict] = []
    for r in prelim[: config.prefilter_top]:
        peak_uv = _bandpass_peak_uv(
            recording, sort["st"][sort["cl"] == r["cluster_id"]],
            r["peak_channel"], gain, config, rng,
        )
        if peak_uv < config.min_peak_uv:
            continue
        rows.append({**r, "peak_uv": round(peak_uv, 1), "_uv": r["_shape"] * peak_uv})

    picked = _select(rows, config, n_donors)
    templates: dict[str, np.ndarray] = {}
    manifest_rows = []
    for i, r in enumerate(picked, 1):
        tid = f"D{i:02d}"
        templates[tid] = validate_template(
            _taper(r["_uv"], config), edge_guard_samples=config.edge_guard_samples
        )
        manifest_rows.append({
            "template_id": tid,
            **{k: v for k, v in r.items() if not k.startswith("_")},
        })

    df = pd.DataFrame(manifest_rows)
    df.to_csv(out_dir / "donor_manifest.csv", index=False)
    if templates:
        np.savez(out_dir / "donor_templates.npz", **templates)

    result = {
        "schema": DONOR_SCHEMA,
        "config": asdict(config),
        "config_digest": config.digest,
        "sort_dir": str(sort_dir),
        "n_compact_candidates": len(rows),
        "n_donors": len(picked),
        "donor_ids": list(templates),
        "polarity_mix": df["polarity"].value_counts().to_dict() if len(df) else {},
        "amplitude_band_mix": (
            df["amplitude_band"].value_counts().to_dict() if len(df) else {}
        ),
        "peak_uv_range": (
            [float(df["peak_uv"].min()), float(df["peak_uv"].max())] if len(df) else None
        ),
    }
    (out_dir / "cohort_result.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    return result
