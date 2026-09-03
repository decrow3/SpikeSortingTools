"""Synthetic parametric compact spike templates — the D2b-3 sharpness axis.

D2b-2 built 14 real compact donors from the imec0 sort, but they top out near
295 µV and the compact clean units thin above ~250 µV. To ask *"does voltage
interpolation damage sharp, high-SNR waveforms more than broad ones?"* — the
D2b-3 tradeoff question, and the T01 warning from Candidate 2 — the cohort needs
a controlled sharpness × amplitude grid that real data does not supply.

`synthetic_template` builds one: a parametric extracellular action potential
(fast negative trough + slower positive repolarisation) with an explicit
`trough_width_ms` (sharpness) and a `1/(1+(Δy/λ)²)` spatial decay, scaled to a
target µV peak. `build_synthetic_cohort` writes a `donor_templates.npz` +
manifest in the same format as `ladder_donors`, so the two cohorts merge.

Nothing is read from disk; nothing is written under /mnt.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from pipeline.config import fingerprint
from testing.ladder_donors import DonorConfig, _taper
from testing.luke_injected_ground_truth_benchmark import validate_template

SYNTHETIC_DONOR_SCHEMA = "luke-ladder-synthetic-donor-v1"


@dataclass(frozen=True)
class SyntheticSpec:
    peak_uv: float
    trough_width_ms: float          # FWHM of the negative trough — the sharpness knob
    spatial_lambda_um: float = 25.0  # 1/(1+(Δy/λ)²) decay; smaller = more compact
    polarity: str = "neg"
    repol_ratio: float = 0.35        # positive overshoot amplitude / trough amplitude
    trough_to_peak_ms: float | None = None  # default: 0.4 + trough_width_ms

    @property
    def label(self) -> str:
        return (
            f"S_{self.polarity}_{self.peak_uv:.0f}uV_{self.trough_width_ms*1000:.0f}us"
            f"_l{self.spatial_lambda_um:.0f}"
        )


def synthetic_template(
    spec: SyntheticSpec,
    *,
    fs: float,
    n_samples: int = 61,
    n_channels: int = 33,
    channel_pitch_um: float = 10.0,
    config: DonorConfig | None = None,
) -> np.ndarray:
    config = config or DonorConfig()
    t = np.arange(n_samples, dtype=np.float64)
    t_trough = n_samples * 0.42
    sig_tr = (spec.trough_width_ms * fs / 1000.0) / 2.3548  # FWHM -> sigma
    tp_ms = spec.trough_to_peak_ms if spec.trough_to_peak_ms is not None else (
        0.4 + spec.trough_width_ms
    )
    t_peak = t_trough + tp_ms * fs / 1000.0
    sig_pk = 1.8 * sig_tr

    trough = -np.exp(-0.5 * ((t - t_trough) / sig_tr) ** 2)
    repol = spec.repol_ratio * np.exp(-0.5 * ((t - t_peak) / sig_pk) ** 2)
    w = trough + repol
    w = w / np.abs(w).max()
    if spec.polarity == "pos":
        w = -w

    centre = n_channels // 2
    dy = (np.arange(n_channels) - centre) * channel_pitch_um
    spatial = 1.0 / (1.0 + (dy / spec.spatial_lambda_um) ** 2)

    template = (w[:, None] * spatial[None, :]) * spec.peak_uv
    tapered = _taper(template.astype(np.float32), config)
    return validate_template(tapered, edge_guard_samples=config.edge_guard_samples)


def default_grid() -> list[SyntheticSpec]:
    """Sharpness × amplitude, plus a broad-footprint and a positive control.

    Trough width is held in the range Kilosort's universal templates actually
    detect: D2b-3's first pass found 0.55 ms troughs (~16 samples) do not sort
    even stationary — KS4 reads them as LFP, not spikes. Real somatic APs run
    ~0.2-0.45 ms.
    """
    specs: list[SyntheticSpec] = []
    for peak in (120.0, 300.0, 500.0):
        for width in (0.20, 0.30, 0.45):
            specs.append(SyntheticSpec(peak, width, spatial_lambda_um=25.0))
    specs += [
        SyntheticSpec(500.0, 0.20, spatial_lambda_um=18.0),   # very compact footprint
        SyntheticSpec(300.0, 0.20, spatial_lambda_um=45.0),   # sharp but broad footprint
        SyntheticSpec(300.0, 0.25, polarity="pos"),
        SyntheticSpec(120.0, 0.25, polarity="pos"),
    ]
    return specs


def build_synthetic_cohort(
    out_dir: Path | str,
    *,
    fs: float = 29999.835983263598,
    specs: list[SyntheticSpec] | None = None,
    config: DonorConfig | None = None,
) -> dict:
    out_dir = Path(out_dir)
    if str(out_dir).startswith("/mnt/"):
        raise ValueError("refusing to write a synthetic cohort under /mnt")
    out_dir.mkdir(parents=True, exist_ok=True)
    config = config or DonorConfig()
    specs = specs or default_grid()

    import pandas as pd

    templates: dict[str, np.ndarray] = {}
    rows = []
    for i, spec in enumerate(specs, 1):
        tid = f"S{i:02d}"
        tmpl = synthetic_template(spec, fs=fs, config=config)
        templates[tid] = tmpl
        e = np.sqrt((tmpl.astype(np.float64) ** 2).sum(axis=0))
        pk = int(np.argmax(e))
        rows.append({
            "template_id": tid, "source": "synthetic", "label": spec.label,
            "peak_uv": round(float(np.abs(tmpl).max()), 1),
            "trough_width_ms": spec.trough_width_ms,
            "spatial_lambda_um": spec.spatial_lambda_um,
            "polarity": spec.polarity,
            "energy_frac_pm3": round(float(e[max(0, pk - 3): pk + 4].sum() / e.sum()), 3),
        })
    np.savez(out_dir / "donor_templates.npz", **templates)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "donor_manifest.csv", index=False)

    result = {
        "schema": SYNTHETIC_DONOR_SCHEMA,
        "config_digest": fingerprint({"stage": "synthetic_donor", **asdict(config),
                                      "specs": [s.label for s in specs]}),
        "n_donors": len(templates),
        "donor_ids": list(templates),
        "polarity_mix": df["polarity"].value_counts().to_dict(),
        "peak_uv_range": [float(df["peak_uv"].min()), float(df["peak_uv"].max())],
        "trough_width_ms_values": sorted(df["trough_width_ms"].unique().tolist()),
    }
    (out_dir / "cohort_result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
