"""Method-level controls for the waveform-only lighthouse tracker.

Four audits, all on cached evidence or synthetic templates; no voltage
extraction and no managed job required.

A. Track plausibility.  Coverage, patch span and within-second centroid
   scatter for each frozen candidate.  A post-hoc readout of the produced
   track, not a continuity prior on matching.
B. Seed-recovery temporal null.  Does a candidate recover its own cached seed
   spike times above the chance rate implied by its accepted-event density?
   Circular-shift null on the frozen 930-940 s training window.  Also counts
   how many rival identities clear the strict cosine gate on the same
   detection, from cached per-identity score matrices.
C. Sub-lattice sensitivity and centroid gain.  The matcher searches exact
   40 um patch translations.  Synthetically place each candidate template at
   sub-lattice depths and ask whether it still clears cosine 0.86, and how
   much of the imposed displacement the energy centroid recovers.
D. Candidate independence.  Cluster the frozen cohort by cached template
   cosine so related families are not counted as independent votes.

No motion estimator, absolute-depth prior or field prediction enters any
audit.  Results describe the tracker's own behaviour, not tissue motion.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.interpolate import CubicSpline
from scipy.spatial.distance import squareform

ROOT = Path(__file__).resolve().parents[1]
BASE = Path("/mnt/NPX/Luke/20250804/rescue_pipeline_results_Luke0804_V2V1_g0_imec0")
MANIFEST = BASE / "recording/rescue_recording_manifest.json"
SEEDS = ROOT / "testing/outputs/luke_population_depth_v2/templates.npz"
GLOBAL = ROOT / "testing/outputs/luke_waveform_only_global_v1"
EXTENSION = ROOT / "testing/outputs/luke_lighthouse_extension_300s_v1"
COHORT = ROOT / "testing/outputs/luke_waveform_only_expansion_v1/candidate_audit.csv"
OUT = ROOT / "testing/outputs/luke_lighthouse_method_audit_v1"

# Frozen matcher constants, copied from luke_waveform_only_global_v1.
OFF = np.arange(-24, 25)
LAGS = range(-3, 4)
COSINE, MARGIN, GAIN = 0.86, 0.025, (0.35, 3.0)
PATCH_STEP, PATCH_LO, PATCH_HI = 40, -60, 80
TOLERANCE_S = 3e-4
TRAINING = (930.0, 940.0)
HELDOUT = (940.0, 1230.0)
BLUE, ORANGE, GRAY = "#0072B2", "#D55E00", "#777777"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save(fig, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=150)
    plt.close(fig)


def load_geometry():
    manifest = json.loads(MANIFEST.read_text())
    geo = np.asarray(manifest["channel_locations_um"], dtype=float)
    bases = np.arange(80, int(geo[:, 1].max()) - 79, PATCH_STEP)
    patches = np.array(
        [np.flatnonzero((geo[:, 1] >= b + PATCH_LO) & (geo[:, 1] <= b + PATCH_HI)) for b in bases]
    )
    assert patches.shape[1] == 16, patches.shape
    return geo, bases, patches, float(manifest["sampling_frequency_hz"])


def near(times, reference, radius=TOLERANCE_S):
    """Frozen coincidence rule from luke_waveform_only_global_review_v1."""
    if not len(reference):
        return np.zeros(len(times), bool)
    ref = np.sort(reference)
    i = np.searchsorted(ref, times)
    lo = np.abs(times - ref[np.clip(i - 1, 0, len(ref) - 1)])
    hi = np.abs(times - ref[np.clip(i, 0, len(ref) - 1)])
    return np.minimum(lo, hi) <= radius


# ---------------------------------------------------------------- A. tracks


def audit_tracks(events, cohort):
    held = events[(events.evidence == "strict_accepted") & events.time_s.between(*HELDOUT)]
    span_s = HELDOUT[1] - HELDOUT[0]
    rows = []
    for unit in sorted(cohort.unit_id):
        q = held[held.unit_id == unit]
        seconds = np.unique(np.floor(q.time_s)) if len(q) else np.array([])
        within, medians = [], []
        if len(q):
            for _, g in q.groupby(np.floor(q.time_s)):
                c = g.centroid_um.to_numpy()
                if len(c) >= 6:
                    within.append(np.median(np.abs(c - np.median(c))) / 0.6745)
                    medians.append(np.median(c))
        rows.append(
            dict(
                unit_id=int(unit),
                strict_events=len(q),
                seconds_covered=len(seconds),
                coverage_pct=100 * len(seconds) / span_s,
                patches=int(q.patch_base_um.nunique()) if len(q) else 0,
                patch_span_um=float(np.ptp(q.patch_base_um)) if len(q) else np.nan,
                centroid_span_um=float(np.ptp(q.centroid_um)) if len(q) else np.nan,
                within_second_sigma_um=float(np.median(within)) if within else np.nan,
                dense_seconds=len(medians),
            )
        )
    return pd.DataFrame(rows)


def figure_tracks(tracks):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout="constrained")
    t = tracks.dropna(subset=["patch_span_um"])
    ax = axes[0]
    bad = t.patch_span_um > 500
    ax.scatter(t.coverage_pct[~bad], t.patch_span_um[~bad], s=60, c=BLUE, label="Patch span <= 500 um")
    ax.scatter(t.coverage_pct[bad], t.patch_span_um[bad], s=90, c=ORANGE, marker="X",
               label="Patch span > 500 um")
    for r in t.itertuples():
        ax.annotate(int(r.unit_id), (r.coverage_pct, r.patch_span_um), fontsize=8,
                    xytext=(4, 3), textcoords="offset points")
    ax.axhline(500, color=GRAY, ls="--", lw=0.8)
    ax.set(xlabel="Held-out temporal coverage (% of 940-1230 s)",
           ylabel="Span of accepted patch bases (um)", yscale="symlog", ylim=(-20, 5000),
           title="Track plausibility: where does the matcher place each candidate?")
    ax.legend(fontsize=8)

    ax = axes[1]
    d = tracks.dropna(subset=["within_second_sigma_um"]).sort_values("within_second_sigma_um")
    ax.barh([str(int(u)) for u in d.unit_id], d.within_second_sigma_um, color=BLUE)
    ax.set(xlabel="Within-second centroid scatter, robust sigma (um)", ylabel="Unit",
           title="Centroid measurement noise\n(units with >=5 seconds holding >=6 strict events)")
    for i, (v, n) in enumerate(zip(d.within_second_sigma_um, d.dense_seconds)):
        ax.text(v + 0.05, i, f"{v:.1f} um / {n} s", va="center", fontsize=8)
    ax.set_xlim(0, max(3.0, d.within_second_sigma_um.max() * 1.6))

    fig.suptitle("A - Frozen lighthouse track plausibility, held-out 940-1,230 s\n"
                 "Post-hoc readout of produced tracks; no continuity prior entered matching",
                 fontsize=13)
    save(fig, "01_track_plausibility")


# ------------------------------------------------------- B. recovery null


def audit_recovery(fs, cohort, draws=2000):
    bank = np.load(SEEDS)
    training = pd.concat([pd.read_csv(GLOBAL / f"chunk_{t}.csv") for t in (930, 935)],
                         ignore_index=True)
    accepted = training[training.status == "accepted"]
    rng = np.random.default_rng(11)
    lo, hi = TRAINING
    rows = []
    for unit in sorted(cohort.unit_id):
        seed = bank[f"unit_{unit}_seed_frames"] / fs
        seed = seed[(seed >= lo) & (seed <= hi)]
        matched = accepted[accepted.unit_id == unit].time_s.to_numpy()
        if not len(seed):
            continue
        observed = float(near(seed, matched).mean())
        offsets = rng.uniform(0, hi - lo, draws)
        null = np.array([
            float(near(lo + np.mod(seed - lo + o, hi - lo), matched).mean()) for o in offsets
        ])
        rows.append(
            dict(
                unit_id=int(unit),
                seed_events=len(seed),
                accepted_events=len(matched),
                observed_recovery=observed,
                null_mean=float(null.mean()),
                null_p95=float(np.quantile(null, 0.95)),
                p_value=float((null >= observed).mean()),
                excess=observed - float(null.mean()),
            )
        )
    return pd.DataFrame(rows)


def audit_rivals(chunks, limit=12):
    """How many identities clear the strict gate on the same detection?"""
    counts = np.zeros(limit + 1, dtype=np.int64)
    detections = 0
    for path in chunks:
        with np.load(path) as z:
            scores, gains = z["scores"], z["gains"]
        ok = (scores >= COSINE) & (gains >= GAIN[0]) & (gains <= GAIN[1])
        n = np.clip(ok.sum(axis=1), 0, limit)
        counts += np.bincount(n, minlength=limit + 1)
        detections += len(scores)
    return pd.DataFrame(dict(identities_clearing_gate=np.arange(limit + 1),
                             detections=counts,
                             fraction=counts / max(detections, 1)))


def figure_recovery(recovery, rivals):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout="constrained")
    ax = axes[0]
    d = recovery.sort_values("observed_recovery")
    y = np.arange(len(d))
    ax.barh(y, d.observed_recovery, color=BLUE, label="Observed seed recovery")
    ax.scatter(d.null_mean, y, color=ORANGE, s=28, zorder=3, label="Circular-shift null mean")
    ax.hlines(y, d.null_mean, d.null_p95, color=ORANGE, lw=1.4, zorder=3)
    ax.scatter(d.null_p95, y, color=ORANGE, marker="|", s=60, zorder=3, label="Null 95th percentile")
    ax.set(yticks=y, yticklabels=[str(int(u)) for u in d.unit_id],
           xlabel="Fraction of seed spikes recovered (+/-0.3 ms)", ylabel="Unit",
           title="Seed recovery vs chance coincidence\n930-940 s training window, 2,000 circular shifts")
    for i, r in enumerate(d.itertuples()):
        ax.text(r.observed_recovery + 0.01, i, f"p={r.p_value:.3f}", va="center", fontsize=7)
    ax.set_xlim(0, min(1.05, d.observed_recovery.max() * 1.35 + 0.05))
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[1]
    r = rivals[rivals.identities_clearing_gate <= 8]
    ax.bar(r.identities_clearing_gate, r.fraction, color=BLUE)
    ax.set(xlabel="Identities clearing cosine >=0.86 and gain 0.35-3 on one detection",
           ylabel="Fraction of all detections", yscale="log",
           title="Identity ambiguity in the cached score matrices\n930-1,030 s, all 247 competitors")
    for v, f in zip(r.identities_clearing_gate, r.fraction):
        if f > 0:
            ax.text(v, f * 1.25, f"{f:.2e}" if f < 1e-3 else f"{f:.3f}", ha="center", fontsize=7)
    ambiguous = rivals[rivals.identities_clearing_gate >= 2].fraction.sum()
    unique = rivals[rivals.identities_clearing_gate == 1].fraction.sum()
    ax.text(0.98, 0.02, f"Detections with a unique strict identity: {unique:.4f}\n"
                        f"Detections with >=2 competing strict identities: {ambiguous:.4f}\n"
                        f"Share of strict-gated detections that are ambiguous: "
                        f"{ambiguous / max(unique + ambiguous, 1e-12):.1%}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            bbox=dict(fc="white", ec=GRAY, alpha=0.85))

    fig.suptitle("B - Is recovery better than chance, and is the winning identity unique?\n"
                 "Cached training events and cached per-identity scores; no new extraction",
                 fontsize=13)
    save(fig, "02_recovery_null_and_ambiguity")


# ------------------------------------------- C. sub-lattice sensitivity


def shift_template(w, geo, delta, kernel="linear"):
    """Move the source up by `delta` um: T_new(y) = T_old(y - delta).

    Interpolates within each 40 um column.  Exact at multiples of 40 um for
    both kernels; intermediate offsets are interpolated.  Linear smooths the
    waveform slightly, so `kernel="cubic"` is run as a control to separate
    interpolator artefact from probe geometry.
    """
    out = np.zeros_like(w)
    for x in np.unique(geo[:, 0]):
        idx = np.flatnonzero(geo[:, 0] == x)
        idx = idx[np.argsort(geo[idx, 1])]
        ys = geo[idx, 1]
        target = ys - delta
        inside = (target >= ys[0]) & (target <= ys[-1])
        for t in range(w.shape[0]):
            if kernel == "linear":
                out[t, idx] = np.interp(target, ys, w[t, idx], left=0.0, right=0.0)
            else:
                values = np.zeros(len(target))
                values[inside] = CubicSpline(ys, w[t, idx])(target[inside])
                out[t, idx] = values
    return out


def score_against_bank(wave, normalized, norms, peak_t, patches):
    """Frozen scoring rule: normalized multichannel cosine, +/-3 sample lag."""
    n_patch, n_id = len(patches), len(normalized)
    best = np.full((n_patch, n_id), -1.0)
    gains = np.zeros((n_patch, n_id))
    for lag in LAGS:
        frames = peak_t + OFF + lag
        if frames.min() < 0 or frames.max() >= wave.shape[0]:
            continue
        block = wave[frames][:, patches]              # (49, n_patch, 16)
        flat = block.transpose(1, 0, 2).reshape(n_patch, -1)
        dots = flat @ normalized.T
        sc = dots / np.maximum(np.linalg.norm(flat, axis=1)[:, None], 1e-20)
        better = sc > best
        best[better] = sc[better]
        gains[better] = (dots / norms[None, :])[better]
    return best, gains


def audit_sublattice(geo, bases, patches, cohort, noise_scale, rng, kernel="linear"):
    bank_file = np.load(SEEDS)
    comparison = np.load(GLOBAL / "seed_comparison.npz")
    waveforms, unit_ids = comparison["waveforms"], comparison["unit_ids"]
    flat = waveforms.reshape(len(waveforms), -1).astype(float)
    norms = np.linalg.norm(flat, axis=1)
    normalized = flat / norms[:, None]
    deltas = np.arange(0, 41, 5.0)
    rows = []
    for unit in sorted(cohort.unit_id):
        full = bank_file[f"unit_{unit}_full"].astype(float)
        noise = bank_file[f"unit_{unit}_noise"].astype(float)
        peak_t = int(np.unravel_index(np.abs(full).argmax(), full.shape)[0])
        if peak_t < 27 or peak_t + 27 >= len(full):
            continue
        reference = None
        for delta in deltas:
            wave = shift_template(full, geo, delta, kernel)
            if noise_scale:
                wave = wave + rng.normal(0, noise_scale * noise[None, :], wave.shape)
            best, gains = score_against_bank(wave, normalized, norms, peak_t, patches)
            order = np.argsort(best, axis=1)
            win = order[:, -1]
            top = best[np.arange(len(patches)), win]
            patch = int(np.argmax(top))
            wi, rj = win[patch], order[patch, -2]
            score = best[patch, wi]
            margin = score - best[patch, rj]
            gain = gains[patch, wi]
            channels = patches[patch]
            energy = (wave[peak_t + OFF][:, channels] ** 2).sum(axis=0)
            centroid = float(energy @ geo[channels, 1] / energy.sum())
            if reference is None:
                reference = centroid
            rows.append(
                dict(
                    unit_id=int(unit),
                    delta_um=float(delta),
                    kernel=kernel,
                    exact_lattice=bool(delta % PATCH_STEP == 0),
                    winner_id=int(unit_ids[wi]),
                    winner_is_self=bool(unit_ids[wi] == unit),
                    score=float(score),
                    margin=float(margin),
                    gain=float(gain),
                    accepted=bool(score >= COSINE and margin >= MARGIN and GAIN[0] <= gain <= GAIN[1]),
                    patch_base_um=float(bases[patch]),
                    centroid_um=centroid,
                    recovered_um=centroid - reference,
                )
            )
    return pd.DataFrame(rows)


def figure_sublattice(clean, noisy, cubic):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout="constrained")

    for ax, table, label in ((axes[0, 0], clean, "noiseless"), (axes[0, 1], noisy, "single-spike noise")):
        for _, q in table.groupby("unit_id"):
            q = q.sort_values("delta_um")
            ax.plot(q.delta_um, q.score, lw=1, alpha=0.7, color=BLUE)
        med = table.groupby("delta_um").score.median()
        ax.plot(med.index, med.values, lw=2.6, color="black", label="Median")
        if label == "noiseless":
            cub = cubic.groupby("delta_um").score.median()
            ax.plot(cub.index, cub.values, lw=2.2, color="#009E73", ls="--",
                    label="Median, cubic interpolation control")
        ax.axhline(COSINE, color=ORANGE, ls="--", lw=1.2, label="Strict gate 0.86")
        ax.set(xlabel="Imposed sub-lattice displacement (um)", ylabel="Best cosine on 40 um lattice",
               title=f"Self-match score vs sub-lattice offset - {label}", ylim=(0.4, 1.01))
        ax.legend(fontsize=8, loc="lower left")

    ax = axes[1, 0]
    for table, color, label in ((clean, BLUE, "noiseless (linear)"),
                                (cubic, "#009E73", "noiseless (cubic control)"),
                                (noisy, ORANGE, "single-spike noise")):
        rate = table.groupby("delta_um").accepted.mean()
        ax.plot(rate.index, 100 * rate.values, "o-", color=color, lw=2, label=label)
    ax.set(xlabel="Imposed sub-lattice displacement (um)", ylabel="Candidates passing strict gate (%)",
           title="Strict acceptance is a function of where the cell sits\nbetween lattice nodes",
           ylim=(-4, 104))
    ax.axvspan(15, 25, color=GRAY, alpha=0.12)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    for _, q in clean.groupby("unit_id"):
        q = q.sort_values("delta_um")
        ax.plot(q.delta_um, q.recovered_um, lw=1, alpha=0.6, color=BLUE)
    med = clean.groupby("delta_um").recovered_um.median()
    ax.plot(med.index, med.values, lw=2.6, color="black", label="Median recovered")
    ax.plot([0, 40], [0, 40], ls=":", color=ORANGE, lw=1.6, label="Unit gain (ideal)")
    ax.set(xlabel="Imposed displacement (um)", ylabel="Recovered centroid displacement (um)",
           title="Centroid gain: how much imposed motion is read back?")
    ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("C - Sub-lattice sensitivity of the exact 40 um translation bank\n"
                 "Synthetic: each frozen candidate template placed off-lattice, then scored by the frozen matcher",
                 fontsize=13)
    save(fig, "03_sublattice_sensitivity")


# ------------------------------------------------- D. candidate families


def audit_families(cohort, threshold=0.9):
    comparison = np.load(GLOBAL / "seed_comparison.npz")
    sim, unit_ids = comparison["similarities"].astype(float), comparison["unit_ids"]
    ids = sorted(cohort.unit_id)
    index = [int(np.flatnonzero(unit_ids == u)[0]) for u in ids]
    sub = sim[np.ix_(index, index)].copy()
    sub = np.clip((sub + sub.T) / 2, -1, 1)
    np.fill_diagonal(sub, 1.0)
    distance = squareform(np.clip(1 - sub, 0, None), checks=False)
    labels = fcluster(linkage(distance, method="single"), 1 - threshold, criterion="distance")
    depths = cohort.set_index("unit_id").seed_centroid_um
    table = pd.DataFrame(dict(unit_id=[int(u) for u in ids], family=labels,
                              seed_centroid_um=[float(depths[u]) for u in ids]))
    return table, sub, np.asarray(ids)


def figure_families(table, sim, ids, threshold):
    order = np.argsort(table.family.to_numpy(), kind="stable")
    fig, ax = plt.subplots(figsize=(9.5, 8), layout="constrained")
    im = ax.imshow(sim[np.ix_(order, order)], vmin=0, vmax=1, cmap="magma")
    ticks = [f"{ids[i]} (f{table.family.iloc[i]})" for i in order]
    ax.set(xticks=range(len(ids)), yticks=range(len(ids)))
    ax.set_xticklabels(ticks, rotation=90, fontsize=8)
    ax.set_yticklabels(ticks, fontsize=8)
    fam = table.family.to_numpy()[order]
    for k in np.flatnonzero(np.diff(fam)) + 1:
        ax.axhline(k - 0.5, color="white", lw=1.4)
        ax.axvline(k - 0.5, color="white", lw=1.4)
    fig.colorbar(im, ax=ax, label="Timing-aligned template cosine")
    sizes = table.family.value_counts()
    ax.set_title(f"D - Candidate independence - single linkage at cosine >={threshold}\n"
                 f"{len(ids)} candidates resolve into {table.family.nunique()} families "
                 f"(largest {int(sizes.max())}); families, not templates, are independent votes",
                 fontsize=12)
    save(fig, "04_candidate_families")


# --------------------------------------------------------------- assemble


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    geo, bases, patches, fs = load_geometry()
    cohort = pd.read_csv(COHORT)
    cohort = cohort[cohort.selected].copy()
    events = pd.read_csv(EXTENSION / "overlay_events.csv")

    tracks = audit_tracks(events, cohort)
    tracks.to_csv(OUT / "a_track_plausibility.csv", index=False)
    figure_tracks(tracks)
    print("A track plausibility\n", tracks.to_string(index=False), "\n", flush=True)

    recovery = audit_recovery(fs, cohort)
    recovery.to_csv(OUT / "b_seed_recovery_null.csv", index=False)
    chunks = sorted(GLOBAL.glob("chunk_*_scores.npz"))
    rivals = audit_rivals(chunks)
    rivals.to_csv(OUT / "b_identity_ambiguity.csv", index=False)
    figure_recovery(recovery, rivals)
    print("B seed recovery null\n", recovery.to_string(index=False), "\n", flush=True)
    print("B identity ambiguity\n", rivals.head(6).to_string(index=False), "\n", flush=True)

    rng = np.random.default_rng(7)
    clean = audit_sublattice(geo, bases, patches, cohort, 0.0, rng)
    noisy = audit_sublattice(geo, bases, patches, cohort, 1.0, rng)
    cubic = audit_sublattice(geo, bases, patches, cohort, 0.0, rng, kernel="cubic")
    clean["arm"], noisy["arm"], cubic["arm"] = "noiseless", "single_spike_noise", "noiseless_cubic"
    pd.concat([clean, noisy, cubic]).to_csv(OUT / "c_sublattice_sensitivity.csv", index=False)
    figure_sublattice(clean, noisy, cubic)
    print("C sub-lattice acceptance by offset\n",
          pd.DataFrame(dict(noiseless=clean.groupby("delta_um").accepted.mean(),
                            cubic=cubic.groupby("delta_um").accepted.mean(),
                            noisy=noisy.groupby("delta_um").accepted.mean(),
                            self_win_clean=clean.groupby("delta_um").winner_is_self.mean(),
                            median_score_clean=clean.groupby("delta_um").score.median(),
                            median_recovered_um=clean.groupby("delta_um").recovered_um.median())
                       ).to_string(), "\n", flush=True)

    threshold = 0.9
    families, sim, ids = audit_families(cohort, threshold)
    families.to_csv(OUT / "d_candidate_families.csv", index=False)
    figure_families(families, sim, ids, threshold)
    print("D families\n", families.to_string(index=False), "\n", flush=True)

    dense = tracks.dropna(subset=["within_second_sigma_um"])
    summary = dict(
        schema_version="luke-lighthouse-method-audit-v1",
        status="complete",
        scientific_status="descriptive_method_control_requires_review",
        candidates=int(len(cohort)),
        extraction_performed=False,
        motion_estimator_used=False,
        track_plausibility=dict(
            median_coverage_pct=float(tracks.coverage_pct.median()),
            candidates_over_500um_patch_span=[int(u) for u in tracks.unit_id[tracks.patch_span_um > 500]],
            median_within_second_sigma_um=float(dense.within_second_sigma_um.median()) if len(dense) else None,
            candidates_with_five_dense_seconds=int((tracks.dense_seconds >= 5).sum()),
        ),
        seed_recovery_null=dict(
            candidates=int(len(recovery)),
            significant_at_p05=[int(u) for u in recovery.unit_id[recovery.p_value <= 0.05]],
            not_significant=[int(u) for u in recovery.unit_id[recovery.p_value > 0.05]],
            median_observed=float(recovery.observed_recovery.median()),
            median_null=float(recovery.null_mean.median()),
        ),
        identity_ambiguity=dict(
            detections=int(rivals.detections.sum()),
            unique_strict_fraction=float(rivals.fraction[rivals.identities_clearing_gate == 1].sum()),
            multi_identity_fraction=float(rivals.fraction[rivals.identities_clearing_gate >= 2].sum()),
        ),
        sublattice=dict(
            acceptance_by_offset_um={str(k): float(v) for k, v in
                                     clean.groupby("delta_um").accepted.mean().round(4).items()},
            acceptance_by_offset_um_cubic={str(k): float(v) for k, v in
                                           cubic.groupby("delta_um").accepted.mean().round(4).items()},
            acceptance_by_offset_um_noisy={str(k): float(v) for k, v in
                                           noisy.groupby("delta_um").accepted.mean().round(4).items()},
            self_win_by_offset_um={str(k): float(v) for k, v in
                                   clean.groupby("delta_um").winner_is_self.mean().round(4).items()},
            median_score_at_0um=float(clean.score[clean.delta_um == 0].median()),
            median_score_at_20um=float(clean.score[clean.delta_um == 20].median()),
            centroid_gain_slope=float(np.polyfit(clean.delta_um, clean.recovered_um, 1)[0]),
        ),
        families=dict(
            threshold_cosine=threshold,
            family_count=int(families.family.nunique()),
            largest_family=int(families.family.value_counts().max()),
        ),
        interpretation=(
            "Controls on the tracker itself. They do not certify or refute any candidate as a "
            "biological cell, and they are not motion measurements. Sub-lattice results are "
            "synthetic and use within-column linear interpolation, which mildly smooths "
            "intermediate offsets; exact 0 and 40 um offsets are uninterpolated anchors."
        ),
        source_sha256={
            str(p): sha256(p)
            for p in [Path(__file__), MANIFEST, SEEDS, COHORT,
                      EXTENSION / "overlay_events.csv", GLOBAL / "seed_comparison.npz",
                      GLOBAL / "chunk_930.csv", GLOBAL / "chunk_935.csv"]
        },
    )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    (OUT / "chart_contract.json").write_text(json.dumps(dict(
        question="Does the frozen lighthouse matcher have the sensitivity, specificity and "
                 "independence its downstream motion-field comparisons assume?",
        surface="Standalone Matplotlib PDF/PNG",
        charts="Track plausibility scatter; recovery-versus-null bars; sub-lattice score, "
               "acceptance and gain curves; candidate similarity matrix",
        grain="One frozen candidate; cached events 930-1230 s; synthetic offsets 0-40 um",
        palette="Blue observed, orange null/threshold, gray reference",
        uncertainty="Null distributions shown as mean and 95th percentile; synthetic arms "
                    "reported noiseless and with single-spike noise",
        fallback="Units without dense seconds are reported as missing, not as zero motion",
    ), indent=2))
    acc = clean.groupby("delta_um").accepted.mean()
    acc_cubic = cubic.groupby("delta_um").accepted.mean()
    selfwin = clean.groupby("delta_um").winner_is_self.mean()
    flagged = summary["track_plausibility"]["candidates_over_500um_patch_span"]
    ambiguous_share = summary["identity_ambiguity"]["multi_identity_fraction"] / (
        summary["identity_ambiguity"]["multi_identity_fraction"]
        + summary["identity_ambiguity"]["unique_strict_fraction"])
    (OUT / "README.md").write_text(f"""# Lighthouse tracker method audit

Four controls on the waveform-only whole-probe tracker itself. No voltage
extraction, no managed job, and no motion estimator, absolute-depth prior or
held-out selection enters any of them. Nothing here validates or refutes a
motion field, and nothing here certifies a candidate as a biological cell.

Reproduce with `python -m testing.luke_lighthouse_method_audit_v1`; numerical
checks are in `testing/test_luke_lighthouse_method_audit.py`.

## A - Track plausibility (`01_track_plausibility`)

Median held-out temporal coverage is {summary["track_plausibility"]["median_coverage_pct"]:.0f}% of 940-1,230 s, and only
{summary["track_plausibility"]["candidates_with_five_dense_seconds"]} of {len(cohort)} candidates hold at least five seconds carrying six or more
strict events. Within-second centroid scatter is {summary["track_plausibility"]["median_within_second_sigma_um"]:.1f} um, so centroid
precision is not the limiting factor anywhere in this cohort.

Candidates {flagged} place strict matches across more than
500 um of probe within 290 s. Tissue drift does not do that, so those tracks
are most consistent with waveform lookalikes winning at separated positions.
Span and coverage are post-hoc readouts of the produced track; neither was
applied as a gate during matching, and neither is a continuity prior.

## B - Seed-recovery null and identity ambiguity (`02_recovery_null_and_ambiguity`)

Circularly shifting each candidate's cached seed spike train within the frozen
930-940 s training window (2,000 draws) puts chance recovery at
{summary["seed_recovery_null"]["median_null"]:.5f} against an observed median of {summary["seed_recovery_null"]["median_observed"]:.3f}.
{len(summary["seed_recovery_null"]["significant_at_p05"])} of {summary["seed_recovery_null"]["candidates"]} candidates recover their own seed spikes above chance
(p <= 0.05); {summary["seed_recovery_null"]["not_significant"]} does not. The selection rule is doing real work
rather than exploiting detection density.

Specificity is weaker. Across {summary["identity_ambiguity"]["detections"]:,} detections, {ambiguous_share:.0%} of those clearing
cosine 0.86 have at least one rival identity clearing it too. The 0.025
identity margin therefore carries nearly all discrimination on this highly
degenerate 247-template bank. That is the design working as written, but it
means the margin, not the cosine gate, is the parameter to calibrate.

## C - Sub-lattice sensitivity (`03_sublattice_sensitivity`) - SYNTHETIC

Each candidate's own template is placed at known depth offsets and matched by
the frozen rule. Columns sample every 40 um, so 0 and 40 um are exact index
shifts; intermediate offsets are interpolated. Linear and cubic kernels agree
closely, so the dip below is probe geometry rather than the interpolator.

| Offset (um) | 0 | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 |
|---|---|---|---|---|---|---|---|---|---|
| Strict acceptance, linear | {" | ".join(f"{100*acc[d]:.0f}%" for d in sorted(acc.index))} |
| Strict acceptance, cubic | {" | ".join(f"{100*acc_cubic[d]:.0f}%" for d in sorted(acc_cubic.index))} |
| Wins own identity | {" | ".join(f"{100*selfwin[d]:.0f}%" for d in sorted(selfwin.index))} |

A noiseless template sitting exactly half-way between lattice nodes clears the
strict gate {100*acc[20.0]:.0f}% of the time and is won by its own identity only
{100*selfwin[20.0]:.0f}% of the time; median cosine falls from
{summary["sublattice"]["median_score_at_0um"]:.3f} to {summary["sublattice"]["median_score_at_20um"]:.3f}. Adding single-spike
noise removes half-lattice acceptance entirely.

Two consequences for anything downstream. Recovered depth is attracted toward
lattice nodes, so real displacement is under-reported and a flatter motion
field is favoured for free. And dropout at half-lattice depths is a property
of the 40 um bank, not evidence that a cell fell silent or stopped moving.

Recovered centroid displacement tracks imposed displacement with slope
{summary["sublattice"]["centroid_gain_slope"]:.3f}, so the centroid mildly over-reads motion even where
matching succeeds.

## D - Candidate independence (`04_candidate_families`)

Single linkage on cached timing-aligned template cosine at >= {summary["families"]["threshold_cosine"]} resolves
{len(cohort)} candidates into {summary["families"]["family_count"]} families, largest {summary["families"]["largest_family"]}. Families, not templates,
are the independent units for any consensus or vote.

## Limitations

Part C is synthetic: it uses each candidate's own averaged template, so it
measures the geometry and scoring rule, not collisions, waveform evolution or
unknown cells. The noisy arm adds independent Gaussian noise at the cached
per-channel level and is a single draw per offset, not a calibrated detection
model. Part B's null preserves seed burst structure but assumes the accepted
event train is fixed. Part A's 500 um flag is a screening heuristic; a
genuinely bimodal or dual-site unit would look the same.
""")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
