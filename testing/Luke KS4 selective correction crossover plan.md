# Luke KS4 selective voltage-correction crossover plan

## Status

Proposed on 2026-08-31 as a narrow follow-up to the failed KS4 4.0.27 native
operator audit. This plan does not reverse that failure and does not authorize
a supplied-trajectory sort.

The completed discovery data show no complete crossover on the tested
`1, 2, 4, 6, 10, 20 µm` grid. Residual and cosine first favor native correction
in every forward-generator/sign stratum at 20 µm, but amplitude preservation
still fails the frozen threshold. The selective-correction hypothesis therefore
remains plausible but unvalidated.

## Refined interpretation of the completed audit

Do not summarize the result as “KS4 voltage warping is bad for Luke.” The
supported conclusion is:

> The tested KS4 4.0.27 `sig_interp=20` operator did not have a favorable
> cost-benefit ratio across the tested displacement amplitudes and continuous
> waveform models.

This is compatible with motion correction being beneficial in regimes where
residual-motion cost exceeds interpolation cost. Published benchmarks include
both benign no-interpolation/no-drift controls and large-motion recordings in
which high-bandwidth voltage correction materially improved sorting. See:

- [Garcia et al., modular drift-correction benchmark](https://pmc.ncbi.nlm.nih.gov/articles/PMC10897502/)
- [Windolf et al., DREDge across species](https://pmc.ncbi.nlm.nih.gov/articles/PMC13055889/)

The relevant design principle is not “motion enabled” versus “motion disabled.”
It is whether the application policy pays interpolation cost only when the
predicted uncorrected-motion penalty is larger.

## Primary question

Can an **exact-identity dead zone** plus a prespecified high-displacement KS4
correction regime outperform both always-off and always-on policies while
preserving waveform amplitude, separability, and temporal continuity?

## Non-negotiable identity behavior

When correction is inactive, the output must be the original preprocessed
voltage exactly—not KS4's `M(0)` matrix and not an approximately identity
interpolation matrix.

For an input batch `X_t` and motion estimate `d_t`, the initial policy class is:

```text
|d_t| <= d_off: output the exact unwarped branch
|d_t| >= d_on:  output the native KS4 corrected branch
d_off < |d_t| < d_on: retain the prior state (hysteresis)
```

Require `d_on > d_off`. Hysteresis prevents state chatter, but it also makes the
operator history-dependent and must be tested explicitly.

The displacement origin must be frozen from an independently justified
reference state. Because estimated motion has an arbitrary offset, a threshold
on raw `|d|` is meaningless without this reference.

## Why the existing grid is insufficient

The original audit was preregistered to judge an always-on operator across all
displacements. Its displacement stratification is now post hoc discovery
evidence. It also tested independent static cases, not transitions between
identity and warped states.

Current discovery summary:

| Magnitude | Worst residual delta | Worst amplitude-error delta | Worst cosine delta | Complete pass |
| ---: | ---: | ---: | ---: | :---: |
| 1 µm | +0.0397 | +0.0341 | −0.0013 | No |
| 2 µm | +0.0731 | +0.0470 | −0.0030 | No |
| 4 µm | +0.1401 | +0.0394 | −0.0109 | No |
| 6 µm | +0.2001 | +0.0616 | −0.0241 | No |
| 10 µm | +0.2512 | +0.0989 | −0.0564 | No |
| 20 µm | −0.1875 | +0.1015 | +0.1496 | No |

Values are the least favorable signed generator-stratum medians. Lower is
better for residual and amplitude error; higher is better for cosine. The exact
reproducible tables are in
[`selective_correction/`](outputs/luke_ks4_native_operator_audit/selective_correction/).

## Confirmatory operator experiment

### Frozen grid

Use a refined signed displacement grid centered on the apparent residual/cosine
transition:

```text
0, ±8, ±10, ±12, ±14, ±16, ±18, ±20, ±24, ±30 µm
```

Retain the original six donor identities, three backgrounds, three independent
forward generators, full geometry, frozen whitening, and exact KS4 4.0.27
matrix construction. Add donor identities only through a separately frozen
replication cohort; do not replace unfavorable donors.

### Primary crossover definition

A magnitude is correction-favorable only if both signs and every forward
generator pass all three original paired-median conditions:

```text
median delta residual <= -0.005
median delta absolute amplitude error <= +0.005
median delta cosine >= 0
```

It must also preserve the original lower-tail pair-separability and projected
separation-to-noise gates. Define a **robust crossover region**, not strict
pointwise mathematical monotonicity. The crossover `d*` is the smallest
magnitude above which the complete condition passes at both signs and all
admissible forward generators for at least three successive displacement
levels, allowing small local reversals within already non-inferior metrics. A
single favorable point is not a usable threshold, and no metric may cross back
into inferiority inside the claimed region.

If no complete `d*` exists, stop. Do not define a selective policy from residual
or cosine alone.

### Relative-support stratification

Report displacement not only in microns but relative to:

- 20-µm vertical site pitch;
- each donor's energy-weighted spatial footprint;
- distance to the usable recording boundary; and
- local count of channels carrying prespecified waveform energy.

This tests the hypothesis that correction value depends on motion relative to
available waveform support rather than displacement alone. It does not permit
selecting a different threshold per donor after outcomes are visible.

## Switching and transition audit

If and only if a complete crossover exists, construct time sequences with known
motion that cross `d_off` and `d_on` at controlled speeds. Include:

- slow ramps;
- 2–6-s excursions matching Luke's rapid-motion regime;
- one-batch impulses;
- repeated threshold crossings; and
- long stationary epochs on both sides of the threshold.

Compare four policies:

1. exact identity, always off;
2. native KS4 correction, always on;
3. hard threshold without hysteresis;
4. frozen `d_off`/`d_on` hysteretic policy.

Score the original waveform and separability metrics plus:

- sample continuity at policy transitions;
- template-state continuity across identity-to-correction and correction-to-
  identity boundaries;
- within-family state distance relative to the nearest competing waveform;
- transient matched-filter score and residual energy;
- artificial common-mode or spatial steps;
- detection-statistic bursts or troughs near switches;
- threshold chatter and corrected-duty fraction; and
- recovery latency after the excursion.

A selective policy fails if switching artifacts erase its steady-state benefit.
It also fails if the corrected branch creates a separate waveform state that
would itself require post-sort merging, even when scalar voltage residuals
improve.

## Frozen real-data segment panel

Any eventual bounded real-voltage comparison must use the six non-overlapping
10-minute segments in
[`Luke identity through motion segment panel plan.md`](Luke%20identity%20through%20motion%20segment%20panel%20plan.md),
not one favorable motion episode. The nested anchors span approximately 8, 15,
and 33 µm median cross-probe DREDGE excursion in the cleaner motion gradient,
plus a 41 µm motion×input-anomaly interaction and sustained-noise/support-
dropout controls.

Report every endpoint separately by segment and test for a graded benefit with
supported motion magnitude. Selective correction should be exactly inactive in
the relative-quiet and unsupported regimes unless the independently qualified
uncertainty rule says otherwise. A gain in the largest-motion segment cannot
compensate for damage in quiet, moderate, or control segments.

Use the shared time-ordered identity audit to determine whether correction
makes candidate waveform trajectories more continuous or creates a new state
at on/off transitions.

## Motion-estimate uncertainty gate

The known-displacement operator experiment cannot establish that a real Luke
batch belongs above the crossover. Before any real trajectory application:

- use an independently qualified rigid estimate;
- freeze its zero/reference state;
- propagate an uncertainty or support interval per batch;
- enter correction only when the lower confidence bound exceeds `d_on`;
- return to identity only when the upper confidence bound falls below `d_off`;
- leave unsupported batches unwarped; and
- report sensitivity to nearby reference-state choices without retuning `d*`.

Sorter labels must not select the threshold or the reference state.

## Advancement ladder

1. Reproduce the post hoc crossover table from the existing artifacts.
2. Run the refined confirmatory operator grid.
3. Require a robust complete crossover region across generator, sign, waveform,
   amplitude, and separability gates.
4. Pass the synthetic switching/transition audit.
5. Validate the real motion estimate and its threshold classification
   independently.
6. Run no-correction versus selective-correction sorts on the frozen multi-
   segment panel, retaining an always-on arm only as a mechanistic comparator.
7. Advance only on reviewed-event recovery, family continuity, refractory and
   duplicate burden, residual explanation, and edge safety—not unit count.

The KS2 native-tracking test remains independent and may proceed in parallel
once its MATLAB/patched-source installation gate is resolved. A selective KS4
result must not be used to tune or reinterpret the KS2 gate.

## Stop rules

Stop the selective path if:

- no robust complete crossover region exists;
- the apparent threshold depends on one forward generator or displacement sign;
- amplitude or lower-tail separability still fails where residual improves;
- exact identity cannot be guaranteed in the dead zone;
- switching creates detection or waveform discontinuities;
- the real trajectory's uncertainty spans the crossover for most candidate
  batches; or
- empirical waveform-versus-depth smoothness remains unvalidated when the plan
  reaches real voltage application.
