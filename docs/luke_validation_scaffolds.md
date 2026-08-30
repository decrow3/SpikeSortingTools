# Luke validation scaffolds

**Implemented:** 2026-08-29  
**Safety scope:** metadata, manifests, small CSVs and synthetic arrays only by
default; no sorter or motion-estimation jobs are launched.

These tools implement the validation guardrails described in
`luke_20250804_rescue_status_and_test_plan.md` without competing with the
active full-duration pilot.

## Sealed holdout planner

`testing/luke_holdout_validation.py` selects 24 untouched 30-second windows:
two probes by four session-time quartiles by three within-cell motion strata.
It enforces the discovery exclusions, session-edge margin and inter-window
spacing with deterministic seeded backtracking. It fails closed if any design
cell cannot be populated.

The manifest preregisters 16 later events per window (384 total): two events
per polarity-by-depth-quartile cell, balanced marginally across 4--6 sigma,
6--8 sigma and at least 8 sigma. The planner does not extract those events.
Existing reviewed events and patched-sort recovery status must not be used to
construct this cohort.

### Prospective holdout v2 draw complete

The production design supersedes the earlier 24-by-30-second planning
scaffold. Six 120-second windows are paired across both probes: relative-quiet
and high-motion epochs in each session-time third. The immutable v2 manifest
SHA-256 is
`01643baa20fd9ee4905a9bd6e9282ab25e1365dd4899bd69244ea63a4a7fcc9b`.
Before raw-voltage access, the extraction method was fixed to the matched
300--6000 Hz filter and 100-micrometer local median reference used by the
Luke--Yates audit; its method SHA-256 is
`630632a7245226f5ae6209d640e9932bd429d013c7ca42bec0cd55f75117d062`.

`testing/luke_draw_prospective_holdout_events.py` scanned both probes without
sorter output or labels. It detected 5,439,067 physically deduplicated
candidates and selected exactly four lowest-SHA candidates in every crossed
probe/window by depth-third, polarity and fixed 50--75, 75--100 and at-least
100 microvolt cell. All 216 cells met quota: 864 events were selected with no
borrowing and zero deficit. The full draw took 1,479.7 seconds using a CUDA
implementation of the sealed median operation.

CUDA was accepted only after the independently sealed six-chunk equivalence
gate in `testing/luke_holdout_backend_equivalence.py`. The representative set
crossed both probes and all three time thirds and contained three quiet and
three high-motion chunks. NumPy and CUDA had 0.0 microvolt maximum referenced
sample and candidate-amplitude differences in all chunks, with exactly equal
candidate coordinates, strata and SHA-selected identities. The protocol and
result are recorded in
`testing/outputs/luke_prospective_holdout/backend_equivalence_{protocol,result}_v2.json`.

Blinding is explicit:

- `holdout_reviewer_candidates_v2.csv` is the only reviewer-facing artifact
  and contains opaque `candidate_id` values only;
- `holdout_candidates_v2.csv` is an internal stratification manifest and is
  not reviewer-facing;
- `holdout_candidate_key_v2.csv` is the sealed coordinate/amplitude key and is
  not reviewer-facing; and
- `holdout_cell_deficits_v2.csv` and `holdout_output_roles_v2.json` record the
  complete quota audit and file roles/hashes.

Future waveform sheets and contact sheets must be generated from opaque IDs
and must not expose probe, window, motion, depth, polarity or amplitude strata
to reviewers. The cohort is prospective for future artifact/motion finalists;
it is not pristine retrospective validation of the already completed
full-session no-motion baseline.

Inspect the interface or run a synthetic planning smoke test with:

```bash
python testing/luke_holdout_validation.py --help
python testing/luke_holdout_validation.py \
  --mock-motion-summary \
  --duration imec0=10473.6 \
  --duration imec1=10473.6
```

Use `--motion-summary` instead of `--mock-motion-summary` for the eventual
sealed manifest. Printing to standard output is the default; the tool writes
only when `--output` is supplied. Freeze the real motion-summary source,
thresholds and manifest before opening finalist sort results.

## Injected-ground-truth scaffold

`testing/luke_injected_ground_truth_benchmark.py` defines the sealed donor,
qualification and evaluation splits; balanced depth, morphology/polarity, SNR,
collision and artifact-proximity strata; paired injected/uninjected controls;
ground-truth metrics; and a gated known-drift second phase.

The numerical contract injects only into an unconditioned `float32` raw-domain
view. It rejects direct `int16` injection, nonzero template edges, clipping and
boundary truncation. The current tool is deliberately a planning and synthetic
validation scaffold; it cannot open Luke data or invoke Kilosort.

```bash
python testing/luke_injected_ground_truth_benchmark.py \
  --plan-only \
  --synthetic-validation
```

The first real pilot should use one quiet and one pathological 30-second
window, 8--12 sealed raw-domain templates and CPU stage-level checks. Only
after those pass should paired no-motion-baseline and finalist short sorts be
run. Motion injection is phase two and must distinguish a fixed imposed field
from a re-estimated field.

### Discovery-only CPU pilot

`testing/luke_injected_ground_truth_pilot.py` advances the scaffold without
opening the prospective holdout. It deterministically selects ten donor events
from the existing manually reviewed neural discovery cohort and pairs every
donor with a different reviewed event from the same discovery unit/window for
qualification. Raw snippets are channelwise baseline-removed, spatially
restricted, tapered to zero, hashed, and injected only into float32 views of
the previously used 3,951 s quiet and 8,160 s pathological windows. Paired
uninjected/injected deltas are traced on CPU through phase correction,
saturation handling, CAR, high-pass, and the current interpolated/materialized
source stage.

The run is intentionally small (two 0.5 s backgrounds), runs no sorter or
motion estimator, and writes to
`testing/outputs/luke_injected_ground_truth_pilot/`. Its amplitude, cosine and
channel-localization results are **diagnostic discovery results**, not a
confirmatory evaluation. In particular, the reviewed cohort was selected by a
negative-peak detector, so this pilot cannot close the positive-dominant donor
gap. Inspect the raw-free plan before running:

```bash
python testing/luke_injected_ground_truth_pilot.py --plan-only
python testing/luke_injected_ground_truth_pilot.py --run
```

The real CPU pilot completed on 2026-08-29. It injected ten donor templates
into both discovery backgrounds (20 injections and 100 stage/event rows). Two
findings limit what may be concluded from it:

- The independent events assigned to the same discovery unit/window were not
  a reliable waveform-identity qualification set. Their median raw
  multichannel cosine was 0.047 (range -0.266 to 0.762); none of ten reached
  0.8 and only two reached 0.5. This directly supports keeping manual labels
  and Kilosort cluster membership descriptive rather than treating either as
  template ground truth.
- Relative to the phase-corrected injected-minus-control delta, the current
  CAR/high-pass/materialized stage retained a median 0.699 of peak amplitude
  and a median waveform cosine of 0.674 across the 20 injections. The median
  absolute peak-channel displacement was four channel indices and 55% exceeded
  two indices. Quiet and pathological backgrounds were similar at this small
  scale (median retention 0.699 versus 0.700; median cosine 0.662 versus
  0.701). These values measure deterministic stage distortion, not spike
  recall, and several localization/SNR/polarity cells contain only one to
  three events.

This pilot therefore validates the float32 injection and paired stage-tracing
adapter, but it does **not** validate the current pipeline or supply an
end-to-end truth set. Before a sorter benchmark, donors need an independently
qualified, polarity- and morphology-balanced waveform family set; the next
version should report physical displacement in micrometers in addition to
channel-index displacement. The compact receipt and tables are in
`testing/outputs/luke_injected_ground_truth_pilot/`.

## Acquisition-integrity and polarity-bank audit

`testing/luke_acquisition_integrity_audit.py` reads SpikeGLX metadata and binary
file statistics by default. It checks declared size, int16-frame divisibility,
duration, AP/LF and cross-probe alignment, channel-map/reference/IMRO/geometry
consistency and disconnected-site declarations. Full binary SHA-1 verification
is isolated behind the explicit slow `--verify-full-bin-sha1` flag and has not
been run.

The metadata/stat smoke test found:

- all Luke AP and LF binary sizes match `fileSizeBytes` and complete frames;
- AP/LF duration differences are 0.333 ms on imec0 and 0.200 ms on imec1;
- AP cross-probe start times span 0.496 ms and LF starts span 0.629 ms;
- gain, reference, IMRO and geometry metadata are consistent; and
- both probe metadata declare the same disconnected site near channel 191 at
  1900 micrometers.

The optional polarity-bank analysis consumes the existing small channel-event
CSV plus an explicit authoritative `channel,electrical_bank` mapping. It
controls a smooth depth trend and reports bank partial R-squared with a
permutation test. Without the mapping it reports the test as unavailable; it
does not infer an NP1 ADC mapping.

A follow-up spatial robustness audit separates **ADC identity** from the
12-step **sampling phase** and adds lateral-position fixed effects, a
piecewise-linear depth trend, ADC-block controls for the phase test, and
geometry-preserving cyclic nulls. The unrestricted bank-label permutation had
reported nominal ADC associations in eight of nine strata. Those associations
do **not** survive the spatial null: ADC-identity cyclic p-values are 0.33--0.97
across all nine strata. Sampling phase is significant in one shank-median
pathological stratum (p=0.010), borderline in the shared shank-median stratum
(p=0.050), and unsupported in the other seven. This is weak, preprocessing-
dependent evidence for a recurring phase pattern, not evidence that a specific
ADC caused the polarity anomaly. ADC identity remains structurally confounded
with a 240-micrometer depth block and channel parity on this single probe.

The compact outputs are
`testing/outputs/luke_acquisition_integrity_audit/polarity_adc_spatial_robustness.{csv,json}`.
They record hashes for the event, ADC mapping, geometry, and mapping-source
files. The aggregate row is descriptive because it reuses stages/windows; the
nine stratum rows are the inferential results.

```bash
python testing/luke_acquisition_integrity_audit.py --help
```

Do not run the full-file SHA-1 option alongside materialization or sorting.

## Rapid motion-residual laboratories

Two discovery-only tools now support millisecond-scale iteration on motion
resampling without launching Kilosort or opening the 648-event confirmatory
remainder.

`testing/luke_motion_snippet_residual_lab.py` applies candidate fields to eight
previously studied two-second snippets, learns provisional-unit templates in
one snippet fold and scores residuals in the other. This empirical route failed
closed: after excluding cross-unit coincidences, enforcing the dominant
detection template and requiring a minimally coherent baseline waveform
family, only four events from one provisional unit remained. Their baseline
residual fractions were 0.77--0.88. That is too little and too incoherent to
rank motion candidates, and reinforces the injection pilot's warning that the
present cluster labels are not an independent truth layer.

`testing/luke_synthetic_motion_residual_lab.py` is the usable first-stage
screen. It imposes the measured Luke time/depth displacement on five
well-supported discovery donor templates under three alternative assumptions
about the continuous waveform between contacts, applies candidate inverse
fields, and compares the result with the exact input waveform. The initial run
covered eight two-second snippet centers, nine candidates and 1,080 cases. It
does not read snippet voltage; the short intervals select measured motion-field
states. Therefore it tests resampling and field scaling, not motion estimation,
detection, collisions or clustering.

The residual-only winner was full nonrigid kriging with 20-micrometer spatial
scale: median residual fell from 0.054 without correction to 0.033. It was not
advanced because median amplitude retention fell from 0.874 to 0.866 and the
worst generator assumption showed additional amplitude error. Full nonrigid
IDW was rejected outright (median residual 0.071 and amplitude retention
0.731). A multi-objective gate requires improvement under every generator
assumption without materially worse median amplitude error or waveform cosine.
The already eligible quarter-strength rigid field passed this screen, as did
two small nonrigid additions to it:

- rigid 0.25 plus nonrigid residual 0.10, kriging p=2, sigma=20 micrometers;
- rigid 0.25 plus nonrigid residual 0.25, kriging p=2, sigma=20 micrometers.

These are candidates for the next real-voltage paired snippet test, not
production finalists. The next tier should inject the exact qualified waveform
into raw float32 snippets, subtract the matched uninjected branch after each
candidate warp, and score residual, amplitude, cosine and localization jointly.
Only candidates surviving that test should receive short paired sorts.

```bash
python testing/luke_synthetic_motion_residual_lab.py --maximum-templates 6
python testing/luke_motion_snippet_residual_lab.py --help
```

Compact receipts and tables are in
`testing/outputs/luke_synthetic_motion_residual_lab/` and the empirical smoke
test is in `testing/outputs/luke_motion_snippet_residual_lab_smoke/`.

## Validation

The isolated lightweight suite passes:

```bash
pytest -q \
  testing/test_luke_holdout_validation.py \
  testing/test_luke_injected_ground_truth_benchmark.py \
  testing/test_luke_acquisition_integrity_audit.py \
  testing/test_luke_motion_snippet_residual_lab.py \
  testing/test_luke_synthetic_motion_residual_lab.py
```

Current result: `30 passed`.
