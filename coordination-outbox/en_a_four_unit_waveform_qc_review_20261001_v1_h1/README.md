# Independent H1 review: saved-array four-unit waveform QC

Review date: 2026-10-01.

## Verdict

**GO for one H5-local execution under the exact gate-only enablement and
external resource/provenance conditions in `ONE_EXECUTION_GO.json`.** This is a
technical-coherence analysis only. It cannot establish biological identity,
purity, motion-estimator correctness or sorter benefit.

The candidate manifest
`5d0663f4a89ac6b1da6060ffcfff24167b92d018620a244525bad78774b21447`
and COMPLETE
`4335253e4dacc5432bbed5d8fbfac16905dcb0c5677694d58091cc59aed69214`
match the delegated values, and all seven manifest members verify. Source
`6c715ac4d0324cb2fa6125d7350ad667b584aed5753ffd9931b2f12252c0db83`
and config
`d3290efd1425f969367aaf2d3e36e7cd0065b614241439b56f6441accbe8b138`
match exactly. The review-pending preflight opened no outcome archive or
recording and found the planned output absent.

## Selection, axes and remap

The hash-bound selection receipt contains 20 unique slots: per unit, two
short-pair, two ordinary and one q0 slot for units 271, 278, 445 and 588. Pair
lags are 1–29 samples and equal `frame_2-frame_1`; singles have lag zero.
Archives are required to contain exactly those keys, `int16` channel-by-time
arrays with shape `(384,121+lag)` for pairs and `(384,121)` for singles.
Event indices are 60 and, for pairs, `60+lag`. Scale is raw ADC counts with
29,999.835983263598 Hz and global frame origin zero.

The exact mapping is target-to-source:
`source=(x_target,y_target+state_um)`, with unmatched targets force-zeroed and
an injective mapping required. The actual metadata preflight checks each
model-derived corrected anchor coordinate and every state-specific original
anchor against the hash-bound `(384,2)` geometry before outcomes are opened.
The configured states and anchors agree with the selection receipt.

These anchors derive from the prior Arm-A exported sorter templates. They are
appropriate frozen measurement locations for this technical QC, but they are
not independent evidence of biological identity. Exact-remap equality likewise
tests archive/adapter provenance and construction integrity, not motion benefit.

## Windows, stages and metrics

Signal windows are `[40,101)` for singles and `[40,101+lag)` for pairs.
Baselines concatenate `[0,30)` and `[111+lag,121+lag)`, always 40 samples and
disjoint from signal. Temporal centering subtracts each channel's full-member
mean. CAR then subtracts the across-channel median at each time. Raw int16 is
promoted to float64 without amplitude scaling.

Per-channel noise is `1.4826*MAD` on the frozen baseline with denominator
floored at one ADC count. PTP, PTP-SNR and RMS-z follow the stated formulas.
Peak-channel ties deterministically choose the lowest exact-max channel. Local
domains use Euclidean 80-um radius around the representation/state anchor;
local maximum/median SNR and global peak distance are reported. Shape
correlation centers the flattened exact-correspondence signal vectors and is
undefined for a zero norm; it is descriptive only and has no pass threshold.

Native high-pass channel claims are correctly excluded. The retained members
have only 60 samples of padding, not the original 512-sample capture padding;
reapplying Kilosort's FFT filter would change boundary semantics. The source
contains only raw, temporal-centered and CAR-centered stages.

## Controls and decision semantics

- All four q0 arrays must be whole-array identical.
- All 20 corrected arrays must exactly equal the frozen state-specific remap,
  including zero fill.
- For each of eight displaced ordinary slots, wrong-sign and no-shift
  full-array equality fractions must each be below 0.99.
- The historical duplicate-read control remains bound through capture lineage;
  the QC correctly does not invent another archived member or reread voltage.
- Only the 12 single-event slots contribute to visibility. Corrected-A
  CAR-centered visibility requires local max PTP-SNR at least 6 and the global
  peak within 80 um of the corrected anchor. A unit needs at least two of its
  three singles visible.
- PASS requires all integrity controls and at least three usable units;
  CAUTION requires intact controls and exactly two; everything else is FAIL.
  Archive hash/key/shape/dtype mismatches instead stop execution without a
  scientific verdict, which is the correct fail-closed distinction.

The real exact-remap and negative controls compare distinct saved outcomes and
fixed transforms, but exact remap is expected by construction. They must not be
described as independent biological validation. Shape correlation cannot rescue
or alter a verdict.

## Synthetic fixtures and circularity

The unmodified suite independently passed 6/6 on H1. The q0 fixture checks the
explicit identity map; force-zero checks explicit boundary indices. The
translation fixture uses the reviewed helper but asserts a separately stated
destination channel and singleton nonzero count, so it fixes direction rather
than merely comparing the helper with itself. Wrong-sign/no-shift fixtures use
distinct transforms and explicit `<0.99` outcomes. Window and robust-metric
fixtures have numeric known answers. Correlation tests identity and zero-norm
degeneracy.

The tests do not constitute a second independent remap implementation and use
a simplified one-column geometry. That is acceptable for unit coverage because
the outcome-independent metadata preflight applies the actual staggered
geometry and checks every frozen anchor. It limits the fixtures to regression
evidence, not scientific validation.

## Execution bounds and provenance conditions

Static inspection shows exactly four top-level archive opens: one hash and one
NPZ load per archive. Hash, exact key-set and per-member shape/dtype checks are
fail-closed before each member is used. Total numeric input must equal or remain
below 3,912,192 bytes; no recording path exists in the config. Output must be a
fresh root and `COMPLETE.json` is written last after CSVs, figures, validation
and decision.

The script enforces numeric-input, wall-after-work and output-size checks, but
does not itself constrain BLAS thread count, process memory or wall time while
work is in progress. It also does not write source/config/archive hashes into
the output root. Therefore GO requires an external managed invocation enforcing
two CPU threads, 512 MiB and 120 seconds, with durable command, enabled-config,
source/input hashes, stdout/stderr and final exit status. Post-run collection
must verify output size/schema and bind every result hash before publication.

One gate-only enabled config may be created outside the immutable candidate by
changing only status and the three gate values specified in
`ONE_EXECUTION_GO.json`. Any invocation, refusal, failure or completion consumes
the authorization. No parameter change, tuning or retry is allowed.

## Implementation checks

- Done: candidate packet, source, config, selection and lineage hashes -> exact
  expected bindings.
- Done: unit/state/channel anchors, axes/scale, windows and remap direction ->
  traced through source/config and metadata-only preflight; no outcome opened.
- Done: centering/CAR, robust metrics/correlations, thresholds, tie-breaking,
  pair exclusion and PASS/CAUTION/FAIL semantics -> code matches frozen rules.
- Done: q0, exact-remap, wrong-sign/no-shift and duplicate-control semantics ->
  technically coherent with scoped non-biological interpretation.
- Done: six synthetic fixtures -> 6/6 pass independently; circularity limits
  explicitly identified.
- Done: archive validation, output freshness/schema and high-pass exclusion ->
  fail-closed structure inspected; no native high-pass channel claim exists.
- Not done: real archive hashes/keys/shapes, metrics, plots or verdict ->
  deliberately deferred to the single authorized H5-local execution.
- Can establish: an outcome-independent, bounded technical-QC implementation
  suitable for one controlled execution.
- Cannot establish: any real waveform result, biological identity/purity,
  motion-estimator correctness, trained-sorter benefit or permission to tune.
