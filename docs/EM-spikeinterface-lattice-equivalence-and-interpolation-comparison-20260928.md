# EM — reproduce the exact lattice remap with SpikeInterface, then compare correction operators

**Status:** EM.1 CLOSED — STOCK FAIL AND EXACT-ADAPTER QUALIFICATION FAIL;
EM.2A/EM.2B STOPPED.

The bounded huklaban5 voltage audit is complete. The flat `q=0` control is
byte-exact after reproducing DD's parent-read chunks, and all tested zero masks
match. The two nonzero-motion snippets retain 1–2-float32-ULP differences because
the required SpikeInterface 0.104.7 execution reconstructs some upstream parent
values differently from DD's 0.104.8 materialization. See the
[huklaban5 result](EM-huklaban5-voltage-equivalence-result-20260928.md). This is
an EM.1 qualification failure, so the conditional operator screen and sorts do
not proceed.

- **Published handoff:**
  `/mnt/NPX/Luke/DARTsort_motion_experiments/em_spikeinterface_lattice_20260928/huklaban1_v1`
  (`MANIFEST.json` SHA-256
  `10cc8045845efcb52adc3b19ba70c06b0e7a4766fc652badd650e9a657b5a27b`;
  all 15 payload files verified before `COMPLETE.json` was written last).

- **Implemented here:** `npx_preprocessing/motion/lattice_remap_si.py`, its
  geometry-audit runner, and synthetic tests. Under the local SpikeInterface
  0.102.1 environment, the stock `nearest + force_zeros` control substitutes a
  surviving electrode at an interior hole, while the exact adapter zero-fills
  it.
- **Actual-geometry result:** the no-voltage audit of all 1,456 realized
  W2 `(q, target)` pairs found 1,448 exact mappings and eight unsupported
  targets. Local SpikeInterface 0.102.1 zeroed the four outer-border targets
  correctly but substituted a neighbor for four AP191-hole targets under both
  local 0.102.1 and the required 0.104.7, so stock equivalence failed with four
  mismatches. See
  [EM local mapping result](EM-lattice-mapping-audit-result-20260928.md).
  The bounded 9-second huklaban5 comparison is complete and failed strict byte
  equivalence for the exact adapter under the required 0.104.7 path. The
  mapping and zero-fill decisions matched, but upstream preprocessing-version
  differences remained in the nonzero-motion source channels.

- **Canonical home:** `/home/huklab/Documents/RyanSorting/SpikeSortingTools` (this repository). Put the reusable helper and tests here; do not modify the DARTsort repository.
- **Execution locality:** raw voltage remains on huklaban5. If an actual-voltage check or sort must run there, execute this repository's reviewed commit (or an exact source snapshot with recorded SHA-256 hashes) in the installed SpikeInterface 0.104.7 environment. No voltage is copied off huklaban5.
- **Scope:** EM authorizes the bounded source audit, geometry fixtures, and frozen voltage snippets below. EM.2a is a conditional CPU follow-up. Each new EM.2b spike sort requires explicit run authorization after arm selection and resource measurement. Nothing here authorizes production changes, RF work, outer-holdout access, or an automatic W3 run.
- **Name:** retain EM unless the experiment ledger shows that EM has already been allocated.

## Question and interpretation

DD applies a rigid, whole-period remap on the NP1.0 lattice. For output site
`j` with coordinate `(x_j, y_j)` and shift `q`, its mapping is

```text
M_q[j] = k  if (x_k, y_k) = (x_j, y_j + q)
         -1 otherwise
```

`-1` is zero-filled. The 40-µm shift preserves the alternating NP1.0 lateral
phase, so ordinary fully supported sites have an exact source electrode.
However, the actual parent contains 383 supported channels with AP191 excluded.
That creates an interior support hole. SpikeInterface's stock `nearest` kernel
selects the closest surviving electrode; it does not by itself mean “exact
coordinate or zero.” `force_zeros` may handle outer probe borders without
reproducing an interior missing site.

EM therefore tests two distinct claims:

1. **Stock equivalence:** does SpikeInterface 0.104.7 `nearest` plus its border
   policy reproduce DD everywhere on the executed 383-channel geometry?
2. **Adapter equivalence:** if stock equivalence fails, can a thin
   SpikeInterface recording/helper in this repository enforce DD's exact-support
   mask while retaining SpikeInterface's lazy recording interface?

A localized, explained stock failure is a useful result. Do not tune parameters
to conceal it or call a custom exact-coordinate adapter “stock nearest.”

The later methods comparison is a comparison of correction pipelines. It must
not attribute a difference to interpolation alone when field quantization also
differs. Rate-rank correlation is a continuity proxy, not proof of biological
identity or purity.

## EM.1 — source, geometry, and exact equivalence (CPU; no sort)

### 1. Freeze sources and conventions

Record repository commit, file hashes, Python and dependency versions, and cite
file/line ranges from the code actually executed.

From installed SpikeInterface 0.104.7, document:

- `interpolate_motion`/`InterpolateMotionRecording` target construction and
  displacement sign;
- conversion of motion centers to temporal-bin edges and frame assignment,
  including midpoint ties and segment/chunk offsets;
- `get_spatial_interpolation_kernel(method="nearest")`, including 2-D distance,
  tie behavior, and any distance cutoff;
- the exact meanings of `force_zeros`, `force_extrapolate`, and
  `remove_channels` for outer and interior unsupported coordinates;
- input/output dtype and whether a nominal one-hot kernel is numerically a pure
  copy.

From the executed DD source/materializer, document:

- the actual 383-channel parent geometry and excluded AP191 coordinate;
- the 40-µm NP1.0 period and alternating lateral coordinates;
- reference offset `r` (W2: −4.69558824159 µm), half-away-from-zero rounding,
  and `q = 40 * round((d-r)/40)` inside the frozen mask;
- sign convention (`corrected depth = observed depth − displacement`);
- exact motion centers and half-open cells `[t-0.125, t+0.125)`, with midpoint
  ties assigned to the later cell;
- exact-coordinate lookup, zero-fill, preprocessing boundary, channel order,
  crop, scaling, and materialized dtype.

Preserve the existing DD limitation in every report: the original actual-voltage
gate failed (6/10 q0 and 9/10 remap catalogue results), although the paired input
integrity check passed. EM must not silently relabel that gate as passing.

### 2. Cheapest decisive check: compare mappings without voltage

Before reading voltage, enumerate every realized W2 `q` value and every parent
target channel. For each `(q, target)` compare:

- DD exact-coordinate source index or `-1`;
- SpikeInterface nearest source index and distance;
- support class: exact, outer-border missing, AP191/interior missing, tie, or
  other substitution.

Write a compact CSV plus summary JSON containing counts and the first mismatch
in each class. Assert equality on every exactly supported target. Stock full
equivalence passes only if every DD `-1` target is also zero in the
SpikeInterface output and no nearest substitution occurs.

This geometry-only result determines the implementation path:

- **Stock pass:** use the stock lazy recording with frozen arguments.
- **Stock fail, localized and explained:** implement a small exact-coordinate
  adapter that uses DD's mapping/mask and zero-fill semantics. Keep the stock
  result as a control and test the adapter independently.
- **Other failure:** stop EM.1 and report it before changing code or reading
  voltage.

### 3. Strong synthetic fixtures

Use asymmetric, channel-coded traces on an NP1-like multirow geometry—not a
single-channel fixture. Test:

- `q in {-80, -40, 0, 40, 80}` and both correction signs;
- a sequence containing multiple `q` transitions;
- exact midpoint samples and samples immediately on either side;
- outer top/bottom support loss;
- an AP191-like interior missing coordinate;
- chunked and unchunked reads of the same interval;
- nonzero segment start frames;
- preservation of channel order, dtype, and `q=0` bytes.

For every case, compare the complete expected mapping and zero mask. If stock
nearest substitutes across the interior hole, the fixture must demonstrate the
failure rather than weakening the expected result.

### 4. Frozen actual-voltage comparison

Only after the mapping audit and fixtures pass for the chosen implementation,
compare against the DD W2 materialization on huklaban5.

- Select deterministically, before reading voltage: 3 s of low-deviation flat
  time, 3 s from an episode core, and 3 s spanning an onset or offset. Save the
  selection rule, source rows, sample bounds, and frame-level `q` vector.
- First assert that the implementation's complete frame-level `q` vector and
  mapping IDs equal DD's, including every seam sample.
- Apply correction at DD's exact preprocessing boundary and on its exact parent
  channel order, then perform the same crop and dtype conversion.
- For a pure permutation of the same native parent values, require bytewise
  equality. If the SpikeInterface interface necessarily converts dtype, also
  compare before conversion and require exact equality where representable;
  report maximum absolute error separately rather than using it to excuse a
  mapping mismatch.
- Require identical zero positions. Report first differing frame/channel,
  values, `q`, mapping IDs, support class, and cause.

EM.1 reports two verdicts separately: **stock pass/fail** and **exact-adapter
pass/fail**. A passing adapter does not convert a stock failure into a stock
pass.

### 5. Repository deliverables

Place the reusable code under, for example,
`npx_preprocessing/motion/lattice_remap_si.py`, with unit tests in this
repository's test tree. Also save the source audit, mapping table, summary JSON,
and frozen snippet manifest. Commit only compact code and evidence; never commit
voltage.

## EM.2a — operator screen before sorting (CPU; cached snippets)

EM.2a is conditional on an EM.1 pass for the selected exact implementation
(stock or adapter). On the same frozen snippets,
construct the following operator matrix with every parameter explicitly pinned
(no unspecified “defaults”):

| Field | Exact lattice | SI nearest | SI IDW | SI kriging |
|---|---:|---:|---:|---:|
| lattice-rounded | primary DD control | stock control | bridge | bridge |
| unrounded two-layer rigid field | not applicable | snapping diagnostic | candidate | primary standard candidate |

For each cell that is well-defined, report mapping/kernel hashes, zero-filled
fraction by time and channel, RMS and maximum trace difference from the
appropriate control, and direct waveform overlays/statistics on cached events.
Inspect transition seams, outer borders, and the AP191 neighborhood separately.

This screen establishes whether the arms are materially distinct and whether
differences are dominated by quantization, interpolation, unsupported sites, or
seams. It does not establish sorting benefit. Do not launch redundant full
sorts whose snippet outputs are numerically identical.

## EM.2b — conditional W2 correction-pipeline comparison (GPU)

Proceed only after EM.2a has a signed report, the exact local scorecard contract
and tolerances are present in this repository, and the coordinator records the
selected nonredundant arms. The minimal candidate set is:

| Arm | Operator and field | Role |
|---|---|---|
| none | existing S0 | no-correction reference |
| lattice | existing verified DD SL, or the byte-equivalent exact adapter—not both | rounded exact-coordinate pipeline |
| kriging | pinned SI kriging on the unrounded rigid two-layer field | primary standard comparator |
| idw | pinned SI IDW on the same unrounded field | secondary comparator |
| snap | pinned SI nearest on the same unrounded field | diagnostic comparator |

If the scientific conclusion needs to distinguish field rounding from kernel
choice, include at least one rounded-field bridge arm selected by EM.2a. Without
that bridge, conclusions are limited to comparisons of complete pipelines; do
not claim an interpolation-specific or quantization-specific effect.

Use the unchanged static DARTsort configuration, seed, preprocessing boundary,
and frozen clustering/linkage policy. Record complete input and configuration
hashes. No RF fitting, rescoring, or outer-holdout access is permitted.

### Endpoints and frozen reading

- Treat state/rate rank `rho` as a continuity proxy only. Do not call it an
  identity or purity metric.
- Report yield and segment-safe short-interval fractions as separate guardrails;
  neither alone establishes recovery or identity.
- Compute arm differences with the frozen common-block bootstrap (2,000 draws)
  and preserve common resamples across arms. The local scorecard must define
  eligibility, exposures, matching/pairing if any, guardrail tolerances, and
  behavior for undefined correlations before launch.
- **Primary comparison:** lattice versus kriging. IDW is secondary and snap is
  diagnostic unless the frozen scorecard explicitly states otherwise.
- **Meaningful lattice advantage over comparator `c`:** point estimate
  `Delta rho = rho_lattice - rho_c >= 0.05`, its 95% CI lower bound is above
  zero, and all frozen yield/short-interval guardrails pass.
- **Practical equivalence to comparator `c`:** the entire 95% CI for
  `Delta rho` lies inside `[-0.05, 0.05]`, with all guardrails passing. A point
  estimate within 0.05 and a CI merely containing zero is inconclusive, not
  equivalence.
- All other outcomes are mixed or inconclusive. Report them without tuning.

W3 is a separately authorized transfer/replication step only after the W2
reading is frozen. The DK exact-lattice injection may be used later as an
explicitly favorable upper-bound control, not as validation on biological
data.

## Execution safety and limits

- EM.1 target: at most 20 CPU minutes and 1 GB of voltage read.
- EM.2a uses only the frozen snippets and cached events.
- The coordinator sets and records the EM.2b resource cap after measuring one
  selected arm; no silent budget expansion.
- Every sort must run under an independent job manager such as a systemd user
  service or batch scheduler. First verify the launch method with a cheap dummy
  job that survives launcher disconnection.
- Persist the exact launch command, resolved settings, repository commit/source
  hashes, job identifier, stdout/stderr, actual process state, and final exit
  status outside the chat. State explicitly whether interruption restarts the
  whole sort; reuse of completed stages is not checkpointing.
- Preserve failed-run evidence before any restart. Honor cancellation and
  run-specific holds.
- Prefer lazy recordings. Materialize only when required by the worker, within
  local scratch caps. No voltage export, production change, or DARTsort source
  modification.

## Report

Lead with:

1. stock-equivalence verdict;
2. exact-adapter verdict;
3. AP191/interior-hole behavior;
4. operator-screen decision and selected/nonselected sort arms;
5. conditional W2 pipeline reading.

Include one compact implementation table, one arm-by-endpoint table with CIs,
one `rho` figure with CIs (Okabe–Ito colors plus redundant markers), limitations,
job evidence for any sorts, and the SpikeSortingTools commit hash.
