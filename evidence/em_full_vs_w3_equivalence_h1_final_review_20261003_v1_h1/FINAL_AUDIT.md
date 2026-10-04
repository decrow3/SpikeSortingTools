# H1 final review — full-session versus W3 rounded-kriging equivalence

## Decision

**MATERIAL_INPUT_MISMATCH; ONE INPUT-HARMONIZED W3 CONTROL IS WARRANTED.**

The exact full-session sort SHA-256 `3b324bbc4ec3c2bcfcc8bc694f0cf9b0237ac0d4f87d1c13ef7b12facfade660`
and W3-trained sort SHA-256 `e82b0402057197a47f16fb5fefd98701fc18c2ebfdfe824f0d89722a62b5ec1b`
are not a pure training-context contrast. They use different rounded displacement
states on 689,995 of 10,199,918 W3 samples (6.764711%, 79 segments) because the
rounding references are -4.695588242 and +4.096230345 um. All seven observed
mismatch transitions select distinct saved effective-kernel hashes. They also
use different `ScaleRecording` gains on all 383 accepted channels (mean absolute
difference 0.0115236; maximum 0.0487316); offsets are identically zero.

The existing 972 active/649 eligible versus 478 active/386 eligible populations
and rho 0.676167 versus 0.752584 must therefore remain descriptive. They cannot
be attributed to training context, rounding, gain scaling, identity, or purity.

## Equivalence table

| Factor | Finding | Classification | Evidence |
|---|---|---|---|
| Bound outputs | Both requested sort hashes recomputed and matched terminal receipts | matched | H5 `AUDIT.json`; audit source lines 76-80 |
| Clock/window | 29,999.759166666667 Hz; global W3 `[239998073,250197991)`; local `j -> 239998073+j` | matched | H5 `AUDIT.json.clock`; source lines 81-82, 108-120 |
| Raw field | Same 1,361 knot indices, times, raw values, and canonical mask | matched | H5 source lines 97-102 |
| Rounded field | References differ; 92/1,361 knots and 689,995/10,199,918 samples differ by 40 um | load-bearing mismatch | H5 `AUDIT.json.field`; exhaustive source lines 95-120 |
| Kernel/support | Effective-kernel hash differs for every mismatched state pair; structural-zero targets remain zero | weighting/mapping mismatch, not support-mask mismatch | H5 `KERNEL_AND_SUPPORT_BINDING.json` |
| Geometry/channels | 384-source and 182-target IDs/order/locations, AP191 treatment, and small property arrays match | matched | H5 `AUDIT.json.geometry_and_support`; source lines 83-87, 143 |
| Preprocessing order | High-pass, phase shift, accepted-channel slice, references, float32 materialization, AP191 insertion, and remap boundary match | matched except gains | H5 `AUDIT.json.preprocessing` |
| Scale | Gains differ on 383/383 accepted channels; zero offsets match | load-bearing mismatch | H5 source lines 89-93, 144 |
| Runtime/descriptors | Both executed with SI 0.104.8; serialized descriptor labels differ (0.104.8 versus 0.104.7) | unresolved descriptor difference | H5 `AUDIT.json.preprocessing` |
| Sorter | Effective/internal configs and frozen DARTsort provenance are byte-identical | matched | hashes `35456cba...`, `b2a5711a...`, `d8da2835...`; source lines 122-126 |
| Training/recovery | Full-session versus W3-only fitting and learned populations differ; full matching checkpoint was recovered after OOM, W3 completed ordinarily | expected context difference | H5 `AUDIT.json.cache_and_recovery` |
| Cache proof | Both cache checkpoints say complete, but their `contract` fields are null and cached-voltage byte equality was prohibited/not checked | genuinely missing provenance | H5 `AUDIT.json.cache_and_recovery`, `preprocessing.voltage_byte_equivalence_checked` |

Packet integrity was independently checked with `sha256sum -c`; all seven H5
manifest members passed. The published `COMPLETE.json` binds H5 audit SHA-256
`63b7fbb206cb362d312b7beb21872b61cb5e99d424712cd087576e834e25bb61`
and was written last. H1 independently re-summed the seven mismatch-pair sample
counts to 689,995, the mismatch-knot counts to 92, and the W3 interval to
10,199,918 samples.

## Sole candidate for a separately frozen medium run

**Candidate:** train one W3-only model on the exact global W3 byte range of the
completed full-session materialized preprocessing cache, using the byte-identical
effective/internal sorter configuration, frozen source provenance, seed, channel
properties, and no refitted preprocessing. Run it as an independently managed
job with its own immutable contract and receipts.

**Cheapest prerequisite:** before launch, hash deterministic beginning, seam,
and ending blocks through both the direct full-cache range and the proposed
W3 recording view; require byte equality, identical float32 shape/channel order,
and exact `[239998073,250197991)` origin. This reads only saved cache bytes, not
recording voltage. Failure stops the candidate before sorting.

**Frozen prediction:** the input-harmonized W3 result will move toward the
full-session W3 slice: `abs(rho_new - 0.676167)` will be less than the existing
gap `0.076417`, and the pre-existing common-block 95% CI for
`rho_new - 0.676167` will lie wholly inside `[-0.05, 0.05]`, with the existing
yield and short-interval guardrails passing. Active/eligible unit counts remain
descriptive, not acceptance endpoints.

**Acceptance:** preflight byte equivalence passes, the frozen practical-
equivalence CI and all existing guardrails pass, and provenance confirms no
input/config difference beyond W3-only versus full-session training context.
This supports standardizing subsequent medium work on the full-session
preprocessing contract; it does not select a scientifically superior rounding
reference or prove identity/purity.

**Stop rule:** stop without a sort if byte equivalence or provenance fails.
After the one run, stop without tuning, repetition, or long/full promotion if
the CI is not wholly inside `[-0.05, 0.05]`, a guardrail fails, or the comparison
is otherwise invalid. Preserve inconclusive and failed evidence.

This review authorizes no launch. It used no voltage, new sort, scoring,
bootstrap, RF/holdout, target-A arrays, or large-array transfer.

## Implementation checks

- Done: exact artifact binding -> both requested sorting hashes were recomputed
  by H5 and tied to terminal receipts (H5 audit source lines 76-80).
- Done: clocks/frames -> exact common W3 interval, sampling rate, origin, and
  exhaustive half-open state assignment were checked (source lines 45-49,
  81-82, 106-120).
- Done: field/kernel/geometry -> 92 knot and 689,995 sample mismatches were
  identified; saved kernel IDs differ while zero support does not; geometry and
  channel ordering match (H5 audit and kernel-binding artifacts).
- Done: preprocessing/config -> order and boundary match, gains do not; executed
  runtime, effective/internal configs, and source provenance were checked
  (source lines 89-93, 122-126, 143-148).
- Done: packet arithmetic/integrity -> H1 verified all manifest hashes and
  independently summed published pair counts.
- Not done: cached-voltage byte equivalence -> prohibited in this audit; the
  proposed saved-cache block preflight is required before the candidate run.
- Not done: cause of the gain/reference differences -> would require historical
  construction tracing; the combined full-cache control avoids needing that
  attribution for the next decision.
- Can establish: the existing contrast is materially confounded and one
  input-harmonized W3 medium control is executable and decision-relevant.
- Cannot establish: which preprocessing choice is scientifically preferable,
  what caused the existing rho/population difference, biological identity or
  purity, or whether the candidate will pass.
