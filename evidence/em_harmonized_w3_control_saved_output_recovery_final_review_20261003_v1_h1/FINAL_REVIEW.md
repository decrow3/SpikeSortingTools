# H1 final independent review — harmonized W3 saved-output recovery

## Verdict

**Validated inconclusive; no progression is supported.**

Exactly one released CPU-only scoring attempt completed with exit status 0,
`NRestarts=0`, and zero retries. The sealed result is internally consistent and
the producer's frozen classification is correct.

The primary H-minus-F point difference is `+0.06599645095644802`, with 2,000
finite paired bootstrap draws and a 95% interval of
`[0.046296547965893387, 0.0962770414589578]`. Under the prospectively frozen
precedence, this is not practical equivalence because the upper endpoint exceeds
`+0.05`; it is not persistent advantage because the lower endpoint does not
exceed `+0.05`; and it is not worse. The required classification is therefore
`inconclusive`.

Quality advancement independently fails. H has 461 assigned units versus F's
972, a relative difference of `-0.5257201646090535`; both the directional and
equivalence yield limits are 5%. All six segment-safe short-interval comparisons
pass their 0.01 limits, but they cannot rescue the failed H-versus-F yield gate.
Thus both all-comparator directional and equivalence guardrails are false, and
the frozen primary-classification quality gate is false.

The secondary H-minus-L difference is `-0.010420122842438206`, with interval
`[-0.024271317263873257, 0.002228123952590269]`. It contains zero and remains a
combined cached-input/rounding/scaling/materialization/training-context contrast;
it cannot decompose a cause. H versus L passes the frozen yield and all three
short-interval guardrails.

## Inputs, populations, and domains

- F, L, and H hashes are respectively `c865faee...d2bf`, `e82b0402...ec1b`,
  and `03966a70...4361`; H's pipeline receipt and sort marker are bound.
- All arms use the same global half-open frames `[239998073,250197991)`, local
  `[0,10199918)`, and 29999.759166666667 Hz clock. Stable normalization resolves
  the saved two-sample inversions in H and L without changing label pairing.
- Populations are arm-local labels `>=0`, not cross-arm identity matches. Yield
  is the number of assigned labels. Eligibility is at least 100
  `outside_mask_flat` events and is fixed across draws: F 649/972, L 386/478,
  H 370/461 eligible/assigned units.
- Shared domain exposures are 68.35180705826133 s negative excursion,
  221.70902639064025 s outside-mask flat, and 49.939162662179115 s operational
  remainder, totaling 339.9999961110807 s. The remainder label is operational,
  not proof of true rest or literal catalogue membership.
- The continuity endpoint is Spearman correlation of arm-local pseudocount-0.5
  log rates in negative versus flat domains. It is a continuity proxy, not
  biological identity. Short-interval fractions are contamination proxies, not
  purity or merge diagnoses.

## Segment-safe short-interval audit

The unit is the fraction `lag9_29 / denominator` among consecutive same-unit
events within the same contiguous domain segment; lag 8 and lag 30 are boundary
controls and are excluded.

| Comparator | Domain | H denominator / 9–29 | Comparator denominator / 9–29 | H−comparator | Directional | Equivalence |
|---|---|---|---|---:|---|---|
| F | negative | 85,409 / 718 | 67,113 / 620 | -0.0008315424 | pass | pass |
| F | flat | 351,851 / 3,377 | 319,863 / 1,328 | +0.0054460352 | pass | pass |
| F | remainder | 66,583 / 656 | 51,261 / 329 | +0.0034342301 | pass | pass |
| L | negative | 85,409 / 718 | 83,563 / 557 | +0.0017409787 | pass | pass |
| L | flat | 351,851 / 3,377 | 350,140 / 3,184 | +0.0005043073 | pass | pass |
| L | remainder | 66,583 / 656 | 65,577 / 682 | -0.0005476231 | pass | pass |

Directional pass requires H-minus-comparator `<=0.01`; equivalence requires its
absolute value `<=0.01`. Every fraction and decision recomputes from the sealed
CSV.

## Implementation checks

- Done: terminal packet and nested score packet -> every manifest member,
  result, receipt, and completion hash reverified; nested and outer COMPLETE
  files are strictly newest.
- Done: invocation -> sealed release chain, job command, attempt 1, retry false,
  exit 0, `Restart=no`, `NRestarts=0`, CPU-only limits, and no RF/holdout bind.
- Done: identities/clocks -> F/L/H hashes, H receipt/marker, source/config hashes,
  local/global frames, sampling frequency, schemas, bounds, and stable ordering
  agree with the reviewed real-input preflight.
- Done: serialized estimates -> point differences, all 4,000 saved contrast
  rows, draw indices, finite-draw counts, percentiles, and classification
  precedence independently recomputed.
- Done: joint bootstrap -> frozen source uses one seeded block-multiplicity
  matrix for F/L/H; H1 regenerated its `(2000,68)` matrix and exact SHA-256
  `440bae62...512`; serialized paired contrast percentiles match RESULT.
- Done: domains/populations/guardrails -> exposures, eligibility semantics,
  yields, every segment denominator/count/fraction, comparator direction, and
  threshold checked.
- Not done: independent replay from H5-private event arrays -> those files are
  not mounted on H1; the sealed scorer source, actual H5 preflight, multiplicity
  regeneration, and serialized outputs were independently checked instead.
- Can establish: the frozen result is correctly serialized and classified as
  inconclusive with failed primary quality-advancement guardrails.
- Cannot establish: biological identity, purity, causal attribution of H versus
  L, benefit beyond this W3 window, or support for long/full/production work.

No additional run, RF/holdout access, or automatic follow-up is authorized by
this result.
