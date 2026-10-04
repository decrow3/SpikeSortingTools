# H1 recovery-delta review — v2

## Decision

**NO_GO_RESEAL_COMPLETE_LAST_ONLY.**

The saved-output recovery implementation and real-input preflight pass the
load-bearing review. Scoring remains blocked only because the published packet
does not establish the mandatory `COMPLETE.json`-last invariant: its completion
receipt mtime is `1791093526216404100`, while eight manifest members are newer;
the latest is `source/frozen_de_common_outcome_scorecard.py` at
`1791093526217652800`.

The minimum repair is to rewrite or recopy the byte-identical `COMPLETE.json`
after all members and verify its nanosecond mtime is strictly later than every
member. No source, config, input, contract, manifest, hash, service, or
scientific change is requested. Do not score during the repair.

## Accepted implementation findings

- All `MANIFEST.sha256` members rehash successfully. The manifest and COMPLETE
  hashes match the review request.
- The derivative score contract differs from the prospectively frozen v3
  contract only by filling H's terminal sorting hash and changing the field and
  catalogue paths to exact hash-identical local copies.
- `harmonized_scorecard.py`, both frozen helpers, and
  `post_sort_score_and_seal.py` are byte-identical to the accepted v3 package.
  Endpoint, domains, clock, eligibility, 2,000-draw joint block bootstrap,
  seed, thresholds, precedence, secondary interpretation, yield checks,
  short-interval checks, and exclusions are unchanged.
- The first real-input receipt-formatting failure is preserved. The executable
  v1-to-v2 preflight delta changes only receipt reporting from an invalid scalar
  conversion of a `(blocks, domains)` matrix to the sum of each domain column.
  The evaluator already consumes the same matrix by summing over blocks; this
  repair does not change evaluator calculations.
- Independent H1 recomputation with the exact packet source closure and
  authoritative field/mask/catalogue reproduced 106 intervals, 69 edges, a
  `(68,3)` exposure matrix, 339.9999961110807 seconds total exposure, every
  domain total, and every accepted-catalogue overlap value in the H5 receipt.
- Four portable focused tests pass on H1. The fifth is intentionally H5-host
  specific and its direct state result is captured by H5's real-input receipt.
- The H5 preflight source directly verifies H/F/L hashes, required arrays,
  shapes, sampling frequency, local bounds, stable ordering, H terminal receipt
  and marker, exact input hashes/schemas, and fresh output absence. Its receipt
  reports the bound H receipt and COMPLETE hashes and no score, bootstrap,
  guardrail, GPU, voltage, or cache work.

## Implementation checks

- Done: packet identity -> manifest hash
  `72f9f1bf719f91713015f2e6ac671f34c14998e6d26715caed6f37f34b636e0e`;
  all members pass.
- Done: recovery delta -> exact config diff and source hashes independently
  compared with the accepted v3 scorer package.
- Done: preflight repair -> v1/v2 source diff traced; known-answer test and H1
  real-input domain recomputation pass.
- Done: H/output proof structure -> actual H5 preflight reads and validates the
  terminal saved outputs and records the expected receipt/marker/H hashes.
- Done: failure preservation -> missing-catalogue and matrix-format failures are
  retained; original failed score namespace remains the required preserved
  input to the H5 host-state test.
- Not done: scoring or result interpretation -> prohibited; no RESULT exists.
- Not done: independent H1 byte-read of H5-private sort files -> H1 cannot mount
  those paths; this review independently checks the executed H5 preflight code,
  sealed receipt, and cross-packet hashes rather than claiming a second host read.
- Can establish: the unchanged saved-output evaluator is technically ready once
  the publication receipt is resealed last and a coordinator separately releases
  one CPU-only attempt.
- Cannot establish: any point estimate, bootstrap interval, domain denominator,
  guardrail verdict, scientific classification, or advancement decision.

Final post-result review must independently verify point estimates, joint
bootstrap pairing and valid draws, domain exposures and segment-safe
denominators, yield/short-interval guardrails, result sealing, and scoped
interpretation before any decision.
