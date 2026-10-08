# Incremental D1 repair review checklist

Started: 2026-10-07 20:43 PDT

Preserved unchanged checks

- D1 v1 packet integrity and source review.
- Accepted matcher snapshot and inclusive-boundary semantics.
- Ten published fixtures and H1's 176,400-case exhaustive DP comparison.
- Half-open support, pair-name separation, non-transitive interpretation, and
  execution-disable boundary.

Changed interface 1: per-arm provenance

- Require distinct A and REF records; reject a shared hash presented as both.
- A: bind patched Kilosort 4.0.27 `io.py`, wrapper/environment, effective
  settings, producer and curator sources to the actual A receipts.
- REF: bind what historical evidence actually preserves (release commit,
  environment lock/version, producer/curator bytes and effective settings).
- If REF's installed `io.py` bytes are unavailable, require an explicit
  `unavailable_historical_bytes` status and prevent exact-executable claims;
  allow prospective array-relationship validation to proceed within that scope.
- Negative fixtures: swap A/REF source receipts; omit arm id; mutate one source
  hash; claim exact REF bytes when status is unavailable.

Changed interface 2: curated parent mapping

- Output one row per curated event with unique / ambiguous / unmatched status.
- Preserve all candidate original parent indices for ambiguous keys.
- No occurrence-rank or greedy uniqueness unless separately frozen.
- Negative/known-answer fixtures: duplicate composite keys, deleted duplicate,
  one-to-many, many-to-one, out-of-order input, wrong detection template,
  unmatched key, repeated time with distinct detection templates.
- Check aggregate closure: unique + ambiguous + unmatched = curated rows.

Changed interface 3: event-level stage relations

- Explicit immutable fields: parent_row_id, matching_label, final_label.
- Matching and final labels may differ; no equality gate or silent overwrite.
- Preserve pair-local matched indices, all-neighbor candidate sets/degrees, and
  per-side unmatched indices.
- Event-level exclusive relation must reproduce accepted aggregate count.
- Negative fixtures: time-row drift, whole-row reorder, feature-only reorder,
  label substitution, duplicate-time collision, cross-pair identifier reuse.

Changed interface 4: DARTsort candidate rule

- Freeze candidate universe before production outcomes.
- Deterministic ranking keys and exact tie-breaking order.
- Freeze minimum support and behavior below threshold.
- Preserve and report multiple candidates rather than selecting by inspection.
- Keep REF--DARTsort and A--DARTsort candidates pair-local; no transitive
  closure through REF--A.
- Negative fixtures: exact score tie, zero candidates, one candidate, multiple
  above threshold, all below threshold, input-order permutation.

Final transition

- Verify provisional source hashes and run only changed-interface fixtures.
- On sealed packet, compare provisional-to-final manifest/member hashes.
- Re-run preserved checks only if their dependency hashes changed.
- Require explicit execution-disabled gate and no production arrays, voltage,
  replay, sort, RF, or holdout access.

