# Independent N0/N1 contract-validator v1 review

Verdict: **NO_GO_PENDING_EXACT_BINDING_AND_LIFECYCLE_GUARDS**.

The validator handles many scalar, mapping, state-partition, cross-arm, and
basic resource boundaries correctly. It is not yet a complete fail-closed gate
for its own frozen `required_bindings`: ordinary structurally accepted documents
can omit load-bearing identities or satisfy lifecycle fields with declarations
that contradict their intended meaning.

## Findings

1. **Four required bindings are absent from the schema.** A document can pass
   without a recording/session clock identity and origin, an explicit
   `half_open` interval convention, a common-event-set hash distinct from the
   cohort hash, or a checkpoint hash keyed to each window. The single global arm
   checkpoint cannot express “one shared N0 checkpoint per window” for a
   multiwindow contract.
2. **K10 v4 is not actually pinned.** The evaluator implementation and review
   fields accept any two syntactically valid SHA-256 strings. The reviewed
   artifacts are evaluator SHA-256
   `c47d834bd9446a0f7be28d34863b2019c1b140cff7abbdcba091e64d5e8f17e3`
   and independent-review COMPLETE SHA-256
   `083afe0d708881b59c78fb5b129cc306b147e8c89e31433a1827068040191b51`.
3. **Placeholder and exact-container checks are incomplete.** Placeholder
   matching is case-sensitive, so `todo`, `unknown`, and `<PATH>` pass. An empty
   NumPy array passes `other_deltas != []` today, with an ambiguous-truth
   deprecation warning, despite not being an empty JSON list.
4. **Managed execution and stopping are free text.** Values such as `none`,
   `nohup`, `bash`, or `interactive terminal` pass as a managed service, and
   “ignore every bound and continue” passes as the sole stop condition. Thus
   positive numeric resource bounds do not bind the mechanism that enforces
   them.
5. **Output separation is raw-string-only.** Lexically equivalent paths such as
   `/scratch/run/N0` and `/scratch/run/./N0`, ancestor/descendant namespaces,
   and reuse of the recording input path all pass as “distinct” create-only
   outputs.

The independent suite has 29 passing and 15 failing parameterized cases. The
passing cases cover exact keys at every represented mapping level, arm binding
equality, the intended boolean delta, scalar dtype/int64 bounds, positive
resource values, contiguous complete state partitions and padding endpoints,
endpoint direction/margin checks, and explicit non-equivalence. The 20 author
fixtures also pass.

## Exact repair specification

Create a new schema version and preserve this v1 failure packet.

- Add a required `session_clock` mapping that binds an exact clock identity and
  integer origin; require every window and the motion-field clock to reference
  it. Add an exact `interval_bounds: "half_open"` field.
- Add `common_events_sha256`. Replace the single arm `checkpoint_sha256` with
  `checkpoints_by_window`, require its key set to exactly equal `window_id`, and
  require identical normalized mappings across N0 and N1.
- Pin evaluator hashes to the reviewed K10 v4 values above, not merely SHA
  syntax. Keep explicit endpoint directions/margins and non-equivalence.
- Case-fold and strip before placeholder comparison. Require
  `type(other_deltas) is list and len(other_deltas) == 0`.
- Replace `managed_service` free text with exact structured fields such as
  `manager: "systemd_user"`, a resolved `.service` unit, and a boolean
  launcher-disconnection verification receipt. Replace arbitrary stop prose
  with an exact required set of enumerated stop-condition IDs tied to runtime,
  memory, disk, invariant, and managed-service failure.
- Normalize paths lexically without filesystem access; reject `..`, equivalent
  aliases, equal or ancestor/descendant arm namespaces, and equality/ancestry
  with recording or motion-field input paths. Preserve exact booleans for
  create-only and failure preservation.
- Add all 15 failing parameterized cases as regressions and rerun the 29 passing
  independent cases and 20 author fixtures unchanged.

## Implementation checks

- Done: full frozen contract/source/test read and field-to-consumer trace -> the
  missing bindings are absent from the exact-key schema at implementation lines
  88–179; lifecycle values are only free-text/raw-string checked at lines
  198–218.
- Done: independent scalar/hash/path/mapping/sequence, cross-arm, resource,
  output, and evasion fixtures -> 29 pass and 15 fail; author suite 20/20.
- Done: source/provenance verification -> working implementation matches frozen
  SHA-256 `8c8799bc7ccfc68b94baf856c48805886a47760a90af74cfd917a7aedcdba837`;
  local and coordination-copy manifest members and hash chains verify.
- Not done: referenced-byte existence/hashes, live resources, recording or
  voltage access, production execution, GPU/service/sort, RF/holdout, or C4 ->
  explicitly excluded.
- Can establish: v1 is a useful partial structural checker but admits documents
  that evade its frozen exact-binding and lifecycle requirements.
- Cannot establish: correctness of referenced bytes, live resource readiness,
  scientific validity, or permission/readiness to execute N0/N1.

