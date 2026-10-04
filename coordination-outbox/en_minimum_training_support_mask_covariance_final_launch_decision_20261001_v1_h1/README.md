# Final H1 launch decision: covariance/W smoke v2

Decision date: 2026-10-01.

## Verdict

**NO-GO for service start from the submitted evidence packet.** H5 must not
start `en_minimum_training_support_mask_covariance_smoke_enabled_v2.service`
and there is no final receipt to copy.

The installed-preflight packet itself is intact: manifest
`6e700eb3b956ba39606ebe8eafcee337626468c98c9d5b7e897805ae8dac5d52`
and COMPLETE
`c2ae570d72c6897e2b92650bdd79b95d0ae5ad1336639ce44f75eec8fa3b76e0`
match, and every packet member verifies. It establishes the reviewed primary
source/config/service hashes, full installed contract validation, CUDA with one
NVIDIA RTX A5000, metadata readability without content open, 34/34 installed
tests, loaded/static inactive/dead state, fresh v2 run paths and unchanged data
and retained-output bounds.

However, the prior conditional GO explicitly required a prelaunch hash check
of the resolved Python executable and all bound Kilosort modules, plus an
output-parent writability check. The submitted packet does not preserve the
resolved Python path/hash or output-parent writability result. The contract
validator transitively checks configured source bindings, including Kilosort
modules, but it does not establish the separately required resolved-Python
binary identity or parent writability. Those mutable-host prerequisites cannot
be inferred from a successful test run or absent run root.

This is an evidence-completeness NO-GO, not a source defect, scientific failure
or renewed-human-approval issue. Preserve this packet and the installed static
unit unchanged. A new immutable H5 evidence packet may cure the gap by recording:

1. the resolved rescue-production Python absolute path and SHA-256, matching
   the previously reviewed `4adb344720ca7a0ccee7cdc77506bae30da64b55a977cd45e15880ddf8bddcc1`;
2. the five frozen Kilosort module paths and hashes from the prior independent
   review;
3. `output_parent_writable=true` for the exact v2 run-root parent;
4. a repeated immediate unit/fresh-namespace check with no voltage access.

No reinstall, parameter change, namespace change, new test run or voltage read
is needed merely to provide those missing metadata checks. This packet did not
install/start a service or access voltage.

## Implementation checks

- Done: installed-preflight packet hashes -> exact expected values; all four
  manifest entries verify.
- Done: installed primary bindings, full contract validator, CUDA metadata,
  34/34 test receipt, unit state, bounds and fresh run paths -> internally
  consistent and match the conditional contract.
- Done: conditional-GO prerequisite comparison -> required resolved-Python
  identity and output-parent writability are absent from the submitted evidence.
- Not done: independent current H5 metadata recheck -> this reviewer is
  intentionally artifact-only and did not contact or mutate H5.
- Can establish: the installed chain passed its reported contract/tests and
  remained unstarted with no voltage opened.
- Cannot establish: every frozen prelaunch prerequisite; therefore a one-start
  decision is not defensible from this packet.
