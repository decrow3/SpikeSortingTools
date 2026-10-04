# Accepted pair producer → Phase3 consumer interface trace

The accepted v3 producer exposes exactly two pair execution states,
`REF384_repeat` and `repaired_B384`. Each state carries execution status, the
arm COMPLETE path, and its content SHA-256; the outer pair COMPLETE embeds this
map.

The producer-defined curated paths are different by arm:

- REF: `<pair root>/REF384_repeat/sorter_output`
- repaired: the reviewed configuration's `native_invocation.results_dir`,
  validated to equal `<trained.run_root>/native_results`

Both arms write deterministic `launch_evidence/artifact_inventory.json`
receipts covering all seven files consumed by Phase3: times, clusters,
`full_st`, `kept_spikes`, positions, ops, and labels.

Exact missing link: neither arm COMPLETE nor pair COMPLETE contains the
inventory receipt hash. The pair receipt therefore identifies arm COMPLETE
content, but not transitively the saved-array bytes. A narrow consumer repair
can bind the two deterministic inventory receipt hashes in its reviewed
request/tokens and verify every consumed file against them. If that binding is
not considered sufficient, the producer must add an inventory hash to arm
completion in a fresh reviewed version.

Resolved-path equality—not basename or suffix matching—is required. Reject
symlinks, parent escapes, swapped states, unrelated same-basename directories,
wrong REF `arm`, wrong repaired `config_sha256`, and any inventory mismatch.

No real outputs, outcomes, recording voltage, RF, or holdout were accessed.
