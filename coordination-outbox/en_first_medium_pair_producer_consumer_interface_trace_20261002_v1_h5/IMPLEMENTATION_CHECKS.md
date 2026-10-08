# Implementation checks

- Done: pair state production -> outer COMPLETE embeds exactly the state map
  created after REF then repaired execution (`en_first_medium_trained_pair_launch.py:615-630`).
- Done: state content -> `_execution_complete` records status, COMPLETE path,
  and SHA-256 (`:337-348`).
- Done: REF path -> producer fixes `sorter_output` and inventory below
  `<planned root>/REF384_repeat` (`:297-316`).
- Done: repaired path -> config validation fixes results to
  `<trained.run_root>/native_results`; completion is `<run_root>/COMPLETE.json`
  (`kilosort_support_mask_trained.py:163-168`, launcher `:320-334`).
- Done: arm association -> REF COMPLETE has `arm=REF384_repeat`; repaired
  COMPLETE binds exact `config_sha256` (`trained.py:659-669`).
- Done: inventory coverage -> both native inventories cover all seven Phase3
  consumed files; repaired inventory also records byte sizes.
- Done: path threat trace -> exact resolve equality plus non-symlink checks are
  necessary to reject escapes, swaps, and unrelated same-basename inputs.
- Not done: real-run receipt inspection -> no pair has run and real outcomes
  were prohibited; this traces reviewed producer source/config only.
- Can establish: the producer defines deterministic state, output, and inventory
  locations sufficient for a narrow consumer check.
- Cannot establish: a transitive cryptographic array binding from pair COMPLETE
  alone, because no COMPLETE hashes either inventory receipt.
