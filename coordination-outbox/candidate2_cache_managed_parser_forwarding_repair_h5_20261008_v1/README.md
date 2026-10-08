# Candidate2 monitor parser-forwarding repair

This prospective v6 derivative preserves the sealed v5 pre-read failure. Its only executable change is `allow_abbrev=False` on the monitor bootstrap's argument parser, so `--target-unit` crosses the bootstrap-to-monitor boundary instead of being consumed as an abbreviation of `--target-unit-file`. A permanent subprocess regression derives the option names and order from the actual frozen monitor service and proves the inert child receives the target unit.

All 30 runtime source members, the target bootstrap, and the preflight bootstrap are byte-identical to v5. The sorter, monitor implementation/classification, Candidate2 science, and intact cache-dropping validation, whole-file DONTNEED/settle/recheck, and exact-target bounded preadv operations are unchanged. This is a fresh attempt, not a retry of the failed namespace, and there is no automatic retry.

Merged or unknown failures remain failed or unresolved. Missing monitor terminal evidence is a documented observability failure, never success. Full sort-and-QC completion requires external reconciliation of the actual manager invocation, real child command and exits, final stage, effective nblocks=1, and valid sort, curation and QC artifacts.
