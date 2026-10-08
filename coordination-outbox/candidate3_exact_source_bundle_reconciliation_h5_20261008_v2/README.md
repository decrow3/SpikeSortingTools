# Candidate3 exact source-bearing bundle reconciliation v2

Verdict: `ACK_BLOCKED_EXACT_WRAPPER_COMMIT_UNAVAILABLE_AND_IMPORT_TOPOLOGY_REVIEW_REQUIRED_NO_LAUNCH`.

This packet closes two ambiguities from the predecessor. First, it carries exact source payload copies of the reviewed v3 runner and contract. Second, it probes the configured SpikeSortingTools remote from an isolated temporary bare clone, including a direct exact-SHA fetch. The required wrapper commit `aea7431c...` remains unavailable: GitHub returned `upload-pack: not our ref`. No occupied worktree or object database was changed.

DARTsort commit `edcfe1b...` remains locally archiveable and its three required modules reproduce the frozen hashes both from the commit and through the actual DARTsort interpreter. A pristine Kilosort 4.0.27 tree supplies the exact required `io.py` hash `767b76a0...` and imports successfully when placed before dependency-only paths.

The frozen v3 composition is nevertheless not runnable as written. Appending its declared rescue Kilosort site imports `io.py` hash `c6093e14...`; the runner's interpreter identity comparison resolves `sys.executable` but not the expected venv symlink, so the declared interpreter rejects itself. Fixing those joins is a narrow successor redesign requiring focused review; it must not weaken exact equality or source checks.

No voltage was read or hashed, no equality fixture or real comparison ran, no service was installed, and no sort/GPU work started.
