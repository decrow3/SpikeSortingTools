# Implementation checks

- Done: exact runner and contract bytes copied from the sealed v3 packet and rehashed to `0bed993b...` and `6cb5c29b...`.
- Done: occupied SpikeSortingTools repository inspected read-only; required commit remains absent locally.
- Done: configured remote mirrored into a temporary bare repository and exact SHA fetched directly; remote returned `not our ref`, leaving the object unavailable.
- Done: DARTsort commit object `edcfe1b...` and tree `8d40728c...` exist; required module hashes reproduce from the commit without changing its dirty worktree.
- Done: actual DARTsort-interpreter import reproduces all three required DARTsort module hashes.
- Done: frozen appended Kilosort site imports wrong `io.py` hash `c6093e14...`; pristine 4.0.27 source imports exact `767b76a0...` only under an unreviewed prepend/dependency topology.
- Done: inspected frozen runner's real-mode path joins; resolved `sys.executable` cannot equal the unresolved expected venv symlink path on this host.
- Not done: wrapper source reconstruction, isolated environment build, runner redesign, equality execution or recording access.
- Can establish: runner/config, DARTsort, and exact Kilosort source bytes are available; exact wrapper source is not locally or remotely retrievable, and frozen v3 import/path topology is not execution-ready.
- Cannot establish: complete source-bundle readiness, wrapper correctness, real-window equality, full-session feasibility or sorting benefit.
