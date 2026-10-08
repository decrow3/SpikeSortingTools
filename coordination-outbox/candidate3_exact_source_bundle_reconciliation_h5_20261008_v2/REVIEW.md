# Independent review — Candidate3 exact source-bundle reconciliation v2

## Verdict

**GO_BLOCKER_ACCURATE.** The frozen packet accurately identifies that Candidate3 v3 is not execution-ready. The exact wrapper commit/source remains unavailable, and the frozen runner also contains two independently reproduced path/import joins that would reject before a valid equality run. No execution or occupied-repository mutation is warranted from this packet.

## Verified findings

### Exact runner and contract

The carried runner SHA-256 is `0bed993b8718f29591b92884e88b65f2bf8a6780cce49490b869ba82823f1f7e`; the carried contract SHA-256 is `6cb5c29b806b63021bf9edef87535718f24efc9ff27842d9e4e34c3e06e56037`. Both are byte-identical to `RUNNER.py` and `CONTRACT.json` in the sealed v3 contract packet. The occupied repository paths frozen by that contract are absent, so installing these copies there would mutate occupied state.

### Missing SpikeSortingTools object

Commit `aea7431c4026fa74e36ad01139d9e596900a1d25` is absent from the occupied repository's object database. The preserved isolated bare mirror at `/tmp/candidate3-sst-remote-SG8ssZ/repo.git` points to the same configured GitHub remote and also lacks that commit. The direct exact-SHA fetch receipt has empty stdout, return code 128, and exact stderr `fatal: remote error: upload-pack: not our ref aea7431c...`. No network request was repeated in this review.

Consequently the required wrapper `npx_preprocessing/motion/kilosort_prewhitening_si.py`, SHA-256 `b7f06da5685d3b8af0c7bd18e7ccfeb3af7db8fe3b22b1fa6a619908f43bb690`, cannot be reconstructed as an exact reviewed source object from the currently available local/configured-remote evidence.

### DARTsort identity

DARTsort commit `edcfe1b51d672b4136eb13cc78c0875da804b851` and tree `8d40728c4cb53b7fcbab726c465afda04ddbe32b` exist locally. Independent `git show` hashing and actual DARTsort-interpreter imports both reproduce:

- `dartsort/main.py`: `e4c03417c1256384c4f8fbf0250560003bd795b7274932ef1e49dcde217bd249`
- `dartsort/util/main_util.py`: `2846a319c19c25b3cad81ab991057c0868b87d760b66f20e8d2bb3db8ffa3bcc`
- `dartsort/util/preprocess_util.py`: `566ce1ab60af12585978bcd7e3192cb2ca4cc220ab5e41895140c00bdbc06674`

The working repository is dirty, but these three working files equal the required commit bytes; no checkout or mutation was performed.

### Kilosort composition and runner defects

The frozen runner appends the declared rescue Kilosort site (`runner:417-420`). Repeating that exact composition imports `kilosort/io.py` SHA-256 `c6093e14cae920e8cfaae6c3fed8d4c6ca032b046d705ef9868ec171c7bf3c99`, not the contract-required `767b76a07647122b45048ba12cdaf49185f87f3fa33334081d4af099825414bd`; `imports_and_sources()` would reject it.

The pristine Kilosort 4.0.27 tree directly hashes to `767b76a0...`. A read-only probe with that root prepended and the dependency site appended imports the exact required `io.py` and Kilosort 4.0.27. This proves availability of the required Kilosort bytes, not approval of that successor topology.

The frozen runner also evaluates `Path(sys.executable).resolve()` but compares it with the unresolved contract path (`runner:433`). On this host, `/home/huklaban5/Documents/DARTsort/.venv/bin/python` is a symlink to `/home/huklaban5/.local/share/uv/python/cpython-3.12.4-linux-x86_64-gnu/bin/python3.12`; the two `Path` values are unequal. The declared interpreter therefore rejects itself in real mode.

These defects are independent of the missing wrapper object. A successor needs immutable wrapper source plus a narrowly reviewed isolated path/import redesign; it cannot simply launch the frozen v3 runner.

## Nonblocking provenance note

`evidence/IMPORT_IDENTITY_PRISTINE_KILOSORT.json` is byte-reproduced by the current `IMPORT_PROBE.py`. The append evidence predates one schema detail: rerunning the current probe adds `"dependency_site": null`, so `IMPORT_IDENTITY_CONTRACT_COMPOSITION.json` is not byte-identical to current output. Its imported paths and every reported hash reproduce exactly, so this does not weaken the blocker. Regenerating that receipt or labeling the schema drift would improve provenance in a later packet.

## Scope

No voltage or recording was opened, read, or hashed. No equality fixture, service, GPU job, or sort ran. No network request was made. No occupied repository or environment source was modified; import verification used bytecode suppression and a temporary Numba cache.

## Implementation checks

- Done: rehashed carried and sealed-v3 runner/contract -> exact byte identity.
- Done: checked occupied and preserved isolated Git object databases plus direct-fetch receipt -> required wrapper commit remains unavailable.
- Done: derived DARTsort commit/tree and file hashes with `git show`, then repeated actual imports -> all required module identities match.
- Done: repeated frozen append and pristine prepend import probes -> `c6093e14...` mismatch and `767b76a0...` exact identity reproduce.
- Done: inspected and evaluated the frozen interpreter guard -> resolved executable cannot equal the unresolved venv symlink path.
- Not done: network refetch, wrapper reconstruction, isolated successor build, equality execution, voltage access, service, GPU, or sort -> prohibited/outside reconciliation scope.
- Can establish: the packet's no-launch blocker and minimum successor direction are accurate for the exact frozen bytes and presently available sources.
- Cannot establish: wrapper correctness, equality, complete bundle readiness, full-session feasibility, or sorting benefit.
