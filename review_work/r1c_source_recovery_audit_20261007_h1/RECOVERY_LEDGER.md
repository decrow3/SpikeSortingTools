# Exact R1c evaluator source recovery audit

Observed: 2026-10-07, H1 (`huklaban1-Precision-5820-Tower`).

## Target

- Declared commit: `ba21bc97ad177ba5f2a1b79fb69d37f4f3eac3ad`
- Path: `testing/full_session_r1_evaluation.py`
- Required file SHA-256: `415f29d7f26102606143d0bd4fbb8ed9a136199a6642e664d50c2a6716b7c423`
- Binding: `/mnt/NPX/Luke/DARTsort_motion_experiments/post_sort_structure_census_h5_20261007_v1/CONTRACT.json`, SHA-256 `e90ad62778b5fa9422102c2f0d9ffeefa0c710a38ceba68d6d79ad9553bf5ea3`.

## Checks

1. Current repository object database: `git cat-file -e ba21bc97...^{commit}` returned status 128; the commit is absent.
2. Shared project tree: a bounded filename search found only the earlier source snapshot in `en_full_session_r1_imec0_rescore_review_request_20261004_v1_h5/source/`, whose file SHA-256 is `a31f1349854eff0e4c732063993be37e8cf53de5d206628f66d68a65ff7f341e`, not the target.
3. H1 home tree: the same earlier snapshot was the only `full_session_r1_evaluation.py` found.
4. Advertised GitHub refs: no ref tip equals `ba21bc97...`; the advertised `huklaban5` branch tip was `c7c0878f62e4d84bdfb35a285ac9b912d06d1f55`.
5. Temporary audit repository: fetched 256 commits from the advertised `huklaban5` branch. Neither the target commit nor the target path history was present.
6. Exact-object fetch: `git fetch --depth=1 ... ba21bc97...` failed with `upload-pack: not our ref`, so the remote does not expose that object by hash.

## Result

`NOT_RECOVERABLE_FROM_H1_LOCAL_SHARED_OR_ADVERTISED_REMOTE_HISTORY`.

The exact source snapshot remains a required artifact from the producing H5 workspace or another immutable packet that can prove SHA-256 `415f29d7...`. The earlier source supports the defect-mechanism trace, while the R1c census contract and SRP audit confirm the same executed timing semantics; neither substitutes for exact R1c source provenance.

No repository ref was changed, no author packet was edited, and no scientific result was computed.

## Implementation checks

- Done: local object lookup -> target commit absent.
- Done: bounded shared/home filename search -> only earlier nonmatching source snapshot found.
- Done: advertised remote refs plus 256-commit H5 branch fetch -> target absent.
- Done: exact object fetch -> remote rejected it as not an exposed ref/object.
- Not done: direct read of H5's unpushed working repository -> unavailable from H1; requires H5 to publish the exact file bytes.
- Can establish: H1 cannot independently recover the exact R1c evaluator from current local/shared/advertised Git history.
- Cannot establish: that no copy exists anywhere on H5, or independently inspect exact R1c source until it is published.
