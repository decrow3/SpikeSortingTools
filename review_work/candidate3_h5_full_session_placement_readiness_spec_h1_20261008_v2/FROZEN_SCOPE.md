# Frozen repair scope

- Parent: immutable v1 specification, MANIFEST `43c670308750c00f9d45e2167267196dd494b74798dd0c883e6c6609aff38f19`, COMPLETE `d51ebe4f9881764408fc7326bbf34afb2de105cb2eeda2242fa724a453084609`.
- Decision: repair only the two blocking review findings: machine-defined runtime storage accounting and managed-kill/timeout whole-stage restart evidence.
- Effective document: v1 plus the exact replacements in `CORRECTIONS.json`; v2 supersedes v1 for H5-06, H5-11 and the named storage stop-condition string only.
- Unchanged: accepted input, source/config boundary, placement roles, capacity numbers, H5 preference, all other gates, exclusions, invalidations and no-launch status.
- Completion: correction packet seals after structural validation and receives focused independent rereview.
- Exclusions: no H5 contact, voltage read/hash, transfer, materialization, service installation, fixture execution, launch, sort, RF/holdout access, or outcome inspection.

