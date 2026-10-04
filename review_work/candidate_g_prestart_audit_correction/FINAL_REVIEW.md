# Candidate-G corrected prestart-audit focused review

## Verdict

`GO_CREATE_EXACT_RUN_PARENT_AND_RERUN_READ_ONLY_AUDIT`

The small unit-query correction is sound. Definition presence is checked through exact filesystem paths and the complete supported `list-unit-files` inventory; concrete `systemctl show` calls are limited to the main unit and exact finalizer instance. All seven expected-answer cases pass independently, including masked/symlinked definitions and both query-error paths.

The original template-query failure remains sealed and preserved. The filtered-inventory failure and corrected parent-absence failure also match their immutable manifest hashes. Source inspection confirms that canonical/unit absence checks precede the parent check and that the audit performs no canonical write; it stopped at the accepted v6 bootstrap's explicit `safe_parent` precondition before state, input, resource or GPU checks.

Creating exactly `/home/huklaban5/DARTsort_experiment_scratch/candidate_g_continuations` is a routine bounded prerequisite, not a change to the reviewed boundary. The v6 canonical map already fixes that directory as `RUN_PARENT`, and the reviewed bootstrap requires it to exist before exclusively creating the separately frozen attempt child.

## Exact permitted next action

On huklaban5 only:

1. Read-only verify `/home/huklaban5/DARTsort_experiment_scratch` is an existing real directory, resolves to itself, is owned by the executing huklaban5 UID, and is not world-writable. Reconfirm the target parent, canonical release, unit files, attempt child and compact output are absent, including symlinks.
2. Create only the direct child `candidate_g_continuations`, non-recursively, as an empty directory owned by the executing huklaban5 UID with mode `0700`. Do not use parent-creating behavior.
3. Preserve a new receipt containing before/after lstat identity, resolved paths, numeric UID/GID, mode and emptiness.
4. Rerun once the exact corrected read-only audit whose SHA-256 is `16330d5551a9a55c8c2b8a68d3ff24cd57ec5701bae5990c5f9905c91fcbbac2`, using an absent/fresh temporary receipt namespace. Seal the outcome and stop whether it passes or fails.

This does not authorize creation of the attempt child or compact output, release construction, unit installation/reload, service start, recording-voltage access or GPU computation.

Implementation checks
- Done: query semantics -> inspected the corrected source and reran all seven unit-definition/instance cases successfully.
- Done: source and receipts -> verified corrected source `16330d5551a9a55c8c2b8a68d3ff24cd57ec5701bae5990c5f9905c91fcbbac2`, filtered failure `5762f371509b7cb8abbeb1ac4987dbbcbe6545d60c4346a4b09efdbe0c52aedd`, and terminal failure `48304294c5ddb4deeb43bd9d27be94b6050c405c83541bd4e9726a9cfd05eb61` against the packet manifest.
- Done: preserved failure -> independently verified the original eight-product failure packet, MANIFEST `ee101cac09b807a60121e41ef5fa9165bcae6a5b1e8e15f4fb2b679e517e5ee5`, and COMPLETE `9dd9244b4229eb718c59a9cff68b7a8f6b04c46a04e48c97b47b75e78118ddb8`.
- Done: boundary -> traced canonical `RUN_PARENT` and `safe_parent`/`provision`; the one empty parent is a required precondition, while the attempt child remains bootstrap-owned.
- Done: shared canonical absence -> release and compact paths are currently absent from huklaban1's shared view.
- Not done: independent current H5 home/systemd inspection -> huklaban5 home and user bus are not visible from huklaban1; exact absence and owner/mode checks are mandatory before creation.
- Not done: full readiness -> state/input/resource/GPU checks were not reached and require the authorized read-only rerun.
- Can establish: the audit correction is valid and the exact empty run-parent creation stays within the accepted v6 boundary.
- Cannot establish: full readiness or permission to construct/install/start production until the new read-only audit is sealed and reviewed/handed off under the existing activation process.
