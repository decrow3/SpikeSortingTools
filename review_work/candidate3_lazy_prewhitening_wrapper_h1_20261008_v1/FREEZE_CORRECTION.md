# Freeze chronology correction

The preliminary workspace contract in `PRELIMINARY_CONTRACT_SUPERSEDED.json` was frozen before the first 10-test wrapper run. That run passed. After inspecting the integration surface, the author strengthened—not relaxed—the DARTsort acceptance rule to require the actual `dartsort.main.dartsort` boundary functions `ds_will_copy_recording` and `ds_all_to_workdir`, added their exact source hashes, and reran the complete fixture.

The strengthened contract is the committed file bound by `FROZEN_CONTRACT.md`. It passed in the final 22-test combined run recorded by `PYTEST.xml`. The preliminary rule and result are not presented as the final acceptance. This correction preserves the original rule rather than retroactively calling the strengthened rule preregistered.

