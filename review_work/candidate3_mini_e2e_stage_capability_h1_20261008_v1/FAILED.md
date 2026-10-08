# Frozen v1 execution failed during input construction

The exact sample clock makes 297,000 frames span approximately 9.900054 s, while the frozen two 4.95 s cells cover only `[0,9.9)`. `ExactLatticeRemapRecording` correctly rejected the first uncovered sample at 9.900020792303355 s. No DARTsort arm ran.

The generated scratch is retained at `/tmp/candidate3_mini_e2e_20261008_v1`. This is an input-support boundary defect, not scientific evidence. A v2 contract will make the sole repair of deriving cell width and centers from `N_FRAMES / FS`; seed, frames, waveforms, shifts, processing and stage settings remain unchanged.

Implementation checks
- Done: actual adapter rejected an uncovered sample at the recording tail.
- Not done: Kilosort boundary and DARTsort stages -> input construction stopped first.
- Can establish: v1 temporal support was short by the exact-clock remainder.
- Cannot establish: any stage capability or arm comparison.
