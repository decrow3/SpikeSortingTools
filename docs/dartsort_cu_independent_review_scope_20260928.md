# CU H1 independent review scope

Frozen before inspecting CU waveform outcomes.

H1 will independently review, without duplicating H5 acquisition:

1. the exact saved W2 shift matrix orientation, selected constituents and
   applied offsets, using the signed source fixture rather than assumed
   antisymmetry;
2. per-pair `s_ab`, applied `u_b-u_a`, residual
   `e_ab = s_ab - (u_b-u_a)`, and observed pre/post event lag;
3. direct-force, transitive, score-reassigned and same-source routes, with exact
   8, 9--29 and 30-sample boundaries kept separate;
4. the frozen raw-evidence target selection: no more than 96 target pairs
   stratified as 32 same-source, 32 cross-source/zero-offset and 32
   cross-source/nonzero-offset, including the sole originally-close 9--29 pair;
5. deterministic parent/state round-robin selection and immutable row IDs;
6. at most 32 same-pair 5--20 ms comparisons, treated as context rather than
   neuronal truth;
7. at most 1,500 independently selected isolated reference events, up to 30 per
   constituent, distinct from target selection;
8. exact source clock, 182-channel windows (42 samples before, 79 after), raw
   byte and 512 MiB caps, hashes, and physical support masks;
9. independent morphology panels and any one frozen minimal descriptive
   one-versus-two residual operator, only if the saved evidence supports it.

Interpretation is precommitted: deterministic lag or pair permutation is not a
physical duplicate classifier. The CS 1/6,250 result rejects the current merge
alignment step as the main source of newly separated pairs; it does not reject
pre-existing detection offsets that another alignment model might correct.

Limits: 1,200 additional all-work CPU seconds, two threads, one reader, 20 GB
RAM, 500 MB final output, at least 30 GB free, and no H1 GPU, raw-voltage read,
sort, prefix, matching, template construction, field fit, generic framework or
calibration. Starting cumulative charge is 16,733.22 s and the 19,300 s ceiling
remains sufficient.
