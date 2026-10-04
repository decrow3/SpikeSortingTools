# Implementation checks

- Done: exact candidate integrity -> MANIFEST `48b32b...`, COMPLETE `d9a326...`,
  coordination status `e15c95...`, and every listed member verify.
- Done: actual managed entry -> seven candidate fixtures independently pass.
- Done: v1/v2 delta -> only explicit waveform pair coverage, schema binding,
  and success-report schema assertion changed; v1 remains preserved superseded.
- Done: report schema -> each successful known answer validates and all three
  pair-coverage states are explicitly `UNMEASURED`.
- Done: REF-first and thresholds -> REF mismatch opens no candidate; lower95
  below/equal/above 0.05 fails/fails/passes, while advancement remains false.
- Done: amplitude ancestry -> loader hashes/reads `full_st.npy` and
  `kept_spikes.npy` together and uses `full_st[kept_spikes][:,2]`; fixture has
  no `amplitudes.npy`.
- Done: missingness/no-false-success -> absent QC/waveform stays UNMEASURED;
  post-start input mutation writes FAILURE without report or COMPLETE.
- Done: independent negative control -> a pair COMPLETE with no execution
  states and repaired/repeat outputs outside its namespace is accepted by the
  actual managed path, which writes Phase3 COMPLETE.
- Not done: real output evaluation -> prohibited and unnecessary to reproduce
  the binding defect.
- Can establish: corrected schema and scientific composition fixtures behave
  as specified, but the current pair receipt does not establish that scored
  repaired/repeat outputs came from the completed launcher pair.
- Cannot establish: trustworthy run attribution or activation readiness until
  the pair-output binding is repaired and independently reviewed.
