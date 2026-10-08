# A four-unit physical-signal capture: execution result

The frozen H1 GO receipt was copied byte-for-byte and exactly one start request was issued for `en_a_four_unit_physical_signal_capture_approved_v3.service`. The capture completed successfully without retry. The final service state is loaded/static and inactive/dead with `Result=success`, exit status 0, PID 0, and zero restarts.

The run made 41 bounded read calls for 32,492,544 logical bytes and retained 3,912,192 numeric bytes. `STAGE_SUMMARIES.csv` contains 40 rows covering the original and corrected-A representations. The two retained `int16` waveform archives remain in the H5-local source output directory; this compact publication records their sizes, array counts, and SHA-256 hashes but does not transfer voltage.

No spike sort, RF evaluation, holdout access, or retry occurred. This packet reports execution and capture integrity; it does not by itself establish biological identity, sorter improvement, or downstream scientific efficacy.

## Implementation checks

- Done: frozen final-GO packet and launch receipt hashes matched the delegated bindings before launch (`PRESTART_CHECK.json`).
- Done: installed config, service, capture source, and launch validator hashes matched before receipt copy and launch (`PRESTART_CHECK.json`).
- Done: fresh-path and inactive-unit checks passed before the single start (`PRESTART_CHECK.json`).
- Done: completion marker, counters, summary-table row count, retained-array dtype/count/bytes, artifact hashes, and terminal service state were checked (`ARTIFACT_INVENTORY.json`, `UNIT_STATUS.json`).
- Not done: scientific waveform interpretation and downstream sorter evaluation; those require separately frozen analyses using these retained arrays.
- Can establish: the reviewed bounded capture executed once, completed successfully, stayed within its recorded read/retention bounds, and produced intact local artifacts.
- Cannot establish: biological identity, purity, motion-correction benefit, covariance validity, or sorter performance.
