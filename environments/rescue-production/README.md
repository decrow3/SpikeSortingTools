# Rescue production environment

This uv project is the only supported production runtime for
`SpikeGLX_ext_ref_rescue.py`. It is intentionally separate from the
legacy pipeline environment. The production run sheet may call the audited
legacy-compatible curation/QC exporters already supported by this lock, but do
not add MEDiCINe, notebook, or experimental sorter dependencies here.

The production contract is:

- CPython 3.12.4;
- SpikeInterface 0.102.1;
- Kilosort 4.0.27;
- DREDGE 0.3.0;
- Neo 0.14.0 (required by SpikeInterface 0.102.1's SpikeGLX reader contract);
- PyTorch 2.6.0 with CUDA 12.4 wheels;
- the exact transitive dependency graph and artifact hashes in `uv.lock`.

Matplotlib 3.9.1 emits a global PyPI yanked-release warning because its Windows
wheels were removed. The manylinux wheel is functional, matches the validated
host runtime, and is retained intentionally with its artifact hash frozen in
the lock. Changing it still requires the normal upgrade validation process.

Neo is intentionally pinned at 0.14.0. SpikeInterface 0.102.1 forwards the
`load_sync_channel` keyword to `SpikeGLXRawIO`; Neo 0.14.1 and later removed
that keyword and fail before a SpikeGLX recording can be opened.

The published Kilosort 4.0.27 wheel also has an empty-clustering-center return
arity defect: `get_data_cpu()` returns four nulls while its caller unpacks
three values. Before sorting, the pipeline applies the source-hash-guarded
`kilosort-4.0.27-empty-clustering-center-v1` compatibility repair. It refuses
unknown source, changes only that empty-center branch, and records the patched
source SHA-256 in the sort request and manifest. This replaces the undocumented
hand edit that existed in the historical Conda environment.

For every production run, start from the repository root. Leave any active
Conda environment, exactly sync the uv runtime to the committed lock, activate
it, and verify package versions and GPU visibility:

```bash
conda deactivate  # only when a Conda environment is active
uv sync \
  --project environments/rescue-production \
  --frozen \
  --no-group test
source environments/rescue-production/.venv/bin/activate
python environments/rescue-production/verify_environment.py --require-cuda
```

Edit the clearly marked settings block at the top of
`SpikeGLX_ext_ref_rescue.py`, then run it as a normal Python script without CLI
arguments:

```bash
python SpikeGLX_ext_ref_rescue.py
```

For each new recording, set `LOCAL_WORK_DIR` to a recording-specific folder on
the local NVMe. The run sheet stages only the selected SpikeGLX stream there,
resumes an interrupted copy from its local byte offset, and then builds the
accepted preprocessed binary in `LOCAL_WORK_DIR/recording`. Channel metrics,
motion estimation, Kilosort, artifact scans, and QC therefore reuse local data
instead of repeatedly reading the server. `OUTPUT_DIR` remains the durable
results location. Never share one `LOCAL_WORK_DIR` between recordings or
streams; its manifests reject a mismatched source rather than overwriting it.

For a completed sort, leave `PIN_COMPLETED_SORT_IDENTITY` and
`RUN_FULL_PROBE_DIAGNOSTICS` enabled. The first downstream pass should use:

```python
WRITE_RAW_ARTIFACT_SIDECAR = True
RUN_SIMILAR_PAIR_AUDIT = True
RUN_CURATION = False
RUN_QC = False
RUN_MATLAB_EXPORT = False
RUN_POSTCURATION_COMPARISON = False
```

Review `diagnostics/artifact_pair_audit/` before enabling curation. Then enable
the last four switches in order and rerun the same script. Completed exact
stages are reused; interrupted stages retain a pinned request and resume from
their existing cache where the underlying implementation supports it. The
important downstream outputs are:

- `rescue_sort_identity.json`: immutable hashes of the sort manifest and core
  sorter arrays;
- `diagnostics/full_probe/`: identity-bound automatic diagnostics and the
  frozen acceptance decision;
- `artifacts/raw_over_500uv.h5`: restart-safe raw artifact sidecar;
- `diagnostics/artifact_pair_audit/`: all current similar good-good pairs;
- `cur/cur_output/`: legacy-compatible curated Kilosort/Phy output;
- `qc/`: waveform, presence, truncation, refractory, and MATLAB artifacts;
- `diagnostics/postcuration_comparison/`: matched new, legacy, and claim-mask
  curated-population metrics;
- `decision/formal_decision.json`: conservative promotion guardrail.

Never copy an older diagnostics directory onto a new sort. Every downstream
receipt contains `sort_identity_digest`; a mismatch is a hard error rather
than a cache miss.

Use `deactivate` when finished. The Python program contains the runtime guard;
the tested processing implementation remains in `pipeline/`, while the entry
point is a readable sequential run sheet with restartable stage switches.

To run repository tests, sync with `--group test` instead of `--no-group test`.
Dependency changes must be made in `pyproject.toml`, followed by an intentional
`uv lock --project environments/rescue-production`; commit the project file and
changed lockfile together. Production commands must never use `uv run --with`,
an active Conda environment, or an unlocked sync.
