# Rescue production environment

This uv project is the only supported production runtime for
`SpikeGLX_ext_ref_rescue_testing.py`. It is intentionally separate from the
legacy pipeline environment. Do not add legacy curation, MEDiCINe, notebook, or
experimental sorter dependencies here.

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

From the repository root, create or exactly update the runtime with:

```bash
uv sync \
  --project environments/rescue-production \
  --frozen \
  --no-group test
```

Verify package versions and GPU visibility before a sort:

```bash
uv run \
  --project environments/rescue-production \
  --frozen \
  --no-group test \
  python environments/rescue-production/verify_environment.py --require-cuda
```

Always invoke the production pipeline through the same project:

```bash
uv run \
  --project environments/rescue-production \
  --frozen \
  --no-group test \
  python SpikeGLX_ext_ref_rescue_testing.py --help
```

The repository-root launcher supplies this exact prefix. A complete configured
run can therefore be started with:

```bash
./run_rescue_pipeline --config configs/rescue/luke0804_imec0_smoke_60s.json
```

To run repository tests in the same locked graph, add `--group test` instead of
`--no-group test`. Dependency changes must be made in `pyproject.toml`, followed
by an intentional `uv lock --project environments/rescue-production`; commit
the project file and changed lockfile together. Production commands must never
use `uv run --with`, an active Conda environment, or an unlocked sync.
