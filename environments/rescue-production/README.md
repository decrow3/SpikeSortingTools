# Rescue production environment

This uv project is the only supported production runtime for
`SpikeGLX_ext_ref_rescue.py`. It is intentionally separate from the
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

Use `deactivate` when finished. The Python program contains the runtime guard;
the tested processing implementation remains in `pipeline/`, while the entry
point is a readable sequential run sheet with restartable stage switches.

To run repository tests, sync with `--group test` instead of `--no-group test`.
Dependency changes must be made in `pyproject.toml`, followed by an intentional
`uv lock --project environments/rescue-production`; commit the project file and
changed lockfile together. Production commands must never use `uv run --with`,
an active Conda environment, or an unlocked sync.
