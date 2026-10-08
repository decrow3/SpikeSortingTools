# Candidate 3 active-path map

## Supported prospective path

| Boundary | Active binding | Status / owner / next touch |
|---|---|---|
| Execution host | H1 and H5 remain candidates. H1 actual context successfully allocated on its RTX A5000; only the task sandbox lacks NVIDIA devices. | H1 input route/storage remain unresolved; H5 is fallback, not the default. No concurrent full sorts. |
| Core checkout | `/home/huklab/Documents/DARTsort`, clean commit `edcfe1b51d672b4136eb13cc78c0875da804b851` when inventoried | DARTsort core owner at next natural edit: H1; keep the patch small and versioned |
| Core entrypoint | prospective `dartsort.main.dartsort`; no candidate-3 production runner exists yet | The real snippet contract must freeze the exact callable and arguments before execution |
| Interpreter | `/home/huklab/Documents/DARTsort/.venv/bin/python`, Python 3.12.4; SpikeInterface 0.104.8; Torch 2.6.0+cu124 | Bind interpreter and imported-source hashes in every execution contract; the installed DARTsort version string is stale and must not substitute for the Git commit |
| Arm-A identity | completed Luke0804 imec0 Arm A, frames `[0,314204894)`, 29,999.835983263598 Hz, 384 channels, 2.34375 uV/count | Supported target; no raw voltage copied by this map |
| Accepted H5 input | `/media/huklaban5/Data/en_rounded_field_ks129_20260929_v1/arm_a_v1/full/recording/traces_cached_seg0.raw`, SHA-256 `672938e942d105af6b1a3f7f1bddf206939535e0e4b57bc01153472a1ccb8b69` | H5-local and absent on H1; avoids transfer only if H5 is selected |
| Input binding | `/mnt/NPX/Luke/DARTsort_motion_experiments/en_a_four_unit_approval_metadata_20261001_v3_h5/INPUT_BINDINGS.json`, SHA-256 `7c045a88be1819d5411534867ee717ceadf76661c3bb3ba176dd525ed5c4e3b1` | Required contract input |
| Remap source | `ExactLatticeRemapRecording` source SHA-256 `0125394629af7bffc6d46a5c26903efd65cd244f30000ae45d3cfa8324abea3a` | Freeze source snapshot/hash in the real run packet |
| Prewhitening producer | installed Kilosort 4.0.27 `BinaryFiltered.filter`, with accepted channel selection, inversion false, per-channel centering, global median CAR, 300 Hz FFT high-pass, infinite/disabled artifact threshold, and `whiten_mat=None` | Luke-specific wrapper/orchestration belongs in SpikeSortingTools, not DARTsort core; owner H1; next touch is the real snippet contract/runner |
| DARTsort consumer boundary | float traces at the Kilosort prewhitening boundary, consumed with `preprocessing="none"` | Synthetic imported-source fixture supports this algebraic boundary; real Arm-A equality and padded-batch equivalence remain unestablished |
| Checkpoint core | `/home/huklab/Documents/DARTsort/src/dartsort/peel/peel_base.py`, SHA-256 `ac357ade5b6f23b690a403e297a2260ea93c2a1b35df13ee7798d8fc8591a2d4` | Current marker-before-payload order permits false resume; owner H1; next touch is a focused append-only transaction repair after review |
| Matching | Existing DARTsort matching implementation, unchanged by this readiness work | Core ownership remains upstream/DARTsort; no relocation. Bind exact source/config only when the snippet naturally touches matching |
| Clock/frame evaluation | Existing SpikeSortingTools evaluation and row-binding tools | H1 at next natural evaluation touch; no new clock estimator and no D3 work |
| Waveform/QC extraction | Existing SpikeSortingTools testing/evidence scripts are historical references, not a frozen candidate-3 runner | Owner to be frozen in the snippet contract; queue consolidation only when this path is next edited |

## Effective configuration status

No full-run effective configuration is frozen. The only currently supported settings are the exact voltage-boundary semantics above and the operational checkpoint rule below. A real snippet contract must additionally freeze recording adapter, padding, geometry, time origin, training context, DARTsort configuration, output namespace, resource bounds, stop condition, and saved provenance before execution.

Until the checkpoint transaction is repaired and reviewed, any interruption requires restarting the affected peel stage. Reuse of prior completed stages is not within-stage checkpointing.

Materializing the whole prewhitening float32 recording would require 482,618,717,184 bytes before filesystem overhead. The next contract must explicitly budget that storage or use a bounded streaming/managed alternative; this map authorizes neither.

## Source and import bindings

- DARTsort `main.py`: SHA-256 `e4c03417c1256384c4f8fbf0250560003bd795b7274932ef1e49dcde217bd249`.
- DARTsort `peel_base.py`: SHA-256 `ac357ade5b6f23b690a403e297a2260ea93c2a1b35df13ee7798d8fc8591a2d4`.
- DARTsort `data_util.py`: SHA-256 `5e5ee25f412a9a077b6d729ccc0a908349aa7399c1c802c9de889f1a83c7e93b`.
- Candidate-3 fixture imports the installed `BasePeeler`, DARTsort preprocessing guard, and installed Kilosort `BinaryFiltered`; its independent verdict is `GO_SCOPED_FIXTURE_CONCLUSIONS`.
- Repository search found many historical/direct DARTsort imports under `testing/`. None is designated as the production Candidate-3 runner by this map.

## Callers, services, dynamic imports, and current jobs

- No candidate-3 production caller or managed service has been installed.
- No live Candidate-3 process was visible in either the task or approved actual process view at map time.
- Approved actual-context `systemctl --user` inspection found historical DARTsort/Luke units but no running Candidate-3 or sorting service. H5 must still check its own managed-job state if selected.
- Repository search found dynamic/local DARTsort imports in testing utilities such as `cross_dataset_fast_motion.py`, `dh_actual_consumer_fixture.py`, and waveform/QC extractors. They are historical/reference consumers, not approved candidate-3 entrypoints.
- No downstream service dependency is frozen. Prospective consumers are candidate-3 saved sorting/evidence, curation/QC, and comparison stages; each must bind the produced run identity and remain downstream of a successful managed sort.

## Supported versus historical

Supported now:

- the clean DARTsort checkout and bound source hashes above;
- the exact H5 input/boundary identities above;
- the candidate-3 readiness inventory and independent review;
- the synthetic boundary and checkpoint fault fixtures, within their independently reviewed scope;
- the H1 CUDA visibility probe as evidence only for the task sandbox, plus the actual-runtime/storage probe showing successful host CUDA allocation and unresolved H1 input/storage routing.

Historical or unsupported for production selection:

- arbitrary older DARTsort scripts/configurations and MEDiCINe/local experiment paths;
- the stale imported DARTsort version string;
- D3 artifacts, which are explicitly deferred;
- any unbound testing script treated as a production runner;
- any claim that synthetic transition batches establish real remap/padding equivalence;
- any assumption that a completed-stage reuse mechanism makes an interrupted peel stage resumable.

## Deferred work and ownership

1. H1: implement and review the focused DARTsort checkpoint transaction patch at the next core touch.
2. H1: select H1 versus H5 only after binding the input route, writable storage, and (if decision-relevant) bounded I/O evidence; then freeze the real snippet runner and exact effective configuration.
3. H1 or H5/Hub scheduling: use a managed job and prove launcher-disconnection survival with a cheap dummy if the launch method is new; never overlap full sorts contrary to the coordinator schedule.
4. Matching, clock/frame, and waveform/QC consolidation: defer until each path's next natural edit; first map its actual callers and consumers, then make only a bounded change.
5. No repo-wide cleanup, dependency upgrade, code relocation, historical deletion, D3 repair, RF loop, or full-session run is part of this packet.

## Implementation checks

- Done: actual source state -> DARTsort checkout was clean at `edcfe1b51d672b4136eb13cc78c0875da804b851`; hashes above bind the inspected core files.
- Done: input and preprocessing boundary -> bound to accepted H5 input, `INPUT_BINDINGS.json`, remap source hash, installed Kilosort prewhitening output, and DARTsort `preprocessing="none"`.
- Done: checkpoint semantics -> imported-source fault fixture and independent review establish a marker-before-payload false-resume hazard for the tested resize failure.
- Done: host CUDA -> exact DARTsort interpreter made a four-byte allocation on H1's RTX A5000 in the approved actual context; the sandbox failure is not a host failure.
- Done: callers/current jobs -> repository imports plus actual user-service/process views were inspected; no candidate-3 production runner or live process was identified.
- Not done: final host/input/storage choice -> H1 lacks the H5-local accepted materialization and cannot fit output plus full float32 materialization on current local targets; H5 current state remains to be checked if selected.
- Not done: real Arm-A equality, padding, and full effective config -> require a separately frozen real snippet contract and H5 execution.
- Can establish: the supported prospective code/data boundary, present ownership, and prerequisites for the next Candidate-3 touch.
- Cannot establish: full-session readiness, scientific performance, real-voltage equivalence, crash durability beyond the scoped fixture, or availability of an H5 launch slot.
