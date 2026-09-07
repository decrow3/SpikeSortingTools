# Hindsight-first ladder implementation map

This map was written before extending the framework. It records the narrow
composition points chosen from the existing repository.

| Required capability | Existing implementation | Disposition |
|---|---|---|
| Accepted recording identity and content validation | `pipeline.preprocess.validate_accepted_recording`, `recording_binary_receipt`, geometry receipt and materialization integrity sampling | Reuse directly |
| Long physical depth strip with halo/interior contract | Production materializer supports time slicing but not physical channel-strip requests | New narrow development adapter in `testing/development_strip.py`; reuse production validation helpers |
| Content-digested standard KS4 variants | `testing.ladder_sorter.SorterConfig`, effective saved-settings checks, atomic cached sort runner | Reuse and add only 10/9, 9/9 and 9/8 no-motion names |
| Native motion-off, rigid and nonrigid arms | Existing `RESCUE`, `RESCUE_RIGID`, `NONRIGID` configurations | Reuse directly |
| Sort identity, identical restartable curation/QC/export | `pipeline.downstream` identity and stage runners | Reuse directly from the development runner |
| Exclusive full-session correspondence | `testing.luke_full_session_compare` | Extract generic calculations into `testing/sort_comparison.py`; historical script imports them |
| Correct amplitude source and fit-status normalization | `testing.luke_amplitude_dropout_audit` loaders and `build_windows_table` | Call narrowly from generic input loader |
| Common-physical-time completeness | Corrected `common_time` implementation in full-session comparison | Extract unchanged semantics into generic comparator |
| Longitudinal and spatial diagnostics | `testing.luke_full_strip_diagnostic_audit`, `testing.luke_full_probe_rescue_diagnostics` | Reuse small calculations/conventions; do not depend on hard-coded runners |
| Chance-aware coincidence | Shift-null logic in full-strip/full-probe diagnostics | Small generic marked-spike + deterministic shift-null calculation |
| Stable comparison artifact set | No generic implementation; current full-session script is Luke/path bound | New `compare_sorts` artifact writer under `testing/` |
| Prospective coverage/efficacy/guardrail/Pareto gates | Distributed across historical prespecs | New machine-readable contract and gate logic in `testing/development_ladder.py` |
| Challenger sorters, custom correction, stitchers, peelers, injections | Historical bakeoff and ladder branches | Explicitly untouched and absent from the first ladder |

The production graph is not refactored. New orchestration and evaluation remain
under `testing/`; only existing generic production primitives are imported.

