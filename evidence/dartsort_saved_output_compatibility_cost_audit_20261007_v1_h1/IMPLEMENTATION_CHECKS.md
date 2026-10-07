# H1 implementation-first compatibility review

Verdict: **NO-GO for the original R1 WIU/production endpoint; conditional GO for a newly named DARTsort-specific descriptive saved-output scorecard.** No new sorter run or raw-voltage read is required for the scoped scorecard.

The strongest reusable signals are final event samples and labels. They support counts, firing rate, fixed-bin presence, duplicate/ISI measures, the reviewed sliding-refractory algorithm, and the segment-safe 9–29-sample short-interval statistic. DARTsort denoised PTP can support a separately named amplitude-truncation diagnostic, but it is not established equivalent to Kilosort amplitude or PC norm. The saved SpikeInterface amplitude is the recording value at the unit template's peak channel and shift (`spike_amplitudes.py:9-20,29-50,93-111`), another different measurement.

The physical 50 µV criterion is invalid here. The executed preprocessing high-pass filters, removes bad channels, common-references, divides every channel by its native-unit noise level, and common-references again (`preprocess_util.py` at DARTsort git object `dbda377f`:23-72). QC then sets `return_in_uV=False` (`source/qc_phy.py:74-87`). Retained gain metadata cannot invert channel-specific normalization.

The exact imec1 sample clock and channel support align with the later full-session contract: 314,204,094 frames, 29,999.759166666667 Hz, AP-frame-zero origin, and AP191 removed. The common 40-µm q-state lattice itself is missing on H1. DARTsort's own four-depth MEDiCINe field is a different estimator/frame and cannot replace it.

All 1,872 exported Phy groups are `unsorted`; the executed QC explicitly preserves every accepted final event and filters no units (`source/qc_phy.py:93-106`). Therefore KS-good counts or a curated population cannot be reproduced from this base archive without a separately frozen curation redesign.

Implementation checks
- Done: what ran -> exact wrapper source independently found in three copies and reproduced SHA-256 `53042b121d74dbea48293d0ba3d21ca6e163d57ade034ab4b5ed148475a874a2`; executed config, internal config, provenance, recovery receipt, QC source, and QC receipt inspected.
- Done: preprocessing/support -> wrapper applies external `ibllikecmr`, then DARTsort runs with internal preprocessing `none`; 383 channels remain and channel `imec1.ap#AP191` is absent; full cached voltage is local but is unnecessary for this route (`source/pipeline.py:273-291,306-365`; recording attributes).
- Done: axes/frames -> HDF5 `point_source_localizations` is `(N,3)` x/y/z_abs from amortized point-source localization; z is column 2. `channels` is a main/template channel and is not accepted as measured waveform depth. DARTsort motion is depth-by-time; corrected depth equals observed minus displacement.
- Done: clocks -> final samples are recording-local from AP frame zero; `fs=29999.759166666667`, frames `314204094`, duration about `10473.553879 s`; QC validates all accepted samples in `[0,frames)` (`source/qc_phy.py:50-70`).
- Done: silent caps/defaults -> QC waveforms are capped at 200 uniform seed-0 spikes/unit, 103 samples, radius 100 µm; PC projections are 3 components/channel and whitened. Full final HDF5 has 66,286,937 rows, of which 999,382 have negative labels and 65,287,555 are accepted.
- Done: matching/counting -> directly portable metrics require explicit within-unit ordering and prohibit cross-unit/segment/block interval counts. The R1 fixed-histogram bootstrap is already independently rejected; exact per-draw amplitude maxima are mandatory.
- Done: state/domain meaning -> DARTsort MEDiCINe continuous field and later Arm-A rounded q lattice are distinct. Common q=0/displaced scoring is disabled until the exact lattice is supplied and hash-bound. Spatial scoring must freeze observed versus registered model-derived z.
- Done: circularity -> no output scores were inspected or produced; this packet classifies inputs and implementations only.
- Done: provenance -> sorting NPZ is receipt-bound to SHA-256 `772acc6a810bd65ea3dd8b99f197c4b8d3fcfb9248091027877d26fca50e7aa7`; executed source/config/QC hashes and key compact-array hashes are recorded. The 43.817 GB feature HDF5 still requires a fresh full-file hash or reviewed chunk manifest before scoring.
- Not done: original R1 reproduction -> impossible from these saved measurements because physical µV amplitude and curated/KS-good semantics are absent; imec0 R1 and imec1 DARTsort also refer to different probes/pipelines.
- Not done: common state score -> exact 41,895-row q lattice (expected SHA-256 `92d7a28ccc808f76924a6ba392eb71c1a9b0b867796615a15307d1c59bc7df4f`) is unavailable on H1/shared storage.
- Not done: scientific outcome -> no scorecard was executed, and no biological identity, purity, motion benefit, or sorter causality is claimed.
- Can establish: existing outputs are sufficient for a bounded, no-sort, no-voltage-read DARTsort-specific descriptive scorecard after the named bindings are frozen.
- Cannot establish: the original R1 WIU endpoint, physical amplitude completeness, KS-good/curated yield, or production superiority/harm.

## Source citations

- Executed wrapper: `/media/huklab/Data/NPX/Ryansorting/Luke/DARTsort_motion_experiments/luke0804-imec1-full-medicine-shared-recovery-20260913-223213/source/pipeline.py`, especially lines 60-82, 273-291, 306-365, and 404-428.
- Executed QC: `/media/huklab/Data/NPX/Ryansorting/Luke/DARTsort_motion_experiments/luke0804-imec1-full-medicine-shared-recovery-20260913-223213/source/qc_phy.py`, especially lines 50-70, 74-110, and 115-130.
- Exact DARTsort amplitude implementation: git object `dbda377f:src/dartsort/transform/amplitudes.py`, lines 9-121; the file is byte-identical at run commit `2162273`.
- Exact DARTsort localization implementation: git object `dbda377f:src/dartsort/transform/amortized_localization.py`, lines 22-40 and 62-103.
- Exact SpikeInterface amplitude implementation: `/home/huklab/Documents/DARTsort/.venv/lib/python3.12/site-packages/spikeinterface/postprocessing/spike_amplitudes.py`, lines 9-20, 29-50, and 93-111.
- R1 contract: `/mnt/NPX/Luke/DARTsort_motion_experiments/FULL_SESSION_EVALUATION_REDESIGN_20261004.md`, lines 31-67 and 88-106.
- Prior R1 implementation correction: `/mnt/NPX/Luke/DARTsort_motion_experiments/en_full_session_r1_imec0_rescore_final_review_20261004_v1_h1/REVIEW.md`.
