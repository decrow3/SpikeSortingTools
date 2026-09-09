# Whole-probe adaptive lighthouse consensus: active run

User priority: collect roughly20–30distinctive candidates along the whole probe, permit depth-aware dropout and variable per-unit support, then form a consensus through time. This run covers930–1030s; it is not session-wide validation.

Thirty candidates were selected in ten depth strata from930–940s waveform proposals, spanning100–3560µm. Five cached whole-probe waveform flags (38,79,186,269,701: local energy fraction<0.35 or distant peak ratio>0.8) remain displayed but do not vote in the primary consensus. The other25are provisional candidates, not certified independent neurons. The980–990s label inventory did not select or centre templates. Spike labels only seed templates; target matches come from independent voltage detections.

Depth search uses exact same-column40µm geometry translations over±120µm where geometry permits, transported waveform support, nearby competing identity/shift hypotheses, and explicit unmatched/identity-ambiguous/depth-ambiguous/search-boundary states. DREDGE does not enter selection or matching. Shift-sign synthetic checks passed. Thresholds are exploratory and have not been independently calibrated for this population search.

Adaptive report bins target20accepted events, require10, span≤5s, and split at>2s gaps or coarse-shift changes. Six640µm regions get separate consensus estimates; a global rigid trace is not assumed. Each candidate family has one vote; at least3families are needed. The0.25s grid includes only actual accepted-spike support; bin spans define effective resolution. Repeated grid entries from the same adaptive window are not independent measurements. Waveform observations are frozen before loading DREDGE, which is sampled at matching event times and referred to each template's exact seed-spike times. A common template epoch still leaves uncertainty from different firing phases.

Deliverables, generated automatically after successful extraction: whole-probe depth/time scatter, all-candidate motion overlays, waveform contact sheet, depth-resolved consensus with coverage, adaptive window/event-support CSV/NPZ, template quality flags, source hashes, and README. Expected report directory: `testing/outputs/luke_population_consensus_v1/`. The current template audit is [available here](../testing/outputs/luke_population_depth_v2/01_template_full_probe_contact.pdf).

Independent services:

- `luke-population-depth-v2`: extraction. Commands/settings/stdout/stderr/exit receipt at `testing/outputs/luke_population_depth_v2_job/`.
- `luke-population-consensus-wait-v1`: waits for successful extraction receipt, verifies every completed-stage checksum, then renders. Durable job evidence at `testing/outputs/luke_population_consensus_wait_v1_job/`.

Both were verified active with nonzero MainPID after launcher disconnection. This note records launch status; consult actual systemd state and final receipts for current progress. Extraction commits hashed template and10s-chunk stages. A resumed run validates completed stages, archives partial-stage evidence and restarts the incomplete chunk. It is not a within-chunk checkpoint. The first v1 attempt was deliberately stopped after review; its partial results and termination logs remain preserved. No sorting or motion-estimator tuning was launched.

Initial930–940training checkpoint:919accepted events across25candidates,7105identity-ambiguous events,43boundary events,19depth-ambiguous events after per-identity duplicate suppression. Training support is not independent validation. A cached pairwise timing audit found no pair with≥5coincident accepted events within0.5ms in that chunk; this is only a preliminary duplicate-vote check, not identity proof.

Synthetic consensus checks passed: duplicate-family invariance, opposite motions in separate depth regions, no artificial gap/grid filling, and no averaging across coarse-shift changes. Actual consensus interpretation remains pending extraction and visual review.
