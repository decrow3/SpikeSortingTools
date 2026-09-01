# Conservative DREDGE Sidecar for the KS4 Production Pipeline

## Status

Proposed implementation plan, 2026-08-31.

This plan adds routine, provenance-safe rigid motion estimation to the accepted
Neuropixels rescue pipeline without changing the voltage presented to Kilosort
4. It deliberately does **not** authorize voltage motion correction.

The initial production behavior is:

> Estimate rigid motion with DREDGE, save the estimate and its quality-control
> evidence, and sort the exact accepted recording with KS4 motion correction
> disabled.

Selective rigid voltage correction remains a future research branch. It may be
implemented only after a separately versioned crossover policy and its operator
have passed the required validation.

---

## 1. Motivation

The repaired production baseline intentionally omits external voltage motion
correction and disables KS4's native correction. Historical testing showed that
automatically applying a fine nonrigid DREDGE field could substantially increase
detection and collision burden while worsening contamination and reviewed-neural
recovery. More detections were not evidence of a better sorting result.

At the same time, broad rigid DREDGE motion was reproducible in supported
regimes and remains useful for:

- diagnosing motion severity;
- identifying unsupported or anomalous epochs;
- motion-aware post-sort coordinates and QC;
- cross-probe and split-half corroboration; and
- deciding which recordings merit future correction research.

The architecture must therefore keep four questions separate:

1. What motion did DREDGE estimate?
2. Is that estimate credible, and during which epochs?
3. Is credible motion large enough that correction is expected to help?
4. Is a validated operator and policy authorized to modify voltage?

Only the first two questions are in scope for the initial production milestone.

---

## 2. Production architecture

```text
accepted materialized rescue recording
    |
    +--> deterministic rigid DREDGE sidecar
    |       |
    |       +--> raw estimate or failure receipt
    |       +--> support and QC metrics
    |       +--> machine-readable summary and figures
    |       +--> optional research-only split-half audit
    |
    +--> the same accepted recording directory
            |
            +--> KS4 with do_correction=False
            +--> standard QC and post-sort reconciliation
            +--> optional qualified motion-coordinate sidecar
```

The DREDGE branch is observational. It does not sit between the accepted
recording and KS4, and its success is not required for a safe identity sort.

### Required invariants

- The KS4 input is the existing accepted materialized recording.
- No identity recording is rematerialized through a motion operator.
- KS4 `do_correction` remains `False` and is checked in saved sorter settings.
- Production DREDGE is rigid-only.
- A nonrigid field cannot be selected by production configuration.
- A DREDGE failure produces a failure receipt and does not silently reuse an
  incompatible cache.
- A DREDGE failure does not prevent KS4 from sorting the unchanged recording.
- No guessed QC or displacement threshold is represented as validated.

---

## 3. Scope

### Initial production milestone

Implement:

1. An immutable, versioned rigid DREDGE configuration.
2. An explicitly defined estimator-input construction.
3. Cache-safe DREDGE estimation with complete effective provenance.
4. Separate raw-estimate and failure-receipt artifacts.
5. Rigid motion extraction with a frozen sign, time, depth, and reference
   convention.
6. Epoch-resolved support and QC metrics.
7. An optional deterministic split-half audit.
8. Machine-readable and human-readable motion reports.
9. Orchestration that always routes the accepted recording directly to KS4.
10. Tests proving identity routing and preventing production nonrigid voltage
    application.

### Explicitly out of scope

- Applying any DREDGE field to voltage.
- Implementing `apply_selective_motion()`.
- Enabling KS4 native motion correction.
- Defining `d_on_um` or `d_off_um` from the current incomplete crossover data.
- Treating a full nonrigid field as a production artifact.
- Filling, extrapolating, or force-zeroing unsupported motion bins.
- Running a new corrected-versus-identity sorting experiment.
- Replacing KS4 with KS2, DARTsort, KIASORT, or another challenger.

---

## 4. Estimator input contract

The estimator input must be reproducible and must not be described only as
"accepted raw conditioning." The implementation must define and record:

- accepted recording manifest digest;
- selected start and end frames;
- time origin relative to the selected recording;
- channel IDs and ordering;
- probe geometry and geometry hash;
- bad-channel identities and policy;
- gain and dtype;
- all estimator-specific filtering and referencing;
- peak detection settings;
- peak selection or subsampling settings and random seed, if any;
- localization method and settings;
- DREDGE settings, including inherited defaults; and
- SpikeInterface, DREDGE, NumPy, and relevant package versions.

Estimator preprocessing may differ from the sorter input only through a named,
versioned estimator view. It must remain a side branch and must never replace
the accepted sorter recording.

Suggested interface:

```python
estimator_view, input_receipt = build_motion_estimator_input(
    accepted_recording,
    config=motion_config.estimator_input,
)
```

The effective receipt, rather than only user-provided overrides, participates in
the request fingerprint.

---

## 5. Production DREDGE mode

Production uses a direct rigid DREDGE estimate. It does not derive a rigid trace
by silently collapsing a nonrigid field.

```yaml
motion:
  estimate: true
  estimator: dredge
  estimator_mode: rigid
  save_nonrigid_for_diagnostics: false
```

If nonrigid estimation is needed for research, it must use a separately named
research configuration and artifact namespace. That configuration is not an
allowed input to any production correction interface.

The implementation must freeze these coordinate conventions:

```text
time_reference: selected_recording_start
depth_reference: probe_y_um
displacement_convention: observed_depth_offset_um
```

It must also record the displacement reference explicitly:

```python
reference_method: str
reference_interval_s: tuple[float, float] | None
reference_displacement_um: float
```

No future threshold may be applied to `abs(displacement)` until the reference
method and sign convention have been validated and matched to the policy.

---

## 6. Artifact schemas

Raw estimates and qualified downstream fields are different artifact types.

### 6.1 Raw estimator artifact

Suggested schema: `dredge-rigid-estimate-v1`.

```python
@dataclass(frozen=True)
class RigidMotionEstimate:
    schema_version: str
    estimator: str
    estimator_version: str
    displacement_um: np.ndarray
    time_s: np.ndarray
    support_by_time: np.ndarray
    reference_method: str
    reference_interval_s: tuple[float, float] | None
    reference_displacement_um: float
    time_reference: str
    depth_reference: str
    displacement_convention: str
    provenance: dict
```

This artifact reports what the estimator returned. Its existence does not mean
the estimate passed qualification.

### 6.2 Failure receipt

Suggested schema: `dredge-estimation-failure-v1`.

The receipt must contain:

- request digest;
- attempted effective configuration;
- source recording manifest digest;
- failure stage;
- exception type and concise message;
- software versions;
- timestamp; and
- `safe_fallback: identity`.

It must not contain fabricated zero motion or reuse a previous trace.

### 6.3 Qualified motion field

The existing `qualified-motion-field-v1` contract remains the only artifact
accepted by downstream qualified coordinate adapters. A raw DREDGE estimate
must not be relabeled as qualified merely because estimation completed.

Conversion from a raw estimate to a qualified field is a later, explicit step
that requires a versioned qualification policy.

---

## 7. Cache and provenance rules

Never accept a cache based only on the existence of `motion.npy`.

The request digest must cover:

- the accepted recording request/manifest digest;
- selected frame range and time origin;
- channel map and geometry hash;
- bad-channel policy;
- complete effective estimator-input settings;
- complete effective peak detection and localization settings;
- complete effective DREDGE settings;
- estimator mode;
- software versions; and
- pipeline/schema versions.

Cache behavior:

```text
matching complete artifact  -> reuse
missing artifact            -> compute
partial artifact            -> refuse and report
mismatched request digest   -> refuse unless explicitly recomputing
legacy artifact provenance  -> refuse
```

Artifacts must be written into a partial directory and accepted atomically only
after structural validation succeeds.

At minimum validate:

- finite, strictly increasing time bins;
- expected array shapes;
- finite displacement in supported bins;
- finite, nonnegative support;
- matching source/configuration digest;
- explicit coordinate conventions; and
- presence of required effective parameters and versions.

---

## 8. Motion QC

QC is epoch-resolved. A single session Boolean must not hide localized support
dropout.

Compute at minimum:

- peak count per time bin;
- peak count per time-by-depth bin;
- fraction of empty and low-support bins;
- depth support span;
- displacement step distribution;
- speed distribution;
- largest step;
- P95 and P99 absolute step;
- rigid range and robust displacement percentiles;
- missing or nonfinite bins; and
- estimator jumps or speeds identified by configured rules.

Suggested object:

```python
@dataclass(frozen=True)
class MotionQC:
    status: Literal[
        "VALID",
        "PARTIALLY_VALID",
        "INVALID",
        "NOT_EVALUATED",
    ]
    valid_by_time: np.ndarray
    uncertainty_by_time_um: np.ndarray
    reason_codes_by_time: np.ndarray
    metrics: dict
    policy_version: str | None
```

Thresholds belong to an immutable, versioned configuration. Until thresholds
are independently justified, the production sidecar reports metrics with
`status="NOT_EVALUATED"`; it does not guess an authoritative pass/fail result.

Reason codes should be stable machine-readable values such as:

```text
ADEQUATE_SUPPORT
LOW_SUPPORT
EMPTY_SUPPORT
MISSING_MOTION
SPLIT_HALF_FAIL
IMPLAUSIBLE_STEP
EXCESSIVE_SPEED
UNKNOWN_ALIGNMENT
UNKNOWN_SIGN
QC_POLICY_NOT_VALIDATED
```

If several reasons apply, preserve all applicable flags or define and document
a deterministic reason-priority rule.

---

## 9. Optional split-half audit

Split-half estimation is an audit, not a default correction gate in the first
milestone.

The split must be deterministic and preserve temporal and depth coverage as far
as practical. Simple global even/odd indexing is acceptable only if coverage
checks demonstrate that it does not introduce a systematic imbalance.

Compare the two rigid traces using:

- correlation after applying the frozen reference convention;
- median absolute difference;
- P95 absolute difference;
- sign agreement during prespecified large excursions; and
- coverage/support differences between halves.

Quiet traces can make correlation unstable or meaningless. The report must not
interpret low correlation in a near-constant trace without also reporting its
dynamic range and absolute disagreement.

Split-half thresholds must be versioned. Before validation, split-half output is
diagnostic and does not authorize correction.

---

## 10. Production orchestration and failure behavior

Conceptually:

```python
accepted_recording_dir = materialize_rescue_recording(...)

motion_result = run_dredge_sidecar_safely(
    accepted_recording_dir,
    output_dir=motion_dir,
    config=motion_config,
)

# Identity is implemented by routing, not by a zero-motion operator.
recording_dir_for_sorting = accepted_recording_dir

sorting = run_kilosort4(
    recording_dir_for_sorting,
    sort_output_dir,
)
```

`run_dredge_sidecar_safely()` returns either a validated raw-estimate receipt or
a validated failure receipt. It must not convert estimator failure into zero
motion.

The production summary must clearly distinguish:

```text
ESTIMATE_COMPLETED
ESTIMATE_FAILED_IDENTITY_SORT_CONTINUED
ESTIMATE_DISABLED_IDENTITY_SORT_CONTINUED
```

An optional strict audit mode may stop on sidecar failure, but strict mode must
be explicit and must not be the default production sorting behavior.

---

## 11. Identity guarantee

When correction is unavailable or disabled, the motion operator is never
constructed or invoked.

Correct:

```python
recording_dir_for_sorting = accepted_recording_dir
```

Incorrect:

```python
recording_dir_for_sorting = materialize(
    ks4_motion_matrix(dshift=0) @ accepted_recording
)
```

The strongest identity assertions are architectural:

```python
assert recording_dir_for_sorting == accepted_recording_dir
assert sort_request.recording_request_digest == accepted_manifest.request_digest
assert sorter_params["do_correction"] is False
```

Sample-level equality checks may supplement these assertions, but an identity
copy is not needed in production.

---

## 12. Required outputs

Suggested layout:

```text
motion/
    request.json
    estimate_manifest.json
    dredge_rigid_estimate.npz
    estimator_input_receipt.json
    support_metrics.json
    motion_qc.json
    motion_summary.md
    figures/
        rigid_trace.png
        support_vs_time.png
        motion_speed.png
        peak_time_depth_support.png
```

On failure:

```text
motion/
    request.json
    estimation_failure.json
    motion_summary.md
```

Optional research audit:

```text
motion/audits/split_half/
    half_a_estimate.npz
    half_b_estimate.npz
    split_half_metrics.json
    split_half_comparison.png
```

The concise report should include:

```text
Motion estimator: DREDGE rigid
Estimator version: ...
Estimate completed: yes/no
QC status: VALID/PARTIALLY_VALID/INVALID/NOT_EVALUATED
Rigid range: ... um
P95 absolute displacement: ... um
P99 step: ... um
Low-support fraction: ...
Maximum speed: ... um/s
Split-half audit: not run / completed
Voltage motion correction applied: NO
KS4 internal motion correction: OFF
Sorter recording digest: ...
```

---

## 13. Configuration

Initial production configuration:

```yaml
motion:
  schema_version: dredge-sidecar-config-v1
  estimate: true
  estimator: dredge
  estimator_mode: rigid

  estimator_input:
    version: <frozen estimator-input version>
    detection: <explicit versioned settings>
    localization: <explicit versioned settings>
    dredge: <complete explicit effective settings>

  qc:
    policy_version: null
    thresholds_validated: false
    split_half: false

  nonrigid:
    enabled: false
    production_selectable: false

  voltage_correction:
    enabled: false
    policy_version: null
    policy_validated: false

  fallback: identity
```

Configuration validation must reject:

- a production estimator mode other than `rigid`;
- nonrigid voltage application;
- voltage correction without a nonempty, validated policy version;
- thresholds without a matching estimator and operator version;
- unknown time, sign, geometry, or reference conventions; and
- simultaneous external correction and KS4 native correction.

---

## 14. Testing requirements

### Configuration and safety tests

- Default configuration selects rigid estimation and no voltage correction.
- Production configuration cannot select a nonrigid field for voltage use.
- Enabling correction without a validated policy is rejected.
- Unknown estimator or schema versions are rejected.
- Effective configuration, not only overrides, participates in the digest.

### Cache and artifact tests

- A matching complete estimate can be reused.
- A parameter, geometry, frame-range, or source-digest change invalidates reuse.
- An incomplete cache is refused.
- A legacy cache without sufficient provenance is refused.
- Unsupported bins cannot be silently converted to zero displacement.
- Estimator failure creates a failure receipt rather than a motion artifact.

### Identity-path tests

- DREDGE disabled: KS4 receives the accepted recording directory.
- DREDGE succeeds: KS4 receives the same accepted recording directory.
- DREDGE fails: KS4 receives the same accepted recording directory.
- QC invalid or not evaluated: KS4 receives the same accepted recording
  directory.
- Saved KS4 settings confirm `do_correction=False` or its effective equivalent.
- The sort manifest references the accepted recording request digest.

### QC tests

- Epoch masks and reason codes have the same temporal shape as the rigid trace.
- Quiet traces do not produce misleading correlation-only conclusions.
- Missing and nonfinite bins are reported and never extrapolated.
- Split halves are deterministic and their coverage is reported.
- Unvalidated thresholds yield `NOT_EVALUATED`, not a guessed Boolean result.

### Regression test for the historical failure mode

Construct or load a nonrigid time-by-depth field and prove that no production
configuration or public production entry point can route it into voltage
application.

---

## 15. Acceptance criteria for the initial milestone

The first milestone is complete when:

1. A rigid DREDGE estimate or explicit failure receipt is produced for an
   accepted recording with complete provenance.
2. Cache reuse is tied to the full estimator request and source identity.
3. Epoch-resolved support and QC metrics are saved without inventing unvalidated
   thresholds.
4. A concise report and required figures are generated on successful estimates.
5. KS4 sorts the exact accepted recording regardless of sidecar outcome.
6. Saved KS4 settings prove native correction is disabled.
7. Production configuration cannot request nonrigid voltage application.
8. All safety, artifact, identity, and regression tests pass.

No corrected-versus-identity sorting experiment is part of this acceptance
decision.

---

## 16. Future selective rigid correction

This section records the intended extension point but does not authorize its
implementation.

Future work may introduce:

```python
@dataclass(frozen=True)
class SelectiveRigidCorrectionPolicy:
    version: str
    validated: bool
    estimator_name: str
    estimator_version: str
    operator_name: str
    operator_version: str
    reference_method: str
    d_on_um: float
    d_off_um: float
    min_support: float
    max_speed_um_s: float
    require_split_half: bool
    fallback: Literal["identity"] = "identity"
```

Activation requires all of the following:

1. A robust crossover across displacement sign, waveform generator, amplitude,
   residual, cosine, and separability criteria.
2. A validated `d_on_um > d_off_um` policy tied to the exact estimator,
   reference convention, geometry class, and operator version.
3. A switching audit covering slow ramps, rapid excursions, impulses, chatter,
   and support dropout.
4. Evidence that switching does not introduce waveform, amplitude, noise,
   detection-statistic, or near-coincidence artifacts.
5. A frozen real-data segment-panel comparison showing improvement in reviewed
   event recovery and unit-family continuity without worse contamination,
   refractory violations, or duplicate burden.
6. A separate accepted-recording schema and provenance chain for corrected
   voltage.

The future corrected recording must be materialized and validated as a distinct
artifact. It must define chunk-boundary behavior, quantization, probe-edge
behavior, unsupported-bin fallback, and the treatment of batches that straddle
identity/corrected transitions.

Until every condition passes:

```text
voltage correction enabled: false
correction state: identity
reason: CROSSOVER_NOT_VALIDATED
```

---

## 17. Advancement metrics for future comparisons

Future correction experiments must not use spike count or unit count as the
primary advancement criterion. Save and compare at least:

- reviewed-event recovery;
- sealed-event recovery where available;
- refractory violations;
- contamination;
- near-zero-lag cross-unit coincidences;
- unit presence and continuity;
- early/late waveform stability;
- unit-family reconciliation burden;
- detection-statistic behavior around transitions; and
- MUA/good-label distributions as secondary descriptive outputs.

The practical objective is a better sorting result, not maximal detection yield
or perfect physical reconstruction of tissue motion.

---

## 18. Final production policy

For the initial implementation, the policy is intentionally simple:

> Run a provenance-safe rigid DREDGE sidecar by default. Save motion and support
> evidence. Regardless of whether DREDGE succeeds, fails, or remains
> unqualified, pass the exact accepted recording to KS4 with all voltage motion
> correction disabled.

This makes severe-motion sessions diagnosable without reopening the historical
path from a nonrigid estimate to automatic voltage warping.
