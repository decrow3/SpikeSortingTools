# DREDGE on screened waveform inputs

The aggressive waveform mask did not improve the bounded DREDGE comparison overall. Relative to shared-response compensation alone, it weakens the central movement estimate and makes the example trajectory more irregular. This is evidence against adopting this particular mask, not against motion correction or artifact cleanup.

## Controlled comparison

Interval4180–4200s; four inputs: original60719 peaks, compensated55912, amplitude-only24118, full waveform mask13424. Each arm uses the same DREDGE configuration: nonrigid Gaussian windows,200µm step/300µm scale,1µm/1s bins,1µm/1s histogram smoothing,60s time horizon,mincorr0.1,CPU. The existing bounded helper explicitly enforces pairwise search within±80µm in every arm, including clipped probe-edge windows. No settings were tuned to these outputs. Cached peak locations are subset exactly; no relocalization is required when only selecting events from the same voltage.

## Observations

Half-to-half displacements at unit445/2260µm: compensated−4.64µm, amplitude-only−1.95µm, waveform screen−1.25µm; provisional lighthouse centroid−7.24µm.

At unit463/2380µm: compensated−3.99µm, amplitude-only−2.09µm, waveform screen−1.32µm; provisional lighthouse−5.55µm. Screened trajectory has larger reversals and endpoint excursions.

At unit317/1740µm: compensation−6.62µm and waveform screen−6.54µm, both below lighthouse centroid magnitude−9.60µm.

At2740–3100µm, waveform screening gives approximately−5.6 to−6.2µm versus compensated−1.8 to−2.0µm. Lighthouse centroids there give−0.3 to−1.1µm. These deeper lighthouse values remain particularly vulnerable to fixed-template selection and should not be treated as calibrated ground truth.

Thus, the amplitude cut already removes useful central registration information; additional morphology screening further changes the depth distribution and temporal stability. The mechanism of the additional deep excursion is unresolved. A smaller or visually cleaner population is not necessarily a better motion-estimation input.

## Scope and next implication

Keep common-response compensation as the stronger candidate from this specific comparison. Do not promote the blanket amplitude/morphology mask to production. Further input cleanup should target demonstrated artifact waveform families and local channel locking, while checking loss of registration-supporting populations. This20s comparison is not full-session validation or an amplitude-completeness test. No motion was applied to source voltage and no sort was launched.

## Provenance

Script `testing/luke_screened_dredge_trial.py`; outputs `testing/outputs/luke_screened_dredge_trial_v1/`: settings, screen-mask hash, all four fields and pairwise constraints, lighthouse_comparison.csv, summary.json, and01_motion_comparison.png/.pdf. Service `luke-screened-dredge-v1` completed with exit0; receipt/logs/exact launch under `testing/outputs/luke_screened_dredge_job_v1/`. Actual final systemd state checked inactive/dead with ExecMainStatus0. Bounded helper asserts finite fields and pairwise search bounds.
