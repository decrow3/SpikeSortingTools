# Luke0804 imec1 DREDGE-LFP motion handoff

This package contains the four existing full-crop rigid LFP motion arrays, their original specification and completion receipt, and the 930–1030 s overlap audit. It does not rerun DREDGE and does not include the large LF context arrays. Source artifacts are copied byte-for-byte.

## Arrays and masks

Each `*_motion.npz` contains a shared 291,191-sample native-rate axis (`time_s`, approximately 249.998 Hz), `displacement_um`, boolean `supported`, one rigid reference depth (`depth_um = 1910 µm`), and 50×50 sampled pairwise audit matrices (`audit_D`, `audit_C`, `audit_S`). The LFP conditioning used all 192 probe contacts (0–3820 µm); the one stored depth is the reference coordinate of the rigid field, not an input-channel crop.

No separate `invalid` array is stored. Construct it for adapter use as `invalid = (~supported) | (~isfinite(displacement_um))`. The displacement arrays are finite even where unsupported; never treat those values as valid or bridge gaps. Preserve and inspect the original `supported` bit exactly. Both second-derivative arms have 24 supported samples across the whole saved array, all outside the DARTsort crop; there are zero supported samples within the crop. Do not interpolate or extrapolate these arms.

For event sampling, require finite displacement and valid support at the selected native sample. The historical nearest-sample rule used a 10 ms maximum distance. If linear interpolation is used instead, require both bracketing samples to be supported and finite; reject events outside the axis or across any unsupported sample.

## Sign and coordinates

`displacement_um` is the unmodified `Motion.displacement` returned by SpikeInterface 0.104.7 `estimate_motion(method="dredge_lfp", rigid=True)`. SpikeInterface's `correct_motion_on_peaks` applies inverse motion as `corrected_depth = observed_depth - displacement`. Thus the correction operator subtracts this field; do not negate the archived values when reproducing that correction. This package is a diagnostic export, not a qualified AP correction field.

Clock fields in `spec.json` distinguish three axes:

- Raw AP stream time: `t_ap = raw_ap_frame / 29999.759166666667` seconds, with stream origin at frame 0.
- DARTsort crop time: `t_crop = (raw_ap_frame - 193737) / 29999.759166666667`; crop interval is `[193737, 34942956)` and is 1158.316598708 s.
- `time_s` in the motion NPZs is absolute acquisition time on the LF clock. Map an AP-frame event to it with `t_field = ap_origin_s + raw_ap_frame / fs`, where `ap_origin_s = 3057.6775463558583` and `fs` is the AP rate above. Equivalently, convert a field sample to AP stream time with `time_s - ap_origin_s`. The LF origin is `3057.6773463542527`, about 0.2 ms before the AP origin.

The saved field begins about 0.2 ms before AP stream frame 0 and ends about 5.4 ms before the crop's exclusive end. It supplies approximately 6.458 s of pre-crop context, less than the customary 10 s warm-up; see `spec.json`. The overlap audit compares identical AP-stream times `[930, 1030)` and must not be spliced into the full-crop field.

## Verification

`handoff_manifest.json` records source identities, producer hashes, axis checks, and support counts inside and outside the sorter crop. `SHA256SUMS` covers every package file except itself. `complete.json` contains the original output hashes and receipts; the large LF contexts referenced there were intentionally not copied.
