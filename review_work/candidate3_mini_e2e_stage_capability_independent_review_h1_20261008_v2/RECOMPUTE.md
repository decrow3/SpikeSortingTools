# Independent recomputation method

Inputs were read directly from retained synthetic scratch and the sealed packet; no sorter was launched.

1. Open `threshold.h5` and `matching1.h5` with h5py and independently read times, seconds, channels, labels, templates, chunk schedules, markers, and sampling frequency.
2. Verify each packet `ACCOUNTING.npz` array byte-for-byte at the NumPy value level against the corresponding H5/NPY/NPZ scratch array.
3. Load every `*labels.npy`; independently count rows, negative labels, and unique nonnegative labels.
4. Load `dartsort_sorting.npz`; independently count final rows/noise/units and check row order and matching-to-final time/channel changes.
5. Construct `collections.Counter(zip(times_samples, channels))` for detection and matching. Compute `sum((detection & matching).values())`, `sum((matching - detection).values())`, and `sum((detection - matching).values())` without using the producer's deque mapper.
6. Re-run only the current `summarize` subcommand against retained scratch. The regenerated JSON and compressed NPZ hashes exactly equal the sealed packet hashes `82a66a28...` and `18264b0c...`.
7. Verify all v1/v2/v3 manifest members and COMPLETE bindings, then compare filesystem birth times to confirm COMPLETE was created after payload and MANIFEST.

The exact executed pre-run wrapper was reconstructed from the retained root-session Add File plus its only two pre-launch patches. Its hash equals the pre-launch receipt's `5d55...` binding. Diff against current `e627...` shows identical `prepare` and `run_arm`; only the post-run `summarize` reporting/accounting block differs. The current source rerun validates the sealed compact accounting.
