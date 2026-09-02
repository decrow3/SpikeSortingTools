"""Amplitude-truncation fits used by production QC.

The fitting functions below were extracted verbatim from
``pipelineold/truncation.py`` at research-repository commit e71b144; only the
definitions reachable from the production entry points were carried over.

Two interpretation helpers were ADDED afterwards (2026-09-02) and have no
legacy counterpart: :func:`is_saturated` and
:func:`missing_pct_from_normalisation`. They do not change any fit or any
stored QC schema.

Estimator behaviour, characterised in ``testing/test_truncation_fitter.py``:

* Unbiased to well under a percentage point for true missing fractions from
  0.5% to about 40%. Every KS-good cohort measured so far sits at 0.6-3%, i.e.
  inside this range.
* **Hard-censored at 50%.** ``fit_truncated_sigmoid`` bounds ``x0`` below by
  ``x_min``, so ``truncated_sigmoid_missing_pct`` cannot exceed 50. A window
  reported at exactly 50.0 is a boundary-pinned fit, not a measurement; true
  70% missing is reported as 50.0. Filter these with :func:`is_saturated`
  before taking any median. In the stored Luke0804 imec0 analyses they are
  54.9% (rescue), 56.3% (legacy) and 16.4% (claim-mask) of all windows, though
  ~0% of windows belonging to KS-good units.
* Scale-invariant: missingness depends on ``k * (x0 - x_min)``, so differing
  amplitude scales between sorts do not by themselves bias the comparison.
* Eligibility is rate-dependent: a unit needs 1000 spikes within a continuous
  block (gaps > ``max_isi``) to produce any window at all.

``fit_truncated_sigmoid`` also swallows fit failures and falls back to
``[mean_amp, 1, 1]``, which typically reports near-0% missing. The fallback is
printed but not flagged in the returned arrays.
"""

import matplotlib.pyplot as plt
import numpy as np


def truncated_sigmoid(x, x0, k, A, x_min):
    ''' 
    A sigmoid which goes from 1-A to 1 with slope k and offset x0
    '''
    return (A / (1 + np.exp(-k * (x - x0))) - A + 1) * (x > x_min)


def fit_truncated_sigmoid(x, y, x_min = 8):
    from scipy.optimize import curve_fit

    f = lambda x, x0, k, A: truncated_sigmoid(x, x0, k, A, x_min)

    x0 = np.sum(x * y) / np.sum(y) # mean amplitude
    A0 = 1 # CDF goes from 0 to 1
    k0 = 1 # slope
    p0 = [x0, k0, A0]
    bounds = ([x_min, 0, 0], [np.inf, np.inf, np.inf])
    try:
        popt, _ = curve_fit(f, x, y, p0=p0, bounds=bounds)
    except Exception as e:
        print(f'Error fitting truncated sigmoid: {e}')
        popt = [x0, k0, A0]

    return popt


def untruncated_sigmoid(x, x0, k):
    return 1 / (1 + np.exp(-k * (x - x0)))


def truncated_sigmoid_missing_pct(popt, x_min=8):
    x0, k, A = popt
    return 100 * untruncated_sigmoid(x_min, x0, k)


def fit_amp_cdf(amps, x_min = None):
    amps = np.sort(amps)
    n = len(amps)
    p = np.arange(n) / n
    if x_min is None:
        x_min = np.min(amps)
    popt = fit_truncated_sigmoid(amps, p, x_min)
    missing_pct = truncated_sigmoid_missing_pct(popt, x_min)
    return popt, missing_pct


def construct_windows(ts, max_isi, spikes_per_window):
    n_spikes = len(ts)
    dts = np.diff(ts)
    blocks = np.stack([
                np.concatenate([[0], np.where(dts > max_isi)[0] + 1]),
                np.concatenate([np.where(dts > max_isi)[0], [n_spikes-1]])
            ], axis=1)
    n_windows = len(blocks)
    valid_blocks = []
    window_blocks = []
    window_block_times = []
    for iW in range(n_windows):
        i0, i1 = blocks[iW]
        n_samples = i1 - i0 + 1
        n_windows = n_samples // spikes_per_window
        n_window_samples = spikes_per_window * n_windows
        if n_windows == 0:
            continue
        # equally space windows in the block centered in the middle
        for iB in range(i0 + (n_samples - n_window_samples) // 2, 
                        i0 + (n_samples - n_window_samples) // 2 + n_window_samples-1,
                        spikes_per_window):
            window_blocks.append((iB, iB + spikes_per_window-1))
            window_block_times.append((ts[iB], ts[iB + spikes_per_window-1]))
        valid_blocks.append((i0 + n_samples // 2 - n_window_samples // 2, 
                             i0 + n_samples // 2 - n_window_samples // 2 + n_window_samples))
    window_blocks = np.array(window_blocks)
    valid_blocks = np.array(valid_blocks)

    return window_blocks, valid_blocks


def analyze_amplitude_truncation(spike_times, spike_amplitudes, max_isi = 10, spikes_per_window = 1000):
    window_blocks, valid_blocks = construct_windows(spike_times, max_isi, spikes_per_window)

    mpcts = np.zeros(len(window_blocks))
    popts = np.zeros((len(window_blocks), 3))
    for iB, (i0, i1) in enumerate(window_blocks):
        amps = spike_amplitudes[i0:i1]
        popts[iB], mpcts[iB] = fit_amp_cdf(amps)
    
    return window_blocks, valid_blocks, popts, mpcts


def plot_amplitude_truncation(spike_times, spike_amplitudes, window_blocks, valid_blocks, mpcts):
    window_block_times = np.array([[spike_times[i0], spike_times[i1]] for i0, i1 in window_blocks])
    if window_block_times.ndim == 1:
        window_block_times = window_block_times[np.newaxis, :]

    valid_mask = np.zeros(len(spike_times), dtype=bool)
    for i0, i1 in valid_blocks:
        valid_mask[i0:i1] = True

    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axs[0].hist2d(spike_times, spike_amplitudes, bins=(200, 50), cmap='Blues')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Amplitude (a.u.)')
    axs[0].set_title(f'Amplitude vs Time')
    if len(window_block_times) > 1:
        axs[1].bar(np.mean(window_block_times, axis=1), mpcts,
            width=np.diff(window_block_times, axis=1).squeeze(), 
            edgecolor='C0', color='C0', alpha=.7)
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Missing %')
    axs[1].set_title('Missing Percentage from Fit')
    axs[1].fill_between(spike_times, 0, 55, where=~valid_mask, color='r', alpha=.25)
    axs[1].set_ylim(0, 55)
    plt.tight_layout()
    return fig, axs


# ---------------------------------------------------------------------------
# Interpretation helpers (added 2026-09-02; no legacy counterpart)
# ---------------------------------------------------------------------------

SATURATION_PCT = 50.0


def is_saturated(mpcts, tol=1e-9):
    """Return a mask of windows whose fit was pinned at the 50% bound.

    These are censored, not measured: the optimiser drove ``x0`` onto its lower
    bound ``x_min``, which forces the reported statistic to exactly 50. Treat
    them as "at least 50% missing" and exclude them from means and medians
    rather than averaging them in as if they were estimates.
    """
    return np.isclose(np.asarray(mpcts, dtype=float), SATURATION_PCT, atol=tol)


def missing_pct_from_normalisation(popts):
    """Second, independent estimate of the same quantity, from ``A``.

    For a sigmoid CDF truncated at ``x_min`` the renormalisation factor is
    ``A = 1 / (1 - F(x_min))``, so the missing fraction is ``1 - 1/A``. The
    production statistic instead evaluates ``F(x_min)`` from ``(x0, k)`` and
    discards ``A`` entirely. Comparing the two is a free goodness-of-fit check:
    large disagreement means the truncated-sigmoid shape does not describe the
    window. On the stored Luke0804 analyses they agree to about 2 percentage
    points at the median, but disagree by more than 5 points in roughly a fifth
    of windows.
    """
    popts = np.atleast_2d(np.asarray(popts, dtype=float))
    A = np.clip(popts[:, 2], 1e-12, None)
    return 100.0 * (1.0 - 1.0 / A)
