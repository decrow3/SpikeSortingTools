"""Production quality-control report.

Extracted verbatim from ``pipelineold/qc.py`` at research-repository commit
e71b144. Only the definitions reachable from the production entry
points are carried over; the legacy module keeps the rest.

``load_qc`` and ``contamination_rate_from_rvl`` stay in the legacy
module; production does not call them.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from .refractory import compute_rvl_tensor, plot_rvl
from .truncation import analyze_amplitude_truncation, plot_amplitude_truncation
from pathlib import Path
from tqdm import tqdm
import os
import tempfile


def truncation_qc(spike_times, spike_clusters, spike_amplitudes, cache_dir, recalc=False):
    '''
    Run the truncation quality control pipeline on the given sorted data.

    Parameters
    ----------
    '''
    if isinstance(cache_dir, str):
        cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    truncation_path = cache_dir / 'truncation_qc.npz'
    present_path = cache_dir / 'present_qc.npz'
    pdf_path = cache_dir / 'truncation_qc.pdf'

    if truncation_path.exists() and present_path.exists() and pdf_path.exists() and not recalc:
        trunc_qc = np.load(truncation_path)
        pres_qc = np.load(present_path)
        return trunc_qc, pres_qc

    cids = np.unique(spike_clusters)

    pdf = PdfPages(pdf_path)

    trunc_qc = {
        'cid': [],
        'window_blocks': [],
        'popts': [],
        'mpcts': []
    }

    pres_qc = {
        'cid': [],
        'valid_blocks': []
    }

    for cid in tqdm(cids):
        cluster_spikes = spike_times[spike_clusters == cid]
        cluster_amps = spike_amplitudes[spike_clusters == cid]
        window_blocks, valid_blocks, popts, mpcts = analyze_amplitude_truncation(cluster_spikes, cluster_amps)

        if len(window_blocks) > 0:
            trunc_qc['cid'].append(np.ones(len(window_blocks)) * cid)

            if window_blocks.ndim == 1:
                window_blocks = window_blocks[np.newaxis, :]
            trunc_qc['window_blocks'].append(window_blocks)
            
            popts = np.array(popts)
            if popts.ndim == 1:
                popts = popts[np.newaxis, :]
            trunc_qc['popts'].append(popts)

            trunc_qc['mpcts'].append(mpcts)

            pres_qc['cid'].append(np.ones(len(valid_blocks)) * cid)
            pres_qc['valid_blocks'].append(valid_blocks)
            
        fig, axs = plot_amplitude_truncation(cluster_spikes, cluster_amps, window_blocks, valid_blocks, mpcts)
        axs[0].set_title(f'Cluster {cid}\nAmplitudes vs Time')
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    pdf.close()

    def _concat_or_empty(chunks, empty_shape, dtype=float):
        if len(chunks) == 0:
            return np.empty(empty_shape, dtype=dtype)
        return np.concatenate(chunks, axis=0)

    trunc_qc = {
        'cid': _concat_or_empty(trunc_qc['cid'], (0,), dtype=float),
        'window_blocks': _concat_or_empty(trunc_qc['window_blocks'], (0, 2), dtype=int),
        'popts': _concat_or_empty(trunc_qc['popts'], (0, 3), dtype=float),
        'mpcts': _concat_or_empty(trunc_qc['mpcts'], (0,), dtype=float),
    }
    pres_qc = {
        'cid': _concat_or_empty(pres_qc['cid'], (0,), dtype=float),
        'valid_blocks': _concat_or_empty(pres_qc['valid_blocks'], (0, 2), dtype=int),
    }

    np.savez(truncation_path, **trunc_qc)
    np.savez(present_path, **pres_qc)

    return trunc_qc, pres_qc


def refractory_qc(spike_times, spike_clusters, cache_dir, recalc=False):
    '''
    Run the refractory period quality control pipeline on the given sorted data.
    
    Parameters
    ----------
    spike_times: array-like (n_spikes,)
        Spike times in seconds.
    spike_clusters: array-like (n_spikes,)
        The cluster assignments of each spike.
    
    Returns
    -------
    qc_results: dict
        The results of the quality control pipeline
    '''
    if isinstance(cache_dir, str):
        cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    npz_path = cache_dir / 'refractory_qc.npz'
    pdf_path = cache_dir / 'refractory_qc.pdf'
    if npz_path.exists() and pdf_path.exists() and not recalc:
        qc_results = np.load(npz_path)
        return qc_results

    qc_results = {}

    min_refrac, max_refrac = 1e-3, 10e-3
    n_refrac = 100
    refractory_periods = np.exp(np.linspace(np.log(min_refrac), np.log(max_refrac), n_refrac))

    min_contam, max_contam = 5e-3, .35
    n_contam = 50
    contamination_test_proportions = np.exp(np.linspace(np.log(min_contam), np.log(max_contam), n_contam))

    cids = np.unique(spike_clusters)

    rvl = compute_rvl_tensor(spike_times, spike_clusters, cids, refractory_periods, contamination_test_proportions, progress=True)

    pdf = PdfPages(cache_dir / 'refractory_qc.pdf')
    for iU in tqdm(range(len(cids)) , desc='Plotting refractory QC'):
        cid = cids[iU]
        likelihoods = rvl[iU].squeeze()
        cluster_spikes = spike_times[spike_clusters == cid]
        fig, axs = plot_rvl(cluster_spikes, likelihoods, refractory_periods, contamination_test_proportions)
        axs[0].set_title(f'Cluster {cid}\nISI Distribution')
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    pdf.close()

    qc_results = {'rvl_tensor': rvl, 'refractory_periods': refractory_periods, 'contamination_test_proportions': contamination_test_proportions}
    np.savez(cache_dir / 'refractory_qc.npz', **qc_results)
    return qc_results


def waveform_qc(seg, spike_samples, spike_clusters, cache_dir, n_waves=512,
                n_samples=82, uV_per_bit=0.195, recalc=False,
                read_chunk_duration_s=1.0):
    """Extract legacy waveform medians using ordered, chunked recording reads.

    The historical implementation issued one random-access ``get_traces`` call
    per sampled spike.  For a full probe that means hundreds of thousands of
    tiny reads from the network recording.  This implementation preserves the
    random sample selection, boundary padding, float64 median calculation, and
    NPZ schema, but services all requests in temporal order with approximately
    one recording read per second.  Sampled waveforms live in an ephemeral
    local memmap so the optimization does not add a large server-side artifact.
    """
    if isinstance(cache_dir, str):
        cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    npz_path = cache_dir / 'waveforms.npz'
    
    if npz_path.exists() and not recalc:
        waveforms = np.load(npz_path)
        return waveforms


    spike_samples = np.asarray(spike_samples).reshape(-1)
    spike_clusters = np.asarray(spike_clusters).reshape(-1)
    if spike_samples.size != spike_clusters.size:
        raise ValueError("spike_samples and spike_clusters must have equal length")

    cids = np.unique(spike_clusters)
    n_clusters = len(cids)
    n_channels = seg.get_num_channels()
    waveforms = np.zeros((n_clusters, n_samples, n_channels), np.float32)
    samples = np.zeros((n_clusters, n_waves),np.int64) - 1
    times = (np.arange(n_samples) - n_samples//2) / seg.get_sampling_frequency()

    # Select in the same unit order and with the same RNG calls as the legacy
    # loop.  Requests for each unit remain contiguous in the temporary store.
    selected = []
    row_starts = np.zeros(n_clusters + 1, dtype=np.int64)
    for iC, cid in enumerate(cids):
        cluster_samples = spike_samples[spike_clusters == cid]
        n_waves_clust = min(n_waves, len(cluster_samples))
        sub_inds = np.random.choice(len(cluster_samples), n_waves_clust, replace=False)
        cluster_samples_sub = cluster_samples[sub_inds].astype(np.int64, copy=False)
        samples[iC, :n_waves_clust] = cluster_samples_sub
        selected.append(cluster_samples_sub)
        row_starts[iC + 1] = row_starts[iC] + n_waves_clust

    centers = np.concatenate(selected) if selected else np.empty(0, dtype=np.int64)
    n_frames = int(seg.get_num_frames())
    if np.any(centers < 0) or np.any(centers >= n_frames):
        raise ValueError("sampled spike index falls outside the recording")
    chunk_frames = max(1, int(round(
        float(read_chunk_duration_s) * seg.get_sampling_frequency()
    )))

    fd, scratch_name = tempfile.mkstemp(prefix="waveform-qc-", suffix=".mmap")
    os.close(fd)
    scratch_path = Path(scratch_name)
    traces = None
    try:
        traces = np.memmap(
            scratch_path, mode="w+", dtype=np.float64,
            shape=(len(centers), n_samples, n_channels),
        )
        traces[:] = 0.0
        if len(centers):
            chunk_ids = centers // chunk_frames
            order = np.argsort(chunk_ids, kind="stable")
            sorted_chunks = chunk_ids[order]
            edges = np.r_[0, np.flatnonzero(np.diff(sorted_chunks)) + 1, len(order)]
            with tqdm(total=len(edges) - 1, desc="Reading waveform chunks") as pbar:
                for left, right in zip(edges[:-1], edges[1:]):
                    request_rows = order[left:right]
                    request_centers = centers[request_rows]
                    read_start = max(0, int(request_centers.min()) - n_samples // 2)
                    read_end = min(
                        n_frames - 1,
                        int(request_centers.max()) + (n_samples - n_samples // 2),
                    )
                    block = seg.get_traces(start_frame=read_start, end_frame=read_end)
                    for row, center in zip(request_rows, request_centers):
                        i0 = max(0, int(center) - n_samples // 2)
                        i1 = min(
                            n_frames - 1,
                            int(center) + (n_samples - n_samples // 2),
                        )
                        o0 = i0 - (int(center) - n_samples // 2)
                        o1 = o0 + i1 - i0
                        b0 = i0 - read_start
                        b1 = b0 + i1 - i0
                        traces[row, o0:o1, :] = block[b0:b1, :] * uV_per_bit
                    pbar.update(1)
        traces.flush()

        with tqdm(total=n_clusters, desc="Computing waveform medians") as pbar:
            for iC in range(n_clusters):
                first, last = row_starts[iC:iC + 2]
                waveforms[iC, ...] = np.median(traces[first:last], axis=0)
                pbar.update(1)
    finally:
        if traces is not None:
            del traces
        scratch_path.unlink(missing_ok=True)

    out = {'waveforms': waveforms, 'samples': samples, 'times': times, 'cids': cids}
    np.savez(npz_path, **out)
    return out


def run_qc(seg, results, cache_dir, recalc=False):
    '''
    Run the quality control pipeline on the given sorted data.
    
    Parameters
    ----------
    seg: spikeinterface recording segment
        The recording segment which was sorted. Used to extract waveforms and other data.
    results: KilosortResults
        The results of the kilosort4 sorting.
    
    Returns
    -------
    qc_results: dict
        The results of the quality control pipeline
    '''

    qc_results = {}

    spike_samples = results.spike_times

    spike_times = results.spike_times / seg.get_sampling_frequency()
    spike_clusters = results.spike_clusters
    spike_amplitudes = results.st[:, 2]

    wave_dir = cache_dir / 'waveforms'
    waveforms = waveform_qc(seg, spike_samples, spike_clusters, wave_dir, recalc=recalc)
    qc_results['waveforms'] = waveforms

    truncation_dir = cache_dir / 'amp_truncation'
    truncation, present = truncation_qc(spike_times, spike_clusters, spike_amplitudes, truncation_dir, recalc=recalc)
    qc_results['truncation'] = truncation
    qc_results['present'] = present

    refractory_dir = cache_dir / 'refractory'
    refractory = refractory_qc(spike_times, spike_clusters, refractory_dir, recalc=recalc)
    qc_results['refractory'] = refractory    

    return qc_results
