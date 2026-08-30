#%%
import numpy as np
import matplotlib.pyplot as plt
import json
import warnings
from pathlib import Path
from .preprocess import get_default_job_kwargs
import medicine
from medicine.plotting import _correct_motion_on_peaks, plot_motion_correction
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_localization import localize_peaks
from spikeinterface.sortingcomponents.peak_selection import select_peaks
from spikeinterface.sortingcomponents.motion import estimate_motion, motion_utils, interpolate_motion
from spikeinterface.preprocessing import astype
from scipy.signal import medfilt
from scipy.signal import welch


from spikeinterface.core.motion import Motion


def _json_safe_config(value):
    if isinstance(value, dict):
        return {str(key): _json_safe_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_config(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return repr(value)


def _recording_time_bounds(recording):
    """Return inclusive start and exclusive end times for a one-segment recording."""
    if recording.get_num_segments() != 1:
        raise ValueError('Motion correction currently requires one recording segment')
    n = recording.get_num_samples(segment_index=0)
    return tuple(recording.sample_index_to_time(np.array([0, n]), segment_index=0))


def _validate_cross_band_recordings(lfp_recording, ap_recording):
    lfp_start, lfp_end = _recording_time_bounds(lfp_recording)
    ap_start, ap_end = _recording_time_bounds(ap_recording)
    tolerance_s = 2.0 / min(
        float(lfp_recording.get_sampling_frequency()),
        float(ap_recording.get_sampling_frequency()),
    )
    if abs(lfp_start - ap_start) > tolerance_s or abs(lfp_end - ap_end) > tolerance_s:
        raise ValueError(
            'The LF and AP recordings are not time-aligned: '
            f'LF=({lfp_start:.6f}, {lfp_end:.6f}) s, '
            f'AP=({ap_start:.6f}, {ap_end:.6f}) s, '
            f'tolerance={tolerance_s:.6f} s'
        )

    lfp_depths = np.asarray(lfp_recording.get_channel_locations())[:, 1]
    if lfp_depths.size != np.unique(lfp_depths).size or np.any(np.diff(lfp_depths) <= 0):
        raise ValueError(
            'DREDGE-LFP requires unique, increasing y positions; '
            'run prepare_lfp_for_motion() first'
        )


def correct_motion_lfp(
    lfp_recording,
    rec_for_sorting,
    cache_dir,
    dredge_lfp_args=None,
    recalc=False,
    median_filter_size=1,
    interpolation_time_bin_size_s=None,
    max_step_um=40.0,
):
    """Estimate high-rate motion from LFP and apply it to an AP recording."""
    print('Starting cross-band DREDGE-LFP motion correction...')
    cache_dir = Path(cache_dir)
    motion_dir = cache_dir / 'dredge-lfp-motion'
    motion_dir.mkdir(parents=True, exist_ok=True)
    _validate_cross_band_recordings(lfp_recording, rec_for_sorting)

    default_args = dict(
        method='dredge_lfp',
        direction='y',
        # Conservative defaults: the example-session validation found that
        # unconstrained nonrigid LFP registration can produce large false jumps.
        rigid=True,
        win_shape='gaussian',
        win_step_um=800.0,
        win_scale_um=850.0,
        win_margin_um=None,
        chunk_len_s=10.0,
        max_disp_um=100.0,
        progress_bar=True,
        verbose=True,
        # Keeping D/C matrices for every 250 Hz chunk is prohibitively large.
        extra_outputs=False,
    )
    dredge_lfp_args = dict(default_args, **(dredge_lfp_args or {}))
    dredge_lfp_args['method'] = 'dredge_lfp'

    motion_file = motion_dir / 'motion.npy'
    time_file = motion_dir / 'time_bins.npy'
    depth_file = motion_dir / 'depth_bins.npy'
    config_file = motion_dir / 'config.json'
    cache_complete = all(path.exists() for path in (motion_file, time_file, depth_file))

    expected_config = {
        'sampling_frequency_hz': float(lfp_recording.get_sampling_frequency()),
        'method_args': _json_safe_config(dredge_lfp_args),
        'max_step_um': None if max_step_um is None else float(max_step_um),
    }
    if cache_complete and not recalc and config_file.exists():
        cached_config = json.loads(config_file.read_text())
        if cached_config != expected_config:
            raise ValueError(
                'Cached DREDGE-LFP parameters do not match this run; rerun with recalc=True'
            )

    if not cache_complete or recalc:
        result = estimate_motion(recording=lfp_recording, **dredge_lfp_args)
        lfp_motion, _extra = result if dredge_lfp_args.get('extra_outputs') else (result, None)
        displacement = lfp_motion.displacement[0]
        if median_filter_size > 1:
            if median_filter_size % 2 == 0:
                raise ValueError('median_filter_size must be odd')
            displacement = medfilt(displacement, kernel_size=(median_filter_size, 1))
        _validate_lfp_displacement(displacement, max_step_um=max_step_um)
        np.save(motion_file, displacement)
        np.save(time_file, lfp_motion.temporal_bins_s[0])
        np.save(depth_file, lfp_motion.spatial_bins_um)
        config_file.write_text(json.dumps(expected_config, indent=2))
    elif config_file.exists():
        cached = json.loads(config_file.read_text())
        cached_fs = float(cached.get('sampling_frequency_hz', np.nan))
        current_fs = float(lfp_recording.get_sampling_frequency())
        if not np.isclose(cached_fs, current_fs):
            raise ValueError(
                f'Cached LFP motion uses {cached_fs:g} Hz but input is {current_fs:g} Hz; '
                'rerun with recalc=True'
            )
    else:
        warnings.warn(
            'Reusing a DREDGE-LFP cache without config metadata; use recalc=True '
            'to regenerate a fully validated cache',
            RuntimeWarning,
        )

    displacement = np.load(motion_file)
    _validate_lfp_displacement(displacement, max_step_um=max_step_um)
    motion = Motion(
        displacement=displacement,
        temporal_bins_s=np.load(time_file),
        spatial_bins_um=np.load(depth_file),
    )
    interpolation_kwargs = {}
    if interpolation_time_bin_size_s is not None:
        interpolation_kwargs['interpolation_time_bin_size_s'] = float(interpolation_time_bin_size_s)
    corrected = interpolate_motion(
        astype(rec_for_sorting, 'float'),
        motion,
        border_mode='force_zeros',
        **interpolation_kwargs,
    )
    print(
        'Finished DREDGE-LFP correction at '
        f'{lfp_recording.get_sampling_frequency():g} motion updates/s'
    )
    return astype(corrected, 'int16')


def _validate_lfp_displacement(displacement, max_step_um=40.0):
    """Reject discontinuous LFP estimates before they are applied to AP data."""
    displacement = np.asarray(displacement)
    if not np.all(np.isfinite(displacement)):
        raise ValueError('DREDGE-LFP produced non-finite displacement values')
    if max_step_um is None or displacement.shape[0] < 2:
        return
    largest_step = float(np.max(np.abs(np.diff(displacement, axis=0))))
    if largest_step > float(max_step_um):
        raise ValueError(
            'DREDGE-LFP estimate failed the discontinuity check: '
            f'largest one-sample step is {largest_step:.1f} um '
            f'(limit {float(max_step_um):.1f} um). This usually indicates an '
            'underconstrained LFP registration; motion was not applied. Set '
            'max_step_um=None only after independent validation.'
        )


def plot_lfp_motion_output(cache_dir, save_dir=None, heartbeat_band_hz=(3.8, 5.8)):
    """Plot DREDGE-LFP displacement traces and their temporal power spectra."""
    cache_dir = Path(cache_dir) / 'dredge-lfp-motion'
    save_dir = Path(save_dir) if save_dir is not None else cache_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    displacement = np.load(cache_dir / 'motion.npy')
    times = np.load(cache_dir / 'time_bins.npy')
    depths = np.load(cache_dir / 'depth_bins.npy')

    fs = 1.0 / np.median(np.diff(times))
    trace_indices = np.unique(np.linspace(0, depths.size - 1, min(5, depths.size)).astype(int))
    fig, axes = plt.subplots(2, 1, figsize=(11, 7))
    for index in trace_indices:
        axes[0].plot(times, displacement[:, index], label=f'{depths[index]:.0f} um')
    axes[0].set(xlabel='Time (s)', ylabel='Displacement (um)', title='DREDGE-LFP motion')
    axes[0].legend(loc='upper right')

    nperseg = min(displacement.shape[0], max(8, int(round(fs * 30))))
    frequencies, psd = welch(displacement, fs=fs, nperseg=nperseg, axis=0)
    axes[1].semilogy(frequencies, np.nanmedian(psd, axis=1))
    axes[1].axvspan(*heartbeat_band_hz, color='tab:red', alpha=0.15, label='Marmoset heartbeat')
    axes[1].set_xlim(0, min(20, fs / 2))
    axes[1].set(xlabel='Frequency (Hz)', ylabel='Median PSD', title='Motion spectrum')
    axes[1].legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(save_dir / 'dredge_lfp_motion.png', dpi=150)
    plt.close(fig)

def correct_motion(seg, cache_dir, detect_peak_args={}, localize_peak_args={}, ks_motion_args={}, dredge_motion_args={},  dc_motion_args={}, med_motion_args={}, job_kwargs={}, recalc=False, method='all', median_filter_size=1, rec_for_sorting=None):

    print('Starting motion correction...')


    if isinstance(cache_dir, str):
        cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    job_kwargs = dict(get_default_job_kwargs(), **job_kwargs)

    ###
    # Detect peaks
    ###

    default_detect_peak_args = dict(
        method = 'locally_exclusive',  #'locally_exclusive', # replace with locally_exclusive_torch to use DetectPeakLocallyExclusiveTorch ???
        radius_um = 50, #was 100, possibly for the nhp probes, default is 50. Larger values make it take a lot longer..
        detect_threshold=5 #default is 5, Ryan had 7. in median average deviations
    )
    detect_peak_args = dict(default_detect_peak_args, **detect_peak_args)

    f_peaks = cache_dir / 'peaks.npy'
    if not f_peaks.exists() or recalc:
        peaks = detect_peaks(seg, **detect_peak_args, **job_kwargs)
        np.save(cache_dir / 'peaks.npy', peaks)
    else:
        peaks = np.load(f_peaks)

    ###
    # Localize peaks
    ###

    default_localize_peak_args = dict(method = 'monopolar_triangulation')
    localize_peak_args = dict(default_localize_peak_args, **localize_peak_args)

    f_peak_locations = cache_dir / 'peak_locations.npy'
    if not f_peak_locations.exists() or recalc:
        peak_locations = localize_peaks(seg, peaks, **localize_peak_args, **job_kwargs)
        np.save(cache_dir / 'peak_locations.npy', peak_locations)
    else:
        peak_locations = np.load(f_peak_locations)

    # somepeaks,some_peak_indices = select_peaks(peaks=peaks, method='smart_sampling_locations_and_time', return_indices=True, peaks_locations=peak_locations, n_peaks=10000, random_state=0)
    # some_peak_locations = peak_locations[some_peak_indices]

    ###
    # Kilosort motion
    ###

    if method == 'ks' or method == 'all':
        print('Estimating Kilosort-like motion...')

        default_ks_motion_args = dict(method = 'iterative_template', direction = 'y', bin_s = 2.0, num_shifts_block = 5)
        ks_motion_args = dict(default_ks_motion_args, **ks_motion_args)
        ks_motion_args['method'] = 'iterative_template'

        ks_motion_dir = cache_dir / 'ks-motion'
        ks_motion_dir.mkdir(parents=True, exist_ok=True)
        if not (ks_motion_dir / "motion.npy").exists() or recalc:
            ks_motion = estimate_motion(
                recording = seg, 
                peaks = peaks,
                peak_locations = peak_locations,
                **ks_motion_args 
            )
            ks_displacement = ks_motion.displacement[0]
            if median_filter_size > 1:
                ks_displacement = medfilt(ks_displacement, kernel_size=(median_filter_size, 1))

            np.save(ks_motion_dir / "motion.npy", ks_displacement)
            np.save(ks_motion_dir / "time_bins.npy", ks_motion.temporal_bins_s[0])
            np.save(ks_motion_dir / "depth_bins.npy", ks_motion.spatial_bins_um)

        # load kilosort motion
        ks_motion = Motion(
            displacement=np.load(ks_motion_dir / "motion.npy"),
            temporal_bins_s=np.load(ks_motion_dir / "time_bins.npy"),
            spatial_bins_um=np.load(ks_motion_dir / "depth_bins.npy"),
        )
        if method != 'all':
            motion = ks_motion
        
        
    ###
    # Varol2021 decentralized motion
    ###
    if method == 'dc' or method == 'all':
        print('Estimating decentralized motion...')

        default_dc_motion_args = dict(method = 'decentralized', direction = 'y', bin_s = 2.0)
        dc_motion_args = dict(default_dc_motion_args, **dc_motion_args)

        decentralized_motion_dir = cache_dir / 'decentralized-motion'
        decentralized_motion_dir.mkdir(parents=True, exist_ok=True)
        if not (decentralized_motion_dir / "motion.npy").exists() or recalc:
            dc_motion = estimate_motion(
                recording = seg, 
                peaks = peaks,
                peak_locations = peak_locations,
                **dc_motion_args
            )
            dc_displacement = dc_motion.displacement[0]
            if median_filter_size > 1:
                dc_displacement = medfilt(dc_displacement, kernel_size=(median_filter_size, 1))
            np.save(decentralized_motion_dir / "motion.npy", dc_displacement)
            np.save(decentralized_motion_dir / "time_bins.npy", dc_motion.temporal_bins_s[0])
            np.save(decentralized_motion_dir / "depth_bins.npy", dc_motion.spatial_bins_um)

        # load decentralized motion
        dc_motion = Motion(
            displacement=np.load(decentralized_motion_dir / "motion.npy"),
            temporal_bins_s=np.load(decentralized_motion_dir / "time_bins.npy"),
            spatial_bins_um=np.load(decentralized_motion_dir / "depth_bins.npy"),
        )
        if method != 'all':
            motion = dc_motion

    ### DREDGE method
    if method == 'dredge'or method == 'all':
        print('Estimating DREDGE motion...')

        default_dredge_motion_args = dict(method = 'dredge', direction = 'y', rigid = False, win_shape = 'gaussian', win_step_um = 100.0, win_scale_um = 150.0, win_margin_um = 50.0, extra_outputs = True, progress_bar = True, verbose = True)
        dredge_motion_args = dict(default_dredge_motion_args, **dredge_motion_args)
        dredge_motion_args['method'] = 'dredge_ap'

        dredge_motion_dir = cache_dir / 'dredge-motion'
        dredge_motion_dir.mkdir(parents=True, exist_ok=True)
        if not (dredge_motion_dir / "motion.npy").exists() or recalc:
                # With extra_outputs=True, estimate_motion returns (motion, extra), where extra is a dict containing intermediate variables that can be useful for plotting and debugging
                dredge_motion, extra = estimate_motion(
                    recording = seg, 
                    peaks = peaks,
                    peak_locations = peak_locations,
                    **dredge_motion_args
                )
                dredge_displacement = dredge_motion.displacement[0]
                if median_filter_size > 1:
                    dredge_displacement = medfilt(dredge_displacement, kernel_size=(median_filter_size, 1))
                np.save(dredge_motion_dir / "motion.npy", dredge_displacement)
                np.save(dredge_motion_dir / "time_bins.npy", dredge_motion.temporal_bins_s[0])
                np.save(dredge_motion_dir / "depth_bins.npy", dredge_motion.spatial_bins_um)

        
        # load dredge motion
        dredge_motion = Motion(
            displacement=np.load(dredge_motion_dir / "motion.npy"),
            temporal_bins_s=np.load(dredge_motion_dir / "time_bins.npy"),
            spatial_bins_um=np.load(dredge_motion_dir / "depth_bins.npy"),
        )
        if method != 'all':
            motion = dredge_motion
        
    
    ###
    # MEDiCINe motion
    ###
    # Getting a little bit of overfitting, might want to increase time width of the bins
    # time_kernel_width = 50 #default is 30
    # num_depth_bins = 2 #default is 2
    # amplitude_threshold_quantile = 0 #default is 0, but may want to threshold higher [-1,1]

    if method == 'med' or method == 'all':
        print('Estimating MEDiCINe motion...')

        default_med_motion_args = dict(time_bin_size = 1.0, num_depth_bins = 2, time_kernel_width = 50, amplitude_threshold_quantile = 0.2)
        med_motion_args = dict(default_med_motion_args, **med_motion_args)

        # Create directory to store MEDiCINe outputs for this recording
        medicine_output_dir = cache_dir / 'medicine'
        medicine_output_dir.mkdir(parents=True, exist_ok=True)

        if seg.get_time_info()['t_start'] is not None:
            peak_times_ = peaks['sample_index'] / seg.get_sampling_frequency() + seg.get_time_info()['t_start']
        else:
            peak_times_ = peaks['sample_index'] / seg.get_sampling_frequency()
        # Run MEDiCINe
        if not (medicine_output_dir / "motion.npy").exists() or recalc:
            medicine.run_medicine(
                peak_amplitudes=peaks['amplitude'],
                peak_depths=peak_locations['y'],
                peak_times=peak_times_,#peaks['sample_index'] / seg.get_sampling_frequency() + seg.get_time_info()['t_start'],
                output_dir=medicine_output_dir,
                **med_motion_args
            )

            # Load MEDiCINe outputs
            med_motion = np.load(medicine_output_dir / "motion.npy")
            med_time_bins = np.load(medicine_output_dir / "time_bins.npy")
            med_depth_bins = np.load(medicine_output_dir / "depth_bins.npy")
            n_append = 5
            dt = med_time_bins[1] - med_time_bins[0]
            med_time_bins = np.concatenate(
                    [med_time_bins, med_time_bins[-1] + np.arange(1, n_append + 1) * dt]
            )
            med_motion = np.concatenate(
                [med_motion, np.ones((n_append, med_motion.shape[1])) * med_motion[-1]]
            )
            if median_filter_size > 1:
                med_motion = medfilt(med_motion, kernel_size=(median_filter_size, 1))
            np.save(medicine_output_dir / "motion.npy", med_motion)
            np.save(medicine_output_dir / "time_bins.npy", med_time_bins)
            np.save(medicine_output_dir / "depth_bins.npy", med_depth_bins)

        # Load MEDiCINe outputs
        med_motion = Motion( 
            displacement=np.load(medicine_output_dir / "motion.npy"),
            temporal_bins_s=np.load(medicine_output_dir / "time_bins.npy"),
            spatial_bins_um=np.load(medicine_output_dir / "depth_bins.npy"),
        )
        if method != 'all':
            motion = med_motion

    # Interpolate motion: choose default for 'all' as DREDGE
    # When running all methods, prefer DREDGE as the default motion
    if method == 'all':
        motion = dredge_motion
    elif method == 'med':
        motion = med_motion
    elif method == 'ks':
        motion = ks_motion
    elif method == 'dc':
        motion = dc_motion
    elif method == 'dredge':
        motion = dredge_motion

    if rec_for_sorting is None:
        rec_for_sorting = seg

    print(f'Applying correction to recording with Sampling Freq: {rec_for_sorting.get_sampling_frequency()}')
    seg_sort = astype(interpolate_motion(astype(rec_for_sorting, "float"), motion, border_mode='force_zeros'), "int16")

    print('Finished motion correction')
    return seg_sort

def plot_motion_output(seg, cache_dir, save_dir=None, plot_stride=30, uV_per_bit=.195, recalc=False):
    if isinstance(cache_dir, str):
        cache_dir = Path(cache_dir)
    if save_dir is not None and isinstance(save_dir, str):
        save_dir = Path(save_dir)
    if save_dir is None:
        save_dir = cache_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    save_files = [
        'depth_raster.png',
        'motion_comparison.png',
        'amplitude_depth_comparison.png',
        'kilosort_motion_correction.png',
        'decentralized_motion_correction.png',
        'medicine_motion_correction.png',
    ]
    if all([(save_dir / f).exists() for f in save_files]) and not recalc:
        print('All plots already exist, returning...')
        return

    peaks = np.load(cache_dir / 'peaks.npy')
    peak_locations = np.load(cache_dir / 'peak_locations.npy')
    ks_loc= (cache_dir / 'ks-motion')
    dc_loc = (cache_dir / 'decentralized-motion')
    med_loc = (cache_dir / 'medicine')
    dredge_loc = (cache_dir / 'dredge-motion')
    if ks_loc.exists():
        ks_motion = Motion(
            displacement=np.load(cache_dir / "ks-motion/motion.npy"),
            temporal_bins_s=np.load(cache_dir / "ks-motion/time_bins.npy"),
            spatial_bins_um=np.load(cache_dir / "ks-motion/depth_bins.npy"),
        )
        method = 'ks'
    if dc_loc.exists():
        dc_motion = Motion(
            displacement=np.load(cache_dir / "decentralized-motion/motion.npy"),
            temporal_bins_s=np.load(cache_dir / "decentralized-motion/time_bins.npy"),
            spatial_bins_um=np.load(cache_dir / "decentralized-motion/depth_bins.npy"),
        )
        method = 'dc'
    if med_loc.exists():
        med_motion = Motion(
            displacement=np.load(cache_dir / "medicine/motion.npy"),
            temporal_bins_s=np.load(cache_dir / "medicine/time_bins.npy"),
            spatial_bins_um=np.load(cache_dir / "medicine/depth_bins.npy"),
        )
        method = 'med'
    if dredge_loc.exists():
        dredge_motion = Motion(
            displacement=np.load(dredge_loc / "motion.npy"),
            temporal_bins_s=np.load(dredge_loc / "time_bins.npy"),
            spatial_bins_um=np.load(dredge_loc / "depth_bins.npy"),
        )
        method = 'dredge'
    if ks_loc.exists() and dc_loc.exists() and med_loc.exists() and dredge_loc.exists():
        method = 'all'
    

    spike_samples = peaks['sample_index']
    if seg.get_time_info()['t_start'] is not None:
        spike_times = spike_samples / seg.get_sampling_frequency() + seg.get_time_info()['t_start']
    else:
        spike_times = spike_samples / seg.get_sampling_frequency()
    spike_depths = peak_locations['y']
    spike_amps = peaks['amplitude'] * uV_per_bit

    # Subsample
    peak_samples = spike_samples[::plot_stride]
    peak_times = spike_times[::plot_stride]
    peak_depths = spike_depths[::plot_stride]
    peak_amplitudes = spike_amps[::plot_stride]

    # Normalize amplitudes by CDF to have uniform distribution
    amp_argsort = np.argsort(np.argsort(peak_amplitudes))
    peak_amplitudes = amp_argsort / len(peak_amplitudes)

    #
    # Plot depth raster
    #
    # Function for plotting neural activity
    def _plot_neural_activity(ax, times, depths, colors):
        plot = ax.scatter(times, depths, s=1, c=colors, alpha=.75)
        ax.set_xlabel("time (s)", fontsize=12)
        ax.set_ylabel("depth from probe tip (um)", fontsize=12)
        return plot

    # Scatterplot peaks
    cmap = plt.get_cmap("winter")
    colors = cmap(peak_amplitudes)
    fig, axs = plt.subplots(1, 1, figsize=(7, 5))
    plot = _plot_neural_activity(axs, peak_times, peak_depths, colors)
    fig.colorbar(plot, ax=axs)
    fig.savefig(save_dir / 'depth_raster.png')

    #
    # Plot motion estimate comparison
    #

    # Use the motion object that was actually computed
    if method == 'ks':
        motion_ref = ks_motion
    elif method == 'dc':
        motion_ref = dc_motion
    elif method == 'med':
        motion_ref = med_motion
    else: #default to DREDGE
        motion_ref = dredge_motion

    depth = motion_ref.spatial_bins_um[0]
    times = motion_ref.temporal_bins_s[0]

    probe = seg.get_probe()
    d_min = np.min(probe.contact_positions[:, 1])
    d_max = np.max(probe.contact_positions[:, 1])
    n_depths = 5
    depths = np.linspace(d_min, d_max, n_depths)
    fig, axs = plt.subplots(5, 1, figsize=(10, 8), sharex=True) 
    ks_motion_depths = np.zeros((len(times), n_depths))
    dc_motion_depths = np.zeros((len(times), n_depths))
    med_motion_depths = np.zeros((len(times), n_depths))
    dredge_motion_depths = np.zeros((len(times), n_depths))

    for i, depth in enumerate(depths):

        dist = (d_max - depth)
        if method == 'ks' or method == 'all':
            ks_motion_interp = ks_motion.get_displacement_at_time_and_depth(times, np.ones(len(times)) * dist)
            ks_motion_depths[:,n_depths-i-1] = ks_motion_interp
            axs[i].plot(times, ks_motion_interp, label='Kilosort')
        
        if method == 'dc' or method == 'all':
            dc_motion_interp = dc_motion.get_displacement_at_time_and_depth(times, np.ones(len(times)) * dist)
            dc_motion_depths[:,n_depths-i-1] = dc_motion_interp
            axs[i].plot(times, dc_motion_interp, label='Decentralized')

        if method == 'med' or method == 'all':
            med_motion_interp = med_motion.get_displacement_at_time_and_depth(times, np.ones(len(times)) * dist)
            med_motion_depths[:,n_depths-i-1] = med_motion_interp
            axs[i].plot(times, med_motion_interp, label='MEDiCINe')

        if method == 'dredge' or method == 'all':
            dredge_motion_interp = dredge_motion.get_displacement_at_time_and_depth(times, np.ones(len(times)) * dist)
            dredge_motion_depths[:,n_depths-i-1] = dredge_motion_interp
            axs[i].plot(times, dredge_motion_interp, label='DREDGE')


        if i == n_depths // 2: 
            axs[i].set_ylabel('Motion (um)')
        if i == n_depths - 1:
            axs[i].set_xlabel('Time (s)')
        axs[i].set_title(f'Motion estimates (depth = {depth} um)')
        if i == 0:
            axs[i].legend()
    plt.tight_layout()
    plt.savefig(save_dir / 'motion_comparison.png')

    #
    # Plot amplitude-depth comparison
    #
    # Get colors and create figure
    cmap = plt.get_cmap('winter')
    colors = cmap(peak_amplitudes)
    fig, axes = plt.subplots(4, 1, figsize=(15, 10), sharex=True, sharey=True)

    if method == 'ks' or method == 'all':
        peak_depth_ks = _correct_motion_on_peaks(
            peak_times,
            peak_depths,
            ks_motion_depths,
            times,
            depths
        )
        _ = _plot_neural_activity(axes[0], peak_times, peak_depth_ks, colors)
        axes[0].set_title("Kilosort")

    if method == 'dc' or method == 'all':
        peak_depth_dc = _correct_motion_on_peaks(
            peak_times,
            peak_depths,
            dc_motion_depths,
            times,
            depths
        )
        _ = _plot_neural_activity(axes[1], peak_times, peak_depth_dc, colors)
        axes[1].set_title("Decentralized")

    if method == 'med' or method == 'all':
        peak_depth_med = _correct_motion_on_peaks(
            peak_times,
            peak_depths,
            med_motion_depths,
            times,
            depths
        )
        plot = _plot_neural_activity(axes[2], peak_times, peak_depth_med, colors)
        axes[2].set_title("MEDiCINe")
        #fig.colorbar(plot, ax=axes[2]) 

    if method == 'dredge' or method == 'all':
        peak_depth_dredge = _correct_motion_on_peaks(
            peak_times,
            peak_depths,
            dredge_motion_depths,
            times,
            depths
        )
        plot = _plot_neural_activity(axes[3], peak_times, peak_depth_dredge, colors)
        axes[3].set_title("DREDGE")
        #fig.colorbar(plot, ax=axes[2])

    plt.tight_layout()
    plt.savefig(save_dir / 'amplitude_depth_comparison.png')

    #
    #   Plot individual motion correction
    #

    # Kilosort
    if method == 'ks' or method == 'all':
        f_ks = plot_motion_correction(
            spike_times,
            spike_depths,
            spike_amps,
            times,
            depths,
            ks_motion_depths,
        )
        f_ks.suptitle('Kilosort')
        f_ks.savefig(save_dir / 'kilosort_motion_correction.png')

    # Decentralized
    if method == 'dc' or method == 'all':
        f_dc = plot_motion_correction(
            spike_times,
            spike_depths,
            spike_amps,
            times,
            depths,
            dc_motion_depths,
        )
        f_dc.suptitle('Decentralized')    
        f_dc.savefig(save_dir / 'decentralized_motion_correction.png')

    # MEDiCINe
    if method == 'med' or method == 'all':
        f_med = plot_motion_correction(
            spike_times,
            spike_depths,
            spike_amps,
            times,
            depths,
            med_motion_depths,
        )
        f_med.suptitle('MEDiCINe')
        f_med.savefig(save_dir / 'medicine_motion_correction.png')

    # DREDGE
    if method == 'dredge' or method == 'all':
        f_dredge = plot_motion_correction(
            spike_times,
            spike_depths,
            spike_amps,
            times,
            depths,
            dredge_motion_depths,
        )
        f_dredge.suptitle('DREDGE')
        f_dredge.savefig(save_dir / 'dredge_motion_correction.png')
    
    plt.close('all')
