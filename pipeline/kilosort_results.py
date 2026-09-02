"""Accessor for a Kilosort 4 sorter_output directory.

Extracted verbatim from ``pipelineold/sorting.py`` at research-repository commit
e71b144. Only the definitions reachable from the production entry
points are carried over; the legacy module keeps the rest.

The legacy ``sort_ks4`` and ``save_binary_recording`` helpers are
deliberately not carried over: production sorting lives in
``pipeline.sorting``.
"""

from pathlib import Path
import numpy as np
import pandas as pd


class KilosortResults:
    def __init__(self, directory):
        if isinstance(directory, str):
            directory = Path(directory)
        assert isinstance(directory, Path), 'directory must be a string or Path object'
        assert directory.exists(), f'{directory} does not exist'
        assert directory.is_dir(), f'{directory} is not a directory'
        self.directory = directory

        # Move directory to sorter_output if it is a kilosort4 output directory
        if (directory / 'sorter_output').exists():
            directory = directory / 'sorter_output'

        self.spike_times_file = directory / 'spike_times.npy'
        assert self.spike_times_file.exists(), f'{self.spike_times_file} does not exist'
        self._spike_times = None

        self.spike_amplitudes_file = directory / 'amplitudes.npy'
        assert self.spike_amplitudes_file.exists(), f'{self.spike_amplitudes_file} does not exist'
        self._spike_amplitudes = None

        self.st_file = directory / 'full_st.npy'
        if not self.st_file.exists():
            print(f'Warning: {self.st_file} does not exist. Use Kilosort4 with save_extra_vars=True to generate.')
        self.kept_spikes_file = directory / 'kept_spikes.npy'
        if not self.kept_spikes_file.exists():
            print(f'Warning: {self.kept_spikes_file} does not exist. Use Kilosort4 with save_extra_vars=True to generate.')
        self._st = None

        self.spike_clusters_file = directory / 'spike_clusters.npy'
        assert self.spike_clusters_file.exists(), f'{self.spike_clusters_file} does not exist'
        self._spike_clusters = None

        self.spike_templates_file = directory / 'spike_templates.npy'
        assert self.spike_templates_file.exists(), f'{self.spike_templates_file} does not exist'
        self._spike_templates = None
        
        self.spike_positions_file = directory / 'spike_positions.npy'
        assert self.spike_positions_file.exists(), f'{self.spike_positions_file} does not exist'
        self._spike_positions = None

        self.cluster_labels_file = directory / 'cluster_KSLabel.tsv'
        assert self.cluster_labels_file.exists(), f'{self.cluster_labels_file} does not exist'
        self._cluster_labels = None
        
    @property
    def spike_times(self):
        if self._spike_times is None:
            self._spike_times = np.load(self.spike_times_file)
        return self._spike_times
    
    @property
    def spike_amplitudes(self):
        if self._spike_amplitudes is None:
            self._spike_amplitudes = np.load(self.spike_amplitudes_file)
        return self._spike_amplitudes

    @property
    def st(self): 
        if self._st is None:
            st = np.load(self.st_file)
            spikes = np.load(self.kept_spikes_file)
            self._st = st[spikes]
        return self._st
    
    @property
    def spike_clusters(self):
        if self._spike_clusters is None:
            self._spike_clusters = np.load(self.spike_clusters_file)
        return self._spike_clusters

    @property
    def spike_templates(self):
        if self._spike_templates is None:
            self._spike_templates = np.load(self.spike_templates_file)
        return self._spike_templates

    @property
    def spike_positions(self):
        if self._spike_positions is None:
            self._spike_positions = np.load(self.spike_positions_file)
        return self._spike_positions

    @property
    def cluster_labels(self):
        if self._cluster_labels is None:
            self._cluster_labels = pd.read_csv(self.cluster_labels_file, sep='\t')
        return self._cluster_labels
