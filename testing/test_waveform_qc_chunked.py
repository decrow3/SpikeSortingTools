import numpy as np

from pipelineold.qc import waveform_qc


class ArrayRecording:
    def __init__(self, traces, sampling_frequency):
        self.traces = traces
        self.sampling_frequency = float(sampling_frequency)
        self.read_count = 0

    def get_num_channels(self):
        return self.traces.shape[1]

    def get_num_frames(self):
        return self.traces.shape[0]

    def get_sampling_frequency(self):
        return self.sampling_frequency

    def get_traces(self, start_frame, end_frame):
        self.read_count += 1
        return self.traces[start_frame:end_frame]


def legacy_reference(seg, spike_samples, spike_clusters, *, n_waves, n_samples,
                     uV_per_bit):
    cids = np.unique(spike_clusters)
    waveforms = np.zeros(
        (len(cids), n_samples, seg.get_num_channels()), dtype=np.float32
    )
    samples = np.zeros((len(cids), n_waves), dtype=np.int64) - 1
    for iC, cid in enumerate(cids):
        cluster_samples = spike_samples[spike_clusters == cid]
        count = min(n_waves, len(cluster_samples))
        sub_inds = np.random.choice(len(cluster_samples), count, replace=False)
        chosen = cluster_samples[sub_inds]
        samples[iC, :count] = chosen
        traces = np.zeros((count, n_samples, seg.get_num_channels()))
        for iW, iS in enumerate(chosen):
            i0 = max(0, iS - n_samples // 2)
            i1 = min(
                seg.get_num_frames() - 1,
                iS + (n_samples - n_samples // 2),
            )
            wave = seg.get_traces(start_frame=i0, end_frame=i1) * uV_per_bit
            o0 = i0 - (iS - n_samples // 2)
            o1 = o0 + i1 - i0
            traces[iW, o0:o1, :] = wave
        waveforms[iC] = np.median(traces, axis=0)
    return waveforms, samples, cids


def test_chunked_waveform_qc_matches_legacy_selection_and_values(tmp_path):
    raw = np.arange(240 * 4, dtype=np.int16).reshape(240, 4)
    spike_samples = np.array([0, 3, 21, 39, 59, 79, 101, 121, 159, 199, 238, 239])
    spike_clusters = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    kwargs = dict(n_waves=5, n_samples=8, uV_per_bit=0.195)

    reference_recording = ArrayRecording(raw, sampling_frequency=20)
    np.random.seed(17)
    expected_waveforms, expected_samples, expected_cids = legacy_reference(
        reference_recording, spike_samples, spike_clusters, **kwargs
    )

    chunked_recording = ArrayRecording(raw, sampling_frequency=20)
    np.random.seed(17)
    result = waveform_qc(
        chunked_recording,
        spike_samples,
        spike_clusters,
        tmp_path,
        read_chunk_duration_s=2.0,
        **kwargs,
    )

    np.testing.assert_array_equal(result["samples"], expected_samples)
    np.testing.assert_array_equal(result["cids"], expected_cids)
    np.testing.assert_array_equal(result["waveforms"], expected_waveforms)
    assert chunked_recording.read_count < reference_recording.read_count

