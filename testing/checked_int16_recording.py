"""SpikeInterface preprocessing node for explicit, checked int16 conversion."""

from __future__ import annotations

import numpy as np
from spikeinterface.preprocessing.basepreprocessor import (
    BasePreprocessor,
    BasePreprocessorSegment,
)


class CheckedInt16Recording(BasePreprocessor):
    """Round to nearest int16 and refuse nonfinite or out-of-range samples."""

    name = "checked_int16"

    def __init__(self, recording):
        super().__init__(recording, dtype=np.dtype("int16"))
        for parent_segment in recording._recording_segments:
            self.add_recording_segment(CheckedInt16RecordingSegment(parent_segment))
        self._kwargs = {"recording": recording}


class CheckedInt16RecordingSegment(BasePreprocessorSegment):
    def __init__(self, parent_recording_segment):
        super().__init__(parent_recording_segment)

    def get_traces(self, start_frame, end_frame, channel_indices):
        if channel_indices is None:
            channel_indices = slice(None)
        traces = np.asarray(
            self.parent_recording_segment.get_traces(
                start_frame, end_frame, channel_indices
            )
        )
        if not np.all(np.isfinite(traces)):
            raise ValueError("nonfinite value encountered during int16 conversion")
        rounded = np.rint(traces)
        limits = np.iinfo(np.int16)
        if rounded.size and (np.min(rounded) < limits.min or np.max(rounded) > limits.max):
            raise OverflowError("out-of-range value encountered during int16 conversion")
        return rounded.astype(np.int16, copy=False)
