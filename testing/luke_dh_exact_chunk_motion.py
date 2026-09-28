"""Pitch-exact motion adapter for the frozen DH matching-chunk contract."""
from __future__ import annotations
import numpy as np


class ExactChunkLatticeMotion:
    """Rigid motion whose value is constant on half-open sample chunks.

    The adapter is intended for DARTsort calls whose times originate from the
    recording-local sample clock. It converts seconds back to the nearest
    integer sample, validates that the reconstructed sample is close to that
    clock, then applies ``sample // chunk_length_samples``. Thus an exact chunk
    boundary belongs to the new chunk. The padded tail of the final declared
    chunk is supported explicitly because DARTsort queries the nominal centre
    of every matching chunk, including a final partial chunk.
    """
    def __init__(self, states_um, *, sampling_frequency_hz, chunk_length_samples,
                 n_samples):
        states=np.asarray(states_um,dtype=np.float64)
        if states.ndim!=1 or not states.size or not np.isfinite(states).all():
            raise ValueError("states must be a nonempty finite vector")
        self.sampling_frequency_hz=float(sampling_frequency_hz)
        self.chunk_length_samples=int(chunk_length_samples); self.n_samples=int(n_samples)
        if self.sampling_frequency_hz <= 0 or self.chunk_length_samples <= 0 or self.n_samples <= 0:
            raise ValueError("frequency, chunk length, and sample count must be positive")
        expected=(self.n_samples+self.chunk_length_samples-1)//self.chunk_length_samples
        if states.size!=expected: raise ValueError("state count does not cover sample chunks")
        self.displacement=states.copy()
        starts=np.arange(states.size,dtype=np.int64)*self.chunk_length_samples
        self.time_bin_centers_s=(starts+self.chunk_length_samples//2)/self.sampling_frequency_hz
        self.time_bin_edges_s=None; self.spatial_bin_centers_um=None; self.spatial_bin_edges_um=None
        self.local_time_origin_s=0.0
        self.query_support_samples=(0, int(states.size*self.chunk_length_samples))
        self.final_padding_samples=int(self.query_support_samples[1]-self.n_samples)

    def _indices(self,t_s):
        times=np.asarray(t_s,dtype=np.float64)
        if not np.isfinite(times).all():
            raise ValueError("query times must be finite")
        sample_float=(times-self.local_time_origin_s)*self.sampling_frequency_hz
        samples=np.rint(sample_float).astype(np.int64)
        # sample_index_to_time round-trips within floating precision. A looser
        # conversion could silently accept session-time queries on a local field.
        if np.any(np.abs(sample_float-samples)>1e-6):
            raise ValueError("query is not on the declared recording-local sample clock")
        lo,hi=self.query_support_samples
        if np.any(samples<lo) or np.any(samples>=hi):
            raise ValueError(f"query outside declared support [{lo}, {hi}) samples")
        return samples//self.chunk_length_samples

    def disp_at_s(self,t_s,depth_um=None,grid=False):
        d=self.displacement[self._indices(t_s)]
        if grid and depth_um is not None:
            d=np.broadcast_to(np.atleast_1d(d)[None,:],(*np.atleast_1d(depth_um).shape,*np.atleast_1d(d).shape))
        return d

    def correct_s(self,t_s,depth_um,grid=False):
        return np.asarray(depth_um)-self.disp_at_s(t_s,depth_um,grid=grid)

    def uncorrect_s(self,t_s,registered_depth_um,grid=False):
        return np.asarray(registered_depth_um)+self.disp_at_s(t_s,registered_depth_um,grid=grid)
