import numpy as np
from typing import Union
from dataclasses import dataclass


@dataclass
class SortingParameters:
    detect_threshold: float=5.5
    detect_channel_radius: Union[float, None]=None
    detect_time_radius_msec: float=0.5
    detect_sign: int=-1
    snippet_T1: int=20
    snippet_T2: int=20
    snippet_mask_radius: Union[float, None]=None
    npca_per_branch: int=12

    def check_valid(self, *, M: int, N: int, sampling_frequency: float, channel_locations: Union[np.ndarray, None]=None):
        if channel_locations is None:
            channel_locations = np.zeros((M, 1), dtype=np.float32)
        assert channel_locations.shape[0] == M, 'Shape mismatch between traces and channel locations'
        D = channel_locations.shape[1]
        assert N >= self.snippet_T1 + self.snippet_T2
        if self.snippet_mask_radius is not None:
            assert self.snippet_mask_radius >= 0
        assert M >= 1 and M < 1e6
        assert D >= 1 and D <= 3
        assert sampling_frequency > 0 and sampling_frequency <= 1e7
        if self.detect_channel_radius is not None:
            assert self.detect_channel_radius > 0
        assert self.detect_time_radius_msec > 0 and self.detect_time_radius_msec <= 1e4
        assert self.detect_threshold > 0
        assert self.detect_sign in [-1, 0, 1]
        assert self.npca_per_branch >= 1 and self.npca_per_branch <= 1e3

