import numpy as np
import numpy.typing as npt
from typing import Union
from dataclasses import dataclass


@dataclass
class Scheme1SortingParameters:
    """Parameters for MountainSort sorting scheme 1
    - detect_threshold: the threshold for detection of whitened data
    - detect_channel_radius: the radius (in units of channel locations) for exluding nearby channels from detection
    - detect_time_radius_msec: the radius (in msec) for excluding nearby events from detection
    - detect_sign: the sign of the threshold for detection (1, -1, or 0)
    - snippet_T1: the number of timepoints before the event to include in the snippet
    - snippet_T2: the number of timepoints after the event to include in the snippet
    - snippet_mask_radius: the radius (in units of channel locations) for making a snippet around the central channel
    - npca_per_channel: the number of PCA components per channel for initial dimension reduction
    - npca_per_subdivision: the number of PCA components to compute for each subdivision of clustering
    - skip_alignment: whether to skip the alignment step (if None, then False)
    """
    detect_threshold: float = 5.5
    detect_channel_radius: Union[float, None] = None
    detect_time_radius_msec: float = 0.5
    detect_sign: int = -1
    snippet_T1: int = 20
    snippet_T2: int = 20
    snippet_mask_radius: Union[float, None] = None
    npca_per_channel: int = 3
    npca_per_subdivision: int = 10
    skip_alignment: Union[bool, None] = None
    pairwise_merge_step: bool = False # deprecated

    def check_valid(self, *, M: int, N: int, sampling_frequency: float, channel_locations: npt.NDArray[np.float32]):
        """Internal function for checking validity of parameters"""
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
        assert self.npca_per_channel >= 1 and self.npca_per_channel <= 1e3
        assert self.npca_per_subdivision >= 1 and self.npca_per_subdivision <= 1e3
