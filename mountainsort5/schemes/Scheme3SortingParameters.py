import numpy as np
from dataclasses import dataclass
from .Scheme2SortingParameters import Scheme2SortingParameters


@dataclass
class Scheme3SortingParameters:
    """Parameters for MountainSort sorting scheme 3

    - block_sorting_parameters: Scheme2SortingParameters for individual blocks
    - block_duration_sec: duration of each block
    """
    block_sorting_parameters: Scheme2SortingParameters
    block_duration_sec: float

    def check_valid(self, *, M: int, N: int, sampling_frequency: float, channel_locations: np.ndarray):
        """Internal function for checking validity of parameters"""
        self.block_sorting_parameters.check_valid(M=M, N=N, sampling_frequency=sampling_frequency, channel_locations=channel_locations)
        assert self.block_duration_sec > 0
