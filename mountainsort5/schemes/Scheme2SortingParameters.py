import numpy as np
import numpy.typing as npt
from typing import Union, Literal
from dataclasses import dataclass


@dataclass
class Scheme2SortingParameters:
    """Parameters for MountainSort sorting scheme 2

    See Scheme1SortingParameters for more details on the parameters below.

    - phase1_detect_channel_radius: detect_channel_radius in phase 1
    - detect_channel_radius: detect_channel_radius in phase 2
    - phase1_detect_threshold: detect_threshold in phase 1
    - phase1_detect_time_radius_msec: detect_time_radius_msec in phase 1
    - detect_time_radius_msec: detect_time_radius_msec in phase 2
    - phase1_npca_per_branch: npca_per_branch in phase 1
    - phase1_pairwise_merge_step: pairwise_merge_step in phase 1
    - detect_sign
    - detect_threshold: detect_threshold in phase 2
    - snippet_T1
    - snippet_T2
    - snippet_mask_radius
    - max_num_snippets_per_training_batch: the maximum number of snippets to use for training the classifier in each batch
    - classifier_npca: the number of principal components to use for each neighborhood classifier
    - training_duration_sec: the duration of the training data (in seconds)
    - training_recording_sampling_mode: how to sample the training data. If 'initial', then the first training_duration_sec of the recording will be used. If 'uniform', then the training data will be sampled uniformly in 10-second chunks from the recording.
    """
    phase1_detect_channel_radius: Union[float, None]
    detect_channel_radius: Union[float, None]
    phase1_detect_threshold: float=5.5
    phase1_detect_time_radius_msec: float=1.5
    detect_time_radius_msec: float=0.5
    phase1_npca_per_branch: int=12
    phase1_pairwise_merge_step: bool=True
    detect_sign: int=-1
    detect_threshold: float=5.5
    snippet_T1: int=20
    snippet_T2: int=20
    snippet_mask_radius: Union[float, None]=None
    max_num_snippets_per_training_batch: int=200
    classifier_npca: Union[int, None]=None
    training_duration_sec: Union[float, None]=None
    training_recording_sampling_mode: Literal['initial', 'uniform']='initial'

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
        if self.phase1_detect_channel_radius is not None:
            assert self.phase1_detect_channel_radius > 0
        assert self.phase1_detect_time_radius_msec > 0 and self.phase1_detect_time_radius_msec <= 1e4
        assert self.phase1_detect_threshold > 0
        assert self.detect_sign in [-1, 0, 1]
        assert self.phase1_npca_per_branch >= 1 and self.phase1_npca_per_branch <= 1e3

