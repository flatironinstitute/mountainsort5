from typing import Union
import numpy as np
import math
import spikeinterface as si
from ..core.SortingParameters import SortingParameters
from ..core.detect_spikes import detect_spikes
from ..core.extract_snippets import extract_snippets
from ..core.cluster_snippets import cluster_snippets


def sorting_scheme1(
    recording: si.BaseRecording, *,
    channel_locations: Union[np.ndarray, None]=None,
    sorting_parameters: Union[SortingParameters, None]=None
):
    M = recording.get_num_channels()
    N = recording.get_num_frames()
    sampling_frequency = recording.sampling_frequency

    if sorting_parameters is None:
        sorting_parameters = SortingParameters()
    sorting_parameters.check_valid(M=M, N=N, sampling_frequency=sampling_frequency, channel_locations=channel_locations)

    if channel_locations is None:
        channel_locations = np.zeros((M, 1), dtype=np.float32)

    print('Loading traces')
    traces = recording.get_traces()

    print('Detecting spikes')
    time_radius = int(math.ceil(sorting_parameters.detect_time_radius_msec / 1000 * sampling_frequency))
    times, channels = detect_spikes(
        traces=traces,
        channel_locations=channel_locations,
        time_radius=time_radius,
        channel_radius=sorting_parameters.detect_channel_radius,
        detect_threshold=sorting_parameters.detect_threshold,
        detect_sign=sorting_parameters.detect_sign,
        margin_left=sorting_parameters.snippet_T1,
        margin_right=sorting_parameters.snippet_T2
    )

    print('Extracting snippets')
    snippets = extract_snippets( # L x T x M
        traces=traces,
        channel_locations=channel_locations,
        mask_radius=sorting_parameters.snippet_mask_radius,
        times=times,
        channels=channels,
        T1=sorting_parameters.snippet_T1,
        T2=sorting_parameters.snippet_T2
    )
    L = snippets.shape[0]
    T = snippets.shape[1]
    assert snippets.shape[2] == M

    print('Clustering snippets')
    labels = cluster_snippets(
        snippets=snippets,
        npca_per_branch=sorting_parameters.npca_per_branch
    )
    K = int(np.max(labels))
    print(f'Found {K} clusters')
    
    sorting = si.NumpySorting.from_times_labels(times_list=[times], labels_list=[labels], sampling_frequency=sampling_frequency)

    return sorting