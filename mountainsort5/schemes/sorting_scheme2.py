from typing import Union, Tuple, List
import numpy as np
import math
import time
import spikeinterface as si
from .Scheme2SortingParameters import Scheme2SortingParameters
from .Scheme1SortingParameters import Scheme1SortingParameters
from ..core.detect_spikes import detect_spikes
from ..core.extract_snippets import extract_snippets
from .sorting_scheme1 import sorting_scheme1
from ..core.SnippetClassifier import SnippetClassifier
from ..core.pairwise_merge_step import remove_duplicate_events


# This is an extension of sorting_scheme1
def sorting_scheme2(
    recording: si.BaseRecording, *,
    sorting_parameters: Scheme2SortingParameters
) -> si.BaseSorting:
    """
    Inputs:
        recording: a recording object
        sorting_parameters: a SortingParameters object
    Returns:
        sorting: a sorting object
    """
    M = recording.get_num_channels()
    N = recording.get_num_frames()
    sampling_frequency = recording.sampling_frequency
    channel_locations = recording.get_channel_locations()

    sorting_parameters.check_valid(M=M, N=N, sampling_frequency=sampling_frequency, channel_locations=channel_locations)

    sorting1 = sorting_scheme1(
        recording=recording,
        sorting_parameters=Scheme1SortingParameters(
            detect_threshold=sorting_parameters.phase1_detect_threshold,
            detect_sign=sorting_parameters.detect_sign,
            detect_time_radius_msec=sorting_parameters.phase1_detect_time_radius_msec,
            detect_channel_radius=sorting_parameters.phase1_detect_channel_radius,
            snippet_mask_radius=sorting_parameters.snippet_mask_radius,
            snippet_T1=sorting_parameters.snippet_T1,
            snippet_T2=sorting_parameters.snippet_T2,
            npca_per_branch=sorting_parameters.phase1_npca_per_branch,
            pairwise_merge_step=sorting_parameters.phase1_pairwise_merge_step
        )
    )

    times, labels = get_times_labels_from_sorting(sorting1)
    K = np.max(labels)

    print('Loading traces')
    traces = recording.get_traces()
    snippets = extract_snippets(
        traces=traces,
        channel_locations=None,
        mask_radius=None,
        times=times,
        channel_indices=None,
        T1=sorting_parameters.snippet_T1,
        T2=sorting_parameters.snippet_T2
    )

    print('Training classifier')
    snippet_classifer = SnippetClassifier(npca=sorting_parameters.classifier_npca)
    for k in range(1, K + 1):
        inds0 = np.where(labels == k)[0]
        snippets0 = snippets[inds0]
        template0 = np.median(snippets0, axis=0) # T x M
        if sorting_parameters.detect_sign < 0:
            AA = -template0
        elif sorting_parameters.detect_sign > 0:
            AA = template0
        else:
            AA = np.abs(template0)
        peak_indices_over_channels = np.argmax(AA, axis=0)
        peak_values_over_channels = np.max(AA, axis=0)
        offsets_to_include = []
        for m in range(M):
            if peak_values_over_channels[m] > sorting_parameters.detect_threshold * 0.5:
                offsets_to_include.append(peak_indices_over_channels[m] - sorting_parameters.snippet_T1)
        offsets_to_include = np.unique(offsets_to_include)
        for offset in offsets_to_include:
            snippet_classifer.add_training_snippets(
                snippets=subsample_snippets(np.roll(snippets0, shift=-offset, axis=1), sorting_parameters.max_num_snippets_per_training_batch),
                label=k,
                offset=offset
            )
    snippets = None # Free up memory
    print('Fitting model')
    snippet_classifer.fit()
        
    print('Detecting spikes')
    time_radius = int(math.ceil(sorting_parameters.detect_time_radius_msec / 1000 * sampling_frequency))
    times2, channel_indices2 = detect_spikes(
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
    snippets2 = extract_snippets(
        traces=traces,
        channel_locations=channel_locations,
        mask_radius=sorting_parameters.snippet_mask_radius,
        times=times2,
        channel_indices=channel_indices2,
        T1=sorting_parameters.snippet_T1,
        T2=sorting_parameters.snippet_T2
    )

    print('Classifying snippets')
    timer = time.time()
    labels2, offsets2 = snippet_classifer.classify_snippets(snippets2)
    times2 = times2 - offsets2
    elapsed = time.time() - timer
    print(f'Elapsed time for classifying snippets: {elapsed} sec')

    # now that we offset them we need to resort
    sort_inds = np.argsort(times2)
    times2 = times2[sort_inds]
    labels2 = labels2[sort_inds]

    print('Remove duplicates')
    times2, labels2 = remove_duplicate_events(times2, labels2, tol=time_radius)

    sorting2 = si.NumpySorting.from_times_labels([times2], [labels2], sampling_frequency=recording.sampling_frequency)

    return sorting2

def subsample_snippets(snippets: np.ndarray, max_num: int) -> np.ndarray:
    """Subsample snippets
    Inputs:
        snippets: 3D array of snippets (num_snippets x T x M)
        max_num: maximum number of snippets to return
    Returns:
        snippets_out: 3D array of snippets (num_snippets x T x M)
    """
    num_snippets = snippets.shape[0]
    if num_snippets > max_num:
        inds = np.arange(0, max_num) * num_snippets // max_num
        snippets_out = snippets[inds]
    else:
        snippets_out = snippets
    return snippets_out


def get_times_labels_from_sorting(sorting: si.BaseSorting) -> Tuple[np.ndarray, np.ndarray]:
    """Get times and labels from a sorting object
    Inputs:
        sorting: a sorting object
    Returns:
        times: 1D array of spike times
        labels: 1D array of spike labels
    Example:
        times, labels = get_times_labels_from_sorting(sorting)
    """
    times_list = []
    labels_list = []
    for unit_id in sorting.get_unit_ids():
        times0 = sorting.get_unit_spike_train(unit_id=unit_id)
        labels0 = np.ones(times0.shape, dtype=np.int32) * unit_id
        times_list.append(times0)
        labels_list.append(labels0)
    times = np.concatenate(times_list)
    labels = np.concatenate(labels_list)
    inds = np.argsort(times)
    times = times[inds]
    labels = labels[inds]
    return times, labels