import numpy as np
import numpy.typing as npt
import math
import spikeinterface as si
from .Scheme1SortingParameters import Scheme1SortingParameters
from ..core.detect_spikes import detect_spikes
from ..core.extract_snippets import extract_snippets
from ..core.isosplit6_subdivision_method import isosplit6_subdivision_method
from ..core.compute_templates import compute_templates
from ..core.compute_pca_features import compute_pca_features


def sorting_scheme1(
    recording: si.BaseRecording, *,
    sorting_parameters: Scheme1SortingParameters
):
    """MountainSort 5 sorting scheme 1

    Args:
        recording (si.BaseRecording): SpikeInterface recording object
        sorting_parameters (Scheme2SortingParameters): Sorting parameters

    Returns:
        si.BaseSorting: SpikeInterface sorting object
    """

    ###################################################################
    # Handle multi-segment recordings
    if recording.get_num_segments() > 1:
        print('Recording has multiple segments. Joining segments for sorting...')
        recording_joined = si.concatenate_recordings(recording_list=[recording])
        sorting_joined = sorting_scheme1(recording_joined, sorting_parameters=sorting_parameters)
        print('Splitting sorting into segments to match original multisegment recording...')
        sorting = si.split_sorting(sorting_joined, recording_joined)
        return sorting
    ###################################################################

    M = recording.get_num_channels()
    N = recording.get_num_frames()
    sampling_frequency = recording.sampling_frequency

    channel_locations = recording.get_channel_locations()

    print(f'Number of channels: {M}')
    print(f'Number of timepoints: {N}')
    print(f'Sampling frequency: {sampling_frequency} Hz')
    for m in range(M):
        print(f'Channel {m}: {channel_locations[m]}')

    sorting_parameters.check_valid(M=M, N=N, sampling_frequency=sampling_frequency, channel_locations=channel_locations)
    
    print('Loading traces')
    traces = recording.get_traces()

    print('Detecting spikes')
    time_radius = int(math.ceil(sorting_parameters.detect_time_radius_msec / 1000 * sampling_frequency))
    times, channel_indices = detect_spikes(
        traces=traces,
        channel_locations=channel_locations,
        time_radius=time_radius,
        channel_radius=sorting_parameters.detect_channel_radius,
        detect_threshold=sorting_parameters.detect_threshold,
        detect_sign=sorting_parameters.detect_sign,
        margin_left=sorting_parameters.snippet_T1,
        margin_right=sorting_parameters.snippet_T2,
        verbose=True
    )

    # this is important because isosplit does not do well with duplicate points
    times, channel_indices = remove_duplicate_times(times, channel_indices)

    print(f'Extracting {len(times)} snippets')
    snippets = extract_snippets( # L x T x M
        traces=traces,
        channel_locations=channel_locations,
        mask_radius=sorting_parameters.snippet_mask_radius,
        times=times,
        channel_indices=channel_indices,
        T1=sorting_parameters.snippet_T1,
        T2=sorting_parameters.snippet_T2
    )
    L = snippets.shape[0]
    T = snippets.shape[1]
    assert snippets.shape[2] == M

    print('Clustering snippets')
    features = compute_pca_features(snippets.reshape((L, T * M)), sorting_parameters.npca_per_channel * M)
    labels = isosplit6_subdivision_method(
        X=features,
        npca_per_subdivision=sorting_parameters.npca_per_subdivision
    )
    K = int(np.max(labels))
    print(f'Found {K} clusters')

    print('Computing templates')
    templates = compute_templates(snippets=snippets, labels=labels) # K x T x M
    peak_channel_indices = [np.argmin(np.min(templates[i], axis=0)) for i in range(K)]

    print('Determining optimal alignment of templates')
    offsets = align_templates(templates)

    print('Aligning snippets')
    snippets = align_snippets(snippets, offsets, labels)
    # this is tricky - we need to subtract the offset to correspond to shifting the template
    times = offset_times(times, -offsets, labels)

    print('Clustering aligned snippets')
    features = compute_pca_features(snippets.reshape((L, T * M)), sorting_parameters.npca_per_channel * M)
    labels = isosplit6_subdivision_method(
        X=features,
        npca_per_subdivision=sorting_parameters.npca_per_subdivision
    )
    K = int(np.max(labels))
    print(f'Found {K} clusters')

    print('Computing templates')
    templates = compute_templates(snippets=snippets, labels=labels) # K x T x M
    peak_channel_indices = [np.argmin(np.min(templates[i], axis=0)) for i in range(K)]

    print('Offsetting times to peak')
    # Now we need to offset the times again so that the spike times correspond to actual peaks
    offsets_to_peak = determine_offsets_to_peak(templates, detect_sign=sorting_parameters.detect_sign, T1=sorting_parameters.snippet_T1)
    print('Offsets to peak:', offsets_to_peak)
    # This time we need to add the offset
    times = offset_times(times, offsets_to_peak, labels)

    # Now we need to make sure the times are in order, because we have offset them
    sort_inds = np.argsort(times)
    times = times[sort_inds]
    labels = labels[sort_inds]

    # also make sure none of the times are out of bounds now that we have offset them a couple times
    inds_okay = np.where((times >= sorting_parameters.snippet_T1) & (times < N - sorting_parameters.snippet_T2))[0]
    times = times[inds_okay]
    labels = labels[inds_okay]

    print('Reordering units')
    # relabel so that units are ordered by channel
    # and we also put any labels that are not used at the end
    aa = peak_channel_indices
    for k in range(1, K + 1):
        inds = np.where(labels == k)[0]
        if len(inds) == 0:
            aa[k - 1] = np.Inf
    new_labels_mapping = np.argsort(np.argsort(aa)) + 1 # too tricky! my head aches right now
    labels = new_labels_mapping[labels - 1]
    
    sorting = si.NumpySorting.from_times_labels(times_list=[times], labels_list=[labels], sampling_frequency=sampling_frequency)

    return sorting

def remove_duplicate_times(times: npt.NDArray[np.int32], labels: npt.NDArray[np.int32]):
    inds = np.where(np.diff(times) > 0)[0]
    inds = np.concatenate([[0], inds + 1])
    times2 = times[inds]
    labels2 = labels[inds]
    return times2, labels2

def align_templates(templates: npt.NDArray[np.float32]):
    K = templates.shape[0]
    T = templates.shape[1]
    M = templates.shape[2]
    offsets = np.zeros((K,), dtype=np.int32)
    pairwise_optimal_offsets = np.zeros((K, K), dtype=np.int32)
    pairwise_inner_products = np.zeros((K, K), dtype=np.float32)
    for k1 in range(K):
        for k2 in range(K):
            offset, inner_product = compute_pairwise_optimal_offset(templates[k1], templates[k2])
            pairwise_optimal_offsets[k1, k2] = offset
            pairwise_inner_products[k1, k2] = inner_product
    for passnum in range(20):
        something_changed = False
        for k1 in range(K):
            weighted_sum = 0
            total_weight = 0
            for k2 in range(K):
                if k1 != k2:
                    offset = pairwise_optimal_offsets[k1, k2] + offsets[k2]
                    weight = pairwise_inner_products[k1, k2]
                    weighted_sum += weight * offset
                    total_weight += weight
            if total_weight > 0:
                avg_offset = int(weighted_sum / total_weight)
            else:
                avg_offset = 0
            if avg_offset != offsets[k1]:
                something_changed = True
                offsets[k1] = avg_offset
        if not something_changed:
            print('Template alignment converged.')
            break
    print('Align templates offsets: ', offsets)
    return offsets
    

def compute_pairwise_optimal_offset(template1: npt.NDArray[np.float32], template2: npt.NDArray[np.float32]):
    T = template1.shape[0]
    best_inner_product = -np.Inf
    best_offset = 0
    for offset in range(T):
        inner_product = np.sum(np.roll(template1, shift=offset, axis=0) * template2)
        if inner_product > best_inner_product:
            best_inner_product = inner_product
            best_offset = offset
    if best_offset > T // 2:
        best_offset = best_offset - T
    return best_offset, best_inner_product

def align_snippets(snippets: npt.NDArray[np.float32], offsets: npt.NDArray[np.int32], labels: npt.NDArray[np.int32]):
    snippets2 = np.zeros_like(snippets)
    for k in range(1, np.max(labels) + 1):
        inds = np.where(labels == k)[0]
        snippets2[inds] = np.roll(snippets[inds], shift=offsets[k - 1], axis=1)
    return snippets2

def offset_times(times: npt.NDArray[np.int32], offsets: npt.NDArray[np.int32], labels: npt.NDArray[np.int32]):
    times2 = np.zeros_like(times)
    for k in range(1, np.max(labels) + 1):
        inds = np.where(labels == k)[0]
        times2[inds] = times[inds] + offsets[k - 1]
    return times2

def determine_offsets_to_peak(templates: npt.NDArray[np.float32], *, detect_sign: int, T1: int):
    K = templates.shape[0]

    if detect_sign < 0:
        A = -templates
    elif detect_sign > 0: # pragma: no cover
        A = templates # pragma: no cover
    else:
        A = np.abs(templates) # pragma: no cover
    
    offsets_to_peak = np.zeros((K,), dtype=np.int32)
    for k in range(K):
        peak_channel = np.argmax(np.max(A[k], axis=0))
        peak_time = np.argmax(A[k][:, peak_channel])
        offset_to_peak = peak_time - T1
        offsets_to_peak[k] = offset_to_peak
    return offsets_to_peak