from typing import Dict, Tuple, List, Union
from packaging import version
import numpy as np
import numpy.typing as npt
import math
import spikeinterface as si
from .Scheme2SortingParameters import Scheme2SortingParameters
from .Scheme1SortingParameters import Scheme1SortingParameters
from ..core.detect_spikes import detect_spikes
from ..core.extract_snippets import extract_snippets, extract_snippets_in_channel_neighborhood
from .sorting_scheme1 import sorting_scheme1
from ..core.SnippetClassifier import SnippetClassifier
from ..core.remove_duplicate_events import remove_duplicate_events
from ..core.get_sampled_recording_for_training import get_sampled_recording_for_training
from ..core.get_times_labels_from_sorting import get_times_labels_from_sorting
from ..core.Timer import Timer


def sorting_scheme2(
    recording: si.BaseRecording, *,
    sorting_parameters: Scheme2SortingParameters,
    return_snippet_classifiers: bool = False, # used in scheme 3
    reference_snippet_classifiers: Union[Dict[int, SnippetClassifier], None] = None, # used in scheme 3
    label_offset: int = 0 # used in scheme 3
) -> Union[si.BaseSorting, Tuple[si.BaseSorting, Dict[int, SnippetClassifier]]]:
    """MountainSort 5 sorting scheme 2

    Args:
        recording (si.BaseRecording): SpikeInterface recording object
        sorting_parameters (Scheme2SortingParameters): Sorting parameters
        return_snippet_classifiers (bool): whether to return the snippet classifiers (used in scheme 3)
        reference_snippet_classifiers: used in scheme 3
        label_offset: used in scheme 3

    Returns:
        si.BaseSorting: SpikeInterface sorting object
            or, if return_snippet_classifiers is True:
        si.BaseSorting, snippet_classifiers
    """

    ###################################################################
    # Handle multi-segment recordings
    if recording.get_num_segments() > 1:
        print('Recording has multiple segments. Joining segments for sorting...')
        recording_joined: si.BaseRecording = si.concatenate_recordings(recording_list=[recording])
        result = sorting_scheme2(
            recording_joined,
            sorting_parameters=sorting_parameters,
            return_snippet_classifiers=True,
            reference_snippet_classifiers=reference_snippet_classifiers,
            label_offset=label_offset
        )
        assert isinstance(result, tuple)
        sorting_joined, snippet_classifiers = result
        print('Splitting sorting into segments to match original multisegment recording...')
        sorting = si.split_sorting(sorting_joined, recording_joined)
        if return_snippet_classifiers:
            return sorting, snippet_classifiers
        else:
            return sorting
    ###################################################################

    M = recording.get_num_channels()
    N = recording.get_num_frames()
    sampling_frequency = recording.sampling_frequency
    channel_locations = recording.get_channel_locations()

    # check that the sorting parameters are valid
    sorting_parameters.check_valid(M=M, N=N, sampling_frequency=sampling_frequency, channel_locations=channel_locations)

    # Subsample the recording for training
    if sorting_parameters.training_duration_sec is not None:
        print(f'Using training recording of duration {sorting_parameters.training_duration_sec} sec with the sampling mode {sorting_parameters.training_recording_sampling_mode}')
        tt = Timer('SCHEME2 get_sampled_recording_for_training')
        training_recording = get_sampled_recording_for_training(
            recording=recording,
            training_duration_sec=sorting_parameters.training_duration_sec,
            mode=sorting_parameters.training_recording_sampling_mode
        )
        tt.report()
    else:
        print(f'Using the full recording for training: {recording.get_num_frames() / recording.get_sampling_frequency()} sec')
        training_recording = recording

    # Run the first phase of spike sorting (same as sorting_scheme1)
    print('Running phase 1 sorting')
    tt = Timer('SCHEME2 sorting_scheme1')
    sorting1 = sorting_scheme1(
        recording=training_recording,
        sorting_parameters=Scheme1SortingParameters(
            detect_threshold=sorting_parameters.phase1_detect_threshold,
            detect_sign=sorting_parameters.detect_sign,
            detect_time_radius_msec=sorting_parameters.phase1_detect_time_radius_msec,
            detect_channel_radius=sorting_parameters.phase1_detect_channel_radius,
            snippet_mask_radius=sorting_parameters.snippet_mask_radius,
            snippet_T1=sorting_parameters.snippet_T1,
            snippet_T2=sorting_parameters.snippet_T2,
            npca_per_channel=sorting_parameters.phase1_npca_per_channel,
            npca_per_subdivision=sorting_parameters.phase1_npca_per_subdivision
        )
    )
    assert isinstance(sorting1, si.BaseSorting)
    tt.report()

    # Get the times and labels from the first phase sorting
    tt = Timer('SCHEME2 get_times_labels_from_sorting')
    times, labels = get_times_labels_from_sorting(sorting1)
    times: np.ndarray = times
    labels: np.ndarray = labels
    if len(labels) > 0:
        K = np.max(labels) # number of clusters
        labels = labels + label_offset # used in scheme 3
    else:
        K = 0
    tt.report()

    print('Loading training traces')
    # Load the traces from the training recording
    tt = Timer('SCHEME2 training_recording.get_traces')
    training_traces: np.ndarray = training_recording.get_traces()
    training_snippets = extract_snippets(
        traces=training_traces,
        channel_locations=None,
        mask_radius=None,
        times=times,
        channel_indices=None,
        T1=sorting_parameters.snippet_T1,
        T2=sorting_parameters.snippet_T2
    )
    tt.report()

    print('Training classifier')
    # Train the classifier based on the labels obtained from the first phase sorting
    tt = Timer('SCHEME2 training classifier step 1')
    channel_masks: Dict[int, Union[List[int], None]] = {} # by channel
    for m in range(M):
        if sorting_parameters.snippet_mask_radius is not None:
            x: List[int] = []
            channel_masks[m] = []
            for m2 in range(M):
                if np.linalg.norm(channel_locations[m] - channel_locations[m2]) <= sorting_parameters.snippet_mask_radius:
                    x.append(m2)
            channel_masks[m] = x
        else:
            channel_masks[m] = None
    snippet_classifiers: Dict[int, SnippetClassifier] = {} # by channel
    for m in range(M):
        snippet_classifiers[m] = SnippetClassifier(npca=sorting_parameters.classifier_npca)
        # Add random snippets to classifier with label 0 (a noise cluster for classification)
        uniformly_spread_times = np.floor(np.linspace(sorting_parameters.snippet_T1, training_traces.shape[0] - sorting_parameters.snippet_T2 - 1, sorting_parameters.max_num_snippets_per_training_batch)).astype(np.int32)
        random_snippets = extract_snippets_in_channel_neighborhood(
            traces=training_traces,
            times=uniformly_spread_times,
            neighborhood=channel_masks[m],
            T1=sorting_parameters.snippet_T1,
            T2=sorting_parameters.snippet_T2
        )
        snippet_classifiers[m].add_training_snippets(random_snippets, label=0, offset=0)
    tt.report()

    # Add the snippets from the first phase sorting to the classifier
    print('Adding snippets from phase 1 sorting')
    tt = Timer('SCHEME2 adding snippets from phase 1 sorting')
    for k in range(label_offset + 1, label_offset + K + 1):
        inds0 = np.where(labels == k)[0]
        snippets0 = training_snippets[inds0]
        template0 = np.median(snippets0, axis=0) # T x M
        if sorting_parameters.detect_sign < 0:
            AA = -template0
        elif sorting_parameters.detect_sign > 0: # pragma: no cover
            AA = template0 # pragma: no cover
        else:
            AA = np.abs(template0) # pragma: no cover
        peak_indices_over_channels = np.argmax(AA, axis=0)
        peak_values_over_channels = np.max(AA, axis=0)
        peaks_to_include: List[dict] = []
        for m in range(M):
            if peak_values_over_channels[m] >= sorting_parameters.detect_threshold * 0.4: # should be a parameter
                peaks_to_include.append({
                    'channel': m,
                    'offset': peak_indices_over_channels[m] - sorting_parameters.snippet_T1
                })
        for peak in peaks_to_include:
            m = peak['channel']
            offset = peak['offset']
            if channel_masks[m] is not None:
                snippets0_masked = snippets0[:, :, channel_masks[m]]
            else:
                snippets0_masked = snippets0
            snippet_classifiers[m].add_training_snippets(
                snippets=subsample_snippets(np.roll(snippets0_masked, shift=-offset, axis=1), sorting_parameters.max_num_snippets_per_training_batch),
                label=k,
                offset=offset
            )
    training_snippets = None # Free up memory

    print('Fitting models')
    tt = Timer('SCHEME2 fitting models')
    for m in range(M):
        snippet_classifiers[m].fit()
    tt.report()

    # Now that we have the classifier, we can do the full sorting
    # Iterate over time chunks, detect and classify all spikes, and collect the results

    if sorting_parameters.classification_chunk_sec is None:
        chunk_size = int(math.ceil(100e6 / recording.get_num_channels())) # size of chunks in samples
    else:
        chunk_size = int(math.ceil(sorting_parameters.classification_chunk_sec * recording.sampling_frequency))

    print(f'Chunk size: {chunk_size / recording.sampling_frequency} sec')
    chunks = get_time_chunks(np.int64(recording.get_num_samples()), chunk_size=np.int32(chunk_size), padding=np.int32(1000))
    times_list: list[npt.NDArray[np.int64]] = []
    labels_list: list[npt.NDArray] = []
    labels_reference_list = [] if reference_snippet_classifiers is not None else None
    for i, chunk in enumerate(chunks):
        print(f'Time chunk {i + 1} of {len(chunks)}')
        print('Loading traces')
        tt = Timer('SCHEME2 loading traces')
        traces_chunk: np.ndarray = recording.get_traces(start_frame=int(chunk.start - chunk.padding_left), end_frame=int(chunk.end + chunk.padding_right))
        tt.report()

        print('Detecting spikes')
        tt = Timer('SCHEME2 detecting spikes')
        time_radius = int(math.ceil(sorting_parameters.detect_time_radius_msec / 1000 * sampling_frequency))
        times_chunk, channel_indices_chunk = detect_spikes(
            traces=traces_chunk,
            channel_locations=channel_locations,
            time_radius=time_radius,
            channel_radius=sorting_parameters.detect_channel_radius,
            detect_threshold=sorting_parameters.detect_threshold,
            detect_sign=sorting_parameters.detect_sign,
            margin_left=sorting_parameters.snippet_T1,
            margin_right=sorting_parameters.snippet_T2,
            verbose=False
        )
        print(f'Scheme 2 detected {len(times_chunk)} spikes in chunk {i + 1} of {len(chunks)}')
        tt.report()

        print('Extracting and classifying snippets')
        tt = Timer('SCHEME2 extracting and classifying snippets')
        labels_chunk = np.zeros(len(times_chunk), dtype='int32')
        labels_reference_chunk = np.zeros(len(times_chunk), dtype='int32') if reference_snippet_classifiers is not None else None
        for m in range(M):
            inds = np.where(channel_indices_chunk == m)[0]
            if len(inds) > 0:
                snippets2 = extract_snippets_in_channel_neighborhood(
                    traces=traces_chunk,
                    times=times_chunk[inds],
                    neighborhood=channel_masks[m],
                    T1=sorting_parameters.snippet_T1,
                    T2=sorting_parameters.snippet_T2
                )
                labels_chunk_m, offsets_chunk_m = snippet_classifiers[m].classify_snippets(snippets2)
                if labels_chunk_m is not None and offsets_chunk_m is not None:
                    labels_chunk[inds] = labels_chunk_m
                    times_chunk[inds] = times_chunk[inds] - offsets_chunk_m
                if reference_snippet_classifiers is not None:
                    assert labels_reference_chunk is not None
                    labels_reference_chunk_m, _ = reference_snippet_classifiers[m].classify_snippets(snippets2)
                    if labels_reference_chunk_m is not None:
                        labels_reference_chunk[inds] = labels_reference_chunk_m
        tt.report()

        print('Updating events')
        tt = Timer('SCHEME2 updating events')

        # remove events with label 0
        valid_inds = np.where(labels_chunk > 0)[0]
        times_chunk: npt.NDArray = times_chunk[valid_inds]
        labels_chunk: npt.NDArray = labels_chunk[valid_inds]
        labels_reference_chunk = labels_reference_chunk[valid_inds] if labels_reference_chunk is not None else None

        # now that we offset them we need to re-sort
        sort_inds2 = np.argsort(times_chunk)
        times_chunk: npt.NDArray = times_chunk[sort_inds2]
        labels_chunk: npt.NDArray = labels_chunk[sort_inds2]
        labels_reference_chunk = labels_reference_chunk[sort_inds2] if labels_reference_chunk is not None else None

        print('Removing duplicates')
        new_inds = remove_duplicate_events(times_chunk, labels_chunk, tol=time_radius)
        times_chunk: npt.NDArray = times_chunk[new_inds]
        labels_chunk: npt.NDArray = labels_chunk[new_inds]
        labels_reference_chunk = labels_reference_chunk[new_inds] if labels_reference_chunk is not None else None

        # remove events in the margins
        valid_inds = np.where((chunk.padding_left <= times_chunk) & (times_chunk < chunk.total_size - chunk.padding_right))[0]
        times_chunk: npt.NDArray = times_chunk[valid_inds]
        labels_chunk: npt.NDArray = labels_chunk[valid_inds]
        labels_reference_chunk = labels_reference_chunk[valid_inds] if labels_reference_chunk is not None else None

        # don't forget to cast to int64 add the chunk start time
        times_list.append(
            times_chunk.astype(np.int64) + chunk.start - chunk.padding_left
        )
        labels_list.append(labels_chunk)
        if reference_snippet_classifiers is not None:
            assert labels_reference_list is not None
            labels_reference_list.append(labels_reference_chunk)

        tt.report()

    # Now concatenate the results
    print('Concatenating results')
    tt = Timer('SCHEME2 concatenating results')
    times_concat: npt.NDArray[np.int64] = np.concatenate(times_list)
    labels_concat: npt.NDArray = np.concatenate(labels_list)
    labels_reference_concat = np.concatenate(labels_reference_list) if labels_reference_list is not None else None
    tt.report()

    print('Perorming label mapping')
    tt = Timer('SCHEME2 label mapping')
    if reference_snippet_classifiers is not None:
        assert labels_reference_concat is not None
        mapping = get_labels_to_reference_labels_mapping(labels_concat, labels_reference_concat, label_offset=label_offset)
        print('==== mapping =======================')
        for k1, k2 in mapping.items():
            print(f'{k1} -> {k2}')
        print('====================================')
        for m in range(M):
            snippet_classifiers[m].apply_label_mapping(mapping)
        for k1, k2 in mapping.items():
            labels_concat[labels_concat == k1] = k2
    tt.report()

    # Now create a new sorting object from the times and labels results
    print('Creating sorting object')
    tt = Timer('SCHEME2 creating sorting object')
    # spikeinterface changed function name in version 0.102.2. They also stopped using the dev tag so parsing with packaging is safer
    if version.parse(si.__version__) < version.parse("0.102.2"):
        sorting2 = si.NumpySorting.from_times_labels([times_concat], [labels_concat], sampling_frequency=recording.sampling_frequency)
    else:
        sorting2 = si.NumpySorting.from_samples_and_labels([times_concat], [labels_concat], sampling_frequency=recording.sampling_frequency)
    tt.report()

    if return_snippet_classifiers:
        return sorting2, snippet_classifiers
    else:
        return sorting2

# Here's what this function does:
# 1. For each unit, find the matching unit in the reference (has to be a MUTUAL >0.5 overlap)
# 2. If the matching unit is found, then map the unit to the matching unit
# 3. If the matching unit is not found, then map it to the smallest unused label starting with label_offset+1
def get_labels_to_reference_labels_mapping(labels: npt.NDArray, labels_reference: npt.NDArray, *, label_offset: int) -> Dict[int, int]:
    mapping: Dict[int, Union[int, None]] = {}
    unique_labels = np.sort(np.unique(labels))
    last_used_k = label_offset
    for k in unique_labels:
        mapping[k] = None # initialize to None, if it stays as None, then we will need to create a new label
        inds = np.where(labels == k)[0]
        a = labels_reference[inds]
        k_refs, k_ref_counts = np.unique(a, return_counts=True)
        for ii in range(len(k_refs)):
            if k_ref_counts[ii] > 0.5 * len(inds): # the 0.5 is chosen so we don't map to the same unit twice
                inds_ref = np.where(labels_reference == k_refs[ii])[0] # import to test the other way around
                if k_ref_counts[ii] > 0.5 * len(inds_ref): # mutual overlap
                    mapping[k] = k_refs[ii] # map to the reference label
                    break
        if mapping[k] is None: # if not mapped to reference label, then create a new label
            mapping[k] = last_used_k + 1
            last_used_k = last_used_k + 1

    # do this for typing reasons
    mapping2: Dict[int, int] = {}
    for k, k2 in mapping.items():
        assert k2 is not None
        mapping2[k] = k2
    return mapping2

class TimeChunk:
    def __init__(self, start: np.int64, end: np.int64, padding_left: np.int32, padding_right: np.int32):
        self.start = start
        self.end = end
        self.padding_left = padding_left
        self.padding_right = padding_right
        self.total_size = self.end - self.start + np.int64(padding_left) + np.int64(padding_right)

def get_time_chunks(num_samples: np.int64, chunk_size: np.int32, padding: np.int32, max_num_blocks: Union[int, None] = None) -> List[TimeChunk]:
    """Get time chunks
    Inputs:
        num_samples: number of samples in the recording
        chunk_size: size of each chunk in samples
        padding: padding on each side of the chunk in samples
        max_num_blocks: maximum number of blocks to return or None for no limit - if not None, spacing between blocks will be added
    Returns:
        chunks: list of TimeChunk objects
    """
    spacing = 0
    if max_num_blocks is not None:
        if chunk_size * max_num_blocks < num_samples:
            spacing = np.int64(np.floor((num_samples - chunk_size * max_num_blocks) / (max_num_blocks - 1)))
    chunks = []
    start = np.int64(0)
    while start < num_samples:
        end = np.int64(start) + np.int64(chunk_size)
        if end > num_samples:
            end = num_samples
        if end + chunk_size / 2 > num_samples:
            # if the next chunk will be too small, then just make this the last chunk
            end = num_samples
        padding_left = np.minimum(padding, start)
        padding_right = np.minimum(padding, num_samples - end)
        chunks.append(TimeChunk(start=start, end=end, padding_left=padding_left, padding_right=padding_right))
        start = end
        if spacing > 0:
            start = start + spacing
        if max_num_blocks is not None and len(chunks) >= max_num_blocks:
            break
    return chunks

def subsample_snippets(snippets: npt.NDArray[np.float32], max_num: int) -> np.ndarray:
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
