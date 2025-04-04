from typing import Dict, Union
from packaging import version
import numpy as np
import numpy.typing as npt
import spikeinterface as si
from .Scheme3SortingParameters import Scheme3SortingParameters
from .sorting_scheme2 import get_time_chunks
from .sorting_scheme2 import sorting_scheme2
from ..core.get_block_recording_for_scheme3 import get_block_recording_for_scheme3
from ..core.SnippetClassifier import SnippetClassifier
from ..core.get_times_labels_from_sorting import get_times_labels_from_sorting


def sorting_scheme3(
    recording: si.BaseRecording, *,
    sorting_parameters: Scheme3SortingParameters
) -> si.BaseSorting:
    """MountainSort 5 sorting scheme 3

    Args:
        recording (si.BaseRecording): SpikeInterface recording object
        sorting_parameters (Scheme3SortingParameters): Sorting parameters

    Returns:
        si.BaseSorting: SpikeInterface sorting object
    """

    ###################################################################
    # Handle multi-segment recordings
    if recording.get_num_segments() > 1:
        print('Recording has multiple segments. Joining segments for sorting...')
        recording_joined = si.concatenate_recordings(recording_list=[recording])
        sorting_joined = sorting_scheme3(recording_joined, sorting_parameters=sorting_parameters)
        print('Splitting sorting into segments to match original multisegment recording...')
        sorting = si.split_sorting(sorting_joined, recording_joined)
        return sorting
    ###################################################################

    M = recording.get_num_channels()
    N = recording.get_num_frames()
    sampling_frequency = recording.sampling_frequency
    channel_locations = recording.get_channel_locations()

    sorting_parameters.check_valid(M=M, N=N, sampling_frequency=sampling_frequency, channel_locations=channel_locations)

    block_size = int(sorting_parameters.block_duration_sec * sampling_frequency) # size of chunks in samples
    blocks = get_time_chunks(np.int64(recording.get_num_samples()), chunk_size=np.int32(block_size), padding=np.int32(1000))

    times_list: list[npt.NDArray[np.int64]] = []
    labels_list: list[npt.NDArray] = []
    last_label_used = 0
    previous_snippet_classifiers: Union[Dict[int, SnippetClassifier], None] = None
    for i, chunk in enumerate(blocks):
        print('')
        print('=============================================')
        print(f'Processing block {i + 1} of {len(blocks)}...')
        subrecording = get_block_recording_for_scheme3(recording=recording, start_frame=int(chunk.start) - int(chunk.padding_left), end_frame=int(chunk.end) + int(chunk.padding_right))
        result = sorting_scheme2(
            subrecording,
            sorting_parameters=sorting_parameters.block_sorting_parameters,
            return_snippet_classifiers=True,
            reference_snippet_classifiers=previous_snippet_classifiers,
            label_offset=last_label_used
        )
        assert isinstance(result, tuple)
        subsorting, snippet_classifiers = result
        previous_snippet_classifiers = snippet_classifiers
        times0, labels0 = get_times_labels_from_sorting(subsorting)
        valid_inds = np.where((times0 >= chunk.padding_left) & (times0 < chunk.padding_left + (chunk.end - chunk.start)))[0]
        times0: npt.NDArray[np.int64] = times0[valid_inds]
        labels0: npt.NDArray = labels0[valid_inds]
        times0 = times0.astype(np.int64) + chunk.start - np.int64(chunk.padding_left)
        if len(labels0) > 0:
            last_label_used = max(last_label_used, np.max(labels0.astype(int)))
        times_list.append(times0)
        labels_list.append(labels0)

    times_concat = np.concatenate(times_list)
    labels_concat = np.concatenate(labels_list)

    # Now create a new sorting object from the times and labels results
    # spikeinterface changed function name in version 0.102.2. They also stopped using the dev tag so parsing with packaging is safer
    if version.parse(si.__version__) < version.parse("0.102.2"):
        sorting2 = si.NumpySorting.from_times_labels([times_concat], [labels_concat], sampling_frequency=recording.sampling_frequency)
    else:
        sorting2 = si.NumpySorting.from_samples_and_labels([times_concat], [labels_concat], sampling_frequency=recording.sampling_frequency)

    return sorting2
