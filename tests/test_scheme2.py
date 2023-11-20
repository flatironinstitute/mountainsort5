import time
from tempfile import TemporaryDirectory
import numpy as np
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import mountainsort5 as ms5
from mountainsort5.util import create_cached_recording


def test_scheme2():
    recording, sorting_true = se.toy_example(
        duration=40,
        num_channels=8,
        num_units=16,
        sampling_frequency=30000,
        num_segments=1,
        seed=0
    ) # type: ignore
    recording: si.BaseRecording = recording
    sorting_true: si.BaseSorting = sorting_true

    # lazy preprocessing
    recording_filtered = spre.bandpass_filter(recording, freq_min=300, freq_max=6000, dtype=np.float32)
    recording_preprocessed: si.BaseRecording = spre.whiten(recording_filtered)

    with TemporaryDirectory() as tmpdir:
        # cache the recording to a temporary directory for efficient reading
        recording_cached = create_cached_recording(recording_preprocessed, folder=tmpdir)

        # sorting
        print('Starting MountainSort5 (sorting1)')
        timer = time.time()
        result = ms5.sorting_scheme2(
            recording_cached,
            sorting_parameters=ms5.Scheme2SortingParameters(
                phase1_detect_channel_radius=150,
                detect_channel_radius=50,
                max_num_snippets_per_training_batch=3, # for improving test coverage
                snippet_mask_radius=150,
                training_duration_sec=15
            ),
            return_snippet_classifiers=True # for coverage
        )
        assert isinstance(result, tuple)
        sorting1, classifer1 = result

    elapsed_sec = time.time() - timer
    duration_sec = recording.get_total_duration()
    print(f'Elapsed time for sorting: {elapsed_sec:.2f} sec -- x{(duration_sec / elapsed_sec):.2f} speed compared with real time for {recording.get_num_channels()} channels')

    print('Starting MountainSort5 (sorting2)')
    timer = time.time()
    sorting2 = ms5.sorting_scheme2( # noqa
        recording_preprocessed,
        sorting_parameters=ms5.Scheme2SortingParameters(
            phase1_detect_channel_radius=150,
            detect_channel_radius=50,
            training_duration_sec=25,
            training_recording_sampling_mode='uniform'
        )
    )
    elapsed_sec = time.time() - timer
    duration_sec = recording.get_total_duration()
    print(f'Elapsed time for sorting: {elapsed_sec:.2f} sec -- x{(duration_sec / elapsed_sec):.2f} speed compared with real time for {recording.get_num_channels()} channels')

    # commenting out because this step takes a while
    # print('Comparing with truth')
    # comparison = sc.compare_sorter_to_ground_truth(gt_sorting=sorting_true, tested_sorting=sorting)
    # print(comparison.get_performance())

# for coverage
def test_scheme2_single_segment():
    recording, sorting_true = se.toy_example(
        duration=4,
        num_channels=4,
        num_units=4,
        sampling_frequency=30000,
        num_segments=1,
        seed=0
    ) # type: ignore
    recording: si.BaseRecording = recording
    sorting_true: si.BaseSorting = sorting_true

    sorting1 = ms5.sorting_scheme2( # noqa
        recording,
        sorting_parameters=ms5.Scheme2SortingParameters(
            phase1_detect_channel_radius=150,
            detect_channel_radius=50,
            max_num_snippets_per_training_batch=3, # for improving test coverage
            snippet_mask_radius=150,
            training_duration_sec=15,
            classifier_npca=4
        )
    )
