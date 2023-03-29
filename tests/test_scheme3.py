import time
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.comparison as sc
import mountainsort5 as ms5


def test_scheme3():
    recording, sorting_true = se.toy_example(
        duration=20,
        num_channels=8,
        num_units=16,
        sampling_frequency=30000,
        num_segments=2,
        seed=0
    )

    timer = time.time()

    # lazy preprocessing
    recording_filtered = spre.bandpass_filter(recording, freq_min=300, freq_max=6000)
    recording_preprocessed: si.BaseRecording = spre.whiten(recording_filtered)

    # sorting
    print('Starting MountainSort5')
    sorting = ms5.sorting_scheme3(
        recording_preprocessed,
        sorting_parameters=ms5.Scheme3SortingParameters(
            block_sorting_parameters=ms5.Scheme2SortingParameters(
                phase1_detect_channel_radius=150,
                detect_channel_radius=50
            ),
            block_duration_sec=3
        )
    )
    
    elapsed_sec = time.time() - timer
    duration_sec = recording.get_total_duration()
    print(f'Elapsed time for sorting: {elapsed_sec:.2f} sec -- x{(duration_sec / elapsed_sec):.2f} speed compared with real time for {recording.get_num_channels()} channels')

    # commenting out because this step takes a while
    # print('Comparing with truth')
    # comparison = sc.compare_sorter_to_ground_truth(gt_sorting=sorting_true, tested_sorting=sorting)
    # print(comparison.get_performance())