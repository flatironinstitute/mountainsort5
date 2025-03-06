import time
from tempfile import TemporaryDirectory
import numpy as np
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.comparison as sc
import mountainsort5 as ms5
from mountainsort5.util import create_cached_recording
import spikeinterface as si

def main():
    recording, sorting_true = se.toy_example(
        duration=60 * 30,
        num_channels=8,
        num_units=16,
        sampling_frequency=30000,
        num_segments=1,
        seed=0
    ) # type: ignore
    recording: si.BaseRecording = recording
    sorting_true: si.BaseSorting = sorting_true

    timer = time.time()

    # lazy preprocessing
    recording_filtered = spre.bandpass_filter(recording, freq_min=300, freq_max=6000, dtype=np.float32)
    recording_preprocessed: si.BaseRecording = spre.whiten(recording_filtered)

    with TemporaryDirectory() as tmpdir:
        # cache the recording to a temporary directory for efficient reading
        recording_cached = create_cached_recording(recording_preprocessed, folder=tmpdir)

        # sorting
        print('Starting MountainSort5 (scheme 3)')
        sorting = ms5.sorting_scheme3(
            recording_cached,
            sorting_parameters=ms5.Scheme3SortingParameters(
                block_sorting_parameters=ms5.Scheme2SortingParameters(
                    phase1_detect_channel_radius=150,
                    detect_channel_radius=50,
                    training_duration_sec=60
                ),
                block_duration_sec=60 * 5
            )
        )

    elapsed_sec = time.time() - timer
    duration_sec = recording.get_total_duration()
    print(f'Elapsed time for sorting: {elapsed_sec:.2f} sec -- x{(duration_sec / elapsed_sec):.2f} speed compared with real time for {recording.get_num_channels()} channels')

    print('Comparing with truth')
    comparison = sc.compare_sorter_to_ground_truth(gt_sorting=sorting_true, tested_sorting=sorting)
    print(comparison.get_performance())


if __name__ == '__main__':
    main()
