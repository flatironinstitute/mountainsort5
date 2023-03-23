import time
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.comparison as sc
import mountainsort5 as ms5

def main():
    recording, sorting_true = se.toy_example(duration=60 * 5, num_channels=16, num_units=32, sampling_frequency=30000, num_segments=1, seed=0)

    timer = time.time()

    # lazy preprocessing
    recording_filtered = spre.bandpass_filter(recording, freq_min=300, freq_max=6000)
    recording_preprocessed = spre.whiten(recording_filtered)

    # sorting
    print('Starting MountainSort5')
    sorting = ms5.sorting_scheme1(recording_preprocessed)
    
    elapsed_sec = time.time() - timer
    duration_sec = recording.get_total_duration()
    print(f'Elapsed time for sorting: {elapsed_sec:.2f} sec -- x{(duration_sec / elapsed_sec):.2f} speed compared with real time for {recording.get_num_channels()} channels')

    print('Comparing with truth')
    comparison = sc.compare_sorter_to_ground_truth(gt_sorting=sorting_true, tested_sorting=sorting)
    print(comparison.get_performance())

if __name__ == '__main__':
    main()