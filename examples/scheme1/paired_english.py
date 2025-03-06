import time
from tempfile import TemporaryDirectory
import numpy as np
import spikeinterface as si
import spikeinterface.preprocessing as spre
import spikeinterface.comparison as sc
import mountainsort5 as ms5
from mountainsort5.util import create_cached_recording
import spikeforest as sf

def main():
    paired_english_uri = 'sha1://dfb1fd134bfc209ece21fd5f8eefa992f49e8962?paired-english-spikeforest-recordings.json'

    recordings = sf.load_spikeforest_recordings(paired_english_uri)

    # list recordings
    # for rec in recordings:
    #     print(f'{rec.study_name}/{rec.recording_name} {rec.num_channels} channels; {rec.duration_sec} sec')

    # select recording
    # rec = recordings[1] # m57_191105_160026 32 channels; 343.4533333333333 sec - amplitude too low
    rec = recordings[13] # m15_190315_152315_cell1 32 channels; 669.3997 sec # classic example of need for final alignment merge step

    print(f'{rec.study_name}/{rec.recording_name} {rec.num_channels} channels; {rec.duration_sec} sec')

    # this will download the recording/sorting_true from kachery
    print('Loading recording and sorting_true')
    recording = rec.get_recording_extractor()
    sorting_true = rec.get_sorting_true_extractor()

    channel_locations = recording.get_channel_locations()
    for m in range(channel_locations.shape[0]):
        print(f'Channel {recording.channel_ids[m]}: {channel_locations[m, 0]} {channel_locations[m, 1]}')

    timer = time.time()

    # lazy preprocessing
    recording_filtered = spre.bandpass_filter(recording, freq_min=300, freq_max=6000, dtype=np.float32)
    recording_preprocessed: si.BaseRecording = spre.whiten(recording_filtered)

    with TemporaryDirectory() as tmpdir:
        # cache the recording to a temporary directory for efficient reading
        recording_cached = create_cached_recording(recording_preprocessed, folder=tmpdir)

        # sorting
        print('Starting MountainSort5')
        sorting = ms5.sorting_scheme1(
            recording_cached,
            sorting_parameters=ms5.Scheme1SortingParameters(
                detect_channel_radius=50,
                snippet_mask_radius=100
            )
        )
        assert isinstance(sorting, si.BaseSorting)

    elapsed_sec = time.time() - timer
    duration_sec = recording.get_total_duration()
    print(f'Elapsed time for sorting: {elapsed_sec:.2f} sec -- x{(duration_sec / elapsed_sec):.2f} speed compared with real time for {recording.get_num_channels()} channels')

    print('Comparing with truth')
    comparison: sc.GroundTruthComparison = sc.compare_sorter_to_ground_truth(gt_sorting=sorting_true, tested_sorting=sorting)
    print(comparison.get_performance())

if __name__ == '__main__':
    main()
