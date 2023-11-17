import os
import time
import spikeinterface as si
import spikeinterface.preprocessing as spre
import spikeinterface.comparison as sc
import mountainsort5 as ms5
import spikeforest as sf
from generate_visualization_output import generate_visualization_output

def main():
    paired_kampff_uri = 'sha1://b8b571d001f9a531040e79165e8f492d758ec5e0?paired-kampff-spikeforest-recordings.json'

    recordings = sf.load_spikeforest_recordings(paired_kampff_uri)

    # list recordings
    # for rec in recordings:
    #     print(f'{rec.study_name}/{rec.recording_name} {rec.num_channels} channels; {rec.duration_sec} sec')

    # select recording
    rec = recordings[1] # 2015_09_03_Pair_9_0A 32 channels; 593.248 sec # example of bursting with lower amplitude spikes after the initial spike

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
    recording_filtered = spre.bandpass_filter(recording, freq_min=300, freq_max=6000)
    recording_preprocessed: si.BaseRecording = spre.whiten(recording_filtered, dtype='float32')

    # sorting
    print('Starting MountainSort5')
    sorting = ms5.sorting_scheme1(
        recording_preprocessed,
        sorting_parameters=ms5.Scheme1SortingParameters(
            detect_channel_radius=100,
            snippet_mask_radius=100
        )
    )

    elapsed_sec = time.time() - timer
    duration_sec = recording.get_total_duration()
    print(f'Elapsed time for sorting: {elapsed_sec:.2f} sec -- x{(duration_sec / elapsed_sec):.2f} speed compared with real time for {recording.get_num_channels()} channels')

    print('Comparing with truth')
    comparison: sc.GroundTruthComparison = sc.compare_sorter_to_ground_truth(gt_sorting=sorting_true, tested_sorting=sorting)
    print(comparison.get_performance())

    #######################################################################

    if os.getenv('GENERATE_VISUALIZATION_OUTPUT') == '1':
        generate_visualization_output(rec=rec, recording_preprocessed=recording_preprocessed, sorting=sorting, sorting_true=sorting_true)

if __name__ == '__main__':
    main()
