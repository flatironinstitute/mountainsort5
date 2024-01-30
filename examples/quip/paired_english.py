import mountainsort5.quip as quip
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
    # sorting_true = rec.get_sorting_true_extractor()

    output = quip.estimate_units(
        recording,
    )

    print(output)


if __name__ == "__main__":
    main()
