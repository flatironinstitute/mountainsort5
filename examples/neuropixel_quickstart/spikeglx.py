# This is an example script provided by Joshua Melander, March 2024

# Imports
import mountainsort5 as ms5
from mountainsort5.util import load_binary_recording, save_binary_recording
import spikeinterface.full as si

# Constants
scheme = 2
save_rec = True
sort = True
extract_waveforms = True

# Path definitions

imec_path = '/path/to/spikeglx/data_g0'

rec_path = '/path/to/sorting/rec'
sorter_path = '/path/to/sorting/sorter_output'
waveforms_path = '/path/to/sorting/waveforms'

# Preprocess recording and save
if save_rec:
    # Preprocess
    raw_recording = si.read_spikeglx(imec_path, stream_id="imec0.ap")
    recording = si.phase_shift(raw_recording)

    # recording = recording.frame_slice(start_frame=2221*30000, end_frame=recording.get_num_frames())

    recording = si.bandpass_filter(recording, freq_min=500, freq_max=12000)
    bad_channel_ids, channel_labels = si.detect_bad_channels(recording)
    recording = recording.remove_channels(bad_channel_ids)

    print("Bad channels: ", bad_channel_ids)
    recording = si.whiten(recording, dtype="float32")

    # Save recording
    print("    + Preprocessing complete...")
    save_binary_recording(recording, rec_path, n_jobs=64)
    del recording
    print("    + Recording saved...")


# Load cached recording no-matter-what
cached_recording = load_binary_recording(rec_path)
assert isinstance(cached_recording, si.BaseRecording)
print("    + Cached recording loaded...")

if sort:
    sorting_params = {}

    sorting_params["max_num_snippets_per_training_batch"] = 1000
    sorting_params["snippet_mask_radius"] = 60
    sorting_params["phase1_npca_per_channel"] = 3
    sorting_params["phase1_npca_per_subdivision"] = 10
    sorting_params["classifier_npca"] = 10
    sorting_params["detect_channel_radius"] = 60
    sorting_params["phase1_detect_channel_radius"] = 60
    sorting_params["training_recording_sampling_mode"] = "uniform"
    sorting_params["training_duration_sec"] = 350
    sorting_params["phase1_detect_threshold"] = 5.5
    sorting_params["detect_threshold"] = 5.25
    sorting_params["snippet_T1"] = 15
    sorting_params["snippet_T2"] = 40
    sorting_params["detect_sign"] = 0
    sorting_params["phase1_detect_time_radius_msec"] = 0.5
    sorting_params["detect_time_radius_msec"] = 0.5
    sorting_params["classification_chunk_sec"] = 100

    if scheme == 2:
        sorting = ms5.sorting_scheme2(
            recording=cached_recording,
            sorting_parameters=ms5.Scheme2SortingParameters(**sorting_params),
        )
        assert isinstance(sorting, si.BaseSorting)

    elif scheme == 3:
        sorting = ms5.sorting_scheme3(
            recording=cached_recording,
            sorting_parameters=ms5.Scheme3SortingParameters(
                block_sorting_parameters=ms5.Scheme2SortingParameters(**sorting_params),
                block_duration_sec=60 * 5,
            ),
        )
        assert isinstance(sorting, si.BaseSorting)
    else:
        raise ValueError(f"Scheme not supported: {scheme}")

    print("    + Sorting completed...")
    # max_num_snippets_per_training_batch=1000
    sorting.save_to_folder(sorter_path)

    print("    + Sorting saved...")


# Extract waveforms
if extract_waveforms:
    print("    + Sorting loaded...")
    sorting = si.read_numpy_sorting_folder(sorter_path)
    assert isinstance(sorting, si.BaseSorting)

    job_kwargs = dict(n_jobs=64, chunk_duration="5s", progress_bar=True)
    cached_recording.annotate(is_filtered=True)

    waveforms = si.extract_waveforms(
        cached_recording,
        sorting,
        folder=waveforms_path,
        max_spikes_per_unit=1500,
        dtype="float32",
        ms_before=0.5,
        ms_after=1.5,
    )

    locations = si.compute_unit_locations(waveforms)

# Load the results
waveforms = si.load_waveforms(waveforms_path)
sorting = waveforms.sorting
noise = si.compute_noise_levels(waveforms)
print("    + Waveforms extracted...")
