import os
import spikeinterface as si


def create_cached_recording(
    recording: si.BaseRecording,
    *,
    folder: str,
    n_jobs: int = 1,
    chunk_duration: str = '1s',
) -> si.BaseRecording:
    if not os.path.exists(folder):
        os.mkdir(folder)
    fname = f'{folder}/recording.dat'
    if recording.get_num_segments() != 1:
        raise NotImplementedError("Can only write recordings with a single segment")

    si.BinaryRecordingExtractor.write_recording(
        recording=recording,
        file_paths=[fname],
        dtype='float32',
        n_jobs=n_jobs,
        chunk_duration=chunk_duration,
    )
    ret = si.BinaryRecordingExtractor(
        file_paths=[fname],
        sampling_frequency=recording.get_sampling_frequency(),
        channel_ids=recording.get_channel_ids(),
        num_channels=recording.get_num_channels(),
        dtype='float32'
    )
    ret.set_channel_locations(recording.get_channel_locations())
    return ret
