import spikeinterface as si
import numpy as np
from typing import Literal


def get_sampled_recording_for_training(
    recording: si.BaseRecording, *,
    training_duration_sec: float,
    mode: Literal['initial', 'uniform'] = 'initial'
) -> si.BaseRecording:
    """Get a sampled recording for the purpose of training

    Args:
        recording (si.BaseRecording): SpikeInterface recording object
        training_duration_sec (float): Duration of the training in seconds
        mode (str): 'initial' or 'uniform'

    Returns:
        si.BaseRecording: SpikeInterface recording object
    """
    if training_duration_sec * recording.sampling_frequency >= recording.get_num_frames():
        # if the training duration is longer than the recording, then just use the entire recording
        return recording
    if mode == 'initial':
        traces = recording.get_traces(start_frame=0, end_frame=int(training_duration_sec * recording.sampling_frequency))
    elif mode == 'uniform':
        # use chunks of 10 seconds
        chunk_size = int(recording.sampling_frequency * min(10, training_duration_sec))
        # the number of chunks depends on the training duration
        num_chunks = int(np.ceil(training_duration_sec * recording.sampling_frequency / chunk_size))
        chunk_sizes = [chunk_size for i in range(num_chunks)]
        chunk_sizes[-1] = int(training_duration_sec * recording.sampling_frequency - (num_chunks - 1) * chunk_size)
        if num_chunks == 1:
            # if only 1 chunk, then just use the initial chunk
            traces = recording.get_traces(start_frame=0, end_frame=int(training_duration_sec * recording.sampling_frequency))
        else:
            # the spacing between the chunks
            spacing = int((recording.get_num_frames() - np.sum(chunk_sizes)) / (num_chunks - 1))
            traces_list: list[np.ndarray] = []
            tt = 0
            for i in range(num_chunks):
                start_frame = tt
                end_frame = int(start_frame + chunk_sizes[i])
                traces_list.append(recording.get_traces(start_frame=start_frame, end_frame=end_frame))
                tt += int(chunk_sizes[i] + spacing)
            traces = np.concatenate(traces_list, axis=0)
    else:
        raise Exception('Invalid mode: ' + mode) # pragma: no cover

    rec = si.NumpyRecording(
        traces_list=[traces],
        sampling_frequency=recording.sampling_frequency,
        channel_ids=recording.get_channel_ids()
    )
    rec.set_channel_locations(recording.get_channel_locations())
    return rec
