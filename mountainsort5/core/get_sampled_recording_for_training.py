import spikeinterface as si
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
    if mode == 'initial':
        traces = recording.get_traces(start_frame=0, end_frame=int(training_duration_sec * recording.sampling_frequency))
    elif mode == 'uniform':
        raise Exception('Not implemented yet')
    else:
        raise Exception('Invalid mode: ' + mode)
    
    rec = si.NumpyRecording(
        traces_list=[traces],
        sampling_frequency=recording.sampling_frequency,
        channel_ids=recording.get_channel_ids()
    )
    rec.set_channel_locations(recording.get_channel_locations())
    return rec