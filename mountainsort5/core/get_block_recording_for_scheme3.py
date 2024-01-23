from typing import Union
import spikeinterface as si


def get_block_recording_for_scheme3(
    recording: si.BaseRecording, *,
    start_frame: int,
    end_frame: int
) -> si.BaseRecording:
    return BlockRecording(
        recording=recording,
        start_frame=start_frame,
        end_frame=end_frame
    )

class BlockRecording(si.BaseRecording):
    def __init__(self, recording: si.BaseRecording, start_frame: int, end_frame: int):
        sampling_frequency = recording.get_sampling_frequency()
        dtype = recording.get_dtype()
        channel_ids = recording.get_channel_ids()

        si.BaseRecording.__init__(self, sampling_frequency, [ch for ch in channel_ids], dtype)

        self.start_frame = start_frame
        self.end_frame = end_frame

        self.set_channel_locations(recording.get_channel_locations())
        self.is_dumpable = False

        self.add_recording_segment(
            BlockRecordingSegment(recording=recording, start_frame=start_frame, end_frame=end_frame)
        )

        self._kwargs = {'recording': recording, 'start_frame': start_frame, 'end_frame': end_frame}

class BlockRecordingSegment(si.BaseRecordingSegment):
    def __init__(self, recording: si.BaseRecording, start_frame: int, end_frame: int):
        si.BaseRecordingSegment.__init__(
            self,
            sampling_frequency=recording.get_sampling_frequency(),
            t_start=0
        )
        self._recording = recording
        self._start_frame = start_frame
        self._end_frame = end_frame

    def get_num_samples(self):
        return self._end_frame - self._start_frame

    def get_traces(self, start_frame: Union[int, None] = None,
        end_frame: Union[int, None] = None,
        channel_indices=None
    ):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.get_num_samples()

        # Get the traces from the parent recording
        return self._recording._recording_segments[0].get_traces(
            start_frame=start_frame + self._start_frame,
            end_frame=end_frame + self._start_frame,
            channel_indices=channel_indices
        )
