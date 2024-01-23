import os
import spikeinterface.full as si


def save_binary_recording(recording: si.BaseRecording, folder: str, n_jobs: int = 1):
    # Check whether the recording is already saved
    if os.path.exists(folder):
        print("Recording is already saved at {}".format(folder))
        print(
            "If you would like to load that recording, please use the load_binary_recording function"
        )
        return

    recording.save(folder=folder, format="binary", progress_bar=True, n_jobs=n_jobs)
    return recording


def load_binary_recording(folder: str):
    # Check whether the recording is already saved
    if not os.path.exists(folder):
        print("No recording saved at {}".format(folder))
        print(
            "If you would like to save that recording, please use the save_binary_recording function"
        )
        return

    recording = si.load_extractor(folder)
    return recording
