import spikeinterface.extractors as se
import mountainsort5.quip as quip
import spikeinterface as si


def main():
    recording, sorting_true = se.toy_example(duration=60 * 2, num_channels=8, num_units=16, sampling_frequency=30000, num_segments=1, seed=0)  # type: ignore
    recording: si.BaseRecording = recording
    sorting_true: si.BaseSorting = sorting_true

    output = quip.estimate_units(
        recording,
    )

    print(output)


if __name__ == "__main__":
    main()
