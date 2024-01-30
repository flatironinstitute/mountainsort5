# Quip - Quick, preliminary neural unit estimator

Quip can be used to quickly estimate the number of units in a recording, together with estimated firing rates, peak channel IDs and peak SNRs.

Important note: Unlike with mountainsort5 spike sorting, you do not need to preprocess the recording prior to running quip.estimate_units().

## Usage

```python
import spikeinterface.extractors as se
import mountainsort5.quip as quip
import spikeinterface as si


recording, sorting_true = se.toy_example(duration=60 * 2, num_channels=8, num_units=16, sampling_frequency=30000, num_segments=1, seed=0)  # type: ignore
recording: si.BaseRecording = recording
sorting_true: si.BaseSorting = sorting_true

output = quip.estimate_units(
    recording
)

print(output)
```