# Scheme 3

This scheme is designed to handle long recordings that may involve spike waveform drift. The recording is divided into blocks, and each is spike sorted using scheme 2. Then the snippet classifiers are used to associate matching units between blocks.

## Usage

```python
import spikeinterface as si
import spikeinterface.preprocessing as spre
import mountainsort5 as ms5

recording = ... # load your recording using SpikeInterface

# Make sure the recording is preprocessed appropriately
# lazy preprocessing
recording_filtered = spre.bandpass_filter(recording, freq_min=300, freq_max=6000)
recording_preprocessed: si.BaseRecording = spre.whiten(recording_filtered, dtype='float32')

# use scheme 3
sorting = ms5.sorting_scheme3(
    recording=recording_preprocessed,
    sorting_parameters=ms5.Scheme3SortingParameters(
        block_sorting_parameters=ms5.Scheme2SortingParameters(...),
        block_duration_sec=60 * 10
    )
)
```

## Parameters

**block_sorting_parameters**

The sorting parameters to use for each block. See [scheme 2](./scheme2.md) for details.

**block_duration_sec**

The duration of each block, in seconds.

## Algorithm

The recording is subdivided into blocks, and each block is spike sorted using scheme 2. Let $B_n$ represent the $n^{th}$ block. For each $n=0,1,2,\dots$ the clusters from $B_{n+1}$ are then associated with the clusters from block $B_n$ according to the following rule. The events in $B_{n+1}$ are classified using both the classifiers from $B_n$ and block $B_{n+1}$. If the number of events classified as cluster $x$ of $B_n$ and cluster $y$ of $B_{n+1}$ is greater than half the total number of events in cluster $x$ and is also greater than half the total number of events in cluster $y$, then $y$ and $x$ are considered to be the same cluster. Note that some clusters of $B_{n+1}$ will not be associated with any clusters in the previous block.

For more details, see the documented source code in [sorting_scheme3.py](../mountainsort5/schemes/sorting_scheme3.py).