# Scheme 1

This is the simplest of the MountainSort sorting schemes and is useful for quick tests. The entire recording is loaded into memory, and clustering is performed in two passes. In general, scheme 2 should be used intead since it has better handling of events that overlap in time, and works with larger datasets on limited RAM systems. Nevertheless, scheme 1 can be useful for testing and debugging, and is used as the first phase of scheme 2.

## Usage

```python
import numpy as np
import spikeinterface as si
import spikeinterface.preprocessing as spre
import mountainsort5 as ms5
from mountainsort5.util import TemporaryDirectory, create_cached_recording

recording = ... # load your recording using SpikeInterface

# Make sure the recording is preprocessed appropriately

# Note that if the recording traces are of float type, you may need to scale
# it to a reasonable voltage range in order for whitening to work properly
# recording = spre.scale(recording, gain=...)

# lazy preprocessing
recording_filtered = spre.bandpass_filter(recording, freq_min=300, freq_max=6000, dtype=np.float32)
recording_preprocessed: si.BaseRecording = spre.whiten(recording_filtered)

with TemporaryDirectory() as tmpdir:
    # cache the recording to a temporary directory for efficient reading
    recording_cached = create_cached_recording(recording_preprocessed, folder=tmpdir)

    # use scheme 1
    sorting = ms5.sorting_scheme1(
        recording=recording_cached,
        sorting_parameters=ms5.Scheme1SortingParameters(
            detect_sign=-1,
            # other parameters...
        )
    )

# Now you have a sorting object that you can save to disk or use for further analysis
```

## Parameters

**detect_sign**

The `detect_sign` parameter determines whether to detect positive-going or negative-going spikes. The default value is -1, which is appropriate for most recordings. The value 1 is appropriate for recordings in which the spikes are positive-going. You can also use a value of 0 to detect both positive and negative-going spikes.

**detect_threshold**

The detection threshold (`detect_threshold`) is one of the more difficult parameters to optimize. If it is set too low, then you may end up with false merges because clusters will overlap for spikes with lower amplitudes. This is especially relevant for bursting cells that exhibit a range of spike amplitudes. On the other hand, if this parameter is set too high, then you'll miss some low-amplitude events. I would recommend leaving this at the default value (5.5 for whitened data) since this seems to perform well on a variety of examples I have tried. You may think that 5.5 sounds high, but note that detection is performed on the ZCA-whitened data for which the spike peaks tend to be better separated from noise compared with the pre-whitened bandpass-filtered data.

**detect_time_radius_msec**

This parameter determines the minimum allowable time interval between detected spikes in the same spatial region (see `detect_channel_radius`). This is useful for avoiding double-detection of individual spikes. The default value is 0.5 msec for scheme 1. The first phase of scheme 2 uses scheme 1, and in that setting the default value is 1.5 msec in order to avoid including overlapping spikes in the intitial clustering pass. More details on this can be found in the [documentation for scheme 2](./scheme2.md).

**detect_channel_radius**

This parameter is a companion to `detect_time_radius_msec`. It determines the spatial radius (in units of the channel locations) within which the detection time radius applies. In general, a spike is detected at timepoint `t` on channel `m` if it is the largest amplitude event within a time window size `detect_time_radius_msec` around `t` and within a spatial neighborhood around channel `m` defined by `detect_channel_radius`. The default is `None` which effectively gives an infinite spatial radius.

**snippet_T1** and **snippet_T2**

These parameters determine the time duration (units of samples) of the extracted spike snippets. The total duration of the snippet is `snippet_T1 + snippet_T2` samples, with `snippet_T1` samples before the peak and `snippet_T2` samples after the peak.

**snippet_mask_radius**

If specified, this parameter determines the spatial radius (in units of the channel locations) of a mask that is applied to the extracted spike snippets. For spikes centered at channel `m` the mask includes all channels within `snippet_mask_radius` of channel $m$, and the signal outside of the mask region is set to zero. This is useful for removing signal from time-overlapping, spatially-separated events during the clustering phase.

**npca_per_channel**

This parameter affects dimension reduction prior to clustering. If the dataset has `M` channels, then `M * npca_per_channel` PCA features are computed for each spike snippet. The default value is 3, which is appropriate for most recordings.

**npca_per_subdivision**

MountainSort utilizes a subdivision method for clustering. After extracting spike snippets from the preprocessed traces, the data undergo dimension reduction through PCA before clustering. Initially, `npca_per_subdivision` PCA features are computed, followed by Isosplit clustering. If more than one cluster is detected, the dataset is split into two subdivisions, and feature extraction is performed again within each subdivision separately, followed by Isosplit clustering to subdivide the clusters further. This process is repeated recursively until each leaf subdivision returns a single cluster. At each stage, the same number of PCA components (npca_per_subdivision) is used. Recomputing features on each subdivision offers an advantage, as it allows the refined components to capture features that can better differentiate between clusters within the same overall feature space region.

## Algorithm

The algorithm is as follows:

* Detect spikes in the preprocessed traces
* Extract spike snippets
* Cluster snippets using isosplit6 and the subdivision method (includes PCA dimension reduction)
* Use cluster templates to align snippets for a second pass of clustering
* Cluster aligned snippets using isosplit6 and the subdivision method in a second pass

The subdivision method of clustering is described above.

[Learn more about Isosplit](https://github.com/magland/isosplit6)

