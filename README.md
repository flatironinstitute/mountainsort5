# MountainSort 5

[![latest-release](https://img.shields.io/pypi/v/mountainsort5.svg)](https://pypi.org/project/mountainsort5)
![tests](https://github.com/magland/mountainsort5/actions/workflows/integration_tests.yml/badge.svg)
[![codecov](https://codecov.io/gh/magland/mountainsort5/branch/main/graph/badge.svg?token=RTENQMNXKQ)](https://codecov.io/gh/magland/mountainsort5)

This is the most recent version of the [MountainSort](https://www.sciencedirect.com/science/article/pii/S0896627317307456) spike sorting algorithm. An implementation of the previous version of this algorithm can be [found here](https://github.com/magland/mountainsort4).

* Uses [Isosplit clustering](https://github.com/magland/isosplit6)
* Runs much faster than previous versions
* Works well on large datasets
* Better handles time-overlapping events and drifting waveforms
* Designed to be easy to use and to work well out of the box
* Runs fast on a CPU
* Uses SpikeInterface for I/O and preprocessing
* Supports multiple sorting schemes, each suited for different experimental setups

![image](https://user-images.githubusercontent.com/3679296/227960322-0723b527-4356-45fb-a045-5ecd6a8269b7.png)

## Installation

```bash
pip install --upgrade mountainsort5
```

**Dependencies**:

Python, SpikeInterface, scikit-learn, isosplit6

## Usage

MountainSort5 utilizes [SpikeInterface](https://github.com/spikeinterface/spikeinterface) recording and sorting objects. You can get started by reading the [SpikeInterface documentation](https://spikeinterface.readthedocs.io/en/latest/).

Once you have a recording object, you can run MountainSort5 using the following code:

```python
import spikeinterface as si
import spikeinterface.preprocessing as spre
import mountainsort5 as ms5

recording = ... # load your recording using SpikeInterface

# Make sure the recording is preprocessed appropriately
# lazy preprocessing
recording_filtered = spre.bandpass_filter(recording, freq_min=300, freq_max=6000)
recording_preprocessed: si.BaseRecording = spre.whiten(recording_filtered)

# use scheme 1
sorting = ms5.sorting_scheme1(
    recording=recording_preprocessed,
    sorting_parameters=ms5.Scheme1SortingParameters(...)
)

# or use scheme 2
sorting = ms5.sorting_scheme2(
    recording=recording_preprocessed,
    sorting_parameters=ms5.Scheme2SortingParameters(...)
)

# or use scheme 3
sorting = ms5.sorting_scheme3(
    recording=recording_preprocessed,
    sorting_parameters=ms5.Scheme3SortingParameters(...)
)

# Now you have a sorting object that you can save to disk or use for further analysis
```

To give it a try with simulated data, run the following scripts in the examples directory:

Scheme 1: [examples/scheme1/toy_example.py](./examples/scheme1/toy_example.py)

Scheme 2: [examples/scheme2/toy_example.py](./examples/scheme2/toy_example.py)

Scheme 3: [examples/scheme3/toy_example.py](./examples/scheme3/toy_example.py)

## Preprocessing

MountainSort5 is designed to operate on preprocessed data. You should bandpass filter and whiten the recording as shown in the examples. SpikeInterface provides a variety of [lazy preprocessing tools](https://spikeinterface.readthedocs.io/en/latest/modules/preprocessing.html), so that intermediate files do not need to be stored to disk.

## Sorting schemes

MountainSort5 is organized into multiple *sorting schemes*. Different experimental setups will be best served by using different schemes.

### Sorting scheme 1

This is the simplest sorting scheme and is useful for quick tests. The entire recording is loaded into memory, and clustering is performed in a single pass. In general, scheme 2 should be used intead since it has better handling of events that overlap in time, and works with larger datasets on limited RAM systems. Nevertheless, scheme 1 can be useful for testing and debugging, and is used as the first pass in scheme 2.

### Sorting scheme 2

The second sorting scheme is generally preferred over scheme 1 because it can handle larger datasets that cannot be fully loaded into memory, and also has other potential advantages in terms of accurately detecting and labeling spikes.

In phase 1, the first scheme is used as a training step, performing unsupervised clustering on a subset of the dataset. Then in phase 2, a set of classifiers are trained based on the labels of the training step. The classifiers are then used to label the remaining data.

### Sorting scheme 3

Sorting scheme 3 is designed to handle long recordings that may involve waveform drift. The recording is divided into blocks, and each being is spike sorted using scheme 2. Then the snippet classifiers are used to associate matching units between blocks.

## General parameters

Unlike most clustering methods, the Isosplit algorithm by design does not have any adjustable parameters. Below are some spike sorting parameters that may affect the accuracy of spike sorting, depending on the type of dataset.

**Detection threshold (detect_threshold)**

One of the trickiest parameters to set is the detection threshold (detect_threshold). If it is set too low, then you will end up with many false merges because the clusters will overlap for spikes with lower amplitudes. This is especially the case for bursting cells that exhibit a range of spike amplitudes. Of course, if it is set too high, then you'll miss some low-amplitude events. I would recommend leaving this at the default value (5.5 for whitened data) since this seems to perform well on a variety of examples I have tried. You may think that 5.5 sounds high, but I will emphasize that the detection is performed on the ZCA-whitened data for which the spikes peaks tend to be better separated from noise compared with the pre-whitened bandpass-filtered data.

**Number of PCA features per branch (npca_per_branch)**

MountainSort utilizes a branching method for clustering. After extracting spike snippets from the preprocessed traces, the data undergoes dimension reduction through PCA before clustering. Initially, `npca_per_branch` PCA features are computed, followed by Isosplit clustering. If more than one cluster is detected, each of them becomes a new branch, and feature extraction is performed again within each branch separately, followed by Isosplit clustering to subdivide the clusters further. This process is repeated until each leaf branch returns a single cluster. At each stage, the same number of PCA components (npca_per_branch) is used. Recomputing features on each branch offers an advantage, as it allows the refined components to capture features that can better differentiate between clusters within the same overall feature space region.

## Citing MountainSort

Until there is a new publication, please cite the original MountainSort paper:

```bitex
@article{chung2017fully,
  title={A fully automated approach to spike sorting},
  author={Chung, Jason E and Magland, Jeremy F and Barnett, Alex H and Tolosa, Vanessa M and Tooker, Angela C and Lee, Kye Y and Shah, Kedar G and Felix, Sarah H and Frank, Loren M and Greengard, Leslie F},
  journal={Neuron},
  volume={95},
  number={6},
  pages={1381--1394},
  year={2017},
  publisher={Elsevier}
}
```

In addition, if you use the SpikeInterface framework, please cite the following paper:

```bibtex
@article{buccino2020spikeinterface,
  title={SpikeInterface, a unified framework for spike sorting},
  author={Buccino, Alessio Paolo and Hurwitz, Cole Lincoln and Garcia, Samuel and Magland, Jeremy and Siegle, Joshua H and Hurwitz, Roger and Hennig, Matthias H},
  journal={Elife},
  volume={9},
  pages={e61834},
  year={2020},
  publisher={eLife Sciences Publications Limited}
}
```

## Author

Jeremy Magland, Center for Computational Mathematics, Flatiron Institute

## License

Apache-2.0
