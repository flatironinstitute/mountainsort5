# MountainSort 5

An updated version of the algorithm described in [this paper](https://www.sciencedirect.com/science/article/pii/S0896627317307456). An implementation of the previous version of this algorithm can be [found here](https://github.com/magland/mountainsort4).

* Uses a somewhat improved version of [Isosplit clustering](https://arxiv.org/abs/1508.04841).
* Runs on CPU, not GPU, and is still very fast

## Installation

During the development stage, get the latest version by cloning this repo

```bash
git clone https://github.com/magland/mountainsort5
cd mountainsort5
pip install -e .

# update periodically
git pull
```

## Usage

Start with [examples/scheme1/toy_example.py](./examples/scheme1/toy_example.py)

## General parameters

Unlike most clustering methods, the Isosplit algorithm by design does not have any adjustable parameters. Below are some spike sorting parameters that may affect the accuracy of spike sorting, depending on the type of dataset.

## Detection threshold (detect_threshold)

One of the trickiest parameters to set is the detection threshold (detect_threshold). If it is set too low, then you will end up with many false merges because the clusters will overlap for spikes with lower amplitudes. This is especially the case for bursting cells that exhibit a range of spike amplitudes. Of course, if it is set too high, then you'll miss some low-amplitude events. I would recommend leaving this at the default value (5.5 for whitened data) since this seems to perform well on a variety of examples I have tried. You may think that 5.5 sounds high, but I will emphasize that the detection is performed on the ZCA-whitened data for which the spikes peaks tend to be better separated from noise compared with the pre-whitened bandpass-filtered data.

## Number of PCA features per branch (npca_per_branch)

MountainSort uses a branch method for clustering. After spike snippets are extracted from the preprocessed traces, dimension reduction via PCA is used prior to clustering. In the first step, npca_per_branch PCA features are computed, and then Isosplit clustering is performed. After this, assuming more than one cluster is found, each cluster becomes a new branch, and feature extraction from the original snippets is performed again within each branch separately, and Isosplit is used to subdivide the clusters further. This procedure is repeated until clustering returns a single cluster for each branch. The same number of PCA components (npca_per_branch) is used at each stage. The advantage of recomputing features on each branch is that the refined components can capture features that can more effectively distinguish between clusters within the same general region of overall feature space.

## Sorting schemes

The MountainSort5 algorithm is organized into multiple *sorting schemes*. Different experimental setups will be best served by using different schemes. For example, if you dataset is relatively small, say 10 minutes with 32 channels, then you'll probably want to use the simplest scheme: scheme 1, which processes everything in memory.

### Sorting scheme 1

This is the simplest sorting scheme and is suited for relatively small datasets. All processing is done in memory, so if your system has more RAM resources, then it will be capable of handling larger datasets. One of the example datasets is ~10 minutes with 32-channels and the peak RAM consumption was around 13 GB. Sorting completed in 1.5 minutes on my laptop without any trouble.

### Sorting scheme 2

This scheme does not exist yet, but I have plans

## Author

Jeremy Magland, Center for Computational Mathematics, Flatiron Institute

## License

Apache-2.0
