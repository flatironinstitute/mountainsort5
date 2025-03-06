from typing import Tuple
import numpy as np
import numpy.typing as npt
import spikeinterface as si


def get_times_labels_from_sorting(sorting: si.BaseSorting) -> Tuple[npt.NDArray[np.int64], npt.NDArray]:
    """Get times and labels from a sorting object
    Inputs:
        sorting: a sorting object
    Returns:
        times: 1D array of spike times
        labels: 1D array of spike labels
    Example:
        times, labels = get_times_labels_from_sorting(sorting)
    """
    times_list = []
    labels_list = []
    for unit_id in sorting.get_unit_ids():
        times0 = sorting.get_unit_spike_train(unit_id=unit_id)
        labels0 = np.ones(times0.shape, dtype=np.int32) * int(unit_id)
        times_list.append(times0.astype(np.int64))
        labels_list.append(labels0)
    if len(times_list) > 0:
        times = np.concatenate(times_list).astype(np.int64)
        labels = np.concatenate(labels_list)
        inds = np.argsort(times)
        times = times[inds]
        labels = labels[inds]
        return times, labels
    else:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int32)
