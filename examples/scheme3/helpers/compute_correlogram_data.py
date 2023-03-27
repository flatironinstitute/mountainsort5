from typing import Union
import spikeinterface as si
import numpy as np

def compute_correlogram_data(*, sorting: si.BaseSorting, unit_id1: int, unit_id2: Union[int, None]=None, window_size_msec: float, bin_size_msec: float):
    times1 = sorting.get_unit_spike_train(unit_id=unit_id1, segment_index=0)
    num_bins = int(window_size_msec / bin_size_msec)
    if num_bins % 2 == 0: num_bins = num_bins - 1 # odd number of bins
    num_bins_half = int((num_bins + 1) / 2)
    bin_edges_msec = np.array((np.arange(num_bins + 1) - num_bins / 2) * bin_size_msec, dtype=np.float32)
    bin_counts = np.zeros((num_bins,), dtype=np.int32)
    if unit_id2 is None or unit_id1 == unit_id2:
        # autocorrelogram
        offset = 1
        while True:
            if offset >= len(times1): break
            deltas_msec = (times1[offset:] - times1[:-offset]) / sorting.get_sampling_frequency() * 1000
            deltas_msec = deltas_msec[deltas_msec <= bin_edges_msec[-1]]
            if len(deltas_msec) == 0: break
            for i in range(num_bins_half):
                start_msec = bin_edges_msec[num_bins_half - 1 + i]
                end_msec = bin_edges_msec[num_bins_half + i]
                ct = len(deltas_msec[(start_msec <= deltas_msec) & (deltas_msec < end_msec)])
                bin_counts[num_bins_half - 1 + i] += ct
                bin_counts[num_bins_half - 1 - i] += ct
            offset = offset + 1
    else:
        # cross-correlogram
        times2 = sorting.get_unit_spike_train(segment_index=0, unit_id=unit_id2)
        all_times = np.concatenate((times1, times2))
        all_labels = np.concatenate((1 * np.ones(times1.shape), 2 * np.ones(times2.shape)))
        sort_inds = np.argsort(all_times)
        all_times = all_times[sort_inds]
        all_labels = all_labels[sort_inds]
        offset = 1
        while True:
            if offset >= len(all_times): break
            deltas_msec = (all_times[offset:] - all_times[:-offset]) / sorting.get_sampling_frequency() * 1000

            deltas12_msec = deltas_msec[(all_labels[offset:] == 2) & (all_labels[:-offset] == 1)]
            deltas21_msec = deltas_msec[(all_labels[offset:] == 1) & (all_labels[:-offset] == 2)]
            deltas11_msec = deltas_msec[(all_labels[offset:] == 1) & (all_labels[:-offset] == 1)]
            deltas22_msec = deltas_msec[(all_labels[offset:] == 2) & (all_labels[:-offset] == 2)]

            deltas12_msec = deltas12_msec[deltas12_msec <= bin_edges_msec[-1]]
            deltas21_msec = deltas21_msec[deltas21_msec <= bin_edges_msec[-1]]
            deltas11_msec = deltas11_msec[deltas11_msec <= bin_edges_msec[-1]]
            deltas22_msec = deltas22_msec[deltas22_msec <= bin_edges_msec[-1]]

            if (len(deltas12_msec) + len(deltas21_msec) + len(deltas11_msec) + len(deltas22_msec)) == 0: break
            
            for i in range(num_bins_half):
                start_msec = bin_edges_msec[num_bins_half - 1 + i]
                end_msec = bin_edges_msec[num_bins_half + i]
                ct12 = len(deltas12_msec[(start_msec <= deltas12_msec) & (deltas12_msec < end_msec)])
                ct21 = len(deltas21_msec[(start_msec <= deltas21_msec) & (deltas21_msec < end_msec)])
                bin_counts[num_bins_half - 1 + i] += ct12
                bin_counts[num_bins_half - 1 - i] += ct21
            offset = offset + 1
    return {
        'bin_edges_sec': (bin_edges_msec / 1000).astype(np.float32),
        'bin_counts': bin_counts.astype(np.int32)
    }