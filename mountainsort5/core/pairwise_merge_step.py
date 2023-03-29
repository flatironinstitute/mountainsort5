from typing import List, Tuple
import numpy as np
import numpy.typing as npt
from .compute_pca_features import compute_pca_features
from isosplit6 import isosplit6


def pairwise_merge_step(*, snippets: npt.NDArray[np.float32], templates: npt.NDArray[np.float32], labels: npt.NDArray[np.int32], times: npt.NDArray[np.int32], detect_sign: int, unit_ids: np.array, detect_time_radius: int) -> Tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    # L = snippets.shape[0]
    # T = snippets.shape[1]
    # M = snippets.shape[2]
    K = templates.shape[0]

    new_times = np.array(times)
    new_labels = np.array(labels)

    if detect_sign < 0:
        A = -templates
    elif detect_sign > 0: # pragma: no cover
        A = templates # pragma: no cover
    else:
        A = np.abs(templates) # pragma: no cover
    
    merges = {}
    peak_channel_indices = [np.argmax(np.max(A[i], axis=0)) for i in range(K)]
    peak_values_on_channels = [np.max(A[i], axis=0) for i in range(K)]
    peak_time_indices_on_channels = [np.argmax(A[i], axis=0) for i in range(K)]
    for i1 in range(K):
        for i2 in range(i1 - 1, -1, -1): # merge everything into lower ID - important to go in reverse order
            offset1 = peak_time_indices_on_channels[i2][peak_channel_indices[i1]] - peak_time_indices_on_channels[i1][peak_channel_indices[i1]]
            offset2 = peak_time_indices_on_channels[i2][peak_channel_indices[i2]] - peak_time_indices_on_channels[i1][peak_channel_indices[i2]]
            if np.abs(offset1 - offset2) <= 4: # make sure the offsets are roughly consistent
                if peak_values_on_channels[i1][peak_channel_indices[i2]] > 0.5 * peak_values_on_channels[i2][peak_channel_indices[i2]]:
                    if peak_values_on_channels[i2][peak_channel_indices[i1]] > 0.5 * peak_values_on_channels[i1][peak_channel_indices[i1]]:
                        # print(f'Pairwise merge: comparing units {unit_ids[i1]} and {unit_ids[i2]} (offset: {offset1})')
                        if test_merge(offset_snippets(snippets[new_labels == unit_ids[i1]], offset=offset1), snippets[new_labels == unit_ids[i2]]):
                            print(f'Pairwise merge: ** merging {unit_ids[i1]} and {unit_ids[i2]} (offset: {offset1})')
                            merges[unit_ids[i1]] = {
                                'unit_id1': unit_ids[i1],
                                'unit_id2': unit_ids[i2],
                                'offset': offset1
                            }
    for unit_id in unit_ids[::-1]: # important to go in descending order for transitive merges to work
        if unit_id in merges:
            mm = merges[unit_id]
            unit_id1 = mm['unit_id1']
            unit_id2 = mm['unit_id2']
            assert unit_id == unit_id1
            offset = mm['offset']
            inds1 = np.nonzero(new_labels == unit_id1)[0]
            inds2 = np.nonzero(new_labels == unit_id2)[0]
            print(f'Performing merge of units {unit_id1} ({len(inds1)} events) and {unit_id2} ({len(inds2)} events) (offset: {offset})')
            new_times[inds1] = new_times[inds1] + offset
            new_labels[inds1] = unit_id2
    
    # need to re-sort the times after the merges (because of the time offsets)
    sort_inds = np.argsort(new_times)
    new_times = new_times[sort_inds]
    new_labels = new_labels[sort_inds]
    
    # After merges, we may now have duplicate events - let's remove them
    new_inds = remove_duplicate_events(new_times, new_labels, tol=detect_time_radius)
    print(f'Removing {len(times) - len(new_inds)} duplicate events')
    new_times = new_times[new_inds]
    new_labels = new_labels[new_inds]

    return new_times, new_labels

def remove_duplicate_events(times: npt.NDArray[np.int32], labels: npt.NDArray[np.int32], *, tol: int) -> npt.NDArray[np.int32]:
    new_labels = np.array(labels)
    unit_ids = np.unique(new_labels)
    for unit_id in unit_ids:
        unit_inds = np.nonzero(new_labels == unit_id)[0]
        unit_times = times[unit_inds]
        inds_duplicate = find_duplicate_times(unit_times, tol=tol)
        new_labels[unit_inds[inds_duplicate]] = 0
    inds_nonzero = np.nonzero(new_labels)[0]
    return inds_nonzero

def find_duplicate_times(times: npt.NDArray[np.int32], *, tol: int) -> npt.NDArray[np.int32]:
    ret: list[np.int32] = []
    deleted = np.zeros((len(times),), dtype=np.int16)
    for i1 in range(len(times)):
        if not deleted[i1]:
            i2 = i1 + 1
            while i2 < len(times) and times[i2] <= times[i1] + tol:
                ret.append(i2)
                deleted[i2] = True
                i2 += 1
    return np.array(ret, dtype=np.int32)

def test_merge(snippets1: npt.NDArray[np.float32], snippets2: npt.NDArray[np.float32]) -> bool:
    L1 = snippets1.shape[0]
    L2 = snippets2.shape[0]
    T = snippets1.shape[1]
    M = snippets1.shape[2]
    V1 = snippets1.reshape((L1, T * M))
    V2 = snippets2.reshape((L2, T * M))
    Vall = np.concatenate([V1, V2], axis=0)
    features0 = compute_pca_features(Vall, npca=12) # should we hard-code this as 12?
    labels0 = isosplit6(features0)
    # not sure best way to test this, but for now we'll check that we only have single cluster after isosplit
    # after all, each cluster should not split on its own (branch cluster method)
    return np.max(labels0) == 1
    # dominant_label1 = np.argmax(np.bincount(labels0[:L1]))
    # dominant_label2 = np.argmax(np.bincount(labels0[L1:]))
    # return dominant_label1 == dominant_label2

def offset_snippets(snippets: npt.NDArray[np.float32], *, offset: int):
    return np.roll(snippets, shift=offset, axis=1)