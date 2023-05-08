from typing import Union
import numpy as np
import numpy.typing as npt
from isosplit6 import isosplit6
from .compute_pca_features import compute_pca_features


def branch_cluster(
    X: npt.NDArray[np.float32], *, # L x D
    npca_per_branch: int,
    inds: Union[npt.NDArray[np.int32], None]=None # pass in inds so that we don't keep making copies of the array
) -> npt.NDArray[np.int32]:
    if inds is not None:
        X_sub = X[inds]
    else:
        X_sub = X
    L = X_sub.shape[0]
    features = compute_pca_features(X_sub, npca=npca_per_branch)
    X_sub = None # free up memory
    labels = isosplit6(features)
    K = np.max(labels)
    if K <= 1:
        return labels
    print(f'{K} clusters in branch')
    ret = np.zeros((L,), dtype=np.int32)
    last_ret_k = 0
    for k in range(1, K + 1):
        inds0 = np.nonzero(labels == k)[0]
        if inds is not None:
            inds1 = inds[inds0]
        else:
            inds1 = inds0
        labels0 = branch_cluster(X, npca_per_branch=npca_per_branch, inds=inds1)
        K2 = np.max(labels0)
        for k2 in range(1, K2 + 1):
            last_ret_k += 1
            ret[inds0[labels0 == k2]] = last_ret_k
    return ret
    # # test merging of clusters
    # K_total = np.max(ret)
    # to_merge = []
    # for k1 in range(1, K_total + 1):
    #     inds1 = np.nonzero(ret == k1)[0]
    #     for k2 in range(k1 + 1, K_total + 1):
    #         inds2 = np.nonzero(ret == k2)[0]
    #         if len(inds1) == 0 or len(inds2) == 0:
    #             continue
    #         X1 = X[inds1]
    #         X2 = X[inds2]
    #         if _should_merge(X1, X2):
    #             to_merge.append((k1, k2))

    # something_changed = True
    # while something_changed:
    #     something_changed = False
    #     for merge_item in to_merge:
    #         k1, k2 = merge_item
    #         inds1 = np.nonzero(ret == k1)[0]
    #         inds2 = np.nonzero(ret == k2)[0]
    #         if len(inds1) == 0 or len(inds2) == 0:
    #             continue
    #         print(f'Merging {k1} and {k2}')
    #         ret[inds2] = k1
    #         something_changed = True
    # # consolidate labels
    # K_total = np.max(ret)
    # ret2 = np.zeros((L,), dtype=np.int32)
    # k2 = 1
    # for k in range(1, K_total + 1):
    #     inds = np.nonzero(ret == k)[0]
    #     if len(inds) > 0:
    #         ret2[inds] = k2
    #         k2 += 1
    # return ret2

def _should_merge(X1: npt.NDArray[np.float32], X2: npt.NDArray[np.float32]):
    discriminating_direction = np.mean(X2, axis=0) - np.mean(X1, axis=0)
    discriminating_direction /= np.linalg.norm(discriminating_direction)
    proj1 = np.dot(X1, discriminating_direction)
    proj2 = np.dot(X2, discriminating_direction)
    proj_union = np.concatenate((proj1, proj2))
    labels = np.concatenate((np.zeros((len(proj1),), dtype=np.int32), np.ones((len(proj2),), dtype=np.int32)))
    sorted_inds = np.argsort(proj_union)
    sorted_labels = labels[sorted_inds]
    cumsum_labels = np.cumsum(sorted_labels)
    num_violations = np.sum((sorted_labels == 0) * cumsum_labels)
    num_comparisons = len(proj1) * len(proj2)
    if num_violations > 0.3 * num_comparisons:
        return True
    return False

