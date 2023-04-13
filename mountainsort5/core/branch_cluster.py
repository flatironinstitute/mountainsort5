from typing import Union
import numpy as np
import numpy.typing as npt
from isosplit6 import isosplit6
from .compute_pca_features import compute_pca_features


def branch_cluster(
    X: npt.NDArray[np.float32], *,
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