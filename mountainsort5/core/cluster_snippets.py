from typing import Union
import numpy as np
from isosplit5 import isosplit6 # I know this is confusing, sorry :)
from .compute_pca_features import compute_pca_features


def cluster_snippets(
    snippets: np.ndarray, *,
    npca_per_branch: int,
    snippet_inds: Union[np.array, None]=None # pass in inds so that we don't keep making copies of the array
) -> np.array:
    if snippet_inds is not None:
        snippets_sub = snippets[snippet_inds]
    else:
        snippets_sub = snippets
    L = snippets_sub.shape[0]
    T = snippets_sub.shape[1]
    M = snippets_sub.shape[2]
    features = compute_pca_features(snippets_sub.reshape((L, T * M)), npca=npca_per_branch)
    snippets_sub = None # free up memory
    labels = isosplit6(features.T)
    K = np.max(labels)
    if K <= 1:
        return labels
    print(f'{K} clusters in branch')
    ret = np.zeros((L,), dtype=np.int32)
    last_ret_k = 0
    for k in range(1, K + 1):
        inds0 = np.nonzero(labels == k)[0]
        if snippet_inds is not None:
            inds1 = snippet_inds[inds0]
        else:
            inds1 = inds0
        labels0 = cluster_snippets(snippets, npca_per_branch=npca_per_branch, snippet_inds=inds1)
        K2 = np.max(labels0)
        for k2 in range(1, K2 + 1):
            last_ret_k += 1
            ret[inds0[labels0 == k2]] = last_ret_k
    return ret