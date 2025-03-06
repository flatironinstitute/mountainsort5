
import numpy as np
import numpy.typing as npt
import warnings
from scipy.cluster.hierarchy import ClusterWarning
from scipy.spatial.distance import squareform
from isosplit6 import isosplit6
from .compute_pca_features import compute_pca_features


warnings.filterwarnings('ignore', category=ClusterWarning)


from typing import Union

def isosplit6_subdivision_method(
    X: npt.NDArray[np.float32], *,
    npca_per_subdivision: int,
    inds: Union[npt.NDArray, None] = None # pass in inds so that we don't keep making copies of the array
):
    """Isosplit6 subdivision method

    Args:
        X: L x M array where L is the number of data points and M is the number of features
        npca_per_subdivision: number of pca components to use for each subdivision
        inds: indices of the data points to use (optional)

    Returns:
        _type_: _description_
    """
    if inds is not None:
        X_sub = X[inds]
    else:
        X_sub = X
    L = X_sub.shape[0]
    if L == 0:
        return np.zeros((0,), dtype=np.int32)
    features = compute_pca_features(X_sub, npca=npca_per_subdivision)
    labels = isosplit6(features)
    if len(labels) > 0:
        K = int(np.max(labels))
    else:
        K = 0
    if K <= 1:
        return labels
    centroids = np.zeros((K, X.shape[1]), dtype=np.float32)
    for k in range(1, K + 1):
        centroids[k - 1] = np.median(X_sub[labels == k], axis=0)
    X_sub = None # free up memory
    dists_between_centroids = np.sqrt(np.sum((centroids[:, None, :] - centroids[None, :, :]) ** 2, axis=2))

    # Before I added the following line I was getting warnings: "The symmetric non-negative hollow observation matrix looks suspiciously like an uncondensed distance matrix"
    # so I think this is needed
    # because linkage function expects "y is a 1-D condensed distance matrix,"
    dists_between_centroids = squareform(dists_between_centroids)

    # hierarchical clustering
    from scipy.cluster.hierarchy import linkage, cut_tree
    Z = linkage(dists_between_centroids, method='single', metric='euclidean')
    clusters0 = cut_tree(Z, n_clusters=2)
    cluster_inds_1 = np.where(clusters0 == 0)[0] + 1
    cluster_inds_2 = np.where(clusters0 == 1)[0] + 1
    inds1 = np.where(np.isin(labels, cluster_inds_1))[0]
    inds2 = np.where(np.isin(labels, cluster_inds_2))[0]
    if inds is not None:
        inds1_b = inds[inds1]
        inds2_b = inds[inds2]
    else:
        inds1_b = inds1
        inds2_b = inds2
    labels1 = isosplit6_subdivision_method(X, npca_per_subdivision=npca_per_subdivision, inds=inds1_b)
    labels2 = isosplit6_subdivision_method(X, npca_per_subdivision=npca_per_subdivision, inds=inds2_b)
    K1 = int(np.max(labels1))
    # K2 = int(np.max(labels2))
    ret_labels = np.zeros(L, dtype=np.int32)
    ret_labels[inds1] = labels1
    ret_labels[inds2] = labels2 + K1
    return ret_labels
