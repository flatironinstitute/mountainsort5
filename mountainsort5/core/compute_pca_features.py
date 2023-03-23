import numpy as np
from sklearn import decomposition


def compute_pca_features(X: np.ndarray, *, npca: int):
    pca = decomposition.PCA(n_components=np.minimum(npca, X.shape[0]))
    return pca.fit_transform(X)