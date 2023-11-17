import numpy as np
import numpy.typing as npt
from sklearn import decomposition


def compute_pca_features(X: npt.NDArray[np.float32], *, npca: int):
    """Compute PCA features.

    Parameters
    ----------
    X : npt.NDArray[np.float32]
        Input data. L x D matrix, where L is the number of samples and D is the
        number of features.
    npca : int
        Number of PCA features to return.

    Returns
    -------
    npt.NDArray[np.float32]
        PCA features. L x npca_2 matrix where npca_2 is min(npca, L, D).
    """
    L = X.shape[0]
    D = X.shape[1]
    npca_2 = np.minimum(np.minimum(npca, L), D)
    if L == 0 or D == 0:
        return np.zeros((0, npca_2), dtype=np.float32)
    pca = decomposition.PCA(n_components=npca_2)
    return pca.fit_transform(X)
