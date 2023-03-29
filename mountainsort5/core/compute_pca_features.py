import numpy as np
import numpy.typing as npt
from sklearn import decomposition


def compute_pca_features(X: npt.NDArray[np.float32], *, npca: int):
    pca = decomposition.PCA(n_components=np.minimum(npca, X.shape[0]))
    return pca.fit_transform(X)