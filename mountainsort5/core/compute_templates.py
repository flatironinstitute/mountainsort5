import numpy as np
import numpy.typing as npt


def compute_templates(snippets: npt.NDArray[np.float32], labels: np.int32):
    # L = snippets.shape[0]
    T = snippets.shape[1]
    M = snippets.shape[2]
    K = np.max(labels)
    templates = np.zeros((K, T, M), dtype=np.float32)
    for k in range(1, K + 1):
        snippets1 = snippets[labels == k]
        templates[k - 1] = np.median(snippets1, axis=0)
    return templates
        