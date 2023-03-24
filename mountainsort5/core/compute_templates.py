import numpy as np


def compute_templates(snippets: np.ndarray, labels: np.int32):
    # L = snippets.shape[0]
    T = snippets.shape[1]
    M = snippets.shape[2]
    K = np.max(labels)
    templates = np.zeros((K, T, M), dtype=np.float32)
    for k in range(1, K + 1):
        snippets1 = snippets[labels == k]
        templates[k - 1] = np.median(snippets1, axis=0)
    return templates
        