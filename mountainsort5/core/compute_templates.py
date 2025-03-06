import numpy as np
import numpy.typing as npt


def compute_templates(snippets: npt.NDArray[np.float32], labels: npt.NDArray):
    """Compute templates from snippets and labels

    Args:
        snippets: L x T x M where L is the number of snippets, T is the number of timepoints, and M is the number of channels
        labels: array of length L with labels for the snippets

    Returns:
        templates: K x T x M where K is the number of clusters
    """
    L = snippets.shape[0]
    T = snippets.shape[1]
    M = snippets.shape[2]
    if len(labels) != L:
        raise Exception('Length of labels must equal number of snippets')
    if L == 0:
        return np.zeros((0, T, M), dtype=np.float32)
    K = int(np.max(labels))
    templates = np.zeros((K, T, M), dtype=np.float32)
    for k in range(1, K + 1):
        snippets1 = snippets[labels == k]
        templates[k - 1] = np.median(snippets1, axis=0)
    return templates
