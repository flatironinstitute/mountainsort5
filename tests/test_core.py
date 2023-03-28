import numpy as np
from mountainsort5.core.compute_pca_features import compute_pca_features
from mountainsort5.core.compute_templates import compute_templates
from mountainsort5.core.detect_spikes import detect_spikes


def test_compute_pca_features():
    x = np.random.normal(size=(100, 20))
    features = compute_pca_features(x, npca=10)
    assert features.shape == (100, 10)

def test_compute_templates():
    L = 1000
    T = 20
    M = 10
    snippets = np.random.normal(size=(L, T, M))
    labels = np.random.randint(1, 11, size=(L,))
    templates = compute_templates(snippets, labels)
    assert templates.shape == (10, T, M)

def test_detect_spikes():
    N = 1000
    M = 10
    traces= np.random.normal(size=(N, M))
    channel_locations = np.random.normal(size=(M, 2))
    times, labels = detect_spikes(
        traces,
        channel_locations=channel_locations,
        time_radius=10,
        channel_radius=50,
        detect_threshold=3,
        detect_sign=-1,
        margin_left=10,
        margin_right=10,
        verbose=False
    )
    assert len(times) == len(labels)