import importlib
from unittest.mock import patch

import numpy as np
import pytest
import spikeinterface as si
from sklearn import decomposition

import mountainsort5 as ms5
from mountainsort5.core.compute_pca_features import compute_pca_features
from mountainsort5.core.SnippetClassifier import SnippetClassifier


@pytest.mark.parametrize("helper", ["features", "classifier"])
def test_randomized_pca_seed(helper, monkeypatch):
    module = importlib.import_module(
        "mountainsort5.core.compute_pca_features"
        if helper == "features" else "mountainsort5.core.SnippetClassifier"
    )
    # Exercise randomized SVD without allocating more than 8000 features.
    monkeypatch.setattr(module, "deterministic_pca_solver", lambda *args: "randomized")
    X = np.random.default_rng(42).standard_normal((80, 40)).astype(np.float32)

    def fit(**kwargs):
        if helper == "features":
            return compute_pca_features(X, npca=5, **kwargs)
        classifier = SnippetClassifier(npca=5, **kwargs)
        classifier.add_training_snippets(X.reshape(80, 10, 4), label=1, offset=0)
        classifier.fit()
        return classifier.pca_model.components_

    assert np.array_equal(fit(), fit(random_seed=0))
    result = fit(random_seed=17)
    assert np.array_equal(result, fit(random_seed=17))
    assert not np.allclose(result, fit(random_seed=18))


def test_subdivision_propagates_seed():
    module = importlib.import_module("mountainsort5.core.isosplit6_subdivision_method")
    X = np.random.default_rng(42).standard_normal((8, 4)).astype(np.float32)
    with patch.object(module, "compute_pca_features", wraps=compute_pca_features) as pca:
        with patch.object(module, "isosplit6", side_effect=[
            np.array([1, 1, 1, 1, 2, 2, 2, 2]),
            np.ones(4, dtype=np.int32),
            np.ones(4, dtype=np.int32),
        ]):
            labels = module.isosplit6_subdivision_method(
                X, npca_per_subdivision=2, random_seed=17
            )
    assert len(labels) == 8
    assert pca.call_count == 3
    assert all(call.kwargs["random_seed"] == 17 for call in pca.call_args_list)


@pytest.mark.parametrize("scheme", [1, 2, 3])
def test_sorting_schemes_propagate_seed(scheme):
    traces = np.random.default_rng(42).normal(0, 0.1, (2000, 1)).astype(np.float32)
    traces[np.arange(100, 1900, 100), 0] = -10
    recording = si.NumpyRecording(traces_list=[traces], sampling_frequency=1000)
    recording.set_channel_locations(np.array([[0, 0]]))
    if scheme == 1:
        parameters = ms5.Scheme1SortingParameters(
            snippet_T1=4, snippet_T2=4, npca_per_channel=2,
            npca_per_subdivision=2, random_seed=17
        )
        sort = ms5.sorting_scheme1
    else:
        parameters = ms5.Scheme2SortingParameters(
            phase1_detect_channel_radius=None, detect_channel_radius=None,
            snippet_T1=4, snippet_T2=4, phase1_npca_per_channel=2,
            phase1_npca_per_subdivision=2, classifier_npca=2,
            max_num_snippets_per_training_batch=20, random_seed=17
        )
        sort = ms5.sorting_scheme2
        if scheme == 3:
            parameters = ms5.Scheme3SortingParameters(
                block_sorting_parameters=parameters, block_duration_sec=1
            )
            sort = ms5.sorting_scheme3

    with patch.object(decomposition, "PCA", wraps=decomposition.PCA) as pca:
        sorting = sort(recording, sorting_parameters=parameters)

    assert sorting.get_num_units() > 0
    # Two initial/aligned feature and subdivision fits, plus classifier fits in scheme 2/3.
    assert pca.call_count >= (4 if scheme == 1 else 5)
    assert all(call.kwargs["random_state"] == 17 for call in pca.call_args_list)


def test_sorting_parameter_seed_defaults():
    assert ms5.Scheme1SortingParameters().random_seed == 0
    assert ms5.Scheme2SortingParameters(
        phase1_detect_channel_radius=None, detect_channel_radius=None
    ).random_seed == 0
