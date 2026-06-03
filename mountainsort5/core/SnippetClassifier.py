from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import numpy as np
import numpy.typing as npt
from sklearn import decomposition
from sklearn.neighbors import NearestNeighbors

from .pca_solver import deterministic_pca_solver


class SnippetClassifier:
    def __init__(self, npca: Union[int, None]) -> None:
        self.npca = npca
        self.training_batches: List[TrainingBatch] = []
        self.pca_model = None
    def add_training_snippets(self, snippets: npt.NDArray[np.float32], label: int, offset: int):
        self.training_batches.append(TrainingBatch(snippets=snippets, label=label, offset=offset))
    def fit(self):
        if len(self.training_batches) == 0:
            raise Exception('No training batches added for classifier.') # pragma: no cover
        all_training_snippets = np.concatenate([b.snippets for b in self.training_batches], axis=0)
        L = all_training_snippets.shape[0]
        self.T = all_training_snippets.shape[1]
        self.M = all_training_snippets.shape[2]
        self.all_training_labels = np.concatenate([np.ones((b.num_snippets,), dtype=np.int32) * b.label for b in self.training_batches])
        self.all_training_offsets = np.concatenate([np.ones((b.num_snippets,), dtype=np.int32) * b.offset for b in self.training_batches])
        if self.npca is not None:
            effective_npca = self.npca
        else:
            effective_npca = max(12, self.M * 3)
        n_features = self.T * self.M
        n_components = min(effective_npca, L)
        svd_solver = deterministic_pca_solver(L, n_features)
        self.pca_model = decomposition.PCA(n_components=n_components, svd_solver=svd_solver, random_state=0)
        self.pca_model.fit(all_training_snippets.reshape(L, n_features))
        X = self.pca_model.transform(all_training_snippets.reshape(L, n_features))
        self.nearest_neighbor_model = NearestNeighbors(n_neighbors=2)
        self.nearest_neighbor_model.fit(X)
    def classify_snippets(self, snippets: npt.NDArray[np.float32]) -> Tuple[Union[npt.NDArray, None], Union[npt.NDArray, None]]:
        if self.pca_model is None:
            raise Exception('self.pca_model is None, which probably means that fit() was not called.') # pragma: no cover
        Y = self.pca_model.transform(snippets.reshape(snippets.shape[0], self.T * self.M))
        nearest_inds = self.nearest_neighbor_model.kneighbors(Y, n_neighbors=2, return_distance=False)
        assert isinstance(nearest_inds, np.ndarray)
        inds = nearest_inds[:, 1] # don't use the first because that could be an identical match
        return self.all_training_labels[inds], self.all_training_offsets[inds]
    def apply_label_mapping(self, mapping: Dict[int, int]):
        for k1, k2 in mapping.items():
            self.all_training_labels[self.all_training_labels == k1] = k2

@dataclass
class TrainingBatch:
    snippets: npt.NDArray[np.float32]
    label: int
    offset: int
    @property
    def num_snippets(self):
        return self.snippets.shape[0]
