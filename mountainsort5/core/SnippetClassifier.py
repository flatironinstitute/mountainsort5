from typing import List, Tuple, Union, Dict
import numpy as np
from dataclasses import dataclass
from sklearn import decomposition
from sklearn.neighbors import NearestNeighbors


class SnippetClassifier:
    def __init__(self, npca: int) -> None:
        self.npca = npca
        self.training_batches: List[TrainingBatch] = []
        self.pca_model = None
    def add_training_snippets(self, snippets: np.ndarray, label: int, offset: int):
        self.training_batches.append(TrainingBatch(snippets=snippets, label=label, offset=offset))
    def fit(self):
        if len(self.training_batches) == 0:
            raise Exception('No training batches added for classifier.')
        all_training_snippets = np.concatenate([b.snippets for b in self.training_batches], axis=0)
        L = all_training_snippets.shape[0]
        self.T = all_training_snippets.shape[1]
        self.M = all_training_snippets.shape[2]
        self.all_training_labels = np.concatenate([np.ones((b.num_snippets,), dtype=np.int32) * b.label for b in self.training_batches])
        self.all_training_offsets = np.concatenate([np.ones((b.num_snippets,), dtype=np.int32) * b.offset for b in self.training_batches])
        self.pca_model = decomposition.PCA(n_components=min(self.npca, L))
        self.pca_model.fit(all_training_snippets.reshape(L, self.T * self.M))
        X = self.pca_model.transform(all_training_snippets.reshape(L, self.T * self.M))
        self.nearest_neighbor_model = NearestNeighbors(n_neighbors=2)
        self.nearest_neighbor_model.fit(X)
    def classify_snippets(self, snippets: np.ndarray) -> Tuple[Union[np.array, None], Union[np.array, None]]:
        if self.pca_model is None:
            return None, None
        Y = self.pca_model.transform(snippets.reshape(snippets.shape[0], self.T * self.M))
        nearest_inds = self.nearest_neighbor_model.kneighbors(Y, n_neighbors=2, return_distance=False)
        inds = nearest_inds[:, 1] # don't use the first because that could be an identical match
        return self.all_training_labels[inds], self.all_training_offsets[inds]
    def apply_label_mapping(self, mapping: Dict[int, int]):
        for k1, k2 in mapping.items():
            self.all_training_labels[self.all_training_labels == k1] = k2

@dataclass
class TrainingBatch:
    snippets: np.ndarray
    label: int
    offset: int
    @property
    def num_snippets(self):
        return self.snippets.shape[0]