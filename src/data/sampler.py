"""
Sampling strategies for Active Learning and Semi-Supervised Learning.
"""
import numpy as np
import torch
from torch.utils.data import Sampler, Subset
from typing import List, Iterator, Sized

class ActiveSelectionSampler(Sampler):
    """
    Sampler that selects a subset of data based on indices.
    Used for the labeled pool in Active Learning.
    """
    def __init__(self, data_source: Sized, indices: List[int]):
        self.data_source = data_source
        self.indices = indices

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)

def get_unlabeled_indices(total_size: int, labeled_indices: List[int]) -> List[int]:
    """Returns indices that are NOT in the labeled set."""
    all_indices = set(range(total_size))
    labeled_set = set(labeled_indices)
    return list(all_indices - labeled_set)

class BalancedBatchSampler(Sampler):
    """
    Sampler that ensures balanced class distribution in each batch.
    Useful for the small labeled set in Target domain.
    """
    def __init__(self, dataset, labels: List[int], batch_size: int, n_classes: int, n_samples: int):
        self.labels = torch.LongTensor(labels)
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.dataset = dataset

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_samples:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[class_] +
                                                                     self.batch_size // self.n_classes])
                self.used_label_indices_count[class_] += self.batch_size // self.n_classes
                if self.used_label_indices_count[class_] + self.batch_size // self.n_classes > len(
                        self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.batch_size

    def __len__(self):
        return self.n_samples // self.batch_size
