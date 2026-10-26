from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch


class RWRSubgraphSampler:
    def __init__(self, adjacency: sp.csr_matrix, size: int, restart_prob: float, seed: int):
        if size < 2:
            raise ValueError("ANEMONE subgraph_size must be >= 2")
        self.adjacency = adjacency.tocsr()
        self.size = size
        self.restart_prob = restart_prob
        self.rng = np.random.default_rng(seed)
        self.neighbors = [
            self.adjacency.indices[self.adjacency.indptr[i] : self.adjacency.indptr[i + 1]]
            for i in range(adjacency.shape[0])
        ]

    def sample_all(self) -> np.ndarray:
        n, reduced = self.adjacency.shape[0], self.size - 1
        result = np.empty((n, self.size), dtype=np.int64)
        for seed in range(n):
            chosen: list[int] = []
            current = seed
            for _ in range(max(60, self.size * 50)):
                if len(chosen) >= reduced:
                    break
                if self.rng.random() < self.restart_prob:
                    current = seed
                neighbors = self.neighbors[current]
                if len(neighbors) == 0:
                    neighbors = self.neighbors[seed]
                if len(neighbors) == 0:
                    break
                current = int(self.rng.choice(neighbors))
                if current != seed and current not in chosen:
                    chosen.append(current)
            direct = self.neighbors[seed]
            while len(chosen) < reduced:
                chosen.append(int(self.rng.choice(direct)) if len(direct) else seed)
            result[seed, :-1] = chosen[:reduced]
            result[seed, -1] = seed
        return result


def normalize_adjacency_official(adjacency: sp.csr_matrix) -> sp.csr_matrix:
    adjacency = sp.coo_matrix(adjacency)
    degree = np.asarray(adjacency.sum(1)).ravel()
    inverse = np.zeros_like(degree, dtype=np.float32)
    np.power(degree, -0.5, out=inverse, where=degree > 0)
    diagonal = sp.diags(inverse)
    normalized = adjacency.dot(diagonal).transpose().dot(diagonal).tocsr()
    return normalized + sp.eye(adjacency.shape[0], dtype=np.float32, format="csr")


def make_batch(
    indices: np.ndarray,
    subgraphs: np.ndarray,
    features: torch.Tensor,
    adjacency: sp.csr_matrix,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, subgraph_size, feature_dim = len(indices), subgraphs.shape[1], features.shape[1]
    batch_adjacency = np.zeros((batch_size, subgraph_size + 1, subgraph_size + 1), dtype=np.float32)
    batch_features = torch.zeros((batch_size, subgraph_size + 1, feature_dim))
    for row, node in enumerate(indices):
        ids = subgraphs[int(node)]
        batch_adjacency[row, :subgraph_size, :subgraph_size] = adjacency[ids][:, ids].toarray()
        batch_adjacency[row, -1, -1] = 1.0
        local = features[torch.from_numpy(ids)]
        batch_features[row, : subgraph_size - 1] = local[:-1]
        batch_features[row, -1] = local[-1]
    return batch_features.to(device), torch.from_numpy(batch_adjacency).to(device)
