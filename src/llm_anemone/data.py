from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp


@dataclass(frozen=True)
class GraphData:
    node_ids: list[str]
    texts: list[str]
    adjacency: sp.csr_matrix

    @property
    def num_nodes(self) -> int:
        return len(self.texts)


@dataclass(frozen=True)
class GraphLabels:
    attribute: np.ndarray
    structure: np.ndarray
    unified_override: np.ndarray | None = None

    @property
    def unified(self) -> np.ndarray:
        if self.unified_override is not None:
            return self.unified_override
        return self.attribute | self.structure


def binarize_adjacency(adjacency: sp.csr_matrix) -> sp.csr_matrix:
    result = adjacency.copy().astype(np.float32)
    result.data[:] = 1.0
    result = result.maximum(result.T).tocsr()
    result.setdiag(0)
    result.eliminate_zeros()
    return result


def _mat_path(dataset_dir: Path, dataset: str) -> Path:
    """Prefer an explicitly repaired MAT while preserving the original file."""
    repaired = dataset_dir / f"{dataset}_repaired.mat"
    return repaired if repaired.exists() else dataset_dir / f"{dataset}.mat"


def load_graph(dataset_dir: Path, dataset: str) -> GraphData:
    frame = pd.read_csv(dataset_dir / f"{dataset}.csv")
    if "text" not in frame or "node_id" not in frame:
        raise KeyError("Dataset CSV must contain node_id and text columns")
    # Graph construction only needs the sparse adjacency.  Loading every MAT
    # variable is unnecessarily expensive for Arxiv and can fail while reading
    # a large, unrelated Attributes matrix.
    mat = sio.loadmat(
        _mat_path(dataset_dir, dataset),
        variable_names=["Network"],
    )
    graph = GraphData(
        node_ids=frame["node_id"].astype(str).tolist(),
        texts=frame["text"].fillna("").astype(str).tolist(),
        adjacency=binarize_adjacency(mat["Network"].tocsr()),
    )
    if graph.adjacency.shape[0] != graph.num_nodes:
        raise ValueError("CSV node count and MAT adjacency size do not match")
    return graph


def load_labels(dataset_dir: Path, dataset: str) -> GraphLabels:
    """Load labels only at evaluation time to keep the detector label-free."""
    mat = sio.loadmat(
        _mat_path(dataset_dir, dataset),
        variable_names=[
            "attr_anomaly_label",
            "stru_anomaly_label",
            "unified_anomaly_label",
        ],
    )
    return GraphLabels(
        attribute=np.asarray(mat["attr_anomaly_label"]).ravel().astype(bool),
        structure=np.asarray(mat["stru_anomaly_label"]).ravel().astype(bool),
        unified_override=(
            np.asarray(mat["unified_anomaly_label"]).ravel().astype(bool)
            if "unified_anomaly_label" in mat
            else None
        ),
    )
