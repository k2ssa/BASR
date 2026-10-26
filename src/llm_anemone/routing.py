from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from .config import RouterConfig
from .utils import empirical_rank


@dataclass(frozen=True)
class RouterOutput:
    score: np.ndarray
    intra: np.ndarray
    neighbor: np.ndarray
    max_pair: np.ndarray
    candidate_indices: np.ndarray
    center_weight: np.ndarray | None = None
    pair_weight: np.ndarray | None = None

    @property
    def candidate_mask(self) -> np.ndarray:
        mask = np.zeros(len(self.score), dtype=bool)
        mask[self.candidate_indices] = True
        return mask


def split_sentences(text: str, min_chars: int = 12) -> list[str]:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text:
        return [""]
    chunks = re.split(r"(?<=[.!?])\s+|\s*;\s+", text)
    chunks = [part.strip() for part in chunks if len(part.strip()) >= min_chars]
    return chunks or [text]


def neighbor_means(features: torch.Tensor, adjacency: sp.csr_matrix) -> torch.Tensor:
    degree = np.asarray(adjacency.sum(1)).ravel().astype(np.float32)
    inverse = np.zeros_like(degree)
    np.divide(1.0, degree, out=inverse, where=degree > 0)
    values = (sp.diags(inverse) @ adjacency).dot(features.numpy())
    result = torch.from_numpy(np.asarray(values)).float()
    result[degree == 0] = features[degree == 0]
    return result


def _top_mean(values: np.ndarray, fraction: float) -> float:
    if len(values) == 0:
        return 0.0
    count = max(1, int(math.ceil(len(values) * fraction)))
    return float(np.partition(values, len(values) - count)[-count:].mean())


def _mad_tail(values: np.ndarray) -> float:
    """Parameter-free robust tail score used by the dynamic router."""
    if len(values) == 0:
        return 0.0
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    z = (values - median) / (1.4826 * mad + 1e-6)
    positive = z[z > 0]
    return float(positive.max()) if len(positive) else 0.0


def compute_signals(
    sentence_embeddings: torch.Tensor,
    offsets: np.ndarray,
    neighbor_embeddings: torch.Tensor,
    config: RouterConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sentences = F.normalize(sentence_embeddings.float(), p=2, dim=1)
    neighbors = F.normalize(neighbor_embeddings.float(), p=2, dim=1)
    n = len(offsets) - 1
    intra = np.zeros(n, dtype=np.float32)
    neighbor = np.zeros(n, dtype=np.float32)
    max_pair = np.zeros(n, dtype=np.float32)
    aggregate = (
        (lambda x: _top_mean(x, config.sentence_fraction))
        if config.method == "fixed_top_fraction"
        else _mad_tail
    )
    for node in range(n):
        group = sentences[offsets[node] : offsets[node + 1]]
        if len(group) <= 1:
            continue
        center = F.normalize(group.mean(0, keepdim=True), p=2, dim=1)
        doc_distance = (1.0 - group @ center.T).squeeze(1).numpy()
        neighbor_distance = (1.0 - group @ neighbors[node]).numpy()
        intra[node] = aggregate(doc_distance)
        neighbor[node] = aggregate(neighbor_distance)
        max_pair[node] = float(1.0 - (group @ group.T).min())
    return intra, neighbor, max_pair


def _rank_valid(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Rank only observed values; missing evidence keeps probability zero."""
    result = np.zeros(len(values), dtype=np.float32)
    indices = np.flatnonzero(valid)
    if len(indices):
        result[indices] = empirical_rank(values[indices])
    return result


def compute_self_calibrated_score(
    sentence_embeddings: torch.Tensor,
    offsets: np.ndarray,
    neighbor_embeddings: torch.Tensor,
    has_neighbors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parameter-free dual-context router.

    Sentence-level internal and neighborhood distances are independently mapped to
    empirical probabilities.  Probability-OR combines both sources, and mean plus
    standard deviation aggregates sentence evidence into a node score.
    """
    sentences = F.normalize(sentence_embeddings.float(), p=2, dim=1)
    neighbors = F.normalize(neighbor_embeddings.float(), p=2, dim=1)
    sentence_count = len(sentences)
    node_count = len(offsets) - 1

    internal_distance = np.zeros(sentence_count, dtype=np.float32)
    neighbor_distance = np.zeros(sentence_count, dtype=np.float32)
    valid_internal = np.zeros(sentence_count, dtype=bool)
    valid_neighbor = np.zeros(sentence_count, dtype=bool)
    max_pair = np.zeros(node_count, dtype=np.float32)

    for node in range(node_count):
        start, end = int(offsets[node]), int(offsets[node + 1])
        group = sentences[start:end]
        if len(group) > 1:
            # Each sentence is compared with a center that does not contain itself.
            leave_one_out = F.normalize(
                group.sum(dim=0, keepdim=True) - group,
                p=2,
                dim=1,
            )
            internal_distance[start:end] = (
                1.0 - (group * leave_one_out).sum(dim=1)
            ).clamp_min(0).numpy()
            valid_internal[start:end] = True
            # Kept only as an interpretable diagnostic; it does not enter the score.
            max_pair[node] = float(1.0 - (group @ group.T).min())
        if has_neighbors[node]:
            neighbor_distance[start:end] = (
                1.0 - group @ neighbors[node]
            ).clamp_min(0).numpy()
            valid_neighbor[start:end] = True

    internal_probability = _rank_valid(internal_distance, valid_internal)
    neighbor_probability = _rank_valid(neighbor_distance, valid_neighbor)
    sentence_probability = 1.0 - (
        (1.0 - internal_probability) * (1.0 - neighbor_probability)
    )

    internal = np.zeros(node_count, dtype=np.float32)
    neighborhood = np.zeros(node_count, dtype=np.float32)
    raw_score = np.zeros(node_count, dtype=np.float32)
    has_evidence = np.zeros(node_count, dtype=bool)
    for node in range(node_count):
        start, end = int(offsets[node]), int(offsets[node + 1])
        internal_values = internal_probability[start:end]
        neighbor_values = neighbor_probability[start:end]
        joint_values = sentence_probability[start:end]
        internal[node] = float(internal_values.mean() + internal_values.std())
        neighborhood[node] = float(neighbor_values.mean() + neighbor_values.std())
        raw_score[node] = float(joint_values.mean() + joint_values.std())
        has_evidence[node] = bool(
            valid_internal[start:end].any() or valid_neighbor[start:end].any()
        )

    score = empirical_rank(raw_score).astype(np.float32)
    score[~has_evidence] = 0.0
    return score, internal, neighborhood, max_pair


def compute_adaptive_internal_score(
    sentence_embeddings: torch.Tensor,
    offsets: np.ndarray,
    neighbor_embeddings: torch.Tensor,
    has_neighbors: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Attribute-centric router with node-adaptive internal evidence weights.

    The score uses only leave-one-out center deviation and maximum sentence-pair
    conflict. Neighborhood deviation is returned for diagnostics, never scoring.
    """
    sentences = F.normalize(sentence_embeddings.float(), p=2, dim=1)
    neighbors = F.normalize(neighbor_embeddings.float(), p=2, dim=1)
    node_count = len(offsets) - 1
    center_raw = np.zeros(node_count, dtype=np.float32)
    pair_raw = np.zeros(node_count, dtype=np.float32)
    neighbor_raw = np.zeros(node_count, dtype=np.float32)
    valid_internal = np.zeros(node_count, dtype=bool)

    for node in range(node_count):
        start, end = int(offsets[node]), int(offsets[node + 1])
        group = sentences[start:end]
        if len(group) > 1:
            leave_one_out = F.normalize(
                group.sum(dim=0, keepdim=True) - group,
                p=2,
                dim=1,
            )
            distances = (
                1.0 - (group * leave_one_out).sum(dim=1)
            ).clamp_min(0).numpy()
            center_raw[node] = float(distances.mean() + distances.std())
            pair_raw[node] = float(1.0 - (group @ group.T).min())
            valid_internal[node] = True

        # Diagnostic only: this value never enters the adaptive_internal score.
        if has_neighbors[node]:
            distances = (1.0 - group @ neighbors[node]).clamp_min(0).numpy()
            neighbor_raw[node] = float(distances.mean() + distances.std())

    center_probability = _rank_valid(center_raw, valid_internal)
    pair_probability = _rank_valid(pair_raw, valid_internal)
    evidence_sum = center_probability + pair_probability
    center_weight = np.divide(
        center_probability,
        evidence_sum,
        out=np.zeros_like(center_probability),
        where=evidence_sum > 0,
    )
    pair_weight = np.divide(
        pair_probability,
        evidence_sum,
        out=np.zeros_like(pair_probability),
        where=evidence_sum > 0,
    )
    score = (
        center_weight * center_probability + pair_weight * pair_probability
    ).astype(np.float32)
    return score, center_raw, neighbor_raw, pair_raw, center_weight, pair_weight


def route(
    sentence_embeddings: torch.Tensor | None,
    offsets: np.ndarray,
    neighbor_embeddings: torch.Tensor,
    config: RouterConfig,
    has_neighbors: np.ndarray | None = None,
    node_embeddings: torch.Tensor | None = None,
) -> RouterOutput:
    if has_neighbors is None:
        has_neighbors = np.ones(len(offsets) - 1, dtype=bool)
    node_count = len(offsets) - 1
    count = int(math.ceil(node_count * config.budget_ratio))

    if config.method == "random":
        # The score is a seeded random permutation represented as ranks.  It is
        # label-free and makes repeated equal-budget trials exactly reproducible.
        rng = np.random.default_rng(config.seed)
        permutation = rng.permutation(node_count)
        candidates = permutation[:count]
        score = np.empty(node_count, dtype=np.float32)
        score[permutation] = np.linspace(1.0, 0.0, node_count, dtype=np.float32)
        zeros = np.zeros(node_count, dtype=np.float32)
        return RouterOutput(score, zeros, zeros, zeros, candidates)

    if config.method == "whole_document":
        if node_embeddings is None:
            raise ValueError("whole_document routing requires node_embeddings")
        documents = F.normalize(node_embeddings.float(), p=2, dim=1)
        neighbors = F.normalize(neighbor_embeddings.float(), p=2, dim=1)
        raw = (1.0 - (documents * neighbors).sum(dim=1)).clamp_min(0).numpy()
        valid = np.asarray(has_neighbors, dtype=bool)
        score = _rank_valid(raw, valid)
        candidates = np.argsort(-score, kind="stable")[:count]
        zeros = np.zeros(node_count, dtype=np.float32)
        # Store the document--neighborhood distance in the neighbor diagnostic.
        return RouterOutput(score, zeros, raw.astype(np.float32), zeros, candidates)

    if sentence_embeddings is None:
        raise ValueError(f"{config.method} routing requires sentence_embeddings")
    if config.method == "self_calibrated":
        score, intra, neighbor, max_pair = compute_self_calibrated_score(
            sentence_embeddings,
            offsets,
            neighbor_embeddings,
            np.asarray(has_neighbors, dtype=bool),
        )
        candidates = np.argsort(-score, kind="stable")[:count]
        return RouterOutput(score, intra, neighbor, max_pair, candidates)
    if config.method == "adaptive_internal":
        score, intra, neighbor, max_pair, center_weight, pair_weight = (
            compute_adaptive_internal_score(
                sentence_embeddings,
                offsets,
                neighbor_embeddings,
                np.asarray(has_neighbors, dtype=bool),
            )
        )
        candidates = np.argsort(-score, kind="stable")[:count]
        return RouterOutput(
            score,
            intra,
            neighbor,
            max_pair,
            candidates,
            center_weight,
            pair_weight,
        )

    intra, neighbor, max_pair = compute_signals(
        sentence_embeddings, offsets, neighbor_embeddings, config
    )
    within = (
        config.intra_weight * empirical_rank(intra)
        + (1.0 - config.intra_weight) * empirical_rank(max_pair)
    )
    score = (
        config.within_weight * within
        + (1.0 - config.within_weight) * empirical_rank(neighbor)
    ).astype(np.float32)
    candidates = np.argsort(-score, kind="stable")[:count]
    return RouterOutput(score, intra, neighbor, max_pair, candidates)
