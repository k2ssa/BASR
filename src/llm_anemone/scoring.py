from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .config import AttributeConfig
from .utils import empirical_rank, selected_positive_rank


@dataclass(frozen=True)
class AttributeScores:
    semantic_change: np.ndarray
    neighbor_gain: np.ndarray
    attribute: np.ndarray
    repaired_features: torch.Tensor


def compute_attribute_scores(
    original: torch.Tensor,
    repaired_nodes: np.ndarray,
    repaired_embeddings: torch.Tensor,
    neighbor_embeddings: torch.Tensor,
    router_score: np.ndarray,
    config: AttributeConfig,
) -> AttributeScores:
    n = len(original)
    repaired = original.clone()
    if len(repaired_nodes):
        repaired[repaired_nodes] = repaired_embeddings
    active = np.zeros(n, dtype=bool)
    active[repaired_nodes] = True
    change = np.zeros(n, dtype=np.float32)
    gain = np.zeros(n, dtype=np.float32)
    if len(repaired_nodes):
        change[repaired_nodes] = (
            1.0 - F.cosine_similarity(original[repaired_nodes], repaired[repaired_nodes], dim=1)
        ).clamp_min(0).numpy()
        before = F.cosine_similarity(original[repaired_nodes], neighbor_embeddings[repaired_nodes], dim=1)
        after = F.cosine_similarity(repaired[repaired_nodes], neighbor_embeddings[repaired_nodes], dim=1)
        gain[repaired_nodes] = (after - before).clamp_min(0).numpy()
    change_score = selected_positive_rank(change, active)
    gain_score = selected_positive_rank(gain, active)
    if config.fusion == "probability_or":
        # Rank-based noisy-OR: either semantic evidence can raise the
        # attribute anomaly score, without introducing a mixing weight.
        attribute = (1.0 - (1.0 - change_score) * (1.0 - gain_score)).astype(np.float32)
    elif config.fusion == "adaptive_evidence":
        # Each node's calibrated evidence determines its own mixing weights:
        # (change^2 + gain^2) / (change + gain). Missing evidence stays zero.
        evidence_sum = change_score + gain_score
        attribute = np.divide(
            change_score**2 + gain_score**2,
            evidence_sum,
            out=np.zeros_like(evidence_sum),
            where=evidence_sum > 0,
        ).astype(np.float32)
    else:
        weights = np.asarray(
            [config.router_weight, config.semantic_change_weight, config.neighbor_gain_weight],
            dtype=np.float32,
        )
        weights /= weights.sum()
        attribute = (
            weights[0] * router_score
            + weights[1] * change_score
            + weights[2] * gain_score
        ).astype(np.float32)
    return AttributeScores(change, gain, attribute, repaired)


def fuse_scores(attribute: np.ndarray, structure_raw: np.ndarray) -> dict[str, np.ndarray]:
    structure = empirical_rank(structure_raw)
    maximum = np.maximum(attribute, structure)
    probability_or = 1.0 - (1.0 - attribute) * (1.0 - structure)
    return {
        "struct_score": structure,
        "unified_max": maximum,
        "unified_prob_or": probability_or,
    }
