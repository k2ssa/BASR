from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from ..config import AnemoneConfig
from ..utils import choose_device, set_seed
from .model import ANEMONE
from .sampling import RWRSubgraphSampler, make_batch, normalize_adjacency_official


def normalize_features(features: torch.Tensor, mode: str) -> torch.Tensor:
    features = features.float().clone()
    if mode == "none":
        return features
    if mode == "l2":
        return torch.nn.functional.normalize(features, p=2, dim=1)
    if mode != "official-row":
        raise ValueError(f"Unknown feature normalization: {mode}")
    row_sum = features.sum(1, keepdim=True)
    safe = torch.where(row_sum.abs() > 1e-12, row_sum, torch.ones_like(row_sum))
    return features / safe


def _labels(batch_size: int, negatives: int, device: torch.device) -> torch.Tensor:
    return torch.cat((torch.ones(batch_size), torch.zeros(batch_size * negatives))).unsqueeze(1).to(device)


def _component_score(logits: torch.Tensor, batch_size: int, negatives: int) -> np.ndarray:
    probabilities = torch.sigmoid(logits.squeeze(1))
    positive = probabilities[:batch_size]
    negative = probabilities[batch_size:].reshape(negatives, batch_size).mean(0)
    return (negative - positive).detach().cpu().numpy()


def _train(
    model: ANEMONE,
    features: torch.Tensor,
    adjacency: sp.csr_matrix,
    sampler: RWRSubgraphSampler,
    config: AnemoneConfig,
    device: torch.device,
    checkpoint: Path,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    patch_loss = nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor([config.negative_patch], device=device))
    context_loss = nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor([config.negative_context], device=device))
    best, wait = float("inf"), 0
    for epoch in range(config.epochs):
        model.train()
        order, subgraphs = np.random.permutation(len(features)), sampler.sample_all()
        total = 0.0
        for start in range(0, len(features), config.batch_size):
            index = order[start : start + config.batch_size]
            batch_features, batch_adjacency = make_batch(index, subgraphs, features, adjacency, device)
            context_logits, patch_logits = model(batch_features, batch_adjacency)
            loss_context = context_loss(context_logits, _labels(len(index), config.negative_context, device)).mean()
            loss_patch = patch_loss(patch_logits, _labels(len(index), config.negative_patch, device)).mean()
            loss = config.alpha * loss_context + (1.0 - config.alpha) * loss_patch
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(index)
        mean_loss = total / len(features)
        if mean_loss < best:
            best, wait = mean_loss, 0
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict()}, checkpoint)
        else:
            wait += 1
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"[ANEMONE] epoch={epoch + 1:03d} loss={mean_loss:.8f} best={best:.8f}")
        if wait >= config.patience:
            break
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["model"])


@torch.no_grad()
def _score(
    model: ANEMONE,
    features: torch.Tensor,
    adjacency: sp.csr_matrix,
    sampler: RWRSubgraphSampler,
    config: AnemoneConfig,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    n = len(features)
    context = np.zeros((config.eval_rounds, n), dtype=np.float32)
    patch = np.zeros_like(context)
    for round_index in range(config.eval_rounds):
        order, subgraphs = np.random.permutation(n), sampler.sample_all()
        for start in range(0, n, config.batch_size):
            index = order[start : start + config.batch_size]
            batch_features, batch_adjacency = make_batch(index, subgraphs, features, adjacency, device)
            context_logits, patch_logits = model(batch_features, batch_adjacency)
            context[round_index, index] = _component_score(context_logits, len(index), config.negative_context)
            patch[round_index, index] = _component_score(patch_logits, len(index), config.negative_patch)
        if round_index == 0 or (round_index + 1) % 32 == 0:
            print(f"[ANEMONE] evaluation {round_index + 1}/{config.eval_rounds}")
    mixed = config.alpha * context + (1.0 - config.alpha) * patch
    estimate = lambda value: value.mean(0) + value.std(0)
    return {"anemone_score": estimate(mixed), "anemone_context": estimate(context), "anemone_patch": estimate(patch)}


def train_and_score(
    features: torch.Tensor,
    adjacency: sp.csr_matrix,
    config: AnemoneConfig,
    checkpoint: Path,
) -> dict[str, np.ndarray]:
    set_seed(config.seed)
    device = choose_device(config.device)
    features = normalize_features(features, config.feature_normalization)
    normalized_adjacency = normalize_adjacency_official(adjacency)
    sampler = RWRSubgraphSampler(adjacency, config.subgraph_size, config.restart_prob, config.seed)
    model = ANEMONE(
        features.shape[1], config.embedding_dim, config.negative_patch,
        config.negative_context, config.readout,
    ).to(device)
    _train(model, features, normalized_adjacency, sampler, config, device, checkpoint)
    return _score(model, features, normalized_adjacency, sampler, config, device)
