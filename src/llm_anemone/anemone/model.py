from __future__ import annotations

import torch
import torch.nn as nn


class GCN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(output_dim))
        self.activation = nn.PReLU()
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        return self.activation(torch.bmm(adjacency, self.linear(features)) + self.bias)


class Readout(nn.Module):
    def __init__(self, mode: str, hidden_dim: int):
        super().__init__()
        self.mode = mode
        self.hidden_dim = hidden_dim

    def forward(self, sequence: torch.Tensor, query: torch.Tensor | None = None) -> torch.Tensor:
        if self.mode == "avg":
            return sequence.mean(1)
        if self.mode == "max":
            return sequence.max(1).values
        if self.mode == "min":
            return sequence.min(1).values
        if query is None:
            raise ValueError("weighted_sum readout requires query")
        weight = torch.softmax(torch.matmul(sequence, query.permute(0, 2, 1)), dim=1)
        return (sequence * weight.expand(-1, -1, self.hidden_dim)).sum(1)


class Discriminator(nn.Module):
    def __init__(self, hidden_dim: int, negative_rounds: int):
        super().__init__()
        self.bilinear = nn.Bilinear(hidden_dim, hidden_dim, 1)
        self.negative_rounds = negative_rounds
        nn.init.xavier_uniform_(self.bilinear.weight)
        nn.init.zeros_(self.bilinear.bias)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        scores = [self.bilinear(left, right)]
        shifted = right
        for _ in range(self.negative_rounds):
            shifted = torch.cat((shifted[-2:-1], shifted[:-1]), dim=0)
            scores.append(self.bilinear(left, shifted))
        return torch.cat(scores, dim=0)


class ANEMONE(nn.Module):
    """Official two-branch patch/context ANEMONE architecture."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        negative_patch: int,
        negative_context: int,
        readout: str,
    ):
        super().__init__()
        self.context_gcn = GCN(input_dim, hidden_dim)
        self.patch_gcn = GCN(input_dim, hidden_dim)
        self.readout = Readout(readout, hidden_dim)
        self.readout_mode = readout
        self.context_discriminator = Discriminator(hidden_dim, negative_context)
        self.patch_discriminator = Discriminator(hidden_dim, negative_patch)

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor):
        context_h = self.context_gcn(features, adjacency)
        patch_h = self.patch_gcn(features, adjacency)
        query = context_h[:, -2:-1] if self.readout_mode == "weighted_sum" else None
        context = self.readout(context_h[:, :-1], query)
        context_logits = self.context_discriminator(context_h[:, -1], context)
        patch_logits = self.patch_discriminator(patch_h[:, -1], patch_h[:, -2])
        return context_logits, patch_logits
