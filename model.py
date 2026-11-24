from __future__ import annotations

import torch
from torch import nn


class GraphConv(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.lin_rel = nn.Linear(in_features, out_features)
        self.lin_root = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x, edge_index):
        aggregate = torch.zeros_like(x)
        aggregate.index_add_(0, edge_index[1], x[edge_index[0]])
        return self.lin_rel(aggregate) + self.lin_root(x)


class MLP(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.convs = nn.ModuleList(nn.Linear(a, b) for a, b in zip(channels, channels[1:]))

    def forward(self, x):
        for layer in self.convs[:-1]:
            x = torch.relu(layer(x))
        return self.convs[-1](x)


class PolicyNetwork(nn.Module):
    def __init__(self, variant: str):
        super().__init__()
        if variant not in ("up", "nup"):
            raise ValueError("variant must be up or nup")
        self.variant = variant
        self.convs = nn.ModuleList([GraphConv(64 if variant == "nup" else 14, 256)] +
                                   [GraphConv(256, 256) for _ in range(3)])
        self.actor = MLP([256, 512, 128, 1])
        self.critic = MLP([256, 512, 128, 1])
        if variant == "nup":
            self.origin = MLP([14, 256, 128, 64])
            self.neighbor = MLP([128, 512, 256, 64])

    def forward(self, x, edge_index, batch):
        for layer in self.convs[:-1]:
            x = torch.relu(layer(x, edge_index))
        x = self.convs[-1](x, edge_index)
        graph_count = int(batch.max()) + 1
        pooled = torch.zeros((graph_count, x.shape[1]), dtype=x.dtype, device=x.device)
        pooled.index_add_(0, batch, x)
        counts = torch.bincount(batch, minlength=graph_count).unsqueeze(1)
        return self.actor(x), self.critic(pooled / counts)

def greedy_action(scores: torch.Tensor) -> int:
    probabilities = scores.flatten().softmax(dim=0)
    return int(torch.nonzero((probabilities - probabilities.max()).abs() < 1e-4)[0, 0])
