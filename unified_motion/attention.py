"""Attention, rotary embeddings, and adaptive normalization for the motion DiT."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def rotary_table(length: int, width: int, reference: Tensor, base: float) -> tuple[Tensor, Tensor]:
    positions = torch.arange(length, device=reference.device, dtype=reference.dtype)
    exponent = (
        torch.arange(width // 2, device=reference.device, dtype=reference.dtype)
        / (width // 2)
    )
    frequencies = torch.exp(-math.log(base) * exponent)
    angles = torch.outer(positions, frequencies)
    return angles.cos(), angles.sin()


def rotate(x: Tensor, table: tuple[Tensor, Tensor]) -> Tensor:
    cos, sin = (component[None, None].to(x.dtype) for component in table)
    even, odd = x[..., ::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1).flatten(-2)


class SelfAttention(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = width // heads
        self.qkv_proj = nn.Linear(width, 3 * width)
        self.out_proj = nn.Linear(width, width)
        self.attn_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, table: tuple[Tensor, Tensor], valid: Tensor) -> Tensor:
        batch, length, width = x.shape
        q, k, v = (
            item.reshape(batch, length, self.heads, self.head_dim).transpose(1, 2)
            for item in self.qkv_proj(x).chunk(3, -1)
        )
        q, k = rotate(q, table), rotate(k, table)
        scores = (q @ k.transpose(-2, -1)) * (1 / math.sqrt(self.head_dim))
        scores = scores.masked_fill(~valid[:, None, None], float("-inf"))
        weights = self.attn_dropout(scores.softmax(-1))
        values = (weights @ v).transpose(1, 2).reshape(batch, length, width)
        values = values * valid[..., None]
        return self.residual_dropout(self.out_proj(values) * valid[..., None])


class CrossAttention(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = width // heads
        self.query_proj = nn.Linear(width, width)
        self.key_proj = nn.Linear(width, width)
        self.value_proj = nn.Linear(width, width)
        self.out_proj = nn.Linear(width, width)
        self.attn_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, query: Tensor, context: Tensor, valid: Tensor) -> Tensor:
        batch, length, width = query.shape
        context_length = context.shape[1]
        q = self.query_proj(query).reshape(batch, length, self.heads, self.head_dim).transpose(1, 2)
        k = self.key_proj(context).reshape(batch, context_length, self.heads, self.head_dim).transpose(1, 2)
        v = self.value_proj(context).reshape(batch, context_length, self.heads, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) * (1 / math.sqrt(self.head_dim))
        missing = ~valid[:, None, None]
        empty = missing.all(-1, keepdim=True)
        scores = scores.masked_fill(missing & ~empty, float("-inf"))
        weights = torch.where(empty, 0.0, scores.softmax(-1))
        values = (self.attn_dropout(weights) @ v).transpose(1, 2).reshape(batch, length, width)
        # Keep the projection bias for null text, matching the released model.
        return self.residual_dropout(self.out_proj(values))


class AdaptiveNorm(nn.Module):
    def __init__(self, width: int, condition_width: int) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(width, elementwise_affine=False, eps=1e-6)
        self.modulation = nn.Linear(condition_width, 3 * width)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x: Tensor, condition: Tensor) -> tuple[Tensor, Tensor]:
        shift, scale, gate = self.modulation(condition).chunk(3, -1)
        return self.norm(x) * (1 + scale[:, None]) + shift[:, None], gate[:, None]


class FeedForward(nn.Module):
    def __init__(self, width: int, expansion: float, dropout: float) -> None:
        super().__init__()
        inner = int(width * expansion)
        self.input_proj = nn.Linear(width, 2 * inner)
        self.output_proj = nn.Linear(inner, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        hidden, gate = self.input_proj(x).chunk(2, -1)
        return self.dropout(self.output_proj(self.dropout(F.silu(hidden) * gate)))
