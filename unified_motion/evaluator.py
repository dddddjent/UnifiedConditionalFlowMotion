"""SnapMoGen's frozen text/motion embedding architecture, without training machinery."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import Tensor, nn

from unified_motion.text import TextEncoder


class PositionEncoding(nn.Module):
    def __init__(self, width: int = 256) -> None:
        super().__init__()
        angles = torch.arange(5000).float()[:, None] * torch.exp(
            torch.arange(0, width, 2).float() * (-math.log(10000.0) / width)
        )
        table = torch.zeros(5000, width)
        table[:, 0::2], table[:, 1::2] = angles.sin(), angles.cos()
        self.register_buffer("pe", table[:, None], persistent=False)
        self.dropout = nn.Dropout(0.1)

    def forward(self, value: Tensor) -> Tensor:
        return self.dropout(value + self.pe[:value.shape[1]].transpose(0, 1))


class EmbeddingEncoder(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(feature_dim, 256)
        self.tokens = nn.Parameter(torch.randn(2, 256))
        self.sequence_pos_encoding = PositionEncoding()
        layer = nn.TransformerEncoderLayer(
            256, 4, 1024, 0.1, activation="gelu", batch_first=True
        )
        self.seqTransEncoder = nn.TransformerEncoder(layer, 6, enable_nested_tensor=False)
        self.linear = nn.Linear(256, 256)

    def forward(self, features: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        tokens = self.tokens[None].expand(len(features), -1, -1)
        value = torch.cat((tokens, self.projection(features)), 1)
        valid = torch.cat((
            torch.ones(len(features), 2, device=mask.device, dtype=torch.bool), mask
        ), 1)
        hidden = self.seqTransEncoder(
            self.sequence_pos_encoding(value), src_key_padding_mask=~valid
        )
        return hidden[:, 0], self.linear(hidden[:, :2])[:, 0]


class MotionEvaluator:
    def __init__(self, checkpoint: Path, device: torch.device) -> None:
        self.device = device
        self.motion = EmbeddingEncoder(148).to(device).eval()
        self.text = EmbeddingEncoder(768).to(device).eval()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.motion.load_state_dict(payload["latent_enc"], strict=True)
        self.text.load_state_dict(payload["text_enc"], strict=True)
        self.motion.requires_grad_(False)
        self.text.requires_grad_(False)
        self.encoder = TextEncoder("google/t5-v1_1-base", device, 120, legacy=False)

    @torch.no_grad()
    def encode_motion(self, value: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        return self.motion(value.to(self.device), mask.to(self.device))

    @torch.no_grad()
    def encode_text(self, prompts: list[str]) -> Tensor:
        encoded, mask = self.encoder.encode(prompts)
        return self.text(encoded, mask)[1]
