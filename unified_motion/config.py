"""Explicit configuration shared by training and released checkpoints."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Literal


Representation = Literal["smg", "combined"]


@dataclass(frozen=True)
class ModelConfig:
    representation: Representation = "combined"
    joints: int = 24
    hidden_dim: int = 432
    layers: int = 12
    heads: int = 12
    expansion: float = 3.0
    dropout: float = 0.1
    rotary_base: float = 10000.0
    text_model: str = "t5-large"
    text_dim: int = 1024
    max_text_tokens: int = 128

    def __post_init__(self) -> None:
        assert self.representation in {"smg", "combined"}
        assert self.joints == 24, "Only the paper's shared 24-joint topology is supported."
        assert self.hidden_dim % self.joints == 0
        assert self.hidden_dim % self.heads == 0
        assert (self.hidden_dim // self.joints) % 2 == 0
        assert (self.hidden_dim // self.heads) % 2 == 0
        assert self.layers > 0 and self.max_text_tokens > 0

    @property
    def smg_dim(self) -> int:
        return 8 + 12 * self.joints

    @property
    def skeleton_dim(self) -> int:
        return 12 * self.joints

    @property
    def feature_dim(self) -> int:
        return self.smg_dim + (self.skeleton_dim if self.representation == "combined" else 0)

    @classmethod
    def for_feature_width(cls, width: int) -> ModelConfig:
        """Use the known paper backbone with the checkpoint's native projections."""
        paper = cls()
        assert width in {paper.smg_dim, paper.feature_dim}, (
            f"Unsupported checkpoint motion width: {width}; expected 296 or 584."
        )
        return replace(paper, representation="smg" if width == paper.smg_dim else "combined")


def generation_guidance(config: ModelConfig) -> dict[str, float]:
    # Text/skeleton CFG weights of the original evaluations for each model.
    if config.representation == "smg":
        return {"text": 2.3, "skeleton": 1.0}
    return {"text": 2.0, "skeleton": 1.0}


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 500
    global_batch_size: int = 512
    window: int = 320
    learning_rate: float = 5e-5
    weight_decay: float = 5e-3
    betas: tuple[float, float] = (0.9, 0.999)
    ema_decay: float = 0.9995
    max_grad_norm: float = 1.0
    drop_both: float = 0.1
    drop_text: float = 0.1
    workers: int = 4
    seed: int = 42
    checkpoint_interval: int = 100
    time_epsilon: float = 0.001

    def __post_init__(self) -> None:
        assert self.epochs > 0 and self.window > 0 and self.global_batch_size > 0
        assert 0 <= self.drop_both <= 1 and 0 <= self.drop_text <= 1
        assert 0 < self.ema_decay < 1
        assert self.checkpoint_interval > 0 and 0 < self.time_epsilon < 1


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def load(cls, path: Path) -> ExperimentConfig:
        payload = json.loads(path.read_text())
        return cls(ModelConfig(**payload["model"]), TrainingConfig(**payload["training"]))

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")
