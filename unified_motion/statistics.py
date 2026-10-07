"""Dataset metadata supplies normalization in the checkpoint's native feature space."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from unified_motion.config import ModelConfig


@dataclass
class Statistics:
    motion_mean: Tensor
    motion_std: Tensor
    skeleton_mean: Tensor
    skeleton_std: Tensor

    @classmethod
    def load(cls, directory: Path, config: ModelConfig) -> Statistics:
        assert directory.is_dir(), f"Normalization directory does not exist: {directory}"
        names = ("motion_mean", "motion_std", "skeleton_mean", "skeleton_std")
        values = {
            name: torch.from_numpy(
                np.load(directory / f"{name}.npy", allow_pickle=False)
            ).float().flatten()
            for name in names
        }
        result = cls(**values)
        result.validate(config)
        return result

    @classmethod
    def from_payload(cls, values: dict[str, Tensor], config: ModelConfig) -> Statistics:
        result = cls(**values)
        result.validate(config)
        return result

    def validate(self, config: ModelConfig) -> None:
        for name in ("motion_mean", "motion_std", "skeleton_mean", "skeleton_std"):
            value = getattr(self, name)
            width = config.feature_dim if name.startswith("motion") else config.skeleton_dim
            assert value.shape == (width,), (
                f"{name}: expected {width} channels, received {tuple(value.shape)}"
            )
            assert torch.isfinite(value).all(), f"{name} contains non-finite values."
            if name.endswith("std"):
                assert (value >= 0).all(), f"{name} contains negative standard deviations."
                setattr(self, name, torch.where(value == 0, 1.0, value))

    def normalize_motion(self, value: Tensor) -> Tensor:
        return (value - self.motion_mean.to(value)) / self.motion_std.to(value)

    def denormalize_motion(self, value: Tensor) -> Tensor:
        return value * self.motion_std.to(value) + self.motion_mean.to(value)

    def normalize_skeleton(self, value: Tensor) -> Tensor:
        return (value - self.skeleton_mean.to(value)) / self.skeleton_std.to(value)

    def payload(self) -> dict[str, Tensor]:
        return {
            name: getattr(self, name).detach().cpu()
            for name in ("motion_mean", "motion_std", "skeleton_mean", "skeleton_std")
        }

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for name, value in self.payload().items():
            np.save(directory / f"{name}.npy", value.numpy())
