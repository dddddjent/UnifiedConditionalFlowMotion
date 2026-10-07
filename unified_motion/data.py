"""Motion manifests and window batching for the shared SMG + Mixamo dataset."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from unified_motion.config import ModelConfig
from unified_motion.statistics import Statistics


@dataclass(frozen=True)
class MotionRecord:
    id: str
    motion: str
    skeleton: str
    source: str
    character: str
    captions: list[str]
    start: int
    end: int


def resolve_input(data_root: Path, path: Path) -> Path:
    """Motion, skeleton, and pair inputs are relative to --data-root unless absolute."""
    return path if path.is_absolute() else data_root / path


def read_records(path: Path) -> list[MotionRecord]:
    records = [
        MotionRecord(**json.loads(line))
        for line in path.read_text().splitlines() if line.strip()
    ]
    assert records and len({record.id for record in records}) == len(records)
    for record in records:
        assert record.source in {"smg", "mixamo"}
        assert record.captions and all(caption.strip() for caption in record.captions)
        assert 0 <= record.start < record.end
    return records


class MotionDataset(Dataset[dict[str, Any]]):
    def __init__(
        self, root: Path, split: str, config: ModelConfig, statistics: Statistics
    ) -> None:
        self.root, self.config, self.statistics = root, config, statistics
        self.records = read_records(root / "splits" / f"{split}.jsonl")
        self.annotations = [
            (record, caption) for record in self.records for caption in record.captions
        ]

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record, caption = self.annotations[index]
        features = load_motion(self.root / record.motion, self.config)[record.start:record.end]
        assert len(features) == record.end - record.start, (
            f"Clip bounds exceed motion length: {record.id}"
        )
        skeleton = load_skeleton(self.root / record.skeleton, self.config)
        return {
            "motion": self.statistics.normalize_motion(torch.from_numpy(features.copy())),
            "skeleton": self.statistics.normalize_skeleton(torch.from_numpy(skeleton.copy())),
            "caption": caption,
            "id": record.id,
        }


def read_motion(path: Path) -> np.ndarray:
    """Read unnormalized native motion features without changing their representation."""
    value = np.load(path, allow_pickle=False)
    assert value.ndim == 2 and value.shape[0] > 0 and np.isfinite(value).all()
    return np.asarray(value, dtype=np.float32)


def load_motion(path: Path, config: ModelConfig) -> np.ndarray:
    """Select existing motion features for the checkpoint's native input width."""
    value = read_motion(path)
    if config.representation == "smg" and value.shape[1] == config.smg_dim + config.skeleton_dim:
        value = value[:, :config.smg_dim]
    assert value.shape[1] == config.feature_dim, (
        f"Motion has {value.shape[1]} channels; checkpoint needs {config.feature_dim}."
    )
    return value


def load_skeleton(path: Path, config: ModelConfig) -> np.ndarray:
    value = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32).reshape(-1)
    assert value.shape == (config.skeleton_dim,) and np.isfinite(value).all()
    return value


class WindowBatch:
    def __init__(self, window: int, *, random_crop: bool = True) -> None:
        self.window, self.random_crop = window, random_crop

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        batch = len(samples)
        motions = torch.zeros(batch, self.window, samples[0]["motion"].shape[-1])
        mask = torch.zeros(batch, self.window, dtype=torch.bool)
        for index, sample in enumerate(samples):
            value = sample["motion"]
            length = min(len(value), self.window)
            extra = len(value) - length
            start = random.randint(0, extra) if self.random_crop else extra // 2
            motions[index, :length] = value[start:start + length]
            mask[index, :length] = True
        return {
            "motion": motions,
            "mask": mask,
            "skeleton": torch.stack([sample["skeleton"] for sample in samples]),
            "captions": [sample["caption"] for sample in samples],
        }
