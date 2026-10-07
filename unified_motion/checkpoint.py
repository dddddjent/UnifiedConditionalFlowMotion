"""Strict checkpoint loading for the released weights and new training runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from unified_motion.config import ModelConfig, TrainingConfig, generation_guidance
from unified_motion.model import MotionTransformer
from unified_motion.statistics import Statistics


@dataclass
class MotionBundle:
    model: MotionTransformer
    statistics: Statistics
    epoch: int
    time_epsilon: float
    guidance: dict[str, Any]
    format: str
    statistics_source: str

    @classmethod
    def load(
        cls,
        checkpoint: Path,
        data_root: Path,
        device: torch.device,
    ) -> MotionBundle:
        assert checkpoint.is_file(), f"Checkpoint does not exist: {checkpoint}"
        assert data_root.is_dir(), f"Dataset root does not exist: {data_root}"
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        checkpoint_format = payload.get("format")
        assert "ema_model" in payload and "epoch" in payload, (
            "A supported checkpoint must contain ema_model and epoch."
        )
        state, epoch = payload["ema_model"], int(payload["epoch"])
        if checkpoint_format is None:
            prefix = "velocity_model."
            assert state and all(key.startswith(prefix) for key in state), (
                "Unexpected original checkpoint parameter names."
            )
            state = {key.removeprefix(prefix): value for key, value in state.items()}
            projection = state["motion_input_projection.weight"]
            assert projection.ndim == 2
            config = ModelConfig.for_feature_width(projection.shape[1])
            epsilon = TrainingConfig().time_epsilon
            guidance = generation_guidance(config)
            checkpoint_format = "original-research-checkpoint"
        else:
            assert checkpoint_format == "unified-motion-v1", (
                f"Unsupported checkpoint format: {checkpoint_format}"
            )
            config = ModelConfig(**payload["model_config"])
            epsilon, guidance = float(payload["time_epsilon"]), payload["guidance"]
        assert 0 < epsilon < 1
        assert state["motion_input_projection.weight"].shape == (config.hidden_dim, config.feature_dim)
        assert state["motion_output_projection.weight"].shape == (config.feature_dim, config.hidden_dim)
        assert state["skeleton_mlp.0.weight"].shape == (config.hidden_dim, config.skeleton_dim)
        statistics_source = str(data_root / "metadata")
        normalization = Statistics.load(data_root / "metadata", config)
        if checkpoint_format == "unified-motion-v1":
            training_normalization = Statistics.from_payload(payload["statistics"], config)
            for name, value in normalization.payload().items():
                assert torch.equal(value, getattr(training_normalization, name)), (
                    f"Dataset metadata disagrees with checkpoint training normalization: "
                    f"{data_root / 'metadata' / f'{name}.npy'}"
                )
        # Retain EMA tensors while releasing the unused raw weights and optimizer.
        del payload
        model = MotionTransformer(config)
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
        return cls(
            model, normalization, epoch, epsilon, guidance,
            checkpoint_format, statistics_source,
        )


def save_checkpoint(
    path: Path,
    model: MotionTransformer,
    ema: MotionTransformer,
    statistics: Statistics,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    epsilon: float,
    rng_states: list[dict[str, Any]],
    experiment_config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "unified-motion-v1",
        "model_config": asdict(model.config),
        "model": model.state_dict(),
        "ema_model": ema.state_dict(),
        # Snapshot for resume and dataset checks; runtime normalization comes from data_root.
        "statistics": statistics.payload(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "time_epsilon": epsilon,
        "guidance": generation_guidance(model.config),
        "rng_states": rng_states,
        "experiment_config": experiment_config,
    }
    temporary = path.with_suffix(".partial")
    torch.save(payload, temporary)
    temporary.replace(path)
