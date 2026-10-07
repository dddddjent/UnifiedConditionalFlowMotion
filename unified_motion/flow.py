"""Rectified-flow generation and shared-noise FlowEdit transport."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from unified_motion.model import MotionTransformer


@dataclass
class Condition:
    text: Tensor
    text_mask: Tensor
    skeleton: Tensor

    def null_text(self) -> Condition:
        return Condition(
            torch.zeros_like(self.text), torch.zeros_like(self.text_mask), self.skeleton
        )


@dataclass(frozen=True)
class Guidance:
    text: float = 1.5
    skeleton: float = 1.0


def predict_target(
    model: MotionTransformer,
    state: Tensor,
    time: float,
    condition: Condition,
    mask: Tensor,
    guidance: Guidance,
) -> Tensor:
    """Paper Eq. 11 on clean-target predictions: u + w_s(s - u) + w_t(ts - s)."""
    times = state.new_full((state.shape[0],), time)
    empty_text = torch.zeros_like(condition.text)
    empty_mask = torch.zeros_like(condition.text_mask)
    empty_skeleton = torch.zeros_like(condition.skeleton)

    def predict(text: Tensor, text_mask: Tensor, skeleton: Tensor) -> Tensor:
        return model(state, times, text, text_mask, skeleton, mask)

    unconditioned = predict(empty_text, empty_mask, empty_skeleton)
    skeleton = predict(empty_text, empty_mask, condition.skeleton)
    result = unconditioned + guidance.skeleton * (skeleton - unconditioned)
    if guidance.text != 0:
        joint = predict(condition.text, condition.text_mask, condition.skeleton)
        result = result + guidance.text * (joint - skeleton)
    return result * mask[..., None]


def predict_velocity(
    model: MotionTransformer,
    state: Tensor,
    time: float,
    condition: Condition,
    mask: Tensor,
    guidance: Guidance,
    epsilon: float,
) -> Tensor:
    assert 0 < epsilon < 1
    safe_time = min(time, 1 - epsilon)
    target = predict_target(model, state, safe_time, condition, mask, guidance)
    return (target - state) / (1 - safe_time) * mask[..., None]


@torch.no_grad()
def generate(
    model: MotionTransformer,
    condition: Condition,
    mask: Tensor,
    guidance: Guidance,
    generator: torch.Generator,
    *,
    steps: int = 100,
    method: Literal["euler", "heun", "rk4"] = "rk4",
    epsilon: float = 0.001,
) -> Tensor:
    """Integrate the flow, then use the terminal clean-target prediction directly."""
    assert steps > 0 and method in {"euler", "heun", "rk4"}
    state = torch.randn(
        (*mask.shape, model.config.feature_dim), generator=generator, device=mask.device
    ) * mask[..., None]
    delta = 1 / steps
    for step in range(steps - 1):
        time = step / steps
        k1 = predict_velocity(model, state, time, condition, mask, guidance, epsilon)
        if method == "euler":
            update = k1
        elif method == "heun":
            k2 = predict_velocity(
                model, state + delta * k1, time + delta, condition, mask, guidance, epsilon
            )
            update = (k1 + k2) / 2
        else:
            k2 = predict_velocity(
                model, state + delta * k1 / 2, time + delta / 2,
                condition, mask, guidance, epsilon,
            )
            k3 = predict_velocity(
                model, state + delta * k2 / 2, time + delta / 2,
                condition, mask, guidance, epsilon,
            )
            k4 = predict_velocity(
                model, state + delta * k3, time + delta, condition, mask, guidance, epsilon
            )
            update = (k1 + 2 * k2 + 2 * k3 + k4) / 6
        state = (state + delta * update) * mask[..., None]
    # The original sampler calls the clean-target model at t=1 for the final step.
    return predict_target(model, state, 1.0, condition, mask, guidance)


@torch.no_grad()
def transport(
    model: MotionTransformer,
    source: Tensor,
    source_condition: Condition,
    target_condition: Condition,
    mask: Tensor,
    source_guidance: Guidance,
    target_guidance: Guidance,
    generator: torch.Generator,
    *,
    steps: int = 100,
    start_step: int = 10,
    tail_steps: int = 0,
    averages: int = 1,
) -> Tensor:
    """FlowEdit as in the original research code.

    Each step adds the difference of guided clean-target predictions (not velocities),
    evaluated at the unclamped grid time; tail steps add the target prediction alone.
    """
    assert 0 <= start_step < steps
    assert 0 <= tail_steps <= steps - start_step and averages > 0
    source = source * mask[..., None]
    state = source.clone()
    for step in range(start_step, steps):
        time = step / steps
        if steps - step <= tail_steps:
            update = predict_target(model, state, time, target_condition, mask, target_guidance)
        else:
            update = torch.zeros_like(state)
            for _ in range(averages):
                noise = torch.randn(source.shape, device=source.device, generator=generator) * mask[..., None]
                source_state = time * source + (1 - time) * noise
                target_state = state + source_state - source
                source_target = predict_target(
                    model, source_state, time, source_condition, mask, source_guidance
                )
                target_target = predict_target(
                    model, target_state, time, target_condition, mask, target_guidance
                )
                update += target_target - source_target
            update /= averages
        state = (state + update / steps) * mask[..., None]
    return state
