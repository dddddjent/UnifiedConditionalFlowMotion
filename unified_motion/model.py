"""The shared text- and skeleton-conditioned joint/frame transformer."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from unified_motion.attention import (
    AdaptiveNorm, CrossAttention, FeedForward, SelfAttention, rotary_table,
)
from unified_motion.config import ModelConfig


class MotionBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        width, joint_width = config.hidden_dim, config.hidden_dim // config.joints
        self.joint_attention = SelfAttention(joint_width, 1, config.dropout)
        self.frame_attention = SelfAttention(width, config.heads, config.dropout)
        self.feed_forward_frame = FeedForward(width, config.expansion, config.dropout)
        self.ada_norm_joint = AdaptiveNorm(joint_width, width)
        self.ada_norm_frame = AdaptiveNorm(width, width)
        self.ada_norm_ff = AdaptiveNorm(width, width)
        self.joint_cross_attention = CrossAttention(joint_width, 1, config.dropout)
        self.frame_cross_attention = CrossAttention(width, config.heads, config.dropout)
        self.joint_cross_norm = nn.RMSNorm(joint_width, eps=1e-6)
        self.frame_cross_norm = nn.RMSNorm(width, eps=1e-6)

    def forward(
        self,
        text_frames: Tensor,
        text_joints: Tensor,
        motion: Tensor,
        condition: Tensor,
        frame_table: tuple[Tensor, Tensor],
        joint_table: tuple[Tensor, Tensor],
        text_mask: Tensor,
        frame_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch, frames, joints, joint_width = motion.shape
        text_length = text_frames.shape[1]
        joint_valid = frame_mask[..., None].expand(-1, -1, joints).reshape(batch * frames, joints)
        attention_valid = joint_valid.clone()
        attention_valid[~attention_valid.any(-1), 0] = True
        joint_condition = condition[:, None].expand(-1, frames, -1).reshape(batch * frames, -1)
        hidden = motion.reshape(batch * frames, joints, joint_width)
        normalized, gate = self.ada_norm_joint(hidden, joint_condition)
        hidden = (
            hidden + gate * self.joint_attention(normalized, joint_table, attention_valid)
        ) * joint_valid[..., None]
        joint_context = (
            text_joints[:, None].expand(-1, frames, -1, -1)
            .reshape(batch * frames, text_length, joint_width)
        )
        joint_text_mask = (
            text_mask[:, None].expand(-1, frames, -1).reshape(batch * frames, text_length)
        )
        hidden = (
            hidden + self.joint_cross_attention(
                self.joint_cross_norm(hidden), joint_context, joint_text_mask
            )
        ) * joint_valid[..., None]
        motion_frames = hidden.reshape(batch, frames, joints * joint_width)
        combined = torch.cat((text_frames, motion_frames), 1)
        combined_mask = torch.cat((text_mask, frame_mask), 1)
        normalized, gate = self.ada_norm_frame(combined, condition)
        combined = (
            combined + gate * self.frame_attention(normalized, frame_table, combined_mask)
        ) * combined_mask[..., None]
        updated_frames = combined[:, text_length:]
        updated_frames = (
            updated_frames + self.frame_cross_attention(
                self.frame_cross_norm(updated_frames), text_frames, text_mask
            )
        ) * frame_mask[..., None]
        combined = torch.cat((combined[:, :text_length], updated_frames), 1)
        normalized, gate = self.ada_norm_ff(combined, condition)
        combined = (combined + gate * self.feed_forward_frame(normalized)) * combined_mask[..., None]
        return (
            combined[:, :text_length],
            combined[:, text_length:].reshape(batch, frames, joints, joint_width),
        )


class MotionTransformer(nn.Module):
    """One backbone; the representation determines only its input/output width.

    T5 runs separately so frozen language weights do not enter motion checkpoints.
    Parameter names inside the backbone preserve the released checkpoint's mapping.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        width, joint_width = config.hidden_dim, config.hidden_dim // config.joints
        self.motion_input_projection = nn.Linear(config.feature_dim, width)
        self.motion_output_projection = nn.Linear(width, config.feature_dim)
        self.final_norm = nn.RMSNorm(joint_width, eps=1e-6)
        self.time_mlp = nn.Sequential(
            nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.skeleton_mlp = nn.Sequential(
            nn.Linear(config.skeleton_dim, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.condition_mlp = nn.Sequential(nn.SiLU(), nn.Linear(width, width))
        self.text_frame_mlp = nn.Sequential(
            nn.Linear(config.text_dim, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.text_joint_mlp = nn.Sequential(
            nn.Linear(config.text_dim, width), nn.SiLU(), nn.Linear(width, joint_width)
        )
        self.layers = nn.ModuleList(MotionBlock(config) for _ in range(config.layers))
        nn.init.zeros_(self.motion_output_projection.weight)
        nn.init.zeros_(self.motion_output_projection.bias)

    def forward(
        self,
        state: Tensor,
        time: Tensor,
        text: Tensor,
        text_mask: Tensor,
        skeleton: Tensor,
        frame_mask: Tensor,
    ) -> Tensor:
        batch, frames, features = state.shape
        config = self.config
        assert features == config.feature_dim
        assert frames > 0 and frame_mask.shape == (batch, frames) and frame_mask.any(-1).all()
        assert skeleton.shape == (batch, config.skeleton_dim)
        assert text.shape == (batch, text_mask.shape[1], config.text_dim)
        assert time.numel() == batch
        frame_mask, text_mask = frame_mask.bool(), text_mask.bool()
        half = config.hidden_dim // 2
        frequencies = torch.exp(
            -math.log(config.rotary_base)
            * torch.arange(half, device=state.device, dtype=state.dtype) / half
        )
        angles = time.reshape(batch, 1) * frequencies
        time_embedding = torch.cat((angles.sin(), angles.cos()), -1)
        condition = self.condition_mlp(self.time_mlp(time_embedding) + self.skeleton_mlp(skeleton))
        joint_width = config.hidden_dim // config.joints
        hidden = self.motion_input_projection(state).reshape(batch, frames, config.joints, joint_width)
        hidden = hidden * frame_mask[:, :, None, None]
        text_frames = self.text_frame_mlp(text) * text_mask[..., None]
        text_joints = self.text_joint_mlp(text) * text_mask[..., None]
        cos, sin = rotary_table(frames, config.hidden_dim // config.heads, state, config.rotary_base)
        text_length = text.shape[1]
        frame_table = (
            torch.cat((cos[:1].expand(text_length, -1), cos)),
            torch.cat((sin[:1].expand(text_length, -1), sin)),
        )
        joint_table = rotary_table(config.joints, joint_width, state, config.rotary_base)
        for layer in self.layers:
            text_frames, hidden = layer(
                text_frames, text_joints, hidden, condition,
                frame_table, joint_table, text_mask, frame_mask,
            )
        hidden = self.final_norm(hidden) * frame_mask[:, :, None, None]
        return self.motion_output_projection(hidden.flatten(-2)) * frame_mask[..., None]
