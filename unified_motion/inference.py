"""Generate, edit, and retarget with a shared checkpoint and transport engine.

See README.md for complete module commands for every operation.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from unified_motion.checkpoint import MotionBundle
from unified_motion.data import load_motion, load_skeleton, resolve_input
from unified_motion.flow import Condition, Guidance, generate, transport
from unified_motion.geometry import JOINT_NAMES, PARENTS, decode_motion, write_bvh
from unified_motion.text import TextEncoder


class MotionEngine:
    def __init__(self, bundle: MotionBundle, device: torch.device) -> None:
        self.bundle, self.device = bundle, device

    def condition(
        self, skeleton: np.ndarray, prompts: list[str] | None, count: int = 1
    ) -> Condition:
        config = self.bundle.model.config
        value = torch.from_numpy(skeleton.copy()).to(self.device)
        structural = self.bundle.statistics.normalize_skeleton(value)[None].expand(count, -1)
        if prompts is None:
            text = torch.zeros(count, config.max_text_tokens, config.text_dim, device=self.device)
            mask = torch.zeros(count, config.max_text_tokens, dtype=torch.bool, device=self.device)
        else:
            assert len(prompts) == count
            encoder = TextEncoder(config.text_model, self.device, config.max_text_tokens)
            text, mask = encoder.encode(prompts)
        return Condition(text, mask, structural)

    def source_motion(self, path: Path) -> Tensor:
        value = torch.from_numpy(
            load_motion(path, self.bundle.model.config).copy()
        ).to(self.device)
        return self.bundle.statistics.normalize_motion(value)[None]

    def save(self, value: Tensor, skeleton: np.ndarray, output: Path, fps: float) -> None:
        features = self.bundle.statistics.denormalize_motion(value).detach().cpu().numpy()
        output.mkdir(parents=True, exist_ok=True)
        for index, sample in enumerate(features):
            decoded = decode_motion(sample, skeleton, self.bundle.model.config)
            stem = output / f"sample-{index:03d}"
            np.save(stem.with_suffix(".npy"), sample)
            np.savez_compressed(
                stem.with_suffix(".npz"),
                direct_positions=decoded.direct_positions,
                fk_positions=decoded.fk_positions,
                local_rotations=decoded.local_rotations,
                root_positions=decoded.root_positions,
                offsets=decoded.offsets,
                parents=PARENTS,
                joint_names=np.array(JOINT_NAMES),
                fps=np.array(fps),
            )
            write_bvh(stem.with_suffix(".bvh"), decoded, fps)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    for operation in ("generate", "edit", "retarget"):
        command = subparsers.add_parser(operation)
        command.add_argument("--checkpoint", type=Path, required=True)
        command.add_argument("--data-root", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--device", default="cuda")
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--steps", type=int, default=100)
        command.add_argument("--fps", type=float, default=30)
        if operation == "generate":
            command.add_argument("--skeleton", type=Path, required=True)
            command.add_argument("--prompt", required=True)
            command.add_argument("--frames", type=int, default=320)
            command.add_argument("--samples", type=int, default=1)
            command.add_argument("--method", choices=("euler", "heun", "rk4"), default="rk4")
            command.add_argument("--text-guidance", type=float)
            command.add_argument("--skeleton-guidance", type=float)
        else:
            command.add_argument("--source", type=Path, required=True)
            command.add_argument("--source-skeleton", type=Path, required=True)
            command.add_argument("--start-step", type=int, default=10)
            command.add_argument("--tail-steps", type=int, default=0)
            command.add_argument("--averages", type=int, default=1)
            command.add_argument(
                "--skeleton-guidance", type=float, default=1.0 if operation == "edit" else 0.8
            )
            if operation == "edit":
                command.add_argument("--source-prompt", required=True)
                command.add_argument("--target-prompt", required=True)
                command.add_argument("--source-text-guidance", type=float, default=1.5)
                command.add_argument("--target-text-guidance", type=float, default=3.5)
            else:
                command.add_argument("--target-skeleton", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    bundle = MotionBundle.load(args.checkpoint, args.data_root, device)
    engine = MotionEngine(bundle, device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    config = bundle.model.config
    if args.operation == "generate":
        assert args.frames > 0 and args.samples > 0
        skeleton = load_skeleton(resolve_input(args.data_root, args.skeleton), config)
        condition = engine.condition(skeleton, [args.prompt] * args.samples, args.samples)
        mask = torch.ones(args.samples, args.frames, device=device, dtype=torch.bool)
        settings = bundle.guidance.copy()
        for field in ("text", "skeleton"):
            value = getattr(args, f"{field}_guidance")
            if value is not None:
                settings[field] = value
        guidance = Guidance(**settings)
        result = generate(
            bundle.model, condition, mask, guidance, generator,
            steps=args.steps, method=args.method, epsilon=bundle.time_epsilon,
        )
        resolved = {"guidance": asdict(guidance)}
    else:
        source = engine.source_motion(resolve_input(args.data_root, args.source))
        source_skeleton = load_skeleton(
            resolve_input(args.data_root, args.source_skeleton), config
        )
        mask = torch.ones(source.shape[:2], device=device, dtype=torch.bool)
        if args.operation == "edit":
            skeleton = source_skeleton
            encoder = TextEncoder(config.text_model, device, config.max_text_tokens)
            text, text_mask = encoder.encode([args.source_prompt, args.target_prompt])
            structural = engine.condition(skeleton, None).skeleton
            source_condition = Condition(text[:1], text_mask[:1], structural)
            target_condition = Condition(text[1:], text_mask[1:], structural)
            source_guidance = Guidance(
                text=args.source_text_guidance, skeleton=args.skeleton_guidance
            )
            target_guidance = Guidance(
                text=args.target_text_guidance, skeleton=args.skeleton_guidance
            )
        else:
            skeleton = load_skeleton(
                resolve_input(args.data_root, args.target_skeleton), config
            )
            source_condition = engine.condition(source_skeleton, None)
            target_condition = engine.condition(skeleton, None)
            source_guidance = target_guidance = Guidance(
                text=0.0, skeleton=args.skeleton_guidance
            )
        result = transport(
            bundle.model, source, source_condition, target_condition, mask,
            source_guidance, target_guidance, generator,
            steps=args.steps, start_step=args.start_step, tail_steps=args.tail_steps,
            averages=args.averages,
        )
        resolved = {
            "source_guidance": asdict(source_guidance),
            "target_guidance": asdict(target_guidance),
        }
    assert torch.isfinite(result).all(), "Inference produced non-finite features."
    engine.save(result, skeleton, args.output, args.fps)
    metadata = {
        **{
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        **resolved,
        "checkpoint_epoch": bundle.epoch,
        "checkpoint_format": bundle.format,
        "statistics_source": bundle.statistics_source,
        "representation": config.representation,
        "feature_dim": config.feature_dim,
    }
    (args.output / "inference.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
