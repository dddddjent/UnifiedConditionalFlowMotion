"""Evaluate generation on SMG and paired retargeting on Mixamo.

Run single-device or with torchrun; full command templates are in README.md.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as distributed
from tqdm import tqdm

from unified_motion.checkpoint import MotionBundle
from unified_motion.config import ModelConfig
from unified_motion.data import (
    MotionRecord, load_motion, load_skeleton, read_motion, read_records, resolve_input,
)
from unified_motion.evaluator import MotionEvaluator
from unified_motion.flow import Condition, Guidance, generate, transport
from unified_motion.geometry import decode_motion
from unified_motion.inference import MotionEngine
from unified_motion.metrics import diversity, fid, multimodality, position_error, retrieval, summarize
from unified_motion.text import TextEncoder


def gather(values: list[Any], world: int) -> list[Any]:
    if world == 1:
        return values
    gathered: list[Any] = [None] * world
    distributed.all_gather_object(gathered, values)
    return [value for part in gathered for value in part]


def generation_batch(
    root: Path,
    entries: list[tuple[int, MotionRecord, str, int, int]],
    bundle: MotionBundle,
    encoder: TextEncoder,
    device: torch.device,
    window: int,
) -> tuple[Condition, torch.Tensor, torch.Tensor, list[str]]:
    config = bundle.model.config
    features = torch.zeros(len(entries), window, config.feature_dim, device=device)
    mask = torch.zeros(len(entries), window, dtype=torch.bool, device=device)
    skeletons, captions = [], []
    for row, (_, record, caption, start, length) in enumerate(entries):
        raw = load_motion(root / record.motion, config)
        assert record.end <= len(raw), f"Clip exceeds motion bounds: {record.id}"
        features[row, :length] = torch.from_numpy(raw[start:start + length].copy()).to(device)
        mask[row, :length] = True
        skeletons.append(load_skeleton(root / record.skeleton, config))
        captions.append(caption)
    structural = bundle.statistics.normalize_skeleton(
        torch.from_numpy(np.stack(skeletons)).to(device)
    )
    text, text_mask = encoder.encode(captions)
    return Condition(text, text_mask, structural), features, mask, captions


@torch.no_grad()
def evaluate_generation(
    args: argparse.Namespace,
    bundle: MotionBundle,
    device: torch.device,
    rank: int,
    world: int,
) -> dict[str, Any]:
    assert args.replications > 0 and args.batch_size > 0 and args.window >= args.min_frames
    assert args.unit_length > 0 and args.min_frames >= args.unit_length
    if args.multimodality:
        assert args.mm_prompts > 0 and args.mm_repeats > args.mm_pairs > 0
    records = [
        record for record in read_records(args.data_root / "splits" / "test.jsonl")
        if record.source == "smg" and record.end - record.start >= args.min_frames
    ]
    assert len(records) >= args.retrieval_pool and args.retrieval_pool >= 3
    encoder = TextEncoder(bundle.model.config.text_model, device, bundle.model.config.max_text_tokens)
    evaluator = MotionEvaluator(args.evaluator, device)
    evaluator_statistics = args.data_root / "metadata" / "evaluator"
    mean = torch.from_numpy(
        np.load(evaluator_statistics / "mean.npy", allow_pickle=False)[:148].copy()
    ).float().to(device)
    std = torch.from_numpy(
        np.load(evaluator_statistics / "std.npy", allow_pickle=False)[:148].copy()
    ).float().to(device)
    assert mean.shape == std.shape == (148,)
    assert torch.isfinite(mean).all() and torch.isfinite(std).all()
    assert (std >= 0).all()
    std = torch.where(std == 0, 1.0, std)
    settings = bundle.guidance.copy()
    for field in ("text", "skeleton"):
        value = getattr(args, f"{field}_guidance")
        if value is not None:
            settings[field] = value
    guidance = Guidance(**settings)
    replications = []
    for replication in range(args.replications):
        seed = args.seed + replication * 10000
        rng = np.random.default_rng(seed)
        ordered = [records[index] for index in rng.permutation(len(records))]
        entries = []
        for index, record in enumerate(ordered):
            length = min(record.end - record.start, args.window) // args.unit_length * args.unit_length
            assert length > 0
            caption = record.captions[int(rng.integers(len(record.captions)))]
            start = record.start + int(rng.integers(record.end - record.start - length + 1))
            entries.append((index, record, caption, start, length))
        local = entries[rank::world]
        generator = torch.Generator(device=device).manual_seed(seed + rank)
        embedded, repeated = [], []
        batches = tqdm(
            range(0, len(local), args.batch_size),
            desc=f"Generation {replication + 1}", disable=rank != 0,
        )
        for start in batches:
            batch = local[start:start + args.batch_size]
            condition, real, mask, captions = generation_batch(
                args.data_root, batch, bundle, encoder, device, args.window
            )
            sampled = generate(
                bundle.model, condition, mask, guidance, generator,
                steps=args.steps, method=args.method, epsilon=bundle.time_epsilon,
            )
            predicted = bundle.statistics.denormalize_motion(sampled)
            fid_real, retrieval_real = evaluator.encode_motion((real[..., :148] - mean) / std, mask)
            fid_pred, retrieval_pred = evaluator.encode_motion((predicted[..., :148] - mean) / std, mask)
            text = evaluator.encode_text(captions)
            for row, entry in enumerate(batch):
                embeddings = (fid_real, fid_pred, retrieval_real, retrieval_pred, text)
                embedded.append((
                    entry[0], *(value[row].cpu().numpy() for value in embeddings)
                ))
            if args.multimodality:
                selected = [row for row, entry in enumerate(batch) if entry[0] < args.mm_prompts]
                if selected:
                    mm_condition = Condition(
                        condition.text[selected], condition.text_mask[selected],
                        condition.skeleton[selected],
                    )
                    mm_mask, outputs = mask[selected], []
                    for _ in range(args.mm_repeats):
                        values = generate(
                            bundle.model, mm_condition, mm_mask, guidance, generator,
                            steps=args.steps, method=args.method, epsilon=bundle.time_epsilon,
                        )
                        raw = bundle.statistics.denormalize_motion(values)
                        outputs.append(
                            evaluator.encode_motion(
                                (raw[..., :148] - mean) / std, mm_mask
                            )[0].cpu().numpy()
                        )
                    values = np.stack(outputs, 1)
                    repeated.extend((batch[row][0], values[index]) for index, row in enumerate(selected))
        all_embeddings = sorted(gather(embedded, world), key=lambda entry: entry[0])
        all_repeats = sorted(gather(repeated, world), key=lambda entry: entry[0])
        if rank == 0:
            real_fid, predicted_fid, real_motion, predicted_motion, text = (
                np.stack([entry[index] for entry in all_embeddings])
                for index in range(1, 6)
            )
            result = {
                "fid": fid(real_fid, predicted_fid),
                "diversity": diversity(predicted_fid, np.random.default_rng(seed + 23)),
                "real_diversity": diversity(real_fid, np.random.default_rng(seed + 17)),
                **retrieval(text, predicted_motion, args.retrieval_pool),
                **{
                    f"real_{key}": value
                    for key, value in retrieval(text, real_motion, args.retrieval_pool).items()
                },
            }
            if args.multimodality:
                assert all_repeats
                result["multimodality"] = multimodality(
                    np.stack([entry[1] for entry in all_repeats]),
                    np.random.default_rng(seed + 29), args.mm_pairs,
                )
            replications.append(result)
            print(json.dumps({"replication": replication + 1, **result}), flush=True)
    if rank != 0:
        return {}
    return {
        "metrics": summarize(replications),
        "replications": replications,
        "samples_per_replication": len(records),
        "resolved_guidance": settings,
    }


def clip_features(root: Path, record: MotionRecord, bundle: MotionBundle) -> np.ndarray:
    value = load_motion(root / record.motion, bundle.model.config)
    assert record.end <= len(value)
    return value[record.start:record.end]


@torch.no_grad()
def evaluate_retargeting(
    args: argparse.Namespace,
    bundle: MotionBundle,
    device: torch.device,
    rank: int,
    world: int,
) -> dict[str, Any]:
    assert np.isfinite(args.normalization_height) and args.normalization_height > 0
    assert args.repeats > 0
    assert args.start_steps and all(0 <= step < args.steps for step in args.start_steps)
    records = {record.id: record for record in read_records(args.data_root / "splits" / "test.jsonl")}
    pairs = [json.loads(line) for line in args.pairs.read_text().splitlines() if line.strip()]
    assert pairs
    engine = MotionEngine(bundle, device)
    guidance = Guidance(text=0.0, skeleton=args.skeleton_guidance)
    results = []
    for pair_index in tqdm(range(rank, len(pairs), world), desc="Retargeting", disable=rank != 0):
        pair = pairs[pair_index]
        source_record, target_record = records[pair["source"]], records[pair["target"]]
        assert source_record.source == target_record.source == "mixamo"
        assert source_record.character != target_record.character
        source_skeleton = load_skeleton(
            args.data_root / source_record.skeleton, bundle.model.config
        )
        target_skeleton = load_skeleton(
            args.data_root / target_record.skeleton, bundle.model.config
        )
        source_features = clip_features(args.data_root, source_record, bundle)
        # Decode ground truth in the dataset's native representation.
        full_target = read_motion(args.data_root / target_record.motion)
        ground_config = ModelConfig.for_feature_width(full_target.shape[1])
        assert target_record.end <= len(full_target)
        full_target = full_target[target_record.start:target_record.end]
        assert len(full_target) == len(source_features), (
            "Retarget pairs must have matching frame ranges."
        )
        truth = decode_motion(full_target, target_skeleton, ground_config).direct_positions
        source = bundle.statistics.normalize_motion(
            torch.from_numpy(source_features.copy()).to(device)
        )[None]
        mask = torch.ones(source.shape[:2], dtype=torch.bool, device=device)
        source_condition = engine.condition(source_skeleton, None)
        target_condition = engine.condition(target_skeleton, None)
        baseline = decode_motion(source_features, target_skeleton, bundle.model.config)
        candidates = []
        for start_step in args.start_steps:
            errors = []
            for repetition in range(args.repeats):
                generator = torch.Generator(device=device).manual_seed(args.seed + pair_index * 10000 + repetition)
                value = transport(
                    bundle.model, source, source_condition, target_condition, mask,
                    guidance, guidance, generator, steps=args.steps, start_step=start_step,
                    averages=args.averages,
                )
                raw = bundle.statistics.denormalize_motion(value)[0].cpu().numpy()
                decoded = decode_motion(raw, target_skeleton, bundle.model.config)
                errors.append((
                    position_error(decoded.direct_positions, truth, args.normalization_height),
                    position_error(decoded.fk_positions, truth, args.normalization_height),
                ))
            direct, fk = np.mean(errors, axis=0)
            candidates.append({"start_step": start_step, "direct": float(direct), "fk": float(fk)})
        results.append({
            "pair_index": pair_index,
            **pair,
            "ground_truth_representation": ground_config.representation,
            "copy_fk": position_error(baseline.fk_positions, truth, args.normalization_height),
            "candidates": candidates,
            "best": min(candidates, key=lambda item: item["direct"]),
        })
    results = sorted(gather(results, world), key=lambda item: item["pair_index"])
    if rank != 0:
        return {}
    return {
        "metrics": {
            "direct": float(np.mean([item["best"]["direct"] for item in results])),
            "fk": float(np.mean([item["best"]["fk"] for item in results])),
            "copy_fk": float(np.mean([item["copy_fk"] for item in results])),
        },
        "selection": "Per pair, the start step with the lowest direct error; FK is reported at that step.",
        "error_units": "Mean squared 3D joint distance / normalization_height^2, multiplied by 1000.",
        "pairs": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    for operation in ("generation", "retarget"):
        command = commands.add_parser(operation)
        command.add_argument("--checkpoint", type=Path, required=True)
        command.add_argument("--data-root", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--steps", type=int, default=100)
        if operation == "generation":
            command.add_argument("--evaluator", type=Path, default=Path("checkpoints/evaluator.pt"))
            command.add_argument("--replications", type=int, default=20)
            command.add_argument("--batch-size", type=int, default=16)
            command.add_argument("--retrieval-pool", type=int, default=100)
            command.add_argument("--window", type=int, default=320)
            command.add_argument("--min-frames", type=int, default=128)
            command.add_argument("--unit-length", type=int, default=8)
            command.add_argument("--method", choices=("euler", "heun", "rk4"), default="rk4")
            command.add_argument("--text-guidance", type=float)
            command.add_argument("--skeleton-guidance", type=float)
            command.add_argument("--multimodality", action="store_true")
            command.add_argument("--mm-prompts", type=int, default=100)
            command.add_argument("--mm-repeats", type=int, default=30)
            command.add_argument("--mm-pairs", type=int, default=10)
        else:
            command.add_argument(
                "--pairs", type=Path, default=Path("splits/retarget-test.jsonl")
            )
            command.add_argument("--normalization-height", type=float)
            command.add_argument(
                "--start-steps", type=int, nargs="+", default=[5, 10, 15, 20, 25, 30, 35, 40]
            )
            command.add_argument("--skeleton-guidance", type=float, default=0.8)
            command.add_argument("--repeats", type=int, default=1)
            command.add_argument("--averages", type=int, default=1)
    args = parser.parse_args()
    if args.operation == "retarget":
        args.pairs = resolve_input(args.data_root, args.pairs)
        if args.normalization_height is None:
            settings = json.loads((args.data_root / "metadata" / "evaluation.json").read_text())
            args.normalization_height = float(settings["normalization_height"])
    rank, local_rank, world = (
        int(os.environ.get(name, default))
        for name, default in (("RANK", "0"), ("LOCAL_RANK", "0"), ("WORLD_SIZE", "1"))
    )
    device = torch.device(f"cuda:{local_rank}" if args.device == "cuda" else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world > 1:
        distributed.init_process_group("nccl" if device.type == "cuda" else "gloo")
    bundle = MotionBundle.load(args.checkpoint, args.data_root, device)
    if args.operation == "generation":
        result = evaluate_generation(args, bundle, device, rank, world)
    else:
        result = evaluate_retargeting(args, bundle, device, rank, world)
    if rank == 0:
        result["metadata"] = {
            **{
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "world_size": world,
            "checkpoint_epoch": bundle.epoch,
            "checkpoint_format": bundle.format,
            "statistics_source": bundle.statistics_source,
            "representation": bundle.model.config.representation,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    if world > 1:
        distributed.destroy_process_group()


if __name__ == "__main__":
    main()
