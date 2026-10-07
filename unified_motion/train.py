"""Train the combined-feature paper model on both data sources.

Run from the repository root:
torchrun --standalone --nproc_per_node=8 -m unified_motion.train \
    --config configs/default.json --data-root data/paper --output runs/paper --device cuda
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as distributed
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from unified_motion.checkpoint import save_checkpoint
from unified_motion.config import ExperimentConfig
from unified_motion.data import MotionDataset, WindowBatch
from unified_motion.model import MotionTransformer
from unified_motion.statistics import Statistics
from unified_motion.text import TextEncoder


def worker_seed(worker_id: int) -> None:
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def block_loss(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    # Preserve the research code's sum over channels / valid frames reduction.
    errors = ((prediction - target).square() * mask[..., None]).sum((1, 2))
    return (errors / mask.sum(1).clamp_min(1)).mean()


def batch_loss(
    network: torch.nn.Module,
    batch: dict[str, Any],
    encoder: TextEncoder,
    config: ExperimentConfig,
    device: torch.device,
    *,
    drop_conditions: bool,
) -> Tensor:
    target, mask = batch["motion"].to(device), batch["mask"].to(device)
    skeleton = batch["skeleton"].to(device)
    text, text_mask = encoder.encode(batch["captions"])
    if drop_conditions:
        both = torch.rand(len(target), device=device) < config.training.drop_both
        text_only = torch.rand(len(target), device=device) < config.training.drop_text
        text_mask = text_mask & ~(both | text_only)[:, None]
        skeleton = skeleton * ~both[:, None]
    noise = torch.randn_like(target) * mask[..., None]
    time = torch.rand(len(target), device=device)
    state = time[:, None, None] * target + (1 - time[:, None, None]) * noise
    prediction = network(state, time, text, text_mask, skeleton, mask)
    split = config.model.smg_dim
    return (
        block_loss(prediction[..., :split], target[..., :split], mask)
        + block_loss(prediction[..., split:], target[..., split:], mask)
    )


def rng_state(device: torch.device) -> dict[str, Any]:
    state = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    return state


def restore_rng(state: dict[str, Any], device: torch.device) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.json"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    config = ExperimentConfig.load(args.config)
    assert config.model.representation == "combined", (
        "Training is restricted to the combined-feature paper model."
    )
    if args.resume is None:
        assert not args.output.exists() or not any(args.output.iterdir()), (
            "Use a fresh output directory, or --resume for an existing run."
        )
    rank, local_rank, world = (
        int(os.environ.get(name, default))
        for name, default in (("RANK", "0"), ("LOCAL_RANK", "0"), ("WORLD_SIZE", "1"))
    )
    device = torch.device(f"cuda:{local_rank}" if args.device == "cuda" else "cpu")
    if device.type == "cuda":
        assert torch.cuda.is_available(), "CUDA requested but unavailable."
        torch.cuda.set_device(device)
    if world > 1:
        distributed.init_process_group("nccl" if device.type == "cuda" else "gloo")
    assert config.training.global_batch_size % world == 0
    random.seed(config.training.seed + rank)
    torch.manual_seed(config.training.seed + rank)
    statistics = Statistics.load(args.data_root / "metadata", config.model)
    dataset = MotionDataset(args.data_root, "train", config.model, statistics)
    assert {record.source for record in dataset.records} == {"smg", "mixamo"}, (
        "Training requires both SMG and Mixamo in the manifest."
    )
    sampler = DistributedSampler(
        dataset, num_replicas=world, rank=rank, seed=config.training.seed
    )
    loader_generator = torch.Generator()
    loader = DataLoader(
        dataset,
        batch_size=config.training.global_batch_size // world,
        sampler=sampler,
        collate_fn=WindowBatch(config.training.window),
        num_workers=config.training.workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=worker_seed,
        generator=loader_generator,
    )
    model = MotionTransformer(config.model).to(device)
    network: torch.nn.Module = model
    if world > 1:
        network = DistributedDataParallel(
            model, device_ids=[local_rank] if device.type == "cuda" else None
        )
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    encoder = TextEncoder(config.model.text_model, device, config.model.max_text_tokens)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        betas=tuple(config.training.betas),
    )
    start_epoch, step = 0, 0
    if args.resume is not None:
        payload = torch.load(args.resume, map_location="cpu", weights_only=True)
        assert payload["format"] == "unified-motion-v1"
        assert payload["experiment_config"] == asdict(config)
        assert len(payload["rng_states"]) == world, "Resume with the original world size."
        for name, value in statistics.payload().items():
            assert torch.equal(value, payload["statistics"][name]), f"Resume statistics changed: {name}"
        model.load_state_dict(payload["model"], strict=True)
        ema.load_state_dict(payload["ema_model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        start_epoch, step = int(payload["epoch"]), int(payload["step"])
        restore_rng(payload["rng_states"][rank], device)
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        config.save(args.output / "config.json")
    for epoch in range(start_epoch, config.training.epochs):
        sampler.set_epoch(epoch)
        loader_generator.manual_seed(config.training.seed + epoch * world + rank)
        network.train()
        total = torch.zeros(2, device=device)
        iterator = tqdm(
            loader, desc=f"Epoch {epoch + 1}/{config.training.epochs}", disable=rank != 0
        )
        for batch in iterator:
            optimizer.zero_grad(set_to_none=True)
            loss = batch_loss(network, batch, encoder, config, device, drop_conditions=True)
            assert torch.isfinite(loss), f"Non-finite loss at step {step}."
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.training.max_grad_norm, error_if_nonfinite=True
            )
            optimizer.step()
            with torch.no_grad():
                for averaged, current in zip(ema.parameters(), model.parameters(), strict=True):
                    averaged.mul_(config.training.ema_decay).add_(
                        current, alpha=1 - config.training.ema_decay
                    )
            count = len(batch["captions"])
            total += torch.tensor((float(loss.detach()) * count, count), device=device)
            step += 1
        if world > 1:
            distributed.all_reduce(total)
        if rank == 0:
            metrics = {
                "epoch": epoch + 1, "step": step,
                "training_loss": float(total[0] / total[1]),
            }
            print(json.dumps(metrics), flush=True)
            with (args.output / "training.jsonl").open("a") as stream:
                stream.write(json.dumps(metrics) + "\n")
        if (
            (epoch + 1) % config.training.checkpoint_interval == 0
            or epoch + 1 == config.training.epochs
        ):
            states: list[Any] = [rng_state(device)]
            if world > 1:
                states = [None] * world
                distributed.all_gather_object(states, rng_state(device))
            if rank == 0:
                save_checkpoint(
                    args.output / f"epoch-{epoch + 1:04d}.pt",
                    model, ema, statistics, optimizer, epoch + 1, step,
                    config.training.time_epsilon, states, asdict(config),
                )
        if world > 1:
            distributed.barrier()
    if world > 1:
        distributed.destroy_process_group()


if __name__ == "__main__":
    main()
