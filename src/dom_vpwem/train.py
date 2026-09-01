from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import fields
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .data import MikasaNpzDataset
from .model import VPWEM, make_ema_model, update_ema
from .normalization import NormalizationStats

CHECKPOINT_VERSION = 2

_RESUME_MUTABLE_TRAIN_FIELDS = {
    "output_dir",
    "log_every",
    "save_every",
    "keep_last_checkpoints",
    "resume",
    "vision_encoder_checkpoint",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def cycle(loader: DataLoader[Any]) -> Iterator[Mapping[str, Any]]:
    while True:
        yield from loader


def _move(value: Tensor, device: torch.device) -> Tensor:
    return value.to(device=device, non_blocking=True)


def prepare_batch(
    batch: Mapping[str, Any],
    stats: NormalizationStats,
    device: torch.device,
) -> dict[str, Tensor]:
    observation = batch["obs"]
    return {
        "actions": stats.normalize_action(_move(batch["action"], device)),
        "obs_rgb": _move(observation["rgb"], device),
        "obs_proprio": stats.normalize_proprio(
            _move(observation["proprio"], device)
        ),
        "memory_rgb": _move(batch["memory_rgb"], device),
        "memory_proprio": stats.normalize_proprio(
            _move(batch["memory_proprio"], device)
        ),
        "memory_timestep": _move(batch["memory_timestep"], device),
        "memory_mask": _move(batch["memory_mask"], device),
        "action_mask": _move(batch["action_mask"], device),
    }


def _make_grad_scaler(enabled: bool) -> Any:
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)  # pragma: no cover


def _capture_rng_state(data_generator: torch.Generator) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "data_generator": data_generator.get_state(),
    }


def _restore_rng_state(
    state: Mapping[str, Any], data_generator: torch.Generator
) -> None:
    """Restore state outside the DataLoader's in-flight worker queues.

    PyTorch exposes neither a serializable mid-epoch sampler cursor nor worker
    prefetch queues. This restores future process/generator streams, but cannot
    replay samples that had already been assigned to workers at interruption.
    """

    required = {"python", "numpy", "torch", "cuda", "data_generator"}
    missing = sorted(required.difference(state))
    if missing:
        raise ValueError(
            "Resume checkpoint is missing RNG state: " + ", ".join(missing)
        )

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    cuda_states = state["cuda"]
    if cuda_states is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in cuda_states])
    data_generator.set_state(state["data_generator"].cpu())


def _validate_resume_compatibility(
    saved: ExperimentConfig, current: ExperimentConfig
) -> None:
    if saved.task != current.task:
        raise ValueError("Resume checkpoint task configuration does not match")
    if saved.model != current.model:
        raise ValueError("Resume checkpoint model configuration does not match")

    mismatches = [
        item.name
        for item in fields(saved.train)
        if item.name not in _RESUME_MUTABLE_TRAIN_FIELDS
        and getattr(saved.train, item.name) != getattr(current.train, item.name)
    ]
    if mismatches:
        raise ValueError(
            "Resume checkpoint training configuration does not match: "
            + ", ".join(mismatches)
        )


def load_vision_encoder_checkpoint(
    model: VPWEM,
    path: str | Path,
    *,
    device: torch.device,
) -> None:
    """Load only observation-encoder weights from a VPWEM or encoder state dict."""

    source_path = Path(path).expanduser()
    checkpoint = torch.load(source_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Vision checkpoint must be a mapping: {source_path}")
    state = checkpoint.get("ema_model", checkpoint.get("model", checkpoint))
    if not isinstance(state, Mapping):
        raise ValueError(
            "Vision checkpoint 'ema_model'/'model' entry must be a state dict: "
            f"{source_path}"
        )

    prefix = "encoder."
    encoder_state = {
        name[len(prefix) :]: value
        for name, value in state.items()
        if isinstance(name, str) and name.startswith(prefix)
    }
    if not encoder_state:
        expected = set(model.encoder.state_dict())
        if set(state) == expected:
            encoder_state = dict(state)
        else:
            raise ValueError(
                f"Vision checkpoint contains no {prefix!r}-prefixed weights: "
                f"{source_path}"
            )
    model.encoder.load_state_dict(encoder_state, strict=True)


def save_checkpoint(
    path: Path,
    *,
    step: int,
    config: ExperimentConfig,
    model: VPWEM,
    ema_model: VPWEM,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: Any,
    stats: NormalizationStats,
    data_generator: torch.Generator,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": CHECKPOINT_VERSION,
        "step": step,
        "config": config.to_dict(),
        "normalization": stats.to_dict(),
        "model": model.state_dict(),
        "ema_model": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "rng_state": _capture_rng_state(data_generator),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_training_checkpoint(
    path: Path,
    *,
    config: ExperimentConfig,
    model: VPWEM,
    ema_model: VPWEM,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: Any,
    device: torch.device,
    data_generator: torch.Generator,
) -> tuple[int, NormalizationStats]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("format_version") != CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint format {checkpoint.get('format_version')!r}"
        )
    saved_config = ExperimentConfig.from_dict(checkpoint["config"])
    _validate_resume_compatibility(saved_config, config)
    model.load_state_dict(checkpoint["model"], strict=True)
    ema_model.load_state_dict(checkpoint["ema_model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint.get("scaler", {}))
    rng_state = checkpoint.get("rng_state")
    if not isinstance(rng_state, Mapping):
        raise ValueError("Resume checkpoint does not contain a valid RNG state")
    _restore_rng_state(rng_state, data_generator)
    return int(checkpoint["step"]), NormalizationStats.from_dict(
        checkpoint["normalization"]
    )


def _remove_old_checkpoints(output_dir: Path, keep: int) -> None:
    if keep <= 0:
        return
    checkpoints = sorted(
        output_dir.glob("checkpoint_*.pt"),
        key=lambda path: int(path.stem.rsplit("_", 1)[-1]),
    )
    for path in checkpoints[:-keep]:
        path.unlink()


def train(config: ExperimentConfig) -> Path:
    config.validate()
    set_seed(config.train.seed)
    device = torch.device(config.train.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    maximum_memory_steps = math.ceil(
        config.task.max_episode_steps / config.model.memory_subsample_ratio
    )
    dataset = MikasaNpzDataset(
        config.task.dataset_dir,
        obs_steps=config.model.obs_steps,
        horizon=config.model.action_horizon,
        memory_steps=maximum_memory_steps,
        memory_subsample_ratio=config.model.memory_subsample_ratio,
        random_memory_sampling=True,
        preload=config.train.preload_dataset,
        cache_size=2,
    )
    stats = NormalizationStats.from_dataset(
        config.task.dataset_dir,
        action_mode=config.model.action_normalization,
    )
    generator = torch.Generator().manual_seed(config.train.seed)
    loader_options: dict[str, Any] = {
        "batch_size": config.train.batch_size,
        "shuffle": True,
        "num_workers": config.train.num_workers,
        "pin_memory": device.type == "cuda",
        "drop_last": len(dataset) >= config.train.batch_size,
        "persistent_workers": config.train.num_workers > 0,
        "worker_init_fn": seed_worker,
        "generator": generator,
    }
    if config.train.num_workers > 0:
        loader_options["prefetch_factor"] = config.train.prefetch_factor
    loader = DataLoader(
        dataset,
        **loader_options,
    )

    model = VPWEM(config.model).to(device)
    # A full resume checkpoint is authoritative. The optional encoder checkpoint
    # is initialization for a new run and must precede the EMA/optimizer copies.
    if config.train.vision_encoder_checkpoint and not config.train.resume:
        load_vision_encoder_checkpoint(
            model,
            config.train.vision_encoder_checkpoint,
            device=device,
        )
    ema_model = make_ema_model(model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.train.gradient_steps
    )
    use_amp = config.train.mixed_precision and device.type == "cuda"
    scaler = _make_grad_scaler(use_amp)
    start_step = 0
    if config.train.resume:
        resume_path = Path(config.train.resume).expanduser()
        start_step, stats = load_training_checkpoint(
            resume_path,
            config=config,
            model=model,
            ema_model=ema_model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            data_generator=generator,
        )
        if start_step > config.train.gradient_steps:
            raise ValueError(
                f"Resume checkpoint step {start_step} exceeds target "
                f"{config.train.gradient_steps}"
            )
        if start_step == config.train.gradient_steps:
            return resume_path

    train_model: Any = model
    if config.train.compile_model:
        if not hasattr(torch, "compile"):
            raise RuntimeError("compile_model requires torch.compile")
        train_model = torch.compile(model)

    output_dir = Path(config.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metrics_path = output_dir / "metrics.jsonl"
    batches = cycle(loader)
    model.train()
    running_loss = 0.0
    running_count = 0
    interval_start = time.monotonic()

    for step in range(start_step + 1, config.train.gradient_steps + 1):
        batch = prepare_batch(next(batches), stats, device)
        optimizer.zero_grad(set_to_none=True)
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if use_amp
            else nullcontext()
        )
        with autocast:
            loss = train_model.diffusion_loss(**batch)
        scaler.scale(loss).backward()
        if config.train.grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.train.grad_clip_norm
            )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        update_ema(ema_model, model, config.model.ema_decay)
        running_loss += float(loss.detach())
        running_count += 1

        if step % config.train.log_every == 0 or step == 1:
            elapsed = time.monotonic() - interval_start
            metric = {
                "step": step,
                "loss": running_loss / max(running_count, 1),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "seconds": elapsed,
            }
            rendered = json.dumps(metric, sort_keys=True)
            print(rendered, flush=True)
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(rendered + "\n")
            running_loss = 0.0
            running_count = 0
            interval_start = time.monotonic()

        if step % config.train.save_every == 0 or step == config.train.gradient_steps:
            checkpoint_path = output_dir / f"checkpoint_{step}.pt"
            save_checkpoint(
                checkpoint_path,
                step=step,
                config=config,
                model=model,
                ema_model=ema_model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                stats=stats,
                data_generator=generator,
            )
            _remove_old_checkpoints(output_dir, config.train.keep_last_checkpoints)

    final_path = output_dir / f"checkpoint_{config.train.gradient_steps}.pt"
    return final_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train VPWEM on the MIKASA shell game")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML configuration; built-in task defaults are used when omitted.",
    )
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--vision-encoder-checkpoint", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = (
        ExperimentConfig.from_yaml(args.config)
        if args.config is not None
        else ExperimentConfig()
    )
    if args.dataset_dir is not None:
        config.task.dataset_dir = str(args.dataset_dir)
    if args.output_dir is not None:
        config.train.output_dir = str(args.output_dir)
    if args.device is not None:
        config.train.device = args.device
    if args.steps is not None:
        config.train.gradient_steps = args.steps
    if args.batch_size is not None:
        config.train.batch_size = args.batch_size
    if args.resume is not None:
        config.train.resume = str(args.resume)
    if args.vision_encoder_checkpoint is not None:
        config.train.vision_encoder_checkpoint = str(args.vision_encoder_checkpoint)
    config.validate()
    print(train(config))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "load_training_checkpoint",
    "load_vision_encoder_checkpoint",
    "prepare_batch",
    "save_checkpoint",
    "train",
]
