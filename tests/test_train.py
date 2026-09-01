from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

import dom_vpwem.train as train_module
from dom_vpwem.config import ExperimentConfig
from dom_vpwem.normalization import NormalizationStats
from dom_vpwem.train import (
    _validate_resume_compatibility,
    build_arg_parser,
    load_training_checkpoint,
    load_vision_encoder_checkpoint,
    save_checkpoint,
)


class TinyPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 2))
        self.head = nn.Linear(2, 1)


class FakeScaler:
    def __init__(self, value: int = 0) -> None:
        self.value = value

    def state_dict(self) -> dict[str, int]:
        return {"value": self.value}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.value = state["value"]


class TinyDataset:
    def __len__(self) -> int:
        return 1


def _stats() -> NormalizationStats:
    return NormalizationStats(
        proprio_min=(0.0,) * 7,
        proprio_max=(1.0,) * 7,
        action_min=(-1.0,) * 7,
        action_max=(1.0,) * 7,
    )


def _cpu_config() -> ExperimentConfig:
    config = ExperimentConfig()
    config.train.device = "cpu"
    config.train.mixed_precision = False
    config.train.batch_size = 2
    config.train.gradient_steps = 8
    return config


def _training_state(config: ExperimentConfig) -> tuple[
    TinyPolicy,
    TinyPolicy,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LRScheduler,
    FakeScaler,
]:
    model = TinyPolicy()
    ema_model = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.train.gradient_steps
    )
    return model, ema_model, optimizer, scheduler, FakeScaler(7)


def test_checkpoint_round_trip_restores_process_and_data_generator_rng(tmp_path: Path) -> None:
    config = _cpu_config()
    model, ema_model, optimizer, scheduler, scaler = _training_state(config)
    data_generator = torch.Generator().manual_seed(303)
    random.seed(101)
    np.random.seed(202)
    torch.manual_seed(404)
    checkpoint = tmp_path / "checkpoint_4.pt"

    save_checkpoint(
        checkpoint,
        step=4,
        config=config,
        model=model,
        ema_model=ema_model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        stats=_stats(),
        data_generator=data_generator,
    )
    expected_python = random.random()
    expected_numpy = float(np.random.random())
    expected_torch = torch.rand(4)
    expected_data = torch.rand(4, generator=data_generator)

    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    restored = _training_state(config)
    restored_generator = torch.Generator().manual_seed(999)
    step, stats = load_training_checkpoint(
        checkpoint,
        config=config,
        model=restored[0],
        ema_model=restored[1],
        optimizer=restored[2],
        scheduler=restored[3],
        scaler=restored[4],
        device=torch.device("cpu"),
        data_generator=restored_generator,
    )

    assert step == 4
    assert stats == _stats()
    assert restored[4].value == 7
    assert random.random() == expected_python
    assert float(np.random.random()) == expected_numpy
    assert torch.equal(torch.rand(4), expected_torch)
    assert torch.equal(torch.rand(4, generator=restored_generator), expected_data)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("seed", 999),
        ("device", "cuda"),
        ("batch_size", 3),
        ("num_workers", 3),
        ("prefetch_factor", 2),
        ("preload_dataset", False),
        ("gradient_steps", 9),
        ("learning_rate", 2e-4),
        ("weight_decay", 2e-5),
        ("grad_clip_norm", 2.0),
        ("mixed_precision", True),
        ("compile_model", True),
    ],
)
def test_resume_rejects_training_changes(field: str, replacement: Any) -> None:
    saved = _cpu_config()
    current = copy.deepcopy(saved)
    setattr(current.train, field, replacement)

    with pytest.raises(ValueError, match=field):
        _validate_resume_compatibility(saved, current)


def test_resume_rejects_a_different_dataset() -> None:
    saved = _cpu_config()
    current = copy.deepcopy(saved)
    current.task.dataset_dir = "another/dataset"

    with pytest.raises(ValueError, match="task configuration"):
        _validate_resume_compatibility(saved, current)


@pytest.mark.parametrize("include_ema", [False, True])
def test_vision_loader_uses_ema_when_available_and_only_changes_encoder(
    tmp_path: Path, include_ema: bool
) -> None:
    online = TinyPolicy()
    ema = TinyPolicy()
    target = TinyPolicy()
    with torch.no_grad():
        for parameter in online.encoder.parameters():
            parameter.fill_(1.0)
        for parameter in ema.encoder.parameters():
            parameter.fill_(2.0)
    target_head = {name: value.clone() for name, value in target.head.state_dict().items()}
    payload: dict[str, Any] = {"model": online.state_dict()}
    if include_ema:
        payload["ema_model"] = ema.state_dict()
    checkpoint = tmp_path / "vision.pt"
    torch.save(payload, checkpoint)

    load_vision_encoder_checkpoint(target, checkpoint, device=torch.device("cpu"))

    expected = ema.encoder if include_ema else online.encoder
    for name, value in target.encoder.state_dict().items():
        assert torch.equal(value, expected.state_dict()[name])
    for name, value in target.head.state_dict().items():
        assert torch.equal(value, target_head[name])


def test_cli_accepts_vision_encoder_checkpoint() -> None:
    args = build_arg_parser().parse_args(
        ["--vision-encoder-checkpoint", "checkpoints/vision.pt"]
    )

    assert args.vision_encoder_checkpoint == Path("checkpoints/vision.pt")


@pytest.mark.parametrize("num_workers", [0, 2])
def test_completed_resume_returns_input_checkpoint_and_sets_prefetch_conditionally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, num_workers: int
) -> None:
    config = _cpu_config()
    config.train.gradient_steps = 3
    config.train.num_workers = num_workers
    config.train.prefetch_factor = 3
    config.train.output_dir = str(tmp_path / "unused-output")
    model, ema_model, optimizer, scheduler, scaler = _training_state(config)
    checkpoint = tmp_path / "finished.pt"
    save_checkpoint(
        checkpoint,
        step=config.train.gradient_steps,
        config=config,
        model=model,
        ema_model=ema_model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        stats=_stats(),
        data_generator=torch.Generator().manual_seed(config.train.seed),
    )
    config.train.resume = str(checkpoint)

    captured: dict[str, Any] = {}

    class CapturingDataLoader:
        def __init__(self, dataset: TinyDataset, **kwargs: Any) -> None:
            del dataset
            captured.update(kwargs)

    monkeypatch.setattr(train_module, "MikasaNpzDataset", lambda *args, **kwargs: TinyDataset())
    monkeypatch.setattr(
        train_module.NormalizationStats,
        "from_dataset",
        classmethod(lambda cls, *args, **kwargs: _stats()),
    )
    monkeypatch.setattr(train_module, "DataLoader", CapturingDataLoader)
    monkeypatch.setattr(train_module, "VPWEM", lambda config: TinyPolicy())
    monkeypatch.setattr(train_module, "make_ema_model", copy.deepcopy)
    monkeypatch.setattr(train_module, "_make_grad_scaler", lambda enabled: FakeScaler())

    assert train_module.train(config) == checkpoint
    assert not (tmp_path / "unused-output").exists()
    if num_workers:
        assert captured["prefetch_factor"] == 3
    else:
        assert "prefetch_factor" not in captured
