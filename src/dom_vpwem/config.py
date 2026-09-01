from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping, TypeVar

import yaml

from .tasks import DEFAULT_ENV_ID, get_task_spec


@dataclass
class TaskConfig:
    env_id: str = DEFAULT_ENV_ID
    dataset_dir: str | None = None
    max_episode_steps: int | None = None
    obs_mode: str = "rgb"
    control_mode: str = "pd_ee_delta_pose"
    reward_mode: str = "normalized_dense"
    sim_backend: str = "gpu"

    def __post_init__(self) -> None:
        spec = get_task_spec(self.env_id)
        if self.dataset_dir is None:
            self.dataset_dir = spec.default_dataset_dir
        if self.max_episode_steps is None:
            self.max_episode_steps = spec.max_episode_steps


@dataclass
class ModelConfig:
    action_dim: int = 7
    proprio_dim: int = 7
    obs_steps: int = 2
    action_steps: int = 8
    action_horizon: int = 9
    image_height: int = 128
    image_width: int = 128
    crop_height: int = 100
    crop_width: int = 100
    random_crop: bool = True
    embedding_dim: int = 256
    vision_backbone: str = "resnet18"
    share_vision_encoder: bool = False
    group_norm: bool = True
    freeze_vision_encoder: bool = False
    denoiser_layers: int = 8
    denoiser_heads: int = 4
    embedding_dropout: float = 0.0
    denoiser_dropout: float = 0.3
    memory_queries: int = 2
    memory_layers: int = 2
    memory_heads: int = 8
    memory_cache_size: int = 8
    memory_cache_strategy: str = "fifo"
    memory_subsample_ratio: int = 5
    short_condition_dropout: float = 0.3
    long_condition_dropout: float = 0.3
    diffusion_steps: int = 50
    beta_schedule: str = "cosine"
    ema_decay: float = 0.999
    proprio_normalization: str = "minmax"
    action_normalization: str = "identity"

    def validate(self) -> None:
        expected = self.obs_steps + self.action_steps - 1
        if self.action_horizon != expected:
            raise ValueError(
                "action_horizon must equal obs_steps + action_steps - 1 "
                f"({expected}), got {self.action_horizon}"
            )
        if self.action_dim != 7 or self.proprio_dim != 7:
            raise ValueError(
                "The MIKASA pd_ee_delta_pose contract requires 7-D actions and "
                "7-D EEF proprioception."
            )
        if self.embedding_dim % self.denoiser_heads:
            raise ValueError("embedding_dim must be divisible by denoiser_heads")
        if self.embedding_dim % self.memory_heads:
            raise ValueError("embedding_dim must be divisible by memory_heads")
        if not 0.0 <= self.embedding_dropout < 1.0:
            raise ValueError("embedding_dropout must be in [0, 1)")
        if not 0.0 <= self.denoiser_dropout < 1.0:
            raise ValueError("denoiser_dropout must be in [0, 1)")
        if self.memory_cache_strategy not in {"fifo", "adjacent_similarity"}:
            raise ValueError(
                "memory_cache_strategy must be 'fifo' or 'adjacent_similarity'"
            )
        if self.memory_subsample_ratio < 1:
            raise ValueError("memory_subsample_ratio must be positive")
        if self.proprio_normalization != "minmax":
            raise ValueError("Only minmax proprio normalization is implemented")
        if self.action_normalization not in {"identity", "minmax"}:
            raise ValueError("action_normalization must be 'identity' or 'minmax'")


@dataclass
class TrainConfig:
    output_dir: str = "outputs/shell_game_shuffle_color_lamp_touch"
    seed: int = 100
    device: str = "cuda"
    batch_size: int = 16
    num_workers: int = 2
    prefetch_factor: int = 1
    preload_dataset: bool = True
    gradient_steps: int = 600_000
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip_norm: float = 1.0
    log_every: int = 100
    save_every: int = 20_000
    keep_last_checkpoints: int = 3
    mixed_precision: bool = True
    compile_model: bool = False
    resume: str | None = None
    vision_encoder_checkpoint: str | None = None


@dataclass
class ExperimentConfig:
    task: TaskConfig = field(default_factory=TaskConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def validate(self) -> None:
        self.model.validate()
        task_spec = get_task_spec(self.task.env_id)
        if self.task.max_episode_steps != task_spec.max_episode_steps:
            raise ValueError(
                f"{self.task.env_id} has a {task_spec.max_episode_steps}-step "
                f"horizon, got {self.task.max_episode_steps}."
            )
        if not self.task.dataset_dir:
            raise ValueError("dataset_dir must be a non-empty path")
        if self.train.batch_size < 1 or self.train.gradient_steps < 1:
            raise ValueError("batch_size and gradient_steps must be positive")
        if self.train.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.train.prefetch_factor < 1:
            raise ValueError("prefetch_factor must be positive")
        if (
            self.model.freeze_vision_encoder
            and not self.train.vision_encoder_checkpoint
            and not self.train.resume
        ):
            raise ValueError(
                "freeze_vision_encoder requires vision_encoder_checkpoint "
                "unless training is resumed from a full VPWEM checkpoint"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if not isinstance(raw, Mapping):
            raise TypeError("Experiment configuration must be a YAML mapping")
        _reject_unknown(raw, {"task", "model", "train"}, "root")
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ExperimentConfig":
        if not isinstance(raw, Mapping):
            raise TypeError("Experiment configuration must be a mapping")
        _reject_unknown(raw, {"task", "model", "train"}, "root")
        config = cls(
            task=_dataclass_from_mapping(TaskConfig, raw.get("task", {})),
            model=_dataclass_from_mapping(ModelConfig, raw.get("model", {})),
            train=_dataclass_from_mapping(TrainConfig, raw.get("train", {})),
        )
        config.validate()
        return config


T = TypeVar("T")


def _dataclass_from_mapping(cls: type[T], value: Any) -> T:
    if not isinstance(value, Mapping):
        raise TypeError(f"{cls.__name__} configuration must be a mapping")
    valid = {item.name for item in fields(cls)}
    _reject_unknown(value, valid, cls.__name__)
    return cls(**dict(value))


def _reject_unknown(value: Mapping[str, Any], valid: set[str], context: str) -> None:
    unknown = sorted(set(value) - valid)
    if unknown:
        raise ValueError(f"Unknown {context} configuration keys: {', '.join(unknown)}")
