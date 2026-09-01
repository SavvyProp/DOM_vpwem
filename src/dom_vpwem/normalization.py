from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor

from .data import discover_npz_episodes, load_npz_episode


@dataclass(frozen=True)
class NormalizationStats:
    proprio_min: tuple[float, ...]
    proprio_max: tuple[float, ...]
    action_min: tuple[float, ...]
    action_max: tuple[float, ...]
    action_mode: str = "identity"
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        for name in ("proprio_min", "proprio_max", "action_min", "action_max"):
            if len(getattr(self, name)) != 7:
                raise ValueError(f"{name} must contain seven values")
        if self.action_mode not in {"identity", "minmax"}:
            raise ValueError("action_mode must be 'identity' or 'minmax'")

    @classmethod
    def from_dataset(
        cls,
        dataset_dir: str | Path,
        *,
        action_mode: str = "identity",
    ) -> "NormalizationStats":
        paths = discover_npz_episodes(dataset_dir)
        if not paths:
            raise FileNotFoundError(f"No MIKASA NPZ episodes found under {dataset_dir}")
        proprio_min = np.full(7, np.inf, dtype=np.float64)
        proprio_max = np.full(7, -np.inf, dtype=np.float64)
        action_min = np.full(7, np.inf, dtype=np.float64)
        action_max = np.full(7, -np.inf, dtype=np.float64)
        for path in paths:
            episode = load_npz_episode(path)
            proprio_min = np.minimum(proprio_min, episode.proprio.min(axis=0))
            proprio_max = np.maximum(proprio_max, episode.proprio.max(axis=0))
            action_min = np.minimum(action_min, episode.action.min(axis=0))
            action_max = np.maximum(action_max, episode.action.max(axis=0))
        if action_mode == "identity" and (
            float(action_min.min()) < -1.001 or float(action_max.max()) > 1.001
        ):
            raise ValueError(
                "Official pd_ee_delta_pose actions must be normalized to [-1, 1], "
                f"but observed range is [{action_min.min():.5g}, {action_max.max():.5g}]."
            )
        return cls(
            proprio_min=tuple(float(value) for value in proprio_min),
            proprio_max=tuple(float(value) for value in proprio_max),
            action_min=tuple(float(value) for value in action_min),
            action_max=tuple(float(value) for value in action_max),
            action_mode=action_mode,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "NormalizationStats":
        return cls(
            proprio_min=tuple(value["proprio_min"]),
            proprio_max=tuple(value["proprio_max"]),
            action_min=tuple(value["action_min"]),
            action_max=tuple(value["action_max"]),
            action_mode=str(value.get("action_mode", "identity")),
            epsilon=float(value.get("epsilon", 1e-6)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "proprio_min": list(self.proprio_min),
            "proprio_max": list(self.proprio_max),
            "action_min": list(self.action_min),
            "action_max": list(self.action_max),
            "action_mode": self.action_mode,
            "epsilon": self.epsilon,
        }

    @staticmethod
    def _tensor(values: tuple[float, ...], reference: Tensor) -> Tensor:
        return reference.new_tensor(values)

    def _minmax(
        self,
        value: Tensor,
        minimum: tuple[float, ...],
        maximum: tuple[float, ...],
    ) -> Tensor:
        low = self._tensor(minimum, value)
        high = self._tensor(maximum, value)
        span = (high - low).clamp_min(self.epsilon)
        normalized = 2.0 * (value.float() - low) / span - 1.0
        constant = (high - low).abs() < self.epsilon
        return torch.where(constant, torch.zeros_like(normalized), normalized)

    def _inverse_minmax(
        self,
        value: Tensor,
        minimum: tuple[float, ...],
        maximum: tuple[float, ...],
    ) -> Tensor:
        low = self._tensor(minimum, value)
        high = self._tensor(maximum, value)
        span = high - low
        return (value.float() + 1.0) * 0.5 * span + low

    def normalize_proprio(self, value: Tensor) -> Tensor:
        return self._minmax(value, self.proprio_min, self.proprio_max)

    def normalize_action(self, value: Tensor) -> Tensor:
        if self.action_mode == "identity":
            return value.float()
        return self._minmax(value, self.action_min, self.action_max)

    def unnormalize_action(self, value: Tensor) -> Tensor:
        if self.action_mode == "identity":
            return value.float()
        return self._inverse_minmax(value, self.action_min, self.action_max)


__all__ = ["NormalizationStats"]
