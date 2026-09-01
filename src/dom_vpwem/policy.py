from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor

from .config import ExperimentConfig
from .mikasa_env import canonicalize_observation
from .model import VPWEM
from .normalization import NormalizationStats
from .train import CHECKPOINT_VERSION


class VPWEMPolicy:
    """Stateful online wrapper around an EMA VPWEM model."""

    def __init__(
        self,
        model: VPWEM,
        normalization: NormalizationStats,
        *,
        device: str | torch.device,
        num_inference_steps: int | None = None,
        action_chunk_size: int = 1,
        sampling_seed: int = 0,
    ) -> None:
        if action_chunk_size < 1 or action_chunk_size > model.config.action_steps:
            raise ValueError(
                f"action_chunk_size must be in [1, {model.config.action_steps}]"
            )
        self.device = torch.device(device)
        self.model = model.to(self.device).eval().requires_grad_(False)
        self.normalization = normalization
        self.num_inference_steps = num_inference_steps
        self.action_chunk_size = action_chunk_size
        self.sampling_seed = sampling_seed
        self.reset()

    def reset(
        self,
        batch_size: int = 1,
        device: str | None = None,
        seed: int | None = None,
    ) -> None:
        if batch_size != 1:
            raise ValueError("The current MIKASA evaluator uses batch_size=1")
        if device is not None and torch.device(device) != self.device:
            raise ValueError(
                f"Policy was loaded on {self.device}, but reset requested {device}"
            )
        self.working_embeddings: deque[tuple[int, Tensor]] = deque()
        self.memory_state = self.model.memory_compressor.init_state(
            batch_size=1,
            device=self.device,
            dtype=self.model.memory_compressor.query_tokens.dtype,
        )
        self.episode_step = 0
        generator_device = self.device if self.device.type == "cuda" else "cpu"
        self.generator = torch.Generator(device=generator_device)
        self.generator.manual_seed(self.sampling_seed if seed is None else seed)

    @torch.no_grad()
    def act(self, observation: Mapping[str, Any]) -> Tensor:
        canonical = canonicalize_observation(observation)
        rgb = torch.from_numpy(canonical["rgb"]).to(self.device)[None, None]
        proprio = torch.from_numpy(canonical["proprio"]).to(self.device)[None, None]
        proprio = self.normalization.normalize_proprio(proprio)
        embedding = self.model.encode(rgb, proprio)[:, 0]
        self.working_embeddings.append((self.episode_step, embedding))
        if len(self.working_embeddings) > self.model.config.obs_steps:
            expired_timestep, expired_embedding = self.working_embeddings.popleft()
            if expired_timestep % self.model.config.memory_subsample_ratio == 0:
                timestep = torch.tensor(
                    [expired_timestep],
                    device=self.device,
                    dtype=torch.long,
                )
                self.model.memory_compressor.step(
                    expired_embedding,
                    timestep,
                    self.memory_state,
                )

        working_values = [value for _, value in self.working_embeddings]
        while len(working_values) < self.model.config.obs_steps:
            working_values.insert(0, working_values[0])
        working = torch.stack(working_values, dim=1)

        episodic = self.memory_state.last_memory

        normalized_actions = self.model.sample(
            working,
            episodic,
            generator=self.generator,
            num_inference_steps=self.num_inference_steps,
        )
        start = self.model.config.obs_steps - 1
        end = start + self.action_chunk_size
        actions = self.normalization.unnormalize_action(
            normalized_actions[0, start:end]
        ).clamp(-1, 1)
        self.episode_step += self.action_chunk_size
        return actions


def load_policy(
    checkpoint: str | Path,
    device: str = "cuda",
    *,
    num_inference_steps: int | None = None,
    action_chunk_size: int | None = None,
) -> VPWEMPolicy:
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    payload = torch.load(Path(checkpoint), map_location=target, weights_only=False)
    if payload.get("format_version") != CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint format {payload.get('format_version')!r}"
        )
    config = ExperimentConfig.from_dict(payload["config"])
    model = VPWEM(config.model)
    state_dict = payload.get("ema_model") or payload.get("model")
    if state_dict is None:
        raise KeyError("Checkpoint contains neither ema_model nor model weights")
    model.load_state_dict(state_dict, strict=True)
    stats = NormalizationStats.from_dict(payload["normalization"])
    return VPWEMPolicy(
        model,
        stats,
        device=target,
        num_inference_steps=num_inference_steps,
        action_chunk_size=1 if action_chunk_size is None else action_chunk_size,
    )


__all__ = ["VPWEMPolicy", "load_policy"]
