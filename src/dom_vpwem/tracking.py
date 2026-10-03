"""VPWEM with a supervised hidden-ball readout, isolated from baseline VPWEM.

The action denoiser is unchanged. Training adds a query that cross-attends to
clean working/episodic tokens, and bounded temporal gradients for FIFO memory.
Deployment uses the same bounded recurrent memory and the VPWEM action sampler.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping

import torch
import torch.nn.functional as F
import yaml
from torch import Tensor, nn

from .config import ExperimentConfig, ModelConfig
from .model import VPWEM, ContextualMemoryCompressor, ContextualMemoryState
from .tasks import SHELL_GAME_SHUFFLE_TOUCH_CUSTOM_ENV_ID


@dataclass
class TrackingConfig:
    attention_heads: int = 4
    hidden_dim: int = 128
    loss_weight: float = 1.0
    prefix_loss_weight: float = 0.5
    hidden_phase_weight: float = 2.0
    xy_scale: float = 0.25  # metres per normalized coordinate
    unroll_steps: int = 8

    def validate(self, model: ModelConfig):
        for name in ("attention_heads", "hidden_dim", "unroll_steps"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"tracking.{name} must be a positive integer")
        if model.embedding_dim % self.attention_heads:
            raise ValueError("embedding_dim must be divisible by tracking.attention_heads")
        for name in ("loss_weight", "hidden_phase_weight", "xy_scale"):
            value = getattr(self, name)
            if not torch.isfinite(torch.tensor(value)) or value <= 0:
                raise ValueError(f"tracking.{name} must be finite and positive")
        if self.prefix_loss_weight < 0 or not torch.isfinite(torch.tensor(self.prefix_loss_weight)):
            raise ValueError("tracking.prefix_loss_weight must be finite and nonnegative")


@dataclass
class TrackingExperimentConfig(ExperimentConfig):
    tracking: TrackingConfig = field(default_factory=TrackingConfig)

    @classmethod
    def from_dict(cls, value: Mapping):
        base = ExperimentConfig.from_dict(
            {key: val for key, val in value.items() if key != "tracking"}
        )
        config = cls(base.task, base.model, base.train, TrackingConfig(**value.get("tracking", {})))
        config.validate()
        return config

    @classmethod
    def from_yaml(cls, path):
        with Path(path).open(encoding="utf-8") as handle:
            return cls.from_dict(yaml.safe_load(handle) or {})

    def to_dict(self):
        return asdict(self)

    def validate(self):
        super().validate()
        if self.task.env_id != SHELL_GAME_SHUFFLE_TOUCH_CUSTOM_ENV_ID:
            raise ValueError("Tracking VPWEM currently supports only ShellGameShuffleTouchCustom")
        if self.model.memory_subsample_ratio != 1 or self.model.memory_cache_strategy != "fifo":
            raise ValueError("Tracking VPWEM requires memory_subsample_ratio=1 and FIFO memory")
        if self.train.compile_model:
            raise ValueError("Tracking prefix training currently requires compile_model=false")
        self.tracking.validate(self.model)


class BallPositionPredictor(nn.Module):
    def __init__(self, model: ModelConfig, tracking: TrackingConfig):
        super().__init__()
        dim = model.embedding_dim
        self.query = nn.Parameter(torch.randn(1, 1, dim) * dim**-0.5)
        self.memory_types = nn.Parameter(torch.randn(2, dim) * dim**-0.5)
        self.working_positions = nn.Parameter(torch.randn(1, model.obs_steps, dim) * dim**-0.5)
        self.memory_positions = nn.Parameter(
            torch.randn(1, model.memory_queries, dim) * dim**-0.5
        )
        self.context_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, tracking.attention_heads, batch_first=True)
        self.output_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, tracking.hidden_dim), nn.GELU(), nn.Linear(tracking.hidden_dim, 2)
        )

    def forward(self, working: Tensor, episodic: Tensor):
        context = self.context_norm(torch.cat([
            working + self.memory_types[0] + self.working_positions,
            episodic + self.memory_types[1] + self.memory_positions,
        ], dim=1))
        query = self.query.expand(working.shape[0], -1, -1)
        recalled, _ = self.attention(query, context, context, need_weights=False)
        return self.mlp(self.output_norm(query + recalled))[:, 0]


class TrackingMemoryCompressor(ContextualMemoryCompressor):
    """Cache updated summaries, with functional FIFO writes and temporal gradients.

    Caching updated summaries rather than layer inputs lets each layer retain
    earlier observations recursively. Evaluation detaches every cache write.
    """

    @staticmethod
    def detach_state(state: ContextualMemoryState):
        state.observation_cache = state.observation_cache.detach()
        state.summary_caches = [cache.detach() for cache in state.summary_caches]
        state.last_memory = state.last_memory.detach()

    def _append_functional(self, cache, valid, sizes, value, active):
        caches, masks, counts = [], [], []
        for row in range(cache.shape[0]):
            if not bool(active[row]):
                caches.append(cache[row])
                masks.append(valid[row])
                counts.append(sizes[row])
                continue
            row_valid = valid[row] if valid.ndim == 2 else valid[row, :, 0]
            count = int(row_valid.sum())
            if count == self.cache_size:
                caches.append(torch.cat([cache[row, 1:], value[row:row + 1]], dim=0))
                masks.append(valid[row])
                counts.append(sizes[row])
            else:
                caches.append(torch.cat([
                    cache[row, :count], value[row:row + 1], cache[row, count + 1:]
                ], dim=0))
                masks.append(torch.cat([
                    valid[row, :count], torch.ones_like(valid[row, count:count + 1]),
                    valid[row, count + 1:]
                ], dim=0))
                counts.append(torch.cat([
                    sizes[row, :count], torch.ones_like(sizes[row, count:count + 1]),
                    sizes[row, count + 1:]
                ], dim=0))
        return torch.stack(caches), torch.stack(masks), torch.stack(counts)

    def step(self, observation, timestep, state, active=None):
        batch = observation.shape[0]
        self._validate_state(state, batch)
        if active is None:
            active = torch.ones(batch, dtype=torch.bool, device=observation.device)
        if not bool(active.any()):
            return state.last_memory
        positioned = observation + self.position_embedding(timestep).to(observation.dtype)
        state.observation_cache, state.observation_valid, state.observation_sizes = (
            self._append_functional(
                state.observation_cache, state.observation_valid, state.observation_sizes,
                positioned if self.training else positioned.detach(), active,
            )
        )
        query = self.query_tokens.expand(batch, -1, -1)
        for index, layer in enumerate(self.layers):
            summaries = torch.cat([state.summary_caches[index].flatten(1, 2), query], dim=1)
            padding = torch.cat([
                ~state.summary_valid[index].flatten(1, 2),
                torch.zeros(batch, self.num_queries, dtype=torch.bool, device=query.device),
            ], dim=1)
            observation_padding = ~state.observation_valid
            inactive = ~state.observation_valid.any(dim=1)
            if bool(inactive.any()):
                observation_padding = observation_padding.clone()
                observation_padding[inactive, 0] = False
            updated = layer(
                query, summaries, padding, self.input_projection(state.observation_cache),
                observation_padding,
            )
            query = torch.where(active[:, None, None], updated, query)
            state.summary_caches[index], state.summary_valid[index], state.summary_sizes[index] = (
                self._append_functional(
                    state.summary_caches[index], state.summary_valid[index],
                    state.summary_sizes[index], query if self.training else query.detach(), active,
                )
            )
        projected = self.output_projection(self.output_norm(query))
        state.last_memory = torch.where(active[:, None, None], projected, state.last_memory)
        return state.last_memory


class TrackingVPWEM(VPWEM):
    def __init__(self, config: ModelConfig, tracking: TrackingConfig):
        super().__init__(config)
        tracking.validate(config)
        if config.memory_subsample_ratio != 1 or config.memory_cache_strategy != "fifo":
            raise ValueError("Tracking VPWEM requires dense FIFO history")
        self.tracking_config = tracking
        # Reuse the already initialized compressor weights.
        compressor = TrackingMemoryCompressor(config)
        compressor.load_state_dict(self.memory_compressor.state_dict())
        self.memory_compressor = compressor
        self.position_predictor = BallPositionPredictor(config, tracking)
        self.loss_metrics = {}

    def _position_loss(self, prediction, target, hidden, mask):
        error = F.smooth_l1_loss(
            prediction.float(), target.float() / self.tracking_config.xy_scale, reduction="none"
        ).mean(-1)
        weights = mask.float() * torch.where(
            hidden.bool(), self.tracking_config.hidden_phase_weight, 1.0
        )
        return (error * weights).sum() / weights.sum().clamp_min(1.0)

    def diffusion_loss(
        self, actions, obs_rgb, obs_proprio, memory_rgb, memory_proprio,
        memory_timestep, memory_mask, action_mask=None, *,
        tracking_xy, tracking_hidden, tracking_mask,
    ):
        working = self.encoder(obs_rgb, obs_proprio)
        # Keep historical ResNet graphs out of memory, as in baseline VPWEM.
        # Current-frame tracking supervision still trains the shared encoder.
        with torch.no_grad():
            historical = working.new_zeros(
                working.shape[0], memory_rgb.shape[1], self.config.embedding_dim
            )
            if bool(memory_mask.any()):
                historical[memory_mask] = self.encoder(
                    memory_rgb[memory_mask][:, None], memory_proprio[memory_mask][:, None]
                )[:, 0]
        features = torch.cat([historical, working], dim=1)
        if tracking_xy.shape != (*features.shape[:2], 2):
            raise ValueError("tracking_xy must align with history plus working frames")
        if tracking_mask.shape != features.shape[:2] or tracking_hidden.shape != features.shape[:2]:
            raise ValueError("Tracking masks must align with history plus working frames")
        state = self.memory_compressor.init_state(
            actions.shape[0], device=working.device, dtype=working.dtype
        )
        predictions, targets, hidden, masks = [], [], [], []
        time = historical.shape[1]
        for index in range(time):
            if index % self.tracking_config.unroll_steps == 0:
                self.memory_compressor.detach_state(state)
            episodic = self.memory_compressor.step(
                historical[:, index], memory_timestep[:, index], state, memory_mask[:, index]
            )
            # At causal timestep j + obs_steps, memory ends at j and working
            # contains j+1..j+obs_steps, exactly matching online VPWEM.
            target_index = index + self.config.obs_steps
            valid = memory_mask[:, index] & tracking_mask[:, target_index]
            if index < time - 1 and bool(valid.any()):
                predictions.append(self.position_predictor(
                    features[:, index + 1:target_index + 1], episodic
                ))
                targets.append(tracking_xy[:, target_index])
                hidden.append(tracking_hidden[:, target_index])
                masks.append(valid)
        episodic = state.last_memory
        predicted_xy = self.position_predictor(working, episodic)
        current_loss = self._position_loss(
            predicted_xy, tracking_xy[:, -1], tracking_hidden[:, -1], tracking_mask[:, -1]
        )
        prefix_loss = current_loss.new_zeros(())
        if predictions:
            prefix_loss = self._position_loss(
                torch.stack(predictions, dim=1), torch.stack(targets, dim=1),
                torch.stack(hidden, dim=1), torch.stack(masks, dim=1),
            )
        step = torch.randint(
            self.config.diffusion_steps, (actions.shape[0],), device=actions.device
        )
        noise = torch.randn_like(actions)
        alpha_bar = self.alpha_bar[step].view(-1, 1, 1)
        noisy = alpha_bar.sqrt() * actions + (1 - alpha_bar).sqrt() * noise
        predicted_noise = self.denoiser(
            noisy, step,
            self._drop_condition(working, self.config.short_condition_dropout, self.training),
            self._drop_condition(episodic, self.config.long_condition_dropout, self.training),
        )
        squared_error = (predicted_noise - noise).square()
        weights = (torch.ones_like(actions[..., 0]) if action_mask is None else action_mask).float()
        diffusion = (squared_error * weights[..., None]).sum() / (
            weights.sum() * actions.shape[-1]
        ).clamp_min(1.0)
        total = diffusion + self.tracking_config.loss_weight * (
            current_loss + self.tracking_config.prefix_loss_weight * prefix_loss
        )
        error_m = (predicted_xy.detach().float() * self.tracking_config.xy_scale
                   - tracking_xy[:, -1].float()).norm(dim=-1)
        self.loss_metrics = {
            "diffusion_loss": diffusion.detach(), "tracking_loss": current_loss.detach(),
            "tracking_prefix_loss": prefix_loss.detach(), "tracking_error_m": error_m.mean(),
        }
        return total
