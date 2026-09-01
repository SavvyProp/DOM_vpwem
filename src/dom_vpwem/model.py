"""Core VPWEM modules.

This is a compact adaptation of the architecture described in the VPWEM paper:
short-horizon observation embeddings form working memory, while a contextual
Q-Former recursively consolidates older embeddings into fixed-size episodic
tokens.  A transformer DDPM predicts an action chunk from both memories.

The implementation is task-specific only at the observation/action boundary;
the neural modules themselves are generic PyTorch modules.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import ModelConfig


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int, max_period: int = 10_000) -> None:
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, timesteps: Tensor) -> Tensor:
        timesteps = timesteps.float()
        half = self.dim // 2
        frequencies = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=timesteps.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        angles = timesteps.unsqueeze(-1) * frequencies
        embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
        if self.dim % 2:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class DiffusionTimestepEmbedding(nn.Module):
    """CleanDiffuser positional timestep embedding used by ChiTransformerPTP."""

    def __init__(self, dim: int, max_positions: int = 10_000) -> None:
        super().__init__()
        self.dim = dim
        self.max_positions = max_positions

    def forward(self, timesteps: Tensor) -> Tensor:
        half = self.dim // 2
        frequencies = torch.arange(
            half,
            device=timesteps.device,
            dtype=torch.float32,
        ) / max(half, 1)
        frequencies = (1 / self.max_positions) ** frequencies
        angles = timesteps.float().unsqueeze(-1) * frequencies
        embedding = torch.cat((angles.cos(), angles.sin()), dim=-1)
        if self.dim % 2:
            embedding = F.pad(embedding, (0, 1))
        return embedding


def _group_norm(channels: int) -> nn.GroupNorm:
    groups = max(1, channels // 16)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ObservationEncoder(nn.Module):
    """Encode top RGB, wrist RGB, and EEF proprioception into one token/frame."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        try:
            import torchvision.models as models
        except ImportError as exc:  # pragma: no cover - exercised in installed env
            raise ImportError(
                "ObservationEncoder requires torchvision. Install the project "
                "with `pip install -e .`."
            ) from exc

        if config.vision_backbone != "resnet18":
            raise ValueError("This VPWEM port currently supports vision_backbone=resnet18")

        norm_layer = _group_norm if config.group_norm else None

        def make_backbone() -> nn.Module:
            kwargs: dict[str, Any] = {"weights": None}
            if norm_layer is not None:
                kwargs["norm_layer"] = norm_layer
            backbone = models.resnet18(**kwargs)
            backbone.fc = nn.Identity()
            return backbone

        self.top_encoder = make_backbone()
        self.wrist_encoder = (
            self.top_encoder if config.share_vision_encoder else make_backbone()
        )
        self.share_vision_encoder = config.share_vision_encoder
        self.crop_height = config.crop_height
        self.crop_width = config.crop_width
        self.random_crop = config.random_crop
        self.proprio_dim = config.proprio_dim
        self.output_dim = config.embedding_dim
        self.projection = nn.Sequential(
            nn.Linear(512 * 2 + config.proprio_dim, config.embedding_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(config.embedding_dim, config.embedding_dim),
        )

        if config.freeze_vision_encoder:
            for parameter in self.top_encoder.parameters():
                parameter.requires_grad_(False)
            if not self.share_vision_encoder:
                for parameter in self.wrist_encoder.parameters():
                    parameter.requires_grad_(False)

    def _crop(self, images: Tensor) -> Tensor:
        height, width = images.shape[-2:]
        if self.crop_height > height or self.crop_width > width:
            raise ValueError(
                f"Crop {(self.crop_height, self.crop_width)} exceeds image {(height, width)}"
            )
        if self.crop_height == height and self.crop_width == width:
            return images
        if self.training and self.random_crop:
            # One vectorized crop per flattened image.
            max_top = height - self.crop_height
            max_left = width - self.crop_width
            tops = torch.randint(max_top + 1, (images.shape[0],), device=images.device)
            lefts = torch.randint(max_left + 1, (images.shape[0],), device=images.device)
            rows = tops[:, None, None] + torch.arange(
                self.crop_height, device=images.device
            )[None, :, None]
            columns = lefts[:, None, None] + torch.arange(
                self.crop_width, device=images.device
            )[None, None, :]
            batches = torch.arange(images.shape[0], device=images.device)[:, None, None]
            channel_last = images.permute(0, 2, 3, 1)
            return channel_last[batches, rows, columns].permute(0, 3, 1, 2)
        top = (height - self.crop_height) // 2
        left = (width - self.crop_width) // 2
        return images[
            ...,
            top : top + self.crop_height,
            left : left + self.crop_width,
        ]

    def forward(self, rgb: Tensor, proprio: Tensor) -> Tensor:
        """Return ``[batch, time, embedding_dim]`` observation embeddings.

        ``rgb`` must be channel-last with the base camera in channels 0:3 and
        wrist camera in channels 3:6. uint8 input is scaled to [0, 1].
        """

        if rgb.ndim != 5 or rgb.shape[-1] != 6:
            raise ValueError(f"Expected rgb [B,T,H,W,6], got {tuple(rgb.shape)}")
        if proprio.ndim != 3 or proprio.shape[-1] != self.proprio_dim:
            raise ValueError(
                f"Expected proprio [B,T,{self.proprio_dim}], got {tuple(proprio.shape)}"
            )
        if rgb.shape[:2] != proprio.shape[:2]:
            raise ValueError("RGB and proprioception batch/time dimensions differ")

        batch, time = rgb.shape[:2]
        rgb = rgb.float()
        if rgb.detach().amax() > 1.5:
            rgb = rgb / 255.0
        top = rgb[..., :3].permute(0, 1, 4, 2, 3).reshape(-1, 3, *rgb.shape[2:4])
        wrist = rgb[..., 3:6].permute(0, 1, 4, 2, 3).reshape(-1, 3, *rgb.shape[2:4])
        top = self._crop(top)
        wrist = self._crop(wrist)
        top_features = self.top_encoder(top)
        wrist_features = self.wrist_encoder(wrist)
        proprio = proprio.float().reshape(batch * time, self.proprio_dim)
        features = torch.cat((top_features, wrist_features, proprio), dim=-1)
        return self.projection(features).reshape(batch, time, self.output_dim)


class ContextualMemoryLayer(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.summary_attention = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.observation_attention = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
        )
        self.summary_norm = nn.LayerNorm(dim)
        self.observation_norm = nn.LayerNorm(dim)
        self.ffn_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: Tensor,
        summary_tokens: Tensor,
        summary_padding_mask: Tensor,
        observation_tokens: Tensor,
        observation_padding_mask: Tensor,
    ) -> Tensor:
        normalized = self.summary_norm(query)
        attended, _ = self.summary_attention(
            normalized,
            summary_tokens,
            summary_tokens,
            key_padding_mask=summary_padding_mask,
            need_weights=False,
        )
        query = query + self.dropout(attended)
        normalized = self.observation_norm(query)
        attended, _ = self.observation_attention(
            normalized,
            observation_tokens,
            observation_tokens,
            key_padding_mask=observation_padding_mask,
            need_weights=False,
        )
        query = query + self.dropout(attended)
        return query + self.dropout(self.feed_forward(self.ffn_norm(query)))


@dataclass
class ContextualMemoryState:
    """Fixed-capacity, mutable state for online memory compression.

    The tensors are intentionally not module buffers: this is transient episode
    state, not model/checkpoint state.  ``ContextualMemoryCompressor.step``
    updates an instance in place and never grows any of its tensor dimensions.
    """

    observation_cache: Tensor
    observation_valid: Tensor
    observation_sizes: Tensor
    summary_caches: list[Tensor]
    summary_valid: list[Tensor]
    summary_sizes: list[Tensor]
    last_memory: Tensor

    @property
    def batch_size(self) -> int:
        return self.observation_cache.shape[0]


class ContextualMemoryCompressor(nn.Module):
    """Recursive Q-Former memory compressor with bounded detached caches."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        dim = config.embedding_dim
        self.dim = dim
        self.num_queries = config.memory_queries
        self.cache_size = config.memory_cache_size
        self.cache_strategy = config.memory_cache_strategy
        self.position_embedding = SinusoidalEmbedding(dim)
        self.input_projection = nn.Linear(dim, dim)
        self.query_tokens = nn.Parameter(torch.empty(1, self.num_queries, dim))
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.normal_(self.query_tokens, std=dim**-0.5)
        self.layers = nn.ModuleList(
            ContextualMemoryLayer(dim, config.memory_heads, dropout=0.1)
            for _ in range(config.memory_layers)
        )
        self.output_norm = nn.LayerNorm(dim)
        self.output_projection = nn.Linear(dim, dim)

    def init_state(
        self,
        batch_size: int,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> ContextualMemoryState:
        """Allocate empty bounded state for a new batch of episodes."""

        if batch_size < 1:
            raise ValueError("Memory state batch_size must be positive")
        if self.cache_size < 1:
            raise ValueError("memory_cache_size must be positive")
        device = self.query_tokens.device if device is None else torch.device(device)
        dtype = self.query_tokens.dtype if dtype is None else dtype

        def zeros(*shape: int) -> Tensor:
            return torch.zeros(shape, device=device, dtype=dtype)

        def valid(*shape: int) -> Tensor:
            return torch.zeros(shape, device=device, dtype=torch.bool)

        return ContextualMemoryState(
            observation_cache=zeros(batch_size, self.cache_size, self.dim),
            observation_valid=valid(batch_size, self.cache_size),
            observation_sizes=zeros(batch_size, self.cache_size),
            summary_caches=[
                zeros(batch_size, self.cache_size, self.num_queries, self.dim)
                for _ in self.layers
            ],
            # AdjSim treats every query as an independent temporal stream.
            summary_valid=[
                valid(batch_size, self.cache_size, self.num_queries)
                for _ in self.layers
            ],
            summary_sizes=[
                zeros(batch_size, self.cache_size, self.num_queries)
                for _ in self.layers
            ],
            last_memory=zeros(batch_size, self.num_queries, self.dim),
        )

    def _validate_state(self, state: ContextualMemoryState, batch_size: int) -> None:
        expected_observation = (batch_size, self.cache_size, self.dim)
        expected_summary = (
            batch_size,
            self.cache_size,
            self.num_queries,
            self.dim,
        )
        expected_summary_meta = (batch_size, self.cache_size, self.num_queries)
        if state.observation_cache.shape != expected_observation:
            raise ValueError(
                "Memory state observation cache has wrong shape: "
                f"{tuple(state.observation_cache.shape)}, expected {expected_observation}"
            )
        if state.observation_valid.shape != expected_observation[:2]:
            raise ValueError("Memory state observation-valid mask has wrong shape")
        if state.observation_sizes.shape != expected_observation[:2]:
            raise ValueError("Memory state observation sizes have wrong shape")
        if not (
            len(state.summary_caches)
            == len(state.summary_valid)
            == len(state.summary_sizes)
            == len(self.layers)
        ):
            raise ValueError("Memory state has the wrong number of Q-Former layers")
        for cache, mask, sizes in zip(
            state.summary_caches, state.summary_valid, state.summary_sizes
        ):
            if cache.shape != expected_summary:
                raise ValueError("Memory state summary cache has wrong shape")
            if (
                mask.shape != expected_summary_meta
                or sizes.shape != expected_summary_meta
            ):
                raise ValueError("Memory state summary metadata has wrong shape")
        if state.last_memory.shape != (
            batch_size,
            self.num_queries,
            self.dim,
        ):
            raise ValueError("Memory state last-memory tensor has wrong shape")

    def _append_stream(
        self,
        cache: Tensor,
        mask: Tensor,
        sizes: Tensor,
        value: Tensor,
    ) -> None:
        """Append one detached value to one ``[time, feature...]`` stream."""

        count = int(mask.sum().item())
        if count < self.cache_size:
            cache[count].copy_(value)
            mask[count] = True
            sizes[count] = 1.0
            return

        candidates = torch.cat((cache, value.unsqueeze(0)), dim=0)
        candidate_sizes = torch.cat((sizes, sizes.new_ones(1)), dim=0)
        if self.cache_strategy == "fifo":
            cache.copy_(candidates[1:])
            sizes.copy_(candidate_sizes[1:])
            return

        flat = F.normalize(candidates.flatten(1).float(), dim=-1, eps=1e-8)
        merge_index = int((flat[:-1] * flat[1:]).sum(-1).argmax().item())
        left_size = candidate_sizes[merge_index]
        right_size = candidate_sizes[merge_index + 1]
        merged = (
            candidates[merge_index] * left_size
            + candidates[merge_index + 1] * right_size
        ) / (left_size + right_size)
        reduced = torch.cat(
            (
                candidates[:merge_index],
                merged.unsqueeze(0),
                candidates[merge_index + 2 :],
            ),
            dim=0,
        )
        reduced_sizes = torch.cat(
            (
                candidate_sizes[:merge_index],
                (left_size + right_size).unsqueeze(0),
                candidate_sizes[merge_index + 2 :],
            )
        )
        cache.copy_(reduced)
        sizes.copy_(reduced_sizes)

    def _append_detached(
        self,
        cache: Tensor,
        mask: Tensor,
        sizes: Tensor,
        value: Tensor,
        active: Tensor,
    ) -> None:
        """Append a batch to fixed caches, compressing independently per stream."""

        with torch.no_grad():
            detached = value.detach()
            for batch_index in torch.nonzero(active, as_tuple=False).flatten().tolist():
                if cache.ndim == 3:
                    self._append_stream(
                        cache[batch_index],
                        mask[batch_index],
                        sizes[batch_index],
                        detached[batch_index],
                    )
                elif cache.ndim == 4:
                    for query_index in range(self.num_queries):
                        self._append_stream(
                            cache[batch_index, :, query_index],
                            mask[batch_index, :, query_index],
                            sizes[batch_index, :, query_index],
                            detached[batch_index, query_index],
                        )
                else:  # pragma: no cover - internal invariant
                    raise ValueError(f"Unsupported memory cache rank {cache.ndim}")

    def step(
        self,
        observation: Tensor,
        timestep: Tensor,
        state: ContextualMemoryState,
        active: Tensor | None = None,
    ) -> Tensor:
        """Consume one historical frame and update bounded episode state.

        Args:
            observation: One embedding per batch row, shaped ``[B, D]``.
            timestep: Episode timestep for each embedding, shaped ``[B]``.
            state: Persistent state returned by :meth:`init_state`.
            active: Optional mask for rows that contain a real frame.

        Returns:
            The latest episodic tokens, shaped ``[B, num_queries, D]``.
        """

        if observation.ndim != 2 or observation.shape[-1] != self.dim:
            raise ValueError(
                f"Expected observation [B,{self.dim}], got {tuple(observation.shape)}"
            )
        batch = observation.shape[0]
        if timestep.shape != (batch,):
            raise ValueError("Memory timestep must have shape [B]")
        self._validate_state(state, batch)
        if state.observation_cache.device != observation.device:
            raise ValueError("Memory state and observation must be on the same device")
        if active is None:
            active = torch.ones(batch, dtype=torch.bool, device=observation.device)
        else:
            if active.shape != (batch,):
                raise ValueError("Memory active mask must have shape [B]")
            active = active.to(device=observation.device, dtype=torch.bool)
        if not bool(active.any()):
            return state.last_memory

        timestep = timestep.to(device=observation.device)
        positioned = observation + self.position_embedding(timestep).to(observation.dtype)
        self._append_detached(
            state.observation_cache,
            state.observation_valid,
            state.observation_sizes,
            positioned,
            active,
        )
        query = self.query_tokens.expand(batch, -1, -1)
        for layer_index, layer in enumerate(self.layers):
            query_input = query
            # Cache tensors are mutated at later recurrent steps. Clone the
            # attention inputs so autograd keeps stable saved tensor versions.
            cached_summaries = state.summary_caches[layer_index].flatten(1, 2).clone()
            cached_padding = ~state.summary_valid[layer_index].flatten(1, 2)
            summary_tokens = torch.cat((cached_summaries, query_input), dim=1)
            current_padding = torch.zeros(
                batch,
                self.num_queries,
                dtype=torch.bool,
                device=observation.device,
            )
            summary_padding = torch.cat((cached_padding, current_padding), dim=1)

            observation_padding = ~state.observation_valid
            # Upstream QFormer projects the bounded positioned cache on every
            # recurrence. Besides matching its order, this keeps the learned
            # projection trainable although cached observations are detached.
            safe_observations = self.input_projection(
                state.observation_cache.clone()
            )
            inactive = ~state.observation_valid.any(dim=1)
            if bool(inactive.any()):
                observation_padding = observation_padding.clone()
                observation_padding[inactive, 0] = False

            updated = layer(
                query,
                summary_tokens,
                summary_padding,
                safe_observations,
                observation_padding,
            )
            query = torch.where(active[:, None, None], updated, query)
            self._append_detached(
                state.summary_caches[layer_index],
                state.summary_valid[layer_index],
                state.summary_sizes[layer_index],
                query_input,
                active,
            )
        projected = self.output_projection(self.output_norm(query))
        state.last_memory = torch.where(
            active[:, None, None], projected, state.last_memory
        )
        return state.last_memory

    def forward(
        self,
        history: Tensor,
        timesteps: Tensor,
        history_mask: Tensor | None = None,
    ) -> Tensor:
        if history.ndim != 3 or history.shape[-1] != self.dim:
            raise ValueError(f"Expected history [B,T,{self.dim}], got {tuple(history.shape)}")
        batch, time, _ = history.shape
        if timesteps.shape != (batch, time):
            raise ValueError("Memory timesteps must have shape [B,T]")
        if history_mask is None:
            history_mask = torch.ones((batch, time), dtype=torch.bool, device=history.device)
        else:
            history_mask = history_mask.to(device=history.device, dtype=torch.bool)

        if history_mask.shape != (batch, time):
            raise ValueError("Memory mask must have shape [B,T]")
        state = self.init_state(batch, device=history.device, dtype=history.dtype)
        for index in range(time):
            self.step(
                history[:, index],
                timesteps[:, index],
                state,
                history_mask[:, index],
            )
        return state.last_memory


class ActionDenoiser(nn.Module):
    """Transformer DDPM noise predictor conditioned on working/episodic tokens."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        dim = config.embedding_dim
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.obs_steps = config.obs_steps
        self.memory_queries = config.memory_queries
        self.action_embedding = nn.Linear(config.action_dim, dim)
        self.working_embedding = nn.Linear(dim, dim)
        self.memory_embedding = nn.Linear(dim, dim)
        self.action_position = nn.Parameter(torch.zeros(1, config.action_horizon, dim))
        self.condition_position = nn.Parameter(
            torch.zeros(1, 1 + config.memory_queries + config.obs_steps, dim)
        )
        self.time_embedding = DiffusionTimestepEmbedding(dim)
        self.condition_encoder = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.Mish(), nn.Linear(4 * dim, dim)
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=config.denoiser_heads,
            dim_feedforward=4 * dim,
            dropout=config.denoiser_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, config.denoiser_layers)
        # Token/position embedding dropout is distinct from attention/FFN
        # dropout. Older configs omit this field and therefore preserve input
        # embeddings exactly while keeping denoiser_dropout inside the decoder.
        self.embedding_dropout = nn.Dropout(config.embedding_dropout)
        self.output_norm = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, config.action_dim)

        causal = torch.full(
            (config.action_horizon, config.action_horizon),
            float("-inf"),
        )
        causal = torch.triu(causal, diagonal=1)
        self.register_buffer("causal_mask", causal, persistent=False)

        memory_mask = torch.zeros(
            config.action_horizon,
            1 + config.memory_queries + config.obs_steps,
        )
        for action_index in range(config.action_horizon):
            for obs_index in range(config.obs_steps):
                condition_index = 1 + config.memory_queries + obs_index
                if action_index < obs_index:
                    memory_mask[action_index, condition_index] = float("-inf")
        self.register_buffer("condition_mask", memory_mask, persistent=False)
        nn.init.normal_(self.action_position, std=0.02)
        nn.init.normal_(self.condition_position, std=0.02)

    def forward(
        self,
        noisy_actions: Tensor,
        diffusion_step: Tensor,
        working_memory: Tensor,
        episodic_memory: Tensor,
    ) -> Tensor:
        if noisy_actions.shape[1:] != (self.action_horizon, self.action_dim):
            expected = (self.action_horizon, self.action_dim)
            raise ValueError(
                "Noisy actions have wrong shape: "
                f"{tuple(noisy_actions.shape)}, expected [B,{expected[0]},{expected[1]}]"
            )
        time_token = self.time_embedding(diffusion_step).unsqueeze(1)
        memory_tokens = self.memory_embedding(episodic_memory)
        working_tokens = self.working_embedding(working_memory)
        condition = torch.cat((time_token, memory_tokens, working_tokens), dim=1)
        condition = self.condition_encoder(
            self.embedding_dropout(
                condition + self.condition_position[:, : condition.shape[1]]
            )
        )
        actions = self.action_embedding(noisy_actions)
        actions = self.embedding_dropout(
            actions + self.action_position[:, : actions.shape[1]]
        )
        decoded = self.decoder(
            actions,
            condition,
            tgt_mask=self.causal_mask,
            memory_mask=self.condition_mask,
        )
        return self.output(self.output_norm(decoded))


def cosine_beta_schedule(steps: int, offset: float = 0.008) -> Tensor:
    points = torch.linspace(0, steps, steps + 1, dtype=torch.float64)
    cumulative = torch.cos(
        ((points / steps + offset) / (1 + offset)) * math.pi / 2
    ).square()
    cumulative = cumulative / cumulative[0]
    betas = 1 - cumulative[1:] / cumulative[:-1]
    return betas.clamp(1e-5, 0.999).float()


class VPWEM(nn.Module):
    """End-to-end VPWEM diffusion policy used for training and inference."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.encoder = ObservationEncoder(config)
        self.memory_compressor = ContextualMemoryCompressor(config)
        self.denoiser = ActionDenoiser(config)
        if config.beta_schedule != "cosine":
            raise ValueError("Only the cosine beta schedule is implemented")
        beta = cosine_beta_schedule(config.diffusion_steps)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", alpha_bar)

    def encode(self, rgb: Tensor, proprio: Tensor) -> Tensor:
        return self.encoder(rgb, proprio)

    @staticmethod
    def _drop_condition(tokens: Tensor, probability: float, training: bool) -> Tensor:
        if not training or probability <= 0:
            return tokens
        keep = torch.rand(tokens.shape[0], 1, 1, device=tokens.device) >= probability
        return tokens * keep.to(tokens.dtype)

    def condition(
        self,
        obs_rgb: Tensor,
        obs_proprio: Tensor,
        memory_rgb: Tensor,
        memory_proprio: Tensor,
        memory_timestep: Tensor,
        memory_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        working = self.encoder(obs_rgb, obs_proprio)
        if memory_rgb.shape[1] == 0:
            episodic = working.new_zeros(
                working.shape[0], self.config.memory_queries, self.config.embedding_dim
            )
        else:
            # VPWEM detaches out-of-window features before caching them. Avoid
            # constructing a large, unusable ResNet graph for up to twelve
            # historical frames per sample.
            with torch.no_grad():
                historical = self.encoder(memory_rgb, memory_proprio)
            episodic = self.memory_compressor(
                historical, memory_timestep, memory_mask
            )
        working = self._drop_condition(
            working, self.config.short_condition_dropout, self.training
        )
        episodic = self._drop_condition(
            episodic, self.config.long_condition_dropout, self.training
        )
        return working, episodic

    def diffusion_loss(
        self,
        actions: Tensor,
        obs_rgb: Tensor,
        obs_proprio: Tensor,
        memory_rgb: Tensor,
        memory_proprio: Tensor,
        memory_timestep: Tensor,
        memory_mask: Tensor,
        action_mask: Tensor | None = None,
    ) -> Tensor:
        working, episodic = self.condition(
            obs_rgb,
            obs_proprio,
            memory_rgb,
            memory_proprio,
            memory_timestep,
            memory_mask,
        )
        batch = actions.shape[0]
        timestep = torch.randint(
            self.config.diffusion_steps, (batch,), device=actions.device
        )
        noise = torch.randn_like(actions)
        alpha_bar = self.alpha_bar[timestep].view(batch, 1, 1)
        noisy = alpha_bar.sqrt() * actions + (1 - alpha_bar).sqrt() * noise
        predicted = self.denoiser(noisy, timestep, working, episodic)
        squared_error = (predicted - noise).square()
        if action_mask is None:
            return squared_error.mean()
        weights = action_mask.to(squared_error.dtype).unsqueeze(-1)
        denominator = (weights.sum() * squared_error.shape[-1]).clamp_min(1)
        return (squared_error * weights).sum() / denominator

    @torch.no_grad()
    def sample(
        self,
        working_memory: Tensor,
        episodic_memory: Tensor,
        *,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
        num_inference_steps: int | None = None,
    ) -> Tensor:
        batch = working_memory.shape[0]
        actions = torch.randn(
            batch,
            self.config.action_horizon,
            self.config.action_dim,
            device=working_memory.device,
            dtype=working_memory.dtype,
            generator=generator,
        ) * temperature
        requested_steps = num_inference_steps or self.config.diffusion_steps
        if requested_steps < 1 or requested_steps > self.config.diffusion_steps:
            raise ValueError(
                "num_inference_steps must be between 1 and "
                f"{self.config.diffusion_steps}, got {requested_steps}"
            )
        if requested_steps == self.config.diffusion_steps:
            schedule = list(range(self.config.diffusion_steps - 1, -1, -1))
        else:
            schedule = (
                torch.linspace(
                    self.config.diffusion_steps - 1,
                    0,
                    requested_steps,
                    device=working_memory.device,
                )
                .round()
                .long()
                .unique_consecutive()
                .tolist()
            )
        for schedule_index, index in enumerate(schedule):
            timestep = torch.full(
                (batch,), index, device=actions.device, dtype=torch.long
            )
            predicted_noise = self.denoiser(
                actions, timestep, working_memory, episodic_memory
            )
            alpha_bar = self.alpha_bar[index]
            predicted_x0 = (
                actions - (1 - alpha_bar).sqrt() * predicted_noise
            ) / alpha_bar.sqrt()
            predicted_x0 = predicted_x0.clamp(-1, 1)
            if schedule_index == len(schedule) - 1:
                actions = predicted_x0
                continue
            next_index = schedule[schedule_index + 1]
            if requested_steps != self.config.diffusion_steps:
                # Deterministic DDIM update for a strided inference schedule.
                next_alpha_bar = self.alpha_bar[next_index]
                actions = (
                    next_alpha_bar.sqrt() * predicted_x0
                    + (1 - next_alpha_bar).sqrt() * predicted_noise
                )
                continue
            alpha_bar_previous = self.alpha_bar[index - 1]
            beta = self.beta[index]
            alpha = self.alpha[index]
            denominator = 1 - alpha_bar
            mean = (
                beta * alpha_bar_previous.sqrt() / denominator * predicted_x0
                + (1 - alpha_bar_previous) * alpha.sqrt() / denominator * actions
            )
            variance = beta * (1 - alpha_bar_previous) / denominator
            noise = torch.randn(
                actions.shape,
                device=actions.device,
                dtype=actions.dtype,
                generator=generator,
            )
            actions = mean + variance.clamp_min(1e-20).sqrt() * noise
        return actions.clamp(-1, 1)


def make_ema_model(model: VPWEM) -> VPWEM:
    ema = copy.deepcopy(model).eval()
    ema.requires_grad_(False)
    return ema


@torch.no_grad()
def update_ema(ema: nn.Module, model: nn.Module, decay: float) -> None:
    ema_parameters = dict(ema.named_parameters())
    model_parameters = dict(model.named_parameters())
    if ema_parameters.keys() != model_parameters.keys():
        raise RuntimeError("EMA and online models do not have matching parameters")
    for name, ema_parameter in ema_parameters.items():
        ema_parameter.lerp_(model_parameters[name].detach(), 1.0 - decay)
    ema_buffers = dict(ema.named_buffers())
    for name, model_buffer in model.named_buffers():
        if name in ema_buffers:
            ema_buffers[name].copy_(model_buffer)
