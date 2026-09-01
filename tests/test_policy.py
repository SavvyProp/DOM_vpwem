import numpy as np
import torch

from dom_vpwem.config import ModelConfig
from dom_vpwem.model import ContextualMemoryCompressor
from dom_vpwem.normalization import NormalizationStats
from dom_vpwem.policy import VPWEMPolicy


class _LightweightPolicyModel(torch.nn.Module):
    """Exercise policy state without constructing the image backbones."""

    def __init__(self) -> None:
        super().__init__()
        self.config = ModelConfig(
            embedding_dim=16,
            denoiser_heads=4,
            memory_heads=4,
            memory_layers=1,
            memory_cache_size=3,
            memory_cache_strategy="fifo",
            memory_subsample_ratio=1,
        )
        self.memory_compressor = ContextualMemoryCompressor(self.config)

    def encode(self, rgb, proprio):
        del rgb
        token = proprio.mean(dim=-1, keepdim=True)
        return token.expand(*token.shape[:-1], self.config.embedding_dim)

    def sample(self, working, episodic, **kwargs):
        del episodic, kwargs
        return working.new_zeros(
            working.shape[0],
            self.config.action_horizon,
            self.config.action_dim,
        )


def test_policy_keeps_only_bounded_online_memory() -> None:
    model = _LightweightPolicyModel()
    stats = NormalizationStats(
        proprio_min=(0.0,) * 7,
        proprio_max=(1.0,) * 7,
        action_min=(-1.0,) * 7,
        action_max=(1.0,) * 7,
    )
    policy = VPWEMPolicy(model, stats, device="cpu")
    state_identity = id(policy.memory_state)
    initial_shapes = (
        policy.memory_state.observation_cache.shape,
        tuple(cache.shape for cache in policy.memory_state.summary_caches),
        tuple(sizes.shape for sizes in policy.memory_state.summary_sizes),
    )

    for step in range(30):
        observation = {
            "rgb": np.zeros((128, 128, 6), dtype=np.uint8),
            "proprio": np.full(7, step / 30, dtype=np.float32),
        }
        actions = policy.act(observation)
        assert actions.shape == (1, model.config.action_dim)
        assert len(policy.working_embeddings) <= model.config.obs_steps

    assert id(policy.memory_state) == state_identity
    assert not hasattr(policy, "historical_embeddings")
    assert not hasattr(policy, "historical_timesteps")
    assert initial_shapes == (
        policy.memory_state.observation_cache.shape,
        tuple(cache.shape for cache in policy.memory_state.summary_caches),
        tuple(sizes.shape for sizes in policy.memory_state.summary_sizes),
    )
    assert int(policy.memory_state.observation_valid.sum()) == model.config.memory_cache_size
    assert all(
        int(valid.sum())
        == model.config.memory_cache_size * model.config.memory_queries
        for valid in policy.memory_state.summary_valid
    )
