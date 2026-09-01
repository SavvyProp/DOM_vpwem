import pytest
import torch

from dom_vpwem.config import ModelConfig
from dom_vpwem.model import VPWEM, ContextualMemoryCompressor


def tiny_config() -> ModelConfig:
    return ModelConfig(
        crop_height=32,
        crop_width=32,
        embedding_dim=32,
        denoiser_layers=1,
        denoiser_heads=4,
        denoiser_dropout=0.0,
        memory_layers=1,
        memory_heads=4,
        diffusion_steps=2,
    )


def test_vpwem_loss_and_sampling_shapes() -> None:
    config = tiny_config()
    model = VPWEM(config)
    batch, memory_steps = 1, 2
    inputs = {
        "actions": torch.rand(batch, config.action_horizon, 7) * 2 - 1,
        "obs_rgb": torch.randint(
            0, 256, (batch, config.obs_steps, 128, 128, 6), dtype=torch.uint8
        ),
        "obs_proprio": torch.rand(batch, config.obs_steps, 7),
        "memory_rgb": torch.randint(
            0, 256, (batch, memory_steps, 128, 128, 6), dtype=torch.uint8
        ),
        "memory_proprio": torch.rand(batch, memory_steps, 7),
        "memory_timestep": torch.arange(memory_steps)[None],
        "memory_mask": torch.ones(batch, memory_steps, dtype=torch.bool),
        "action_mask": torch.ones(batch, config.action_horizon, dtype=torch.bool),
    }
    loss = model.diffusion_loss(**inputs)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert any(
        parameter.grad is not None
        for parameter in model.memory_compressor.parameters()
    )
    assert any(parameter.grad is not None for parameter in model.encoder.parameters())

    model.eval()
    with torch.no_grad():
        working, episodic = model.condition(
            inputs["obs_rgb"],
            inputs["obs_proprio"],
            inputs["memory_rgb"],
            inputs["memory_proprio"],
            inputs["memory_timestep"],
            inputs["memory_mask"],
        )
        actions = model.sample(working, episodic)
    assert working.shape == (batch, config.obs_steps, config.embedding_dim)
    assert episodic.shape == (batch, config.memory_queries, config.embedding_dim)
    assert actions.shape == (batch, config.action_horizon, 7)
    assert torch.all(actions.abs() <= 1)


@pytest.mark.parametrize("strategy", ["fifo", "adjacent_similarity"])
def test_incremental_memory_matches_forward_in_eval_mode(strategy: str) -> None:
    config = tiny_config()
    config.memory_cache_size = 3
    config.memory_cache_strategy = strategy
    compressor = ContextualMemoryCompressor(config).eval()
    batch, time = 2, 8
    history = torch.randn(batch, time, config.embedding_dim)
    timesteps = torch.arange(time).expand(batch, -1)
    mask = torch.tensor(
        [
            [True, True, True, True, True, True, True, True],
            [True, False, True, True, False, True, True, True],
        ]
    )

    with torch.no_grad():
        expected = compressor(history, timesteps, mask)
        state = compressor.init_state(batch, device=history.device, dtype=history.dtype)
        for index in range(time):
            actual = compressor.step(
                history[:, index], timesteps[:, index], state, mask[:, index]
            )

    torch.testing.assert_close(actual, expected)
    assert state.observation_cache.shape == (
        batch,
        config.memory_cache_size,
        config.embedding_dim,
    )
    assert all(
        cache.shape
        == (
            batch,
            config.memory_cache_size,
            config.memory_queries,
            config.embedding_dim,
        )
        for cache in state.summary_caches
    )


def test_adjacent_similarity_compresses_each_query_stream_independently() -> None:
    config = tiny_config()
    config.memory_cache_size = 3
    config.memory_cache_strategy = "adjacent_similarity"
    compressor = ContextualMemoryCompressor(config)
    cache = torch.zeros(
        1,
        config.memory_cache_size,
        config.memory_queries,
        config.embedding_dim,
    )
    # Query 0's most similar adjacent pair is (0, 1).
    cache[0, 0, 0, 0] = 1
    cache[0, 1, 0, 0] = 1
    cache[0, 2, 0, 1] = 1
    # Query 1's most similar pair becomes (2, new).
    cache[0, 0, 1, 0] = 1
    cache[0, 1, 1, 1] = 1
    cache[0, 2, 1, 2] = 1
    value = torch.zeros(1, config.memory_queries, config.embedding_dim)
    value[0, 0, 1] = -1
    value[0, 1, 2] = 1
    valid = torch.ones(
        1,
        config.memory_cache_size,
        config.memory_queries,
        dtype=torch.bool,
    )
    sizes = torch.ones(
        1,
        config.memory_cache_size,
        config.memory_queries,
    )

    compressor._append_detached(cache, valid, sizes, value, torch.tensor([True]))

    torch.testing.assert_close(sizes[0, :, 0], torch.tensor([2.0, 1.0, 1.0]))
    torch.testing.assert_close(sizes[0, :, 1], torch.tensor([1.0, 1.0, 2.0]))
