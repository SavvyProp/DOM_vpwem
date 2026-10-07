from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from dom_vpwem.collect_demos import CollectConfig, collect
from dom_vpwem.config import ModelConfig
from dom_vpwem.demo_env import DemoEnv
from dom_vpwem.normalization import NormalizationStats
from dom_vpwem.policy import load_policy
from dom_vpwem.tasks import SHELL_GAME_SHUFFLE_TOUCH_CUSTOM_ENV_ID
from dom_vpwem.tracking import (
    BallPositionPredictor,
    TrackingConfig,
    TrackingExperimentConfig,
    TrackingMemoryCompressor,
    TrackingVPWEM,
)
from dom_vpwem.tracking_data import TrackingNpzDataset
from dom_vpwem.train import CHECKPOINT_VERSION, _validate_resume_compatibility, train
from dom_vpwem.train_tracking import prepare_tracking_batch


def model_config():
    return ModelConfig(
        embedding_dim=16, denoiser_layers=1, denoiser_heads=4, memory_layers=2,
        memory_heads=4, memory_cache_size=2, memory_subsample_ratio=1,
        diffusion_steps=2, crop_height=32, crop_width=32, denoiser_dropout=0,
        short_condition_dropout=0, long_condition_dropout=0,
    )


def write_episode(root: Path, *, labels=True, length=5):
    rgb = np.zeros((length, 128, 128, 6), np.uint8)
    for index in range(length):
        rgb[index] = index
    arrays = {
        "rgb": rgb, "proprio": np.zeros((length, 7), np.float32),
        "action": np.zeros((length, 7), np.float32),
    }
    if labels:
        arrays.update(
            tracking_xy=np.stack([np.arange(length), -np.arange(length)], axis=1).astype(
                np.float32
            ) * 0.01,
            tracking_hidden=np.arange(length) > 0,
        )
    np.savez(root / "train_data_0.npz", **arrays)


def test_tracking_labels_align_with_prefix_and_current_observation(tmp_path):
    write_episode(tmp_path)
    dataset = TrackingNpzDataset(
        tmp_path, obs_steps=2, horizon=9, memory_steps=6, preload=True,
    )
    for timestep in range(5):
        sample = dataset[timestep]
        valid = sample["tracking_mask"]
        rgb = torch.cat([sample["memory_rgb"], sample["obs"]["rgb"]])
        torch.testing.assert_close(
            sample["tracking_xy"][valid, 0], rgb[valid, 0, 0, 0].float() * 0.01
        )
        assert sample["tracking_mask"][-1]
        assert sample["tracking_xy"][-1, 0].item() == pytest.approx(timestep * 0.01)
        assert "tracking_xy" not in sample["obs"]


def test_tracking_dataset_rejects_unlabeled_demonstrations(tmp_path):
    write_episode(tmp_path, labels=False)
    with pytest.raises(ValueError, match="Collect fresh"):
        TrackingNpzDataset(tmp_path)


def test_tracking_head_trains_both_memories():
    head = BallPositionPredictor(model_config(), TrackingConfig())
    working = torch.randn(2, 2, 16, requires_grad=True)
    episodic = torch.randn(2, 2, 16, requires_grad=True)
    head(working, episodic).square().mean().backward()
    assert working.grad.abs().sum() > 0
    assert episodic.grad.abs().sum() > 0


def disable_dropout(module):
    for child in module.modules():
        if isinstance(child, nn.Dropout):
            child.p = 0
        if isinstance(child, nn.MultiheadAttention):
            child.dropout = 0


def test_functional_memory_matches_online_memory_with_padding_and_eviction():
    config = model_config()
    original = TrackingMemoryCompressor(config).eval()
    functional = TrackingMemoryCompressor(config).train()
    functional.load_state_dict(original.state_dict())
    disable_dropout(functional)
    history = torch.randn(2, 7, 16)
    times = torch.arange(7).expand(2, -1)
    mask = torch.tensor([[True] * 7, [False, False, True, True, False, True, True]])
    expected_state = original.init_state(2)
    actual_state = functional.init_state(2)
    with torch.no_grad():
        for index in range(7):
            expected = original.step(
                history[:, index], times[:, index], expected_state, mask[:, index]
            )
            actual = functional.step(
                history[:, index], times[:, index], actual_state, mask[:, index]
            )
            torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("detach", [False, True])
def test_memory_temporal_gradients_and_chunk_boundary(detach):
    compressor = TrackingMemoryCompressor(model_config()).train()
    history = torch.randn(1, 5, 16, requires_grad=True)
    state = compressor.init_state(1)
    for index in range(5):
        if detach and index == 3:
            compressor.detach_state(state)
        output = compressor.step(history[:, index], torch.tensor([index]), state)
    output.square().sum().backward()
    assert history.grad[:, -1].abs().sum() > 0
    if detach:
        assert history.grad[:, :3].abs().sum() == 0
    else:
        assert history.grad[:, 0].abs().sum() > 0


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(7, 16)

    def forward(self, rgb, proprio):
        return self.projection(proprio + rgb[..., 0, 0, :1].float() / 255)


class UnconditionalDenoiser(nn.Module):
    def forward(self, noisy, step, working, episodic):
        return torch.zeros_like(noisy)


def test_auxiliary_loss_trains_shared_features_without_action_shortcut(tmp_path):
    write_episode(tmp_path, length=7)
    dataset = TrackingNpzDataset(tmp_path, obs_steps=2, horizon=9, memory_steps=7)
    batch = torch.utils.data.default_collate([dataset[6]])
    stats = NormalizationStats((0,) * 7, (1,) * 7, (-1,) * 7, (1,) * 7)
    model = TrackingVPWEM(model_config(), TrackingConfig(unroll_steps=3))
    model.encoder = TinyEncoder()
    model.denoiser = UnconditionalDenoiser()  # diffusion contributes no feature gradients
    loss = model.diffusion_loss(**prepare_tracking_batch(batch, stats, torch.device("cpu")))
    assert torch.isfinite(loss)
    assert model.loss_metrics["tracking_prefix_loss"] > 0
    loss.backward()
    assert model.encoder.projection.weight.grad.abs().sum() > 0
    assert model.memory_compressor.input_projection.weight.grad.abs().sum() > 0
    assert model.position_predictor.query.grad.abs().sum() > 0


def test_ball_position_token_changes_actions_and_preserves_causal_masks():
    torch.manual_seed(17)
    config = model_config()
    model = TrackingVPWEM(config, TrackingConfig(condition_on_position=True)).eval()
    working = torch.randn(2, config.obs_steps, config.embedding_dim)
    episodic = torch.randn(2, config.memory_queries, config.embedding_dim)
    original = model._action_memory(working, episodic, torch.zeros(2, 2))
    shifted = model._action_memory(working, episodic, torch.ones(2, 2))
    assert original.shape == (2, config.memory_queries + 1, config.embedding_dim)
    torch.testing.assert_close(original[:, :-1], episodic)
    assert model.memory_compressor.init_state(2).last_memory.shape == episodic.shape

    position_index = 1 + config.memory_queries
    mask = model.denoiser.condition_mask
    assert torch.isneginf(mask[:config.obs_steps - 1, position_index]).all()
    assert (mask[config.obs_steps - 1:, position_index] == 0).all()
    assert torch.isneginf(mask[0, position_index + 2])  # latest working frame
    noisy = torch.randn(2, config.action_horizon, config.action_dim)
    timestep = torch.ones(2, dtype=torch.long)
    with torch.no_grad():
        before = model.denoiser(noisy, timestep, working, original)
        after = model.denoiser(noisy, timestep, working, shifted)
    torch.testing.assert_close(before[:, :config.obs_steps - 1], after[:, :config.obs_steps - 1])
    assert not torch.allclose(before[:, config.obs_steps - 1:], after[:, config.obs_steps - 1:])


def test_diffusion_loss_trains_position_predictor_through_token(tmp_path):
    write_episode(tmp_path, length=7)
    dataset = TrackingNpzDataset(tmp_path, obs_steps=2, horizon=9, memory_steps=7)
    batch = torch.utils.data.default_collate([dataset[6]])
    stats = NormalizationStats((0,) * 7, (1,) * 7, (-1,) * 7, (1,) * 7)
    model = TrackingVPWEM(
        model_config(), TrackingConfig(condition_on_position=True, unroll_steps=3),
    )
    model.encoder = TinyEncoder()
    # Isolate action-loss gradients from the auxiliary objective entirely.
    model._position_loss = lambda prediction, *args: prediction.new_zeros(())
    loss = model.diffusion_loss(**prepare_tracking_batch(batch, stats, torch.device("cpu")))
    assert torch.isfinite(loss)
    loss.backward()
    for parameter in (
        model.position_predictor.query,
        model.position_predictor.mlp[-1].weight,
        model.ball_position_embedding.weight,
        model.memory_compressor.input_projection.weight,
        model.encoder.projection.weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_position_token_matches_sampling_and_never_uses_ground_truth(tmp_path):
    class CapturingDenoiser(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []

        def forward(self, noisy, step, working, episodic):
            self.calls.append((working.detach().clone(), episodic.detach().clone()))
            return torch.zeros_like(noisy)

    write_episode(tmp_path, length=3)
    dataset = TrackingNpzDataset(tmp_path, obs_steps=2, horizon=9, memory_steps=0)
    batch = torch.utils.data.default_collate([dataset[1]])
    stats = NormalizationStats((0,) * 7, (1,) * 7, (-1,) * 7, (1,) * 7)
    model = TrackingVPWEM(model_config(), TrackingConfig(condition_on_position=True))
    model.encoder = TinyEncoder()
    model.denoiser = CapturingDenoiser()
    predictions = []
    model.position_predictor.register_forward_hook(
        lambda module, args, output: predictions.append(output.detach().clone())
    )
    prepared = prepare_tracking_batch(batch, stats, torch.device("cpu"))
    model.diffusion_loss(**prepared)
    working, action_memory = model.denoiser.calls[-1]
    torch.testing.assert_close(
        action_memory[:, -1], model.ball_position_embedding(predictions[-1]),
    )
    assert len(predictions) == 1
    # Changing privileged labels affects only supervision, never conditioning.
    model.diffusion_loss(**{**prepared, "tracking_xy": prepared["tracking_xy"] + 10})
    torch.testing.assert_close(model.denoiser.calls[-1][1], action_memory)

    model.eval()
    model.denoiser.calls.clear()
    predictions.clear()
    sampled = model.sample(working, action_memory[:, :-1])
    assert torch.isfinite(sampled).all()
    assert len(predictions) == 1
    assert len(model.denoiser.calls) == model.config.diffusion_steps
    for _, sampled_memory in model.denoiser.calls:
        torch.testing.assert_close(sampled_memory, action_memory)


def test_logical_ball_labels_follow_cup_and_do_not_read_parked_ball():
    cups = [SimpleNamespace(pose=SimpleNamespace(p=torch.tensor([[float(i), -float(i), 0]])))
            for i in range(3)]
    base = SimpleNamespace(
        mug_left=cups[0], mug_center=cups[1], mug_right=cups[2],
        cup_with_ball_number=torch.tensor([2]), elapsed_steps=torch.tensor([4]),
        cue_steps_per_env=torch.tensor([2]),
    )
    env = object.__new__(DemoEnv)
    env.env = SimpleNamespace(unwrapped=base)
    env.num_envs = 1
    labels = env.tracking_labels()
    torch.testing.assert_close(labels["tracking_xy"], torch.tensor([[2., -2.]]))
    assert labels["tracking_hidden"].item()


@pytest.mark.parametrize("condition_on_position", [False, True])
def test_short_training_resume_and_policy_loading(tmp_path, condition_on_position):
    data = tmp_path / "data"
    data.mkdir()
    write_episode(data, length=3)
    config = TrackingExperimentConfig.from_yaml("configs/shell_game_shuffle_touch_tracking.yaml")
    config.model = model_config()
    config.tracking.condition_on_position = condition_on_position
    config.task.dataset_dir = str(data)
    config.train.output_dir = str(tmp_path / "output")
    config.train.device = "cpu"
    config.train.num_workers = 0
    config.train.gradient_steps = 2
    config.train.save_every = 1
    config.train.batch_size = 2
    config.train.mixed_precision = False
    checkpoint = train(
        config, dataset_factory=TrackingNpzDataset,
        model_factory=lambda cfg: TrackingVPWEM(cfg, config.tracking),
        batch_preparer=prepare_tracking_batch,
    )
    payload = torch.load(checkpoint, weights_only=False)
    assert payload["format_version"] == CHECKPOINT_VERSION
    assert "tracking" in payload["config"]
    if not condition_on_position:
        # Simulate a checkpoint produced before position conditioning existed.
        del payload["config"]["tracking"]["condition_on_position"]
        torch.save(payload, checkpoint)
    policy = load_policy(checkpoint, device="cpu")
    assert isinstance(policy.model, TrackingVPWEM)
    assert policy.model.tracking_config.condition_on_position == condition_on_position
    assert policy.model.denoiser.memory_queries == (
        config.model.memory_queries + condition_on_position
    )
    action = policy.act({"rgb": np.zeros((128, 128, 6), np.uint8), "proprio": np.zeros(7)})
    assert action.shape == (1, 7) and torch.isfinite(action).all()
    config.train.resume = str(checkpoint.parent / "checkpoint_1.pt")
    resumed = train(
        config, dataset_factory=TrackingNpzDataset,
        model_factory=lambda cfg: TrackingVPWEM(cfg, config.tracking),
        batch_preparer=prepare_tracking_batch,
    )
    assert resumed == checkpoint
    assert torch.load(resumed, weights_only=False)["step"] == 2
    changed = replace(config, tracking=replace(config.tracking, loss_weight=2))
    with pytest.raises(ValueError, match="tracking configuration"):
        _validate_resume_compatibility(config, changed)


def test_collection_saves_pre_step_labels(tmp_path):
    class Env:
        device = torch.device("cpu")
        schema = {}

        def __init__(self, *args, **kwargs):
            self.t = 0

        def observation(self):
            return {"rgb": np.full((1, 128, 128, 6), self.t, np.uint8),
                    "proprio": np.zeros((1, 7), np.float32)}

        def reset(self, **kwargs):
            self.t = 0
            return self.observation(), {}

        def tracking_labels(self):
            return {"tracking_xy": np.full((1, 2), self.t, np.float32),
                    "tracking_hidden": np.asarray([bool(self.t)])}

        def oracle_observation(self):
            return {"state": torch.zeros(1, 1)}

        def step(self, action):
            self.executed_action = action
            self.t += 1
            return self.observation(), np.zeros(1), np.zeros(1), np.zeros(1), {
                "success": np.asarray([self.t == 2])}

        def close(self):
            pass

    class Expert:
        def get_action(self, *args, **kwargs):
            return torch.zeros(1, 7)

    weights = tmp_path / "expert.pt"
    weights.write_bytes(b"test")
    config = CollectConfig(
        SHELL_GAME_SHUFFLE_TOUCH_CUSTOM_ENV_ID, checkpoint=str(weights),
        data_root=str(tmp_path), episodes=1, num_envs=1, device="cpu",
    )
    destination = collect(config, env_factory=Env, expert_factory=lambda *args: Expert())
    with np.load(destination / "train_data_000000.npz", allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["tracking_xy"][:, 0], [0, 1])
        np.testing.assert_array_equal(archive["tracking_hidden"], [False, True])
