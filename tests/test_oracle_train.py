from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.distributions import Normal

from dom_vpwem.oracle_env import make_oracle_agent, make_oracle_env, oracle_observation_schema
from dom_vpwem.oracle_train import (
    OracleTrainConfig,
    bootstrap_values,
    compute_advantages,
    evaluate_oracle,
    parse_config,
    train_oracle,
)


@pytest.mark.parametrize(
    ("variant", "horizon", "run_suffix"),
    [
        ("intercept_fast_cover", 60, "/fixed_start_v1"),
        ("intercept_fast_cover2", 60, "/fixed_start_v1"),
        ("shell_game_shuffle_touch", 60, ""),
        ("remember_color_sequence3_long", 600, ""),
    ],
)
def test_oracle_config_cli_overrides_and_validates_yaml(tmp_path, variant, horizon, run_suffix):
    config, dry_run = parse_config(
        [
            "--config",
            f"configs/oracles/{variant}.yaml",
            "--num-envs",
            "32",
            "--no-finite-horizon-gae",
            "--dry-run",
        ]
    )
    assert dry_run
    assert config.output_dir == f"outputs/oracles/{variant}{run_suffix}"
    assert config.num_envs == 32
    assert config.num_steps == horizon
    assert not config.finite_horizon_gae
    with pytest.raises(SystemExit):
        parse_config(["--num-envs", "1", "--num-steps", "1"])
    invalid = tmp_path / "bad.yaml"
    invalid.write_text("total_timesteps_typo: 100\n")
    with pytest.raises(SystemExit):
        parse_config(["--config", str(invalid)])


@pytest.mark.parametrize("finite_horizon", [False, True])
def test_advantages_stop_at_episode_boundaries(finite_horizon):
    # t=0 ends an episode, t=1 belongs to another. A huge reward at t=1
    # must not change the advantage assigned to t=0.
    rewards = torch.tensor([[1.0], [1000.0]])
    values = torch.tensor([[2.0], [3.0]])
    next_values = torch.tensor([[5.0], [7.0]])
    dones = torch.tensor([[True], [False]])
    advantages, returns = compute_advantages(
        rewards,
        values,
        next_values,
        dones,
        gamma=0.9,
        gae_lambda=0.95,
        finite_horizon=finite_horizon,
    )
    assert advantages[0].item() == pytest.approx(1 + 0.9 * 5 - 2)
    assert returns[0].item() == pytest.approx(1 + 0.9 * 5)


def test_finite_horizon_gae_matches_weighted_n_step_returns():
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    values = torch.tensor([[4.0], [5.0], [6.0]])
    next_values = torch.tensor([[5.0], [6.0], [7.0]])
    advantages, returns = compute_advantages(
        rewards,
        values,
        next_values,
        torch.zeros(3, 1, dtype=torch.bool),
        gamma=0.9,
        gae_lambda=0.8,
        finite_horizon=True,
    )
    targets = [1 + 0.9 * 5, 1 + 0.9 * 2 + 0.9**2 * 6, 1 + 0.9 * 2 + 0.9**2 * 3 + 0.9**3 * 7]
    expected = np.average(targets, weights=[1, 0.8, 0.8**2])
    assert returns[0].item() == pytest.approx(expected)
    assert advantages[0].item() == pytest.approx(expected - 4)


def test_bootstrap_uses_timeout_final_observation_and_zeroes_true_terminal():
    agent = SimpleNamespace(get_value=lambda obs: obs["state"])
    values = bootstrap_values(
        agent,
        {"state": torch.tensor([[100.0], [200.0], [30.0]])},
        torch.tensor([False, True, False]),
        torch.tensor([True, False, False]),
        {"final_observation": {"state": torch.tensor([[5.0], [6.0], [7.0]])}},
        torch.device("cpu"),
    )
    torch.testing.assert_close(values, torch.tensor([5.0, 0.0, 30.0]))
    with pytest.raises(RuntimeError, match="missing final_observation"):
        bootstrap_values(
            agent,
            {"state": torch.ones(1, 1)},
            torch.tensor([False]),
            torch.tensor([True]),
            {},
            torch.device("cpu"),
        )


class ToyOracle(nn.Module):
    """Small Gaussian actor to exercise the real optimization loop on CPU."""

    def __init__(self, env, device):
        super().__init__()
        self.mean = nn.Parameter(torch.full((7,), 1.5))
        self.logstd = nn.Parameter(torch.full((7,), -2.0))
        self.critic = nn.Linear(1, 1)
        self.optimization_actions = []
        self.to(device)

    def get_value(self, obs):
        return self.critic(obs["state"])

    def get_action(self, obs, deterministic=False):
        mean = self.mean.expand(len(obs["state"]), -1)
        return mean if deterministic else Normal(mean, self.logstd.exp()).sample()

    def get_action_and_value(self, obs, action=None):
        distribution = Normal(self.mean.expand(len(obs["state"]), -1), self.logstd.exp())
        if action is None:
            action = distribution.sample()
        else:
            self.optimization_actions.append(action.detach().clone())
        return (
            action,
            distribution.log_prob(action).sum(-1),
            distribution.entropy().sum(-1),
            self.get_value(obs),
        )


class ToyVectorEnv:
    def __init__(
        self, env_id=None, *, num_envs=1, sim_backend="cpu", auto_reset=True, succeeds=True
    ):
        self.device = torch.device("cpu")
        self.num_envs = num_envs
        self.auto_reset = auto_reset
        self.single_observation_space = {"state": SimpleNamespace(shape=(1,))}
        self.single_action_space = SimpleNamespace(shape=(7,))
        self.succeeds = succeeds
        self.steps = 0
        self.seeds = []
        self.actions = []
        self.closed = False

    def reset(self, seed=None):
        self.steps = 0
        self.seeds.append(seed)
        return {"state": torch.zeros(self.num_envs, 1)}, {}

    def step(self, action):
        self.actions.append(action.clone())
        self.steps += 1
        obs = {"state": torch.full((self.num_envs, 1), self.steps / 60)}
        reward = -(action - 0.5).square().mean(-1)
        term = torch.zeros(self.num_envs, dtype=torch.bool)
        trunc = torch.full((self.num_envs,), self.steps == 60, dtype=torch.bool)
        info = {"success": torch.full((self.num_envs,), self.succeeds and self.steps == 1)}
        if trunc.any() and self.auto_reset:
            info["final_observation"] = obs
            obs, _ = self.reset()
        return obs, reward, term, trunc, info

    def close(self):
        self.closed = True


def _config(tmp_path, **kwargs):
    config = OracleTrainConfig(
        output_dir=str(tmp_path / "oracle"),
        device="cpu",
        sim_backend="cpu",
        num_envs=1,
        num_eval_envs=1,
        num_steps=4,
        total_timesteps=8,
        num_minibatches=2,
        update_epochs=1,
        eval_episodes=2,
        eval_every=1,
        success_evals=2,
    )
    return replace(config, **kwargs)


def test_evaluation_counts_transient_success_and_preserves_rng(tmp_path):
    config = _config(tmp_path, num_eval_envs=2, eval_episodes=3)
    env = ToyVectorEnv(num_envs=2, auto_reset=False)
    agent = ToyOracle(env, "cpu")
    rng = torch.get_rng_state().clone()
    result = evaluate_oracle(agent, env, config, torch.device("cpu"))
    assert result["successes"] == [True, True, True]
    assert result["episode_seeds"] == [config.eval_seed + i for i in range(3)]
    assert env.seeds == [
        [config.eval_seed, config.eval_seed + 1],
        [config.eval_seed + 2, config.eval_seed + 3],
    ]
    assert agent.training
    assert torch.equal(rng, torch.get_rng_state())


def test_training_exports_raw_weights_and_resume_state(tmp_path):
    envs, agents = [], []

    def env_factory(*args, **kwargs):
        env = ToyVectorEnv(*args, succeeds=False, **kwargs)
        envs.append(env)
        return env

    def agent_factory(env, device):
        agent = ToyOracle(env, device)
        agents.append(agent)
        return agent

    config = _config(tmp_path)
    final = train_oracle(config, env_factory=env_factory, agent_factory=agent_factory)
    payload = torch.load(final, weights_only=True)
    restored = ToyOracle(envs[0], "cpu")
    restored.load_state_dict(payload, strict=True)
    assert all(env.closed for env in envs)
    assert all(action.abs().max() <= 1 for env in envs for action in env.actions)
    assert any(action.max() > 1 for action in agents[0].optimization_actions)
    assert not (final.parent / "final_success_ckpt.pt").exists()
    state_path = final.parent / "training_state.pt"
    state = torch.load(state_path, weights_only=False)
    assert state["global_step"] == 8
    assert state["optimizer"]["state"]
    assert state["schema"] == oracle_observation_schema(envs[0])

    resumed = replace(config, resume=str(state_path), total_timesteps=12)
    train_oracle(resumed, env_factory=env_factory, agent_factory=agent_factory)
    state = torch.load(state_path, weights_only=False)
    assert state["global_step"] == 12
    assert state["iteration"] == 3
    assert len(envs[2].actions) == 4
    assert envs[2].seeds[0] == config.seed + 8
    assert all(env.closed for env in envs)
    with pytest.raises(ValueError, match="does not match checkpoint"):
        train_oracle(
            replace(resumed, gamma=0.5, total_timesteps=16),
            env_factory=env_factory,
            agent_factory=agent_factory,
        )
    assert all(env.closed for env in envs)


def test_success_checkpoint_requires_consecutive_evaluations(tmp_path):
    config = _config(tmp_path, total_timesteps=16)
    final = train_oracle(config, env_factory=ToyVectorEnv, agent_factory=ToyOracle)
    assert (final.parent / "final_success_ckpt.pt").exists()
    state = torch.load(final.parent / "training_state.pt", weights_only=False)
    assert state["global_step"] == 8  # Stopped after two qualifying evaluations.
    assert state["success_streak"] == 2


def test_export_network_matches_upstream_collector(tmp_path):
    pytest.importorskip("mikasa_robo_suite")
    from baselines.ppo.ppo_memtasks import AgentStateOnly

    trained = []

    def factory(env, device):
        agent = make_oracle_agent(env, device)
        trained.append(agent)
        return agent

    final = train_oracle(_config(tmp_path), env_factory=ToyVectorEnv, agent_factory=factory)
    collector_agent = AgentStateOnly(ToyVectorEnv())
    collector_agent.load_state_dict(torch.load(final, weights_only=True), strict=True)
    obs = {"state": torch.ones(1, 1)}
    torch.testing.assert_close(
        collector_agent.get_action(obs, deterministic=True),
        trained[0].get_action(obs, deterministic=True),
    )


def test_oracle_factory_and_upstream_agent_on_gpu():
    if not torch.cuda.is_available():
        pytest.skip("Requires a working NVIDIA GPU and Vulkan renderer")
    pytest.importorskip("mikasa_robo_suite")
    env = make_oracle_env("InterceptFastCover-VLA-v0", num_envs=2)
    try:
        obs, _ = env.reset(seed=[123, 124])
        assert set(obs) == {"state"}
        torch.testing.assert_close(obs["state"], env.unwrapped.get_obs())
        agent = make_oracle_agent(env, env.device)
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(obs)
        obs, reward, _, _, _ = env.step(action.clamp(-1, 1))
        assert action.shape == (2, 7)
        assert torch.isfinite(obs["state"]).all()
        assert torch.isfinite(reward).all()
    finally:
        env.close()
