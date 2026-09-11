"""Sequence-task behavior with tensor actors, plus an optional real GPU rollout."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch


class TensorActor:
    """Store poses with ManiSkill's subset-reset assignment semantics."""

    def __init__(self, scene, count):
        self.scene = scene
        self.raw = torch.zeros(count, 7)
        self.raw[:, 3] = 1
        self.linear_velocity = torch.zeros(count, 3)
        self.angular_velocity = torch.zeros(count, 3)

    @property
    def pose(self):
        return SimpleNamespace(raw_pose=self.raw, p=self.raw[:, :3])

    @pose.setter
    def pose(self, value):
        data = value.raw_pose if hasattr(value, "raw_pose") else value
        self.raw[self.scene._reset_mask] = data

    def set_pose(self, value):
        self.pose = value

    def set_linear_velocity(self, value):
        self.linear_velocity = value.clone()

    def set_angular_velocity(self, value):
        self.angular_velocity = value.clone()


@pytest.fixture
def env_class():
    pytest.importorskip("mikasa_robo_suite")
    from dom_vpwem.custom_envs.remember_color_sequence import RememberColorSequence3Long

    return RememberColorSequence3Long


def reset_subset(env, indices, seeds=None):
    from mani_skill.envs.utils.randomization.batched_rng import BatchedRNG

    indices = torch.tensor(indices)
    if seeds is not None:
        env._batched_episode_rng[indices] = BatchedRNG.from_seeds(seeds)
    env.scene._reset_mask[:] = False
    env.scene._reset_mask[indices] = True
    env._elapsed_steps[indices] = 0
    env._initialize_episode(indices, {})
    env.scene._reset_mask[:] = True
    return env.evaluate()


@pytest.fixture
def make_env(env_class):
    from mani_skill.envs.utils.randomization.batched_rng import BatchedRNG

    def make(seeds=(42,), n=2):
        # Only simulator construction is replaced. Exercise the real reset,
        # phase, observation and inherited reward implementations below.
        with patch.object(env_class.__bases__[0], "__init__", return_value=None):
            env = env_class(target_from_end=n)
        count = len(seeds)
        env.device = torch.device("cpu")
        env.num_envs = count
        env.color_dict = dict.fromkeys(range(3))
        env.scene = SimpleNamespace(_reset_mask=torch.ones(count, dtype=torch.bool))
        env.cubes = {key: TensorActor(env.scene, count) for key in range(3)}
        env.table_scene = SimpleNamespace(initialize=lambda indices: None)
        env.agent = SimpleNamespace(
            tcp=TensorActor(env.scene, count),
            reset=lambda qpos: None,
            is_static=lambda threshold: torch.ones(count, dtype=torch.bool),
            robot=SimpleNamespace(
                set_root_pose=lambda pose: None,
                get_qvel=lambda: torch.zeros(count, 9),
            ),
        )
        env.robot_init_qpos_noise = 0.02
        env._obs_mode = "rgb"
        env._elapsed_steps = torch.zeros(count, dtype=torch.int64)
        env._batched_episode_rng = BatchedRNG.from_seeds(seeds)
        env._allocate_episode_buffers()
        reset_subset(env, list(range(count)))
        return env

    return make


@pytest.mark.parametrize("n", [1, 2, 3])
def test_seeded_sampling_and_target_relative_to_random_end(make_env, n):
    env = make_env(tuple(range(128)), n=n)
    repeat = make_env(tuple(range(128)), n=n)
    assert set(env.sequence_lengths.tolist()) == {3, 4, 5, 6, 7}
    for name in ("sequence_colors", "sequence_lengths", "empty_steps_per_env", "oracle_info"):
        torch.testing.assert_close(getattr(env, name), getattr(repeat, name))
    for row, length in enumerate(env.sequence_lengths.tolist()):
        assert env.true_color_indices[row] == env.sequence_colors[row, length - n]
        assert env.cue_steps_per_env[row] == length * 10 + (length - 1) * 5
    assert torch.all(env.cue_steps_per_env + env.empty_steps_per_env <= 550)
    assert torch.all((env.empty_steps_per_env >= 50) & (env.empty_steps_per_env <= 450))
    # Per-episode seeds also produce the same sequence when run on their own.
    single = make_env((73,), n=n)
    torch.testing.assert_close(env.sequence_colors[73], single.sequence_colors[0])
    assert env.sequence_lengths[73] == single.sequence_lengths[0]
    positions = torch.stack(list(env.initial_poses.values()), dim=1)
    assert torch.all(torch.pdist(positions[0]) >= env.MANIP_MIN_CUBE_DISTANCE)
    layouts = positions[:, :, 1].argsort(dim=1)
    assert len(torch.unique(layouts, dim=0)) == 6


def test_repeated_colors_have_blank_gaps_and_independent_answer_times(make_env):
    env = make_env((1, 2))
    env.sequence_colors[:, :4] = torch.tensor([[0, 0, 2, 1], [2, 1, 0, 2]])
    env.sequence_lengths[:] = torch.tensor([3, 4])
    env.cue_steps_per_env[:] = torch.tensor([40, 55])
    env.empty_steps_per_env[:] = 7
    env.true_color_indices[:] = 0
    env.agent.tcp.raw[:, :3] = env.center_pose
    # -1: all hidden, 3: all three answer cubes. Row 0 repeats red and
    # ends in blue: seeing/reaching red during the cue must not win.
    for step, expected in (
        (0, [0, 2]),
        (9, [0, 2]),
        (10, [-1, -1]),
        (14, [-1, -1]),
        (15, [0, 1]),
        (24, [0, 1]),
        (25, [-1, -1]),
        (29, [-1, -1]),
        (30, [2, 0]),
        (39, [2, 0]),
        (40, [-1, -1]),
        (45, [-1, 2]),
        (46, [-1, 2]),
        (47, [3, 2]),
        (54, [3, 2]),
        (55, [3, -1]),
        (61, [3, -1]),
        (62, [3, 3]),
    ):
        env._elapsed_steps[:] = step
        info = env.evaluate()
        reward = env.compute_normalized_dense_reward({}, torch.zeros(2, 7), info)
        for row, value in enumerate(expected):
            visible = [key for key, cube in env.cubes.items() if cube.pose.p[row, 2] < 1]
            assert visible == ([] if value == -1 else [0, 1, 2] if value == 3 else [value])
            if value != 3:
                assert not info["success"][row]
                assert reward[row] == 0
            if value == -1:
                hidden_positions = torch.stack([cube.pose.p[row] for cube in env.cubes.values()])
                assert torch.all(torch.pdist(hidden_positions) >= env.MANIP_MIN_CUBE_DISTANCE)


@pytest.mark.parametrize("numpy_action", [False, True])
@pytest.mark.parametrize("flat_action", [False, True])
def test_cue_action_suppression_covers_gaps_releases_at_end_and_handles_partial_reset(
    make_env, numpy_action, flat_action
):
    env = make_env((1, 2))
    env.cue_steps_per_env[:] = torch.tensor([40, 55])
    env.empty_steps_per_env[:] = 7
    wrapper = env.CURRICULUM_WRAPPER(env)
    action = torch.full((7,) if flat_action else (2, 7), 0.75)
    if numpy_action:
        action = action.numpy()

    for step, frozen in (
        (0, [True, True]),
        (10, [True, True]),  # Blank gap after the first color.
        (29, [True, True]),  # Last step of another inter-color gap.
        (39, [True, True]),  # Last cue step for the first episode.
        (40, [False, True]),  # First episode is now in the final blank delay.
        (54, [False, True]),
        (55, [False, False]),  # Both cue sequences have ended.
        (62, [False, False]),  # Both answer phases have begun.
    ):
        env._elapsed_steps[:] = step
        actual = torch.as_tensor(wrapper.action(action))
        expected = torch.full((2, 7), 0.75)
        expected[frozen] = 0
        if flat_action and not any(frozen):
            expected = expected[0]
        torch.testing.assert_close(actual, expected)
        assert torch.all(torch.as_tensor(action) == 0.75)  # Preserve PPO's sampled actions.

    reset_subset(env, [1], seeds=[91])
    actual = torch.as_tensor(wrapper.action(action))
    torch.testing.assert_close(actual[0], torch.full((7,), 0.75))
    torch.testing.assert_close(actual[1], torch.zeros(7))


def test_only_target_in_answer_phase_wins_and_choices_are_not_reteleported(make_env):
    env = make_env((8,))
    target = int(env.true_color_indices[0])
    env._elapsed_steps[:] = env.cue_steps_per_env + env.empty_steps_per_env
    env.evaluate()
    for key, cube in env.cubes.items():
        env.agent.tcp.raw[:, :3] = cube.pose.p
        info = env.evaluate()
        assert info["success"].item() == (key == target)
        if key == target:
            reward = env.compute_normalized_dense_reward({}, torch.zeros(1, 7), info)
            assert reward.item() == 1
    env.cubes[target].raw[:, 0] += 0.04
    env.cubes[target].linear_velocity[:, 0] = 0.1
    shifted = env.cubes[target].raw.clone()
    env.evaluate()
    torch.testing.assert_close(env.cubes[target].raw, shifted)
    assert env.cubes[target].linear_velocity[0, 0].item() == pytest.approx(0.1)


def test_partial_reset_preserves_other_episodes_and_full_buffer_shapes(make_env):
    env = make_env((11, 12, 13))
    env._elapsed_steps[:] = torch.tensor([10, 600, 15])
    env.evaluate()
    env.cubes[0].raw[1, 0] += 0.03
    env._prev_action = torch.ones(3, 7)
    names = (
        "sequence_colors",
        "sequence_lengths",
        "cue_steps_per_env",
        "empty_steps_per_env",
        "true_color_indices",
        "_choices_revealed",
        "_elapsed_steps",
        "_prev_action",
    )
    snapshots = {name: getattr(env, name).clone() for name in names}
    poses = {key: cube.raw.clone() for key, cube in env.cubes.items()}
    initial = {key: value.clone() for key, value in env.initial_poses.items()}
    reset_subset(env, [2], seeds=[91])
    for name, before in snapshots.items():
        torch.testing.assert_close(getattr(env, name)[:2], before[:2])
        assert getattr(env, name).shape == before.shape
    for key, before in poses.items():
        torch.testing.assert_close(env.cubes[key].raw[:2], before[:2])
        torch.testing.assert_close(env.initial_poses[key][:2], initial[key][:2])
    assert env.oracle_info is env.true_color_indices
    assert torch.all(env._prev_action[2] == 0)
    expected = make_env((91,))
    torch.testing.assert_close(env.sequence_colors[2], expected.sequence_colors[0])
    assert env.true_color_indices[2] == expected.true_color_indices[0]


def test_student_observations_exclude_sequence_answer_and_timing(make_env):
    env = make_env()
    info = env.evaluate()
    assert set(env._get_obs_extra(info)) == {"tcp_pose"}
    env._obs_mode = "state"
    privileged = env._get_obs_extra(info)
    assert privileged["oracle_info"] is env.true_color_indices
    assert privileged["answer_phase"].item() == 0
    assert privileged["steps_until_choices"].item() > 0
    assert "sequence_colors" not in privileged


@pytest.mark.parametrize(
    ("settings", "kwargs", "message"),
    [
        ({}, {"target_from_end": 0}, "positive integer"),
        ({}, {"target_from_end": True}, "positive integer"),
        ({}, {"target_from_end": 1.5}, "positive integer"),
        ({}, {"target_from_end": 4}, "minimum sequence length"),
        ({"GAP_STEPS": 0}, {}, "positive integer"),
        ({"COLOR_STEPS": 0}, {}, "positive integer"),
        ({"SEQUENCE_LENGTH_RANGE": (7, 3)}, {}, "ascending order"),
        ({"EMPTY_PHASE_STEPS": (0, 4)}, {}, "positive integers"),
    ],
)
def test_invalid_settings_fail_before_simulator_construction(env_class, settings, kwargs, message):
    with patch.multiple(env_class, **settings) if settings else nullcontext():
        with patch.object(env_class.__bases__[0], "__init__", return_value=None) as base_init:
            with pytest.raises(ValueError, match=message):
                env_class(**kwargs)
            base_init.assert_not_called()


def test_sequence_oracle_rollout_and_partial_reset_on_gpu(env_class, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("Requires a working NVIDIA GPU and Vulkan renderer")
    from dom_vpwem.oracle_env import make_oracle_agent, make_oracle_env
    from dom_vpwem.tasks import REMEMBER_COLOR_SEQUENCE3_LONG_ENV_ID

    monkeypatch.setattr(env_class, "COLOR_STEPS", 1)
    monkeypatch.setattr(env_class, "GAP_STEPS", 1)
    monkeypatch.setattr(env_class, "EMPTY_PHASE_STEPS", (1, 2))
    env = make_oracle_env(REMEMBER_COLOR_SEQUENCE3_LONG_ENV_ID, num_envs=2)
    try:
        obs, _ = env.reset(seed=[123, 124])
        assert set(obs) == {"state"}
        agent = make_oracle_agent(env, env.device)
        with torch.no_grad():
            assert agent.get_action(obs, deterministic=True).shape == (2, 7)
        for _ in range(16):
            obs, reward, _, _, _ = env.step(torch.zeros((2, 7), device=env.device))
            assert torch.isfinite(obs["state"]).all()
            assert torch.isfinite(reward).all()
        assert env.unwrapped.manip_mask.all()
        sequence_before = env.unwrapped.sequence_colors[0].clone()
        obs, _ = env.reset(options={"env_idx": torch.tensor([1], device=env.device)})
        torch.testing.assert_close(env.unwrapped.sequence_colors[0], sequence_before)
        assert env.unwrapped.manip_mask.tolist() == [True, False]
        assert torch.isfinite(obs["state"]).all()
    finally:
        env.close()
