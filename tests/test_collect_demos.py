import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dom_vpwem.collect_demos import CollectConfig, collect, existing_episodes, load_expert, main
from dom_vpwem.data import load_npz_episode
from dom_vpwem.demo_env import DemoEnv, privileged_observation
from dom_vpwem.tasks import INTERCEPT_FAST_COVER_ENV_ID


class ToyDemoEnv:
    """Mutable image buffers; odd-seeded episodes fail before later success."""

    schema = {"observations": [{"key": "state", "shape": [2]}], "action_shape": [7]}
    device = torch.device("cpu")

    def __init__(self, env_id, *, num_envs, sim_backend):
        self.num_envs = num_envs
        self.rgb = np.zeros((num_envs, 128, 128, 6), dtype=np.uint8)
        self.closed = False
        self.reset_calls = []

    def reset(self, *, seed):
        self.seeds = np.asarray(seed)
        self.reset_calls.append(list(seed))
        self.t = 0
        self.rgb[:] = 0
        return self.observation(), {}

    def observation(self):
        return {"rgb": self.rgb, "proprio": np.full((self.num_envs, 7), self.t, np.float32)}

    def oracle_observation(self):
        return {"state": torch.full((self.num_envs, 2), float(self.t))}

    def step(self, action):
        self.executed_action = action.clone() if self.t else torch.zeros_like(action)
        self.t += 1
        self.rgb[:] = self.t
        success = np.full(self.num_envs, self.t >= 2)
        terminated = (self.seeds % 2 == 1) & (self.t == 1)
        return (
            self.observation(),
            success.astype(float),
            terminated,
            np.zeros_like(success),
            {"success": success},
        )

    def close(self):
        self.closed = True


class ToyExpert:
    def get_action(self, state, deterministic):
        assert deterministic
        return torch.full((state["state"].shape[0], 7), 2.0)


@pytest.fixture
def config(tmp_path):
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"test oracle")
    return CollectConfig(
        env_id=INTERCEPT_FAST_COVER_ENV_ID,
        checkpoint=str(weights),
        data_root=str(tmp_path / "data"),
        episodes=2,
        num_envs=2,
        seed=20,
        device="cpu",
    )


def toy_collect(config, env_factory=ToyDemoEnv):
    return collect(config, env_factory=env_factory, expert_factory=lambda env, cfg: ToyExpert())


def test_collection_alignment_effective_actions_and_individual_success(config):
    envs = []

    def factory(*args, **kwargs):
        env = ToyDemoEnv(*args, **kwargs)
        envs.append(env)
        return env

    destination = toy_collect(config, factory)
    assert envs[0].closed
    files = existing_episodes(config)
    assert len(files) == 2
    for index, path in enumerate(files):
        episode = load_npz_episode(path)
        assert episode.length == 2
        np.testing.assert_array_equal(episode.rgb[:, 0, 0, 0], [0, 1])
        np.testing.assert_array_equal(episode.proprio[:, 0], [0, 1])
        np.testing.assert_array_equal(episode.action[:, 0], [0, 1])
        with np.load(path, allow_pickle=False) as arrays:
            assert arrays["episode_seed"].item() == 20 + index * 2
            np.testing.assert_array_equal(arrays["success"], [False, True])
            np.testing.assert_array_equal(arrays["done"], [False, True])
            assert arrays["env_id"].item() == config.env_id
            assert all(arrays[key].dtype != object for key in arrays.files)
    manifest = json.loads((destination / "collection.json").read_text())
    assert manifest["attempted_episodes"] == 4
    assert manifest["saved_episodes"] == 2
    # No checkpoint or simulator is needed when a complete dataset is reused.
    assert (
        collect(replace(config, checkpoint=None), env_factory=lambda *a, **k: pytest.fail())
        == destination
    )


def test_attempt_limit_retains_data_and_resume_extends_without_overwriting(config):
    config.episodes = 3
    config.max_attempts = 4
    with pytest.raises(RuntimeError, match="2/3 successful episodes"):
        toy_collect(config)
    before = {path.name: path.read_bytes() for path in existing_episodes(config)}
    toy_collect(replace(config, max_attempts=8))
    files = existing_episodes(config)
    assert len(files) == 3
    for path in files[:2]:
        assert path.read_bytes() == before[path.name]
    with np.load(files[-1], allow_pickle=False) as arrays:
        assert arrays["episode_seed"].item() == 24


def test_interrupted_batch_reserves_seeds_and_changed_checkpoint_is_rejected(config):
    class Interrupted(ToyDemoEnv):
        def reset(self, *, seed):
            if self.reset_calls:
                raise RuntimeError("interrupted")
            return super().reset(seed=seed)

    with pytest.raises(RuntimeError, match="interrupted"):
        toy_collect(config, Interrupted)
    files = existing_episodes(config)
    assert len(files) == 1
    other = config.destination.parent / "other.pt"
    other.write_bytes(b"different weights")
    with pytest.raises(ValueError, match="oracle_sha256 differs"):
        toy_collect(replace(config, checkpoint=str(other)))
    toy_collect(config)
    with np.load(existing_episodes(config)[1], allow_pickle=False) as arrays:
        assert arrays["episode_seed"].item() == 24  # 22/23 were reserved by the failed batch.


def test_status_checks_existing_data_without_an_oracle(config):
    arguments = [
        "--env-id",
        config.env_id,
        "--data-root",
        config.data_root,
        "--episodes",
        "2",
        "--status",
    ]
    assert main(arguments) == 1
    toy_collect(config)
    assert main(arguments) == 0
    path = existing_episodes(config)[0]
    with np.load(path, allow_pickle=False) as arrays:
        payload = {key: arrays[key] for key in arrays.files}
    payload["env_id"] = np.asarray("SomeOtherTask-VLA-v0")
    np.savez_compressed(path, **payload)
    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 2


def test_privileged_read_does_not_advance_and_restores_rgb_mode_on_error():
    info = {"elapsed_steps": torch.tensor([5])}
    base = SimpleNamespace(_obs_mode="rgb", task_cue=torch.tensor([7]))

    def get_obs(received):
        assert received is info
        assert base._obs_mode == "state"
        return torch.tensor([[1.0, 2.0]])

    base.get_obs = get_obs
    result = privileged_observation(base, info)
    assert base._obs_mode == "rgb"
    torch.testing.assert_close(result["state"], torch.tensor([[1.0, 2.0]]))
    assert result["task_cue"].shape == (1, 1)

    def fail(received):
        raise RuntimeError("failed state query")

    base.get_obs = fail
    with pytest.raises(RuntimeError, match="failed state query"):
        privileged_observation(base, info)
    assert base._obs_mode == "rgb"


@pytest.mark.parametrize("phase", ["cue_and_delay", "sequence", "intercept"])
def test_demo_env_captures_actions_after_real_curriculum_wrapper(monkeypatch, phase):
    gym = pytest.importorskip("gymnasium")
    pytest.importorskip("mikasa_robo_suite")
    from mikasa_robo_suite.vla.utils.wrappers import CurriculumPhaseNoopActionWrapper

    import dom_vpwem.demo_env as module
    from dom_vpwem.custom_envs.intercept_fast_cover import InterceptCueNoopActionWrapper
    from dom_vpwem.custom_envs.remember_color_sequence import CuePhaseNoopActionWrapper

    wrapper = {
        "cue_and_delay": CurriculumPhaseNoopActionWrapper,
        "sequence": CuePhaseNoopActionWrapper,
        "intercept": InterceptCueNoopActionWrapper,
    }[phase]
    cue_steps = 5 if phase == "intercept" else 1
    blank_steps = 10 if phase == "sequence" else 0

    class RawEnv(gym.Env):
        _obs_mode = "rgb"
        task_cue = None
        device = torch.device("cpu")
        num_envs = 2
        single_action_space = gym.spaces.Box(-1, 1, shape=(7,), dtype=np.float32)
        action_space = gym.spaces.Box(-1, 1, shape=(2, 7), dtype=np.float32)
        observation_space = gym.spaces.Dict({})
        CUE_STEPS = cue_steps
        cue_steps_per_env = torch.ones(2, dtype=torch.int64)
        empty_steps_per_env = torch.full((2,), blank_steps, dtype=torch.int64)

        @property
        def elapsed_steps(self):
            return torch.full((2,), self.t)

        def reset(self, *, seed=None, options=None):
            self.t = 0
            return {}, {"elapsed_steps": torch.full((2,), self.t)}

        def get_obs(self, info):
            assert self._obs_mode == "state"
            assert info["elapsed_steps"][0].item() == self.t
            return torch.full((2, 3), float(self.t))

        def step(self, action):
            self.t += 1
            return (
                {},
                torch.zeros(2),
                torch.zeros(2, dtype=torch.bool),
                torch.zeros(2, dtype=torch.bool),
                {"elapsed_steps": torch.full((2,), self.t)},
            )

    raw = RawEnv()
    monkeypatch.setattr(
        module,
        "_load_runtime",
        lambda: (
            lambda *args, **kwargs: raw,
            lambda env, **kwargs: wrapper(env),
        ),
    )
    env = DemoEnv(INTERCEPT_FAST_COVER_ENV_ID, num_envs=2)
    try:
        for step in range(cue_steps + blank_steps):
            env.step(torch.ones(2, 7))
            assert torch.all(env.executed_action == 0)
            assert torch.all(env.oracle_observation()["state"] == step + 1)
            assert raw.t == step + 1 and raw._obs_mode == "rgb"
        env.step(torch.ones(2, 7))
        assert torch.all(env.executed_action == 1)
    finally:
        env.close()


def test_exported_oracle_loads_for_cover2_and_rejects_wrong_schema(tmp_path):
    gym = pytest.importorskip("gymnasium")
    pytest.importorskip("mikasa_robo_suite")
    from dom_vpwem.oracle_env import make_oracle_agent, oracle_observation_schema

    env = SimpleNamespace(
        single_observation_space=gym.spaces.Dict(
            {"state": gym.spaces.Box(-np.inf, np.inf, shape=(3,))}
        ),
        single_action_space=gym.spaces.Box(-1, 1, shape=(7,)),
    )
    env.schema = oracle_observation_schema(env)
    original = make_oracle_agent(env, torch.device("cpu"))
    path = tmp_path / "oracle.pt"
    torch.save(original.state_dict(), path)
    metadata = {"env_id": INTERCEPT_FAST_COVER_ENV_ID, "schema": env.schema}
    path.with_suffix(".json").write_text(json.dumps(metadata))
    cfg = CollectConfig("InterceptFastCover2-VLA-v0", checkpoint=str(path), device="cpu")
    loaded = load_expert(env, cfg)
    obs = {"state": torch.ones(1, 3)}
    torch.testing.assert_close(loaded.get_action(obs, True), original.get_action(obs, True))
    metadata["schema"] = {}
    path.with_suffix(".json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="schema does not match"):
        load_expert(env, cfg)


@pytest.mark.parametrize(
    "env_id",
    [
        "InterceptFastCover-VLA-v0",
        "InterceptFastCover2-VLA-v0",
        "ShellGameShuffleTouchCustom-VLA-v0",
        "RememberColorSequence3-Long-VLA-v0",
    ],
)
def test_collection_state_matches_ppo_factory_on_gpu(env_id):
    if not torch.cuda.is_available():
        pytest.skip("Requires working NVIDIA/Vulkan drivers")
    pytest.importorskip("mikasa_robo_suite")
    from dom_vpwem.oracle_env import make_oracle_env, oracle_observation_schema

    demo = DemoEnv(env_id, num_envs=2)
    oracle = None
    try:
        oracle = make_oracle_env(env_id, num_envs=2, auto_reset=False)
        visual, _ = demo.reset(seed=[80, 81])
        state, _ = oracle.reset(seed=[80, 81])
        assert visual["rgb"].shape == (2, 128, 128, 6)
        assert visual["proprio"].shape == (2, 7)
        assert demo.schema == oracle_observation_schema(oracle)
        torch.testing.assert_close(demo.oracle_observation()["state"], state["state"])
    finally:
        demo.close()
        if oracle is not None:
            oracle.close()
