from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dom_vpwem import evaluate as evaluate_module
from dom_vpwem import mikasa_env as mikasa_env_module
from dom_vpwem.evaluate import (
    EpisodeResult,
    EvaluationResult,
    VideoResult,
    _action_chunk,
    evaluate_policy,
    run_episode,
)
from dom_vpwem.mikasa_env import (
    DEFAULT_ENV_ID,
    MikasaContractError,
    MikasaEnvAdapter,
    MikasaEnvConfig,
    canonicalize_observation,
    make_mikasa_env,
    split_rgb,
)


def _observation(step: int = 0) -> dict[str, np.ndarray]:
    rgb = np.full((1, 128, 128, 6), step, dtype=np.uint8)
    proprio = np.full((1, 7), float(step), dtype=np.float32)
    return {
        "rgb": rgb,
        "proprio": proprio,
        "oracle_info": np.array([2], dtype=np.uint8),
        "task_cue": np.array([1], dtype=np.uint8),
    }


class FakeEnv:
    def __init__(self, *, horizon: int = 4) -> None:
        self.max_episode_steps = horizon
        self.action_space = SimpleNamespace(shape=(7,))
        self.unwrapped = self
        self.device = None
        self.actions: list[np.ndarray] = []
        self.seeds: list[int | None] = []
        self.closed = False
        self._step = 0

    def reset(self, *, seed=None, options=None):
        self._step = 0
        self.seeds.append(seed)
        return _observation(), {"language_instruction": "shell game"}

    def step(self, action):
        action = np.asarray(action)
        assert action.shape == (1, 7)
        self.actions.append(action.copy())
        self._step += 1
        # Success is deliberately transient; evaluation must OR-latch it.
        info = {"success": np.array([self._step == 2])}
        reward = np.array([float(self._step)], dtype=np.float32)
        terminated = np.array([False])
        truncated = np.array([self._step >= self.max_episode_steps])
        return _observation(self._step), reward, terminated, truncated, info

    def close(self):
        self.closed = True


class ChunkPolicy:
    def __init__(self) -> None:
        self.reset_calls: list[tuple[int, str | None]] = []
        self.act_calls = 0
        self.observation_shapes: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def reset(self, batch_size=1, device=None):
        self.reset_calls.append((batch_size, device))

    def act(self, observation):
        self.act_calls += 1
        self.observation_shapes.append((observation["rgb"].shape, observation["proprio"].shape))
        assert set(observation) == {"rgb", "proprio"}
        return np.full((2, 7), self.act_calls / 10.0, dtype=np.float32)


class SeedOutcomeEnv(FakeEnv):
    def __init__(self, success_seeds, *, horizon=3):
        super().__init__(horizon=horizon)
        self.success_seeds = set(success_seeds)

    def reset(self, **kwargs):
        observation, info = super().reset(**kwargs)
        observation["rgb"].fill(self.seeds[-1])
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        observation["rgb"].fill(self.seeds[-1] + self._step)
        info["success"] = np.array([self.seeds[-1] in self.success_seeds and self._step == 2])
        return observation, reward, terminated, truncated, info


def _patch_evaluation_recorder(monkeypatch):
    created = []

    class Recorder:
        def __init__(self, output, **kwargs):
            self.output = Path(output)
            self.kwargs = kwargs
            self.frames = []
            created.append(self)

        def __enter__(self):
            return self

        def write_frame(self, observation, *, step, success):
            self.frames.append((step, int(observation["rgb"][0, 0, 0]), success))

        def __exit__(self, exc_type, exc, traceback):
            if exc_type is None:
                self.output.parent.mkdir(parents=True, exist_ok=True)
                self.output.write_bytes(b"fake-video")

    monkeypatch.setattr(evaluate_module, "Mp4RolloutRecorder", Recorder)
    return created


def test_canonicalize_observation_unbatches_and_drops_privileged_keys():
    canonical = canonicalize_observation(_observation())

    assert set(canonical) == {"rgb", "proprio"}
    assert canonical["rgb"].shape == (128, 128, 6)
    assert canonical["rgb"].dtype == np.uint8
    assert canonical["proprio"].shape == (7,)
    assert canonical["proprio"].dtype == np.float32

    top, wrist = split_rgb(canonical["rgb"])
    assert top.shape == wrist.shape == (128, 128, 3)


@pytest.mark.parametrize(
    "observation, message",
    [
        ({"rgb": np.zeros((128, 128, 3), np.uint8), "proprio": np.zeros(7)}, "rgb shape"),
        ({"rgb": np.zeros((128, 128, 6), np.float32), "proprio": np.zeros(7)}, "uint8"),
        ({"rgb": np.zeros((128, 128, 6), np.uint8), "proprio": np.zeros(6)}, "proprio shape"),
    ],
)
def test_canonicalize_observation_rejects_contract_violations(observation, message):
    with pytest.raises(MikasaContractError, match=message):
        canonicalize_observation(observation)


def test_adapter_builds_canonical_env_lazily_with_injected_factories():
    calls = {}
    fake = FakeEnv()

    def make(env_id, **kwargs):
        calls["make"] = (env_id, kwargs)
        return fake

    def wrap(env, *, include_overlays):
        calls["wrap"] = (env, include_overlays)
        return env

    adapter = MikasaEnvAdapter(env_factory=make, wrapper_factory=wrap)
    env_id, kwargs = calls["make"]

    assert env_id == DEFAULT_ENV_ID
    assert kwargs == {
        "num_envs": 1,
        "obs_mode": "rgb",
        "control_mode": "pd_ee_delta_pose",
        "reward_mode": "normalized_dense",
        "render_mode": "all",
        "sim_backend": "gpu",
    }
    assert calls["wrap"] == (fake, False)
    adapter.close()
    assert fake.closed


def test_installed_runtime_preflights_lamp_only_for_shuffle_task(monkeypatch):
    fake = FakeEnv()
    checked: list[str] = []

    def make(env_id, **kwargs):
        del env_id, kwargs
        return fake

    def wrap(env, *, include_overlays):
        assert include_overlays is False
        return env

    monkeypatch.setattr(mikasa_env_module, "_load_runtime", lambda: (make, wrap))
    monkeypatch.setattr(
        mikasa_env_module,
        "require_mikasa_lamp_asset",
        lambda: checked.append("lamp"),
    )

    assert make_mikasa_env(MikasaEnvConfig()) is fake
    assert checked == ["lamp"]

    assert make_mikasa_env(MikasaEnvConfig(env_id="ShellGameTouch-VLA-v0")) is fake
    assert checked == ["lamp"]


def test_adapter_clips_batches_and_validates_actions():
    fake = FakeEnv(horizon=1)
    adapter = MikasaEnvAdapter(env=fake)
    adapter.reset(seed=7)

    adapter.step(np.array([2.0, -2.0, 0, 0, 0, 0, 0], dtype=np.float32))
    np.testing.assert_array_equal(fake.actions[0][0, :2], np.array([1.0, -1.0]))

    with pytest.raises(MikasaContractError, match="shape"):
        adapter.step(np.zeros(6, dtype=np.float32))
    with pytest.raises(MikasaContractError, match="NaN"):
        adapter.step(np.full(7, np.nan, dtype=np.float32))


def test_evaluate_policy_executes_fifo_chunks_and_latches_success():
    fake = FakeEnv(horizon=4)
    adapter = MikasaEnvAdapter(env=fake)
    policy = ChunkPolicy()

    result = evaluate_policy(
        policy,
        adapter=adapter,
        n_episodes=2,
        start_seed=100,
        device="cpu",
    )

    assert fake.seeds == [100, 101]
    assert policy.reset_calls == [(1, "cpu"), (1, "cpu")]
    assert policy.act_calls == 4  # two chunks of length two in each episode
    assert policy.observation_shapes == [((128, 128, 6), (7,))] * 4
    assert [episode.success_once for episode in result.episodes] == [True, True]
    assert [episode.episode_return for episode in result.episodes] == [10.0, 10.0]
    assert [episode.n_steps for episode in result.episodes] == [4, 4]
    assert result.success_rate == 1.0
    assert result.to_dict()["episode_seeds"] == [100, 101]
    assert result.to_dict()["split"] == "Short"
    assert result.to_dict()["memory_type"] == "Tracking"
    assert result.to_dict()["action_chunk_size"] == 2
    assert result.to_dict()["model"] == {"name": "dom-vpwem", "config": {}}
    assert not fake.closed  # caller-owned adapters remain open by default

    # The first two executed actions come from one call and are identical.
    np.testing.assert_array_equal(fake.actions[0], fake.actions[1])
    assert not np.array_equal(fake.actions[1], fake.actions[2])


def test_run_episode_records_reset_and_every_post_step_observation():
    class FrameSink:
        def __init__(self):
            self.frames = []

        def write_frame(self, observation, *, step, success):
            self.frames.append((step, int(observation["rgb"][0, 0, 0]), success))

    sink = FrameSink()
    episode = run_episode(
        MikasaEnvAdapter(env=FakeEnv(horizon=4)),
        ChunkPolicy(),
        seed=100,
        frame_sink=sink,
    )

    assert episode.n_steps == 4
    assert sink.frames == [
        (0, 0, False),
        (1, 1, False),
        (2, 2, True),
        (3, 3, True),
        (4, 4, True),
    ]


def test_evaluate_policy_records_only_the_selected_episode(monkeypatch, tmp_path):
    created = []

    class FakeRecorder:
        def __init__(self, output, **kwargs):
            self.output = output
            self.kwargs = kwargs
            self.frames = []
            created.append(self)

        def __enter__(self):
            return self

        def write_frame(self, observation, *, step, success):
            self.frames.append((step, int(observation["rgb"][0, 0, 0]), success))

        def __exit__(self, exc_type, exc, traceback):
            return None

    monkeypatch.setattr(evaluate_module, "Mp4RolloutRecorder", FakeRecorder)
    fake = FakeEnv(horizon=2)

    result = evaluate_policy(
        ChunkPolicy(),
        adapter=MikasaEnvAdapter(env=fake),
        n_episodes=3,
        start_seed=100,
        video_output=tmp_path / "rollout.mp4",
        video_episode=1,
        video_fps=12,
    )

    assert len(result.episodes) == 3
    assert fake.seeds == [100, 101, 102]
    assert len(created) == 1
    assert created[0].kwargs == {
        "fps": 12,
        "episode_index": 1,
        "seed": 101,
        "horizon": 2,
    }
    assert created[0].frames == [(0, 0, False), (1, 1, False), (2, 2, True)]


@pytest.mark.parametrize(
    ("success_seeds", "count", "ensure_success", "expected_indices"),
    [
        ([], 3, True, [0, 1, 2]),
        ([104], 3, True, [0, 1, 4]),
        ([100], 3, True, [0, 1, 2]),
        ([104], 3, False, [0, 1, 2]),
        ([103], 1, True, [3]),
        ([102, 104], 3, True, [0, 1, 2]),
    ],
)
def test_multiple_videos_include_actual_success_without_replays_or_biased_metrics(
    monkeypatch, tmp_path, success_seeds, count, ensure_success, expected_indices,
):
    created = _patch_evaluation_recorder(monkeypatch)
    fake = SeedOutcomeEnv(success_seeds)
    policy = ChunkPolicy()
    result = evaluate_policy(
        policy, adapter=MikasaEnvAdapter(env=fake), n_episodes=5, start_seed=100,
        video_output=tmp_path / "rollout.mp4", video_count=count,
        video_ensure_success=ensure_success,
    )
    assert fake.seeds == [100, 101, 102, 103, 104]
    assert len(policy.reset_calls) == 5
    assert len(result.episodes) == 5
    assert result.success_rate == len(success_seeds) / 5
    assert [video.episode_index for video in result.videos] == expected_indices
    assert len(result.videos) == count
    assert set(tmp_path.glob("*.mp4")) == {Path(video.path) for video in result.videos}
    # Only the initial clips and a needed replacement are encoded.
    assert len(created) == count + (expected_indices[-1] >= count)
    assert len(result.to_dict()["videos"]) == count
    for video in result.videos:
        assert video.seed == 100 + video.episode_index
        assert video.success_once == (video.seed in success_seeds)
        outcome = "success" if video.success_once else "failure"
        assert Path(video.path).name.endswith(f"seed_{video.seed}_{outcome}.mp4")
        recording = next(record for record in created if str(record.output) == video.path)
        assert recording.frames == [
            (0, video.seed, False), (1, video.seed + 1, False),
            (2, video.seed + 2, video.success_once), (3, video.seed + 3, video.success_once),
        ]


@pytest.mark.parametrize("success_seeds", [[], [104]])
def test_cli_multi_video_selection_and_no_success_report(
    monkeypatch, tmp_path, capsys, success_seeds,
):
    _patch_evaluation_recorder(monkeypatch)
    fake = SeedOutcomeEnv(success_seeds)
    adapter = MikasaEnvAdapter(env=fake)
    monkeypatch.setattr(evaluate_module, "_load_cli_policy", lambda args: ChunkPolicy())
    monkeypatch.setattr(evaluate_module, "MikasaEnvAdapter", lambda config: adapter)
    output = tmp_path / "result.json"
    assert evaluate_module.main([
        "--checkpoint", str(tmp_path / "policy.pt"), "--start-seed", "100",
        "--episodes", "5", "--video-output", str(tmp_path / "rollout.mp4"),
        "--video-count", "3", "--video-ensure-success", "--output", str(output),
    ]) == 0
    captured = capsys.readouterr()
    payload = json.loads(output.read_text())
    assert json.loads(captured.out) == payload
    assert payload["n_episodes"] == 5
    assert payload["sr"] == len(success_seeds) / 5
    assert len(payload["videos"]) == 3
    assert sum(video["success_once"] for video in payload["videos"]) == len(success_seeds)
    assert captured.err.count("Saved rollout video:") == 3
    assert ("No successful rollout occurred" in captured.err) == (not success_seeds)


def test_evaluate_policy_does_not_load_video_dependencies_without_output(
    monkeypatch,
):
    monkeypatch.setattr(
        evaluate_module,
        "Mp4RolloutRecorder",
        lambda *args, **kwargs: pytest.fail("recorder should not be constructed"),
    )

    result = evaluate_policy(
        ChunkPolicy(),
        adapter=MikasaEnvAdapter(env=FakeEnv(horizon=1)),
        n_episodes=1,
    )

    assert result.episodes[0].n_steps == 1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"video_output": "rollout.mov"}, "end in .mp4"),
        ({"video_output": "rollout.mp4", "video_episode": 2}, "between 0 and 0"),
        ({"video_output": "rollout.mp4", "video_fps": 0}, "positive integer"),
        ({"video_output": "rollout.mp4", "video_count": 0}, "positive integer"),
        ({"video_output": "rollout.mp4", "video_count": True}, "positive integer"),
        ({"video_output": "rollout.mp4", "video_count": 2}, "cannot exceed"),
        ({"video_count": 2}, "require --video-output"),
        ({"video_ensure_success": True}, "require --video-output"),
        ({"video_output": "rollout.mp4", "video_count": 2, "video_episode": 1,
          "n_episodes": 3}, "Use video_episode=0"),
    ],
)
def test_evaluate_policy_validates_video_options_before_rollout(kwargs, message):
    fake = FakeEnv(horizon=1)
    kwargs = dict(kwargs)
    n_episodes = kwargs.pop("n_episodes", 1)
    with pytest.raises(ValueError, match=message):
        evaluate_policy(
            ChunkPolicy(),
            adapter=MikasaEnvAdapter(env=fake),
            n_episodes=n_episodes,
            **kwargs,
        )
    assert fake.seeds == []


@pytest.mark.parametrize(
    ("env_id", "expected_split", "expected_memory_type"),
    [
        ("ShellGameTouch-VLA-v0", "Short", "Spatial"),
        ("ShellGameShuffleColorLampTouch-VLA-v0", "Short", "Tracking"),
        ("ShellGameShuffleTouchCustom-VLA-v0", "Short", "Tracking"),
        ("RememberColorSequence3-Long-VLA-v0", "Long", "TemporalOrder"),
        ("InterceptFastCover-VLA-v0", "Short", "Spatial"),
        ("InterceptFastCover2-VLA-v0", "Short", "Spatial"),
    ],
)
def test_evaluate_policy_derives_benchmark_metadata_from_env_id(
    env_id,
    expected_split,
    expected_memory_type,
):
    adapter = MikasaEnvAdapter(
        config=MikasaEnvConfig(env_id=env_id),
        env=FakeEnv(horizon=1),
    )

    result = evaluate_policy(ChunkPolicy(), adapter=adapter, n_episodes=1)

    assert result.split == expected_split
    assert result.memory_type == expected_memory_type
    assert result.to_dict()["split"] == expected_split
    assert result.to_dict()["memory_type"] == expected_memory_type


def test_evaluate_policy_tolerates_no_argument_reset():
    class NoArgPolicy:
        def __init__(self):
            self.resets = 0

        def reset(self):
            self.resets += 1

        def act(self, observation):
            return np.zeros(7, dtype=np.float32)

    fake = FakeEnv(horizon=1)
    policy = NoArgPolicy()
    result = evaluate_policy(policy, adapter=MikasaEnvAdapter(env=fake), n_episodes=1)

    assert policy.resets == 1
    assert result.episodes[0].n_steps == 1


def test_evaluate_policy_rejects_inconsistent_emitted_chunk_sizes():
    class InconsistentPolicy:
        def __init__(self):
            self.calls = 0

        def reset(self, **kwargs):
            pass

        def act(self, observation):
            self.calls += 1
            size = 2 if self.calls == 1 else 1
            return np.zeros((size, 7), dtype=np.float32)

    with pytest.raises(ValueError, match="inconsistent action chunk sizes within"):
        evaluate_policy(
            InconsistentPolicy(),
            adapter=MikasaEnvAdapter(env=FakeEnv(horizon=3)),
            n_episodes=1,
        )


def test_evaluate_policy_rejects_chunk_size_changes_across_episodes():
    class PerEpisodeChunkPolicy:
        def __init__(self):
            self.episode = 0

        def reset(self, **kwargs):
            self.episode += 1

        def act(self, observation):
            return np.zeros((self.episode, 7), dtype=np.float32)

    with pytest.raises(ValueError, match="inconsistent action chunk sizes across"):
        evaluate_policy(
            PerEpisodeChunkPolicy(),
            adapter=MikasaEnvAdapter(env=FakeEnv(horizon=1)),
            n_episodes=2,
        )


def test_evaluate_policy_validates_declared_chunk_size():
    class IncorrectlyDeclaredPolicy(ChunkPolicy):
        action_chunk_size = 8

    with pytest.raises(ValueError, match="declares 8.*emitted chunks of size 2"):
        evaluate_policy(
            IncorrectlyDeclaredPolicy(),
            adapter=MikasaEnvAdapter(env=FakeEnv(horizon=1)),
            n_episodes=1,
        )

    class NonIntegralDeclarationPolicy(ChunkPolicy):
        action_chunk_size = 2.5

    with pytest.raises(ValueError, match="must be a positive integer"):
        evaluate_policy(
            NonIntegralDeclarationPolicy(),
            adapter=MikasaEnvAdapter(env=FakeEnv(horizon=1)),
            n_episodes=1,
        )


def test_cli_plumbs_model_and_benchmark_metadata(monkeypatch, tmp_path, capsys):
    loaded = object()
    captured: dict[str, object] = {}

    def fake_load(args):
        captured["load_args"] = args
        return loaded

    def fake_evaluate(policy, **kwargs):
        captured["policy"] = policy
        captured["evaluate_kwargs"] = kwargs
        return EvaluationResult(
            env_id=DEFAULT_ENV_ID,
            start_seed=kwargs["start_seed"],
            episodes=(EpisodeResult(123, True, 1.0, 1, action_chunk_size=3),),
            model=kwargs["model_name"],
            model_config=kwargs["model_config"],
            benchmark_commit=kwargs["benchmark_commit"],
            action_chunk_size=3,
            videos=(VideoResult(str(kwargs["video_output"]), 0, 123, True),),
        )

    monkeypatch.setattr(evaluate_module, "_load_cli_policy", fake_load)
    monkeypatch.setattr(evaluate_module, "evaluate_policy", fake_evaluate)
    output = tmp_path / "result.json"
    video_output = tmp_path / "rollout.mp4"
    checkpoint = tmp_path / "checkpoint.pt"

    assert (
        evaluate_module.main(
            [
                "--checkpoint",
                str(checkpoint),
                "--episodes",
                "1",
                "--num-inference-steps",
                "7",
                "--action-chunk-size",
                "3",
                "--model-name",
                "vpwem-test",
                "--benchmark-commit",
                "509b875",
                "--video-output",
                str(video_output),
                "--video-episode",
                "0",
                "--video-fps",
                "12",
                "--output",
                str(output),
            ]
        )
        == 0
    )

    load_args = captured["load_args"]
    assert load_args.num_inference_steps == 7
    assert load_args.action_chunk_size == 3
    assert captured["policy"] is loaded
    evaluate_kwargs = captured["evaluate_kwargs"]
    assert evaluate_kwargs["model_name"] == "vpwem-test"
    assert evaluate_kwargs["benchmark_commit"] == "509b875"
    assert evaluate_kwargs["video_output"] == video_output
    assert evaluate_kwargs["video_episode"] == 0
    assert evaluate_kwargs["video_fps"] == 12
    assert evaluate_kwargs["video_count"] == 1
    assert evaluate_kwargs["video_ensure_success"] is False
    assert evaluate_kwargs["model_config"] == {
        "checkpoint": str(checkpoint),
        "num_inference_steps": 7,
        "requested_action_chunk_size": 3,
    }

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["model"] == {
        "name": "vpwem-test",
        "config": evaluate_kwargs["model_config"],
    }
    assert payload["benchmark_commit"] == "509b875"
    assert payload["action_chunk_size"] == 3
    captured_output = capsys.readouterr()
    assert json.loads(captured_output.out) == payload
    assert f"Saved rollout video: {video_output}" in captured_output.err


@pytest.mark.parametrize("generated_path", [False, True])
def test_cli_rejects_using_the_same_json_and_video_path(
    monkeypatch, tmp_path, capsys, generated_path,
):
    monkeypatch.setattr(
        evaluate_module,
        "_load_cli_policy",
        lambda args: pytest.fail("policy should not be loaded"),
    )
    video_output = tmp_path / "same.mp4"
    output = (
        tmp_path / "same_episode_001_seed_4242424242_failure.mp4"
        if generated_path else video_output
    )

    with pytest.raises(SystemExit) as error:
        evaluate_module.main(
            [
                "--checkpoint",
                str(tmp_path / "checkpoint.pt"),
                "--output",
                str(output),
                "--video-output",
                str(video_output),
            ] + (["--video-count", "2"] if generated_path else [])
        )

    assert error.value.code == 2
    assert "must be different paths" in capsys.readouterr().err


@pytest.mark.parametrize(
    "output",
    [
        np.zeros((0, 7), dtype=np.float32),
        np.zeros((2, 6), dtype=np.float32),
        np.zeros((2, 1, 7), dtype=np.float32),
    ],
)
def test_action_chunk_rejects_invalid_policy_outputs(output):
    with pytest.raises(ValueError):
        _action_chunk(output)


def test_config_rejects_noncanonical_observation_or_control_modes():
    with pytest.raises(ValueError, match="obs_mode"):
        MikasaEnvConfig(obs_mode="state")
    with pytest.raises(ValueError, match="control_mode"):
        MikasaEnvConfig(control_mode="pd_joint_pos")
    with pytest.raises(ValueError, match="single-env"):
        MikasaEnvConfig(num_envs=2)
