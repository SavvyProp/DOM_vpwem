"""Read expert state and student images from the same MIKASA simulator."""

from __future__ import annotations

from typing import Any

from .mikasa_env import _load_runtime
from .oracle_env import oracle_observation_schema
from .tasks import get_task_spec


def privileged_observation(base_env: Any, info: dict) -> dict:
    """Read the exact state-mode input without advancing or rendering the scene.

    MIKASA gates privileged fields on obs_mode, including in _get_obs_extra.
    Restore RGB mode even on failure. Passing the existing info also avoids
    rerunning evaluate(), which updates scene actors in several memory tasks.
    """
    original_mode = base_env._obs_mode
    try:
        base_env._obs_mode = "state"
        observation = {"state": base_env.get_obs(info)}
        # Match StateOnlyTensorToDictWrapper + FlattenRGBDObservationWrapper:
        # remove the additional oracle label, retain a real task cue if present.
        cue = base_env.task_cue
        if cue is not None:
            observation["task_cue"] = cue[:, None] if cue.ndim == 1 else cue
        return observation
    finally:
        base_env._obs_mode = original_mode


class DemoEnv:
    """Canonical RGB wrappers, privileged state access, and executed actions.

    Full batches reset together. There is no automatic reset, so terminal
    frames cannot be silently replaced with a new episode's observations.
    """

    def __init__(self, env_id: str, *, num_envs: int, sim_backend: str = "gpu"):
        import gymnasium as gym
        import numpy as np
        import torch

        task = get_task_spec(env_id)
        gym_make, apply_wrappers = _load_runtime()

        class CaptureAction(gym.Wrapper):
            # This wrapper sits inside the curriculum action wrapper.
            executed_action = None

            def step(self, action):
                self.executed_action = torch.as_tensor(action).detach().clone()
                return self.env.step(action)

        raw = gym_make(
            env_id,
            num_envs=num_envs,
            obs_mode="rgb",
            control_mode="pd_ee_delta_pose",
            reward_mode="normalized_dense",
            sim_backend=sim_backend,
            render_backend="cpu" if sim_backend == "cpu" else "gpu",
            reconfiguration_freq=0,
            max_episode_steps=task.max_episode_steps,
        )
        try:
            self.capture = CaptureAction(raw)
            self.env = apply_wrappers(self.capture, include_overlays=False)
            self.device = raw.unwrapped.device
            self.num_envs = num_envs
            self.single_action_space = raw.unwrapped.single_action_space
            self.reset(seed=list(range(num_envs)))
            oracle_obs = self.oracle_observation()
            for state in oracle_obs.values():
                if state.ndim != 2 or state.shape[0] != num_envs:
                    raise ValueError(f"Expected batched flat oracle state, got {state.shape}")
            self.single_observation_space = gym.spaces.Dict(
                {
                    key: gym.spaces.Box(
                        -np.inf, np.inf, shape=tuple(state.shape[1:]), dtype=np.float32
                    )
                    for key, state in oracle_obs.items()
                }
            )
            self.schema = oracle_observation_schema(self)
        except Exception:
            raw.close()
            raise

    def reset(self, *, seed):
        obs, self.info = self.env.reset(seed=seed)
        return obs, self.info

    def oracle_observation(self):
        return privileged_observation(self.env.unwrapped, self.info)

    def step(self, action):
        obs, reward, terminated, truncated, self.info = self.env.step(action)
        return obs, reward, terminated, truncated, self.info

    @property
    def executed_action(self):
        return self.capture.executed_action

    def close(self):
        self.env.close()
