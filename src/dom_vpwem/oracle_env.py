"""State observations and MLP oracles compatible with MIKASA's PPO collector.

Simulator imports are lazy. The demonstration collector matches this factory's
privileged observation contract, independently of the RGB-only student adapter.
"""

from __future__ import annotations

from typing import Any

from .mikasa_env import _load_runtime
from .simulator_assets import require_mikasa_lamp_asset
from .tasks import SHELL_GAME_SHUFFLE_COLOR_LAMP_TOUCH_ENV_ID, get_task_spec


def make_oracle_env(
    env_id: str,
    *,
    num_envs: int,
    sim_backend: str = "gpu",
    auto_reset: bool = True,
    reconfiguration_freq: int | None = None,
) -> Any:
    """Build a vector state environment with the collector's action semantics.

    Episodes run to the registered horizon, as in MIKASA oracle training.
    ``auto_reset=False`` lets validation explicitly choose every episode seed.
    State already contains privileged information, so the additional
    ``oracle_info`` field is disabled, matching ``AgentStateOnly`` collection.
    """
    spec = get_task_spec(env_id)
    gym_make, _ = _load_runtime()  # Registers upstream tasks and local variants.
    from baselines.ppo.ppo_memtasks import FlattenRGBDObservationWrapper
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
    from mikasa_robo_suite.vla.utils.apply_wrappers import VLA_WRAPPER_CONFIGS
    from mikasa_robo_suite.vla.utils.wrappers import StateOnlyTensorToDictWrapper

    if SHELL_GAME_SHUFFLE_COLOR_LAMP_TOUCH_ENV_ID in (env_id, spec.base_env_id):
        require_mikasa_lamp_asset()
    env = gym_make(
        env_id,
        num_envs=num_envs,
        obs_mode="state",
        control_mode="pd_ee_delta_pose",
        reward_mode="normalized_dense",
        sim_backend=sim_backend,
        render_backend="cpu" if sim_backend == "cpu" else "gpu",
        reconfiguration_freq=reconfiguration_freq,
    )
    try:
        env = StateOnlyTensorToDictWrapper(env)
        curriculum = VLA_WRAPPER_CONFIGS[env_id].curriculum_wrapper
        if curriculum is not None:
            env = curriculum(env)
        env = FlattenRGBDObservationWrapper(
            env,
            rgb=False,
            depth=False,
            state=True,
            oracle=False,
            joints=False,
        )
        return ManiSkillVectorEnv(
            env,
            auto_reset=auto_reset,
            ignore_terminations=True,
            record_metrics=True,
        )
    except Exception:
        env.close()
        raise


def make_oracle_agent(env: Any, device: Any) -> Any:
    """Use the exact upstream actor/critic architecture and checkpoint keys."""
    from baselines.ppo.ppo_memtasks import AgentStateOnly

    return AgentStateOnly(env).to(device)


def oracle_observation_schema(env: Any) -> dict[str, Any]:
    """Record ordered input keys; input order matters when reusing checkpoints."""
    return {
        "observations": [
            {"key": key, "shape": list(space.shape)}
            for key, space in env.single_observation_space.items()
        ],
        "action_shape": list(env.single_action_space.shape),
        "obs_mode": "state",
        "control_mode": "pd_ee_delta_pose",
        "reward_mode": "normalized_dense",
        "include_oracle": False,
        "include_rgb": False,
    }
