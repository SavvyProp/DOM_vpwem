"""Authoritative metadata for the MIKASA tasks supported by this package."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

DEFAULT_DATASET_ROOT = "data_mikasa_robo/data_npz"

SHELL_GAME_SHUFFLE_COLOR_LAMP_TOUCH_ENV_ID = (
    "ShellGameShuffleColorLampTouch-VLA-v0"
)
SHELL_GAME_TOUCH_ENV_ID = "ShellGameTouch-VLA-v0"
SHELL_GAME_SHUFFLE_TOUCH_CUSTOM_ENV_ID = "ShellGameShuffleTouchCustom-VLA-v0"
INTERCEPT_FAST_COVER_ENV_ID = "InterceptFastCover-VLA-v0"
INTERCEPT_FAST_COVER2_ENV_ID = "InterceptFastCover2-VLA-v0"
REMEMBER_COLOR_SEQUENCE3_LONG_ENV_ID = "RememberColorSequence3-Long-VLA-v0"
DEFAULT_ENV_ID = SHELL_GAME_SHUFFLE_COLOR_LAMP_TOUCH_ENV_ID


@dataclass(frozen=True)
class TaskSpec:
    """Task metadata, including lazy registration for repository-owned variants."""

    env_id: str
    dataset_slug: str
    max_episode_steps: int
    split: str
    memory_type: str
    language_instruction: str
    public_dataset: bool = True
    # An unchanged local alias can reuse an upstream (env ID, dataset folder).
    # Disable public_dataset when changing its behavior or observations.
    public_dataset_source: tuple[str, str] | None = None
    # Local variants reuse the base task's VLA wrapper configuration. Simulator
    # classes are imported only when register_custom_envs() is called.
    base_env_id: str | None = None
    entry_point: str | None = None

    @property
    def default_dataset_dir(self) -> str:
        """Default location produced by the current MIKASA collectors."""

        return f"{DEFAULT_DATASET_ROOT}/{self.dataset_slug}"

    @property
    def dataset_dir(self) -> str:
        """Backward-friendly alias for :attr:`default_dataset_dir`."""

        return self.default_dataset_dir

    @property
    def default_dataset_path(self) -> str:
        """Path-named alias for callers that distinguish slugs from paths."""

        return self.default_dataset_dir

    @property
    def max_horizon(self) -> int:
        """Benchmark-named alias for :attr:`max_episode_steps`."""

        return self.max_episode_steps


_TASK_SPECS = {
    SHELL_GAME_SHUFFLE_COLOR_LAMP_TOUCH_ENV_ID: TaskSpec(
        env_id=SHELL_GAME_SHUFFLE_COLOR_LAMP_TOUCH_ENV_ID,
        dataset_slug="shell_game_shuffle_color_lamp_touch_vla_v0",
        max_episode_steps=60,
        split="Short",
        memory_type="Tracking",
        language_instruction=(
            "Observe which color is under each cup, track the cups as they shuffle, "
            "then touch the cup matching the lamp color."
        ),
    ),
    SHELL_GAME_TOUCH_ENV_ID: TaskSpec(
        env_id=SHELL_GAME_TOUCH_ENV_ID,
        dataset_slug="shell_game_touch_vla_v0",
        max_episode_steps=30,
        split="Short",
        memory_type="Spatial",
        language_instruction=(
            "Observe which cup hides the ball, wait, then touch that cup."
        ),
    ),
    SHELL_GAME_SHUFFLE_TOUCH_CUSTOM_ENV_ID: TaskSpec(
        env_id=SHELL_GAME_SHUFFLE_TOUCH_CUSTOM_ENV_ID,
        dataset_slug="shell_game_shuffle_touch_custom_vla_v0",
        max_episode_steps=60,
        split="Short",
        memory_type="Tracking",
        language_instruction=(
            "Observe which cup hides the ball, track the cups as they shuffle, "
            "then touch the correct cup."
        ),
        public_dataset=True,
        public_dataset_source=("ShellGameShuffleTouch-VLA-v0", "shell_game_shuffle_touch_vla_v0"),
        base_env_id="ShellGameShuffleTouch-VLA-v0",
        entry_point="dom_vpwem.custom_envs.shell_game_shuffle_touch:ShellGameShuffleTouch",
    ),
    INTERCEPT_FAST_COVER_ENV_ID: TaskSpec(
        env_id=INTERCEPT_FAST_COVER_ENV_ID,
        # Solid covers and five-step cue; keep earlier demonstrations separate.
        dataset_slug="intercept_fast_cover_vla_v0_collision_cue5_v1",
        max_episode_steps=60,
        split="Short",
        memory_type="Spatial",
        language_instruction=(
            "Intercept the rolling ball by moving to its path and deflecting it toward the target."
        ),
        public_dataset=False,
        base_env_id="InterceptFast-VLA-v0",
        entry_point="dom_vpwem.custom_envs.intercept_fast_cover:InterceptFastCover",
    ),
    REMEMBER_COLOR_SEQUENCE3_LONG_ENV_ID: TaskSpec(
        env_id=REMEMBER_COLOR_SEQUENCE3_LONG_ENV_ID,
        dataset_slug="remember_color_sequence3_long_vla_v0",
        max_episode_steps=600,
        split="Long",
        memory_type="TemporalOrder",
        language_instruction=(
            "Observe the sequence of colors, wait, then touch the cube matching "
            "the second-to-last color."
        ),
        public_dataset=False,
        base_env_id="RememberColor3-Long-VLA-v0",
        entry_point=(
            "dom_vpwem.custom_envs.remember_color_sequence:RememberColorSequence3Long"
        ),
    ),
    INTERCEPT_FAST_COVER2_ENV_ID: TaskSpec(
        env_id=INTERCEPT_FAST_COVER2_ENV_ID,
        dataset_slug="intercept_fast_cover2_vla_v0_collision_cue5_v1",
        max_episode_steps=60,
        split="Short",
        memory_type="Spatial",
        language_instruction=(
            "Intercept the rolling ball by moving to its path and deflecting it toward the target."
        ),
        public_dataset=False,
        base_env_id="InterceptFast-VLA-v0",
        entry_point="dom_vpwem.custom_envs.intercept_fast_cover2:InterceptFastCover2",
    ),
}

TASK_SPECS: Mapping[str, TaskSpec] = MappingProxyType(_TASK_SPECS)
TASK_REGISTRY = TASK_SPECS


def get_task_spec(env_id: str) -> TaskSpec:
    """Return metadata for ``env_id``, failing clearly for unsupported tasks."""

    try:
        return TASK_SPECS[env_id]
    except KeyError:
        supported = ", ".join(sorted(TASK_SPECS))
        raise ValueError(
            f"Unsupported MIKASA task {env_id!r}. Registered tasks: {supported}."
        ) from None


__all__ = [
    "DEFAULT_DATASET_ROOT",
    "DEFAULT_ENV_ID",
    "INTERCEPT_FAST_COVER_ENV_ID",
    "INTERCEPT_FAST_COVER2_ENV_ID",
    "REMEMBER_COLOR_SEQUENCE3_LONG_ENV_ID",
    "SHELL_GAME_SHUFFLE_COLOR_LAMP_TOUCH_ENV_ID",
    "SHELL_GAME_SHUFFLE_TOUCH_CUSTOM_ENV_ID",
    "SHELL_GAME_TOUCH_ENV_ID",
    "TASK_REGISTRY",
    "TASK_SPECS",
    "TaskSpec",
    "get_task_spec",
]
