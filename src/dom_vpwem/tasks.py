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
DEFAULT_ENV_ID = SHELL_GAME_SHUFFLE_COLOR_LAMP_TOUCH_ENV_ID


@dataclass(frozen=True)
class TaskSpec:
    """Static benchmark and dataset metadata for one supported task."""

    env_id: str
    dataset_slug: str
    max_episode_steps: int
    split: str
    memory_type: str
    language_instruction: str

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
    "SHELL_GAME_SHUFFLE_COLOR_LAMP_TOUCH_ENV_ID",
    "SHELL_GAME_TOUCH_ENV_ID",
    "TASK_REGISTRY",
    "TASK_SPECS",
    "TaskSpec",
    "get_task_spec",
]
