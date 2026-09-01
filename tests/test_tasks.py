from types import SimpleNamespace

import pytest

from dom_vpwem.mikasa_env import MikasaEnvAdapter, MikasaEnvConfig
from dom_vpwem.tasks import TASK_REGISTRY, get_task_spec


def test_task_registry_contains_benchmark_metadata() -> None:
    touch = get_task_spec("ShellGameTouch-VLA-v0")
    tracking = get_task_spec("ShellGameShuffleColorLampTouch-VLA-v0")

    assert touch.dataset_slug == "shell_game_touch_vla_v0"
    assert touch.default_dataset_dir == (
        "data_mikasa_robo/data_npz/shell_game_touch_vla_v0"
    )
    assert (touch.max_episode_steps, touch.split, touch.memory_type) == (
        30,
        "Short",
        "Spatial",
    )
    assert touch.language_instruction == (
        "Observe which cup hides the ball, wait, then touch that cup."
    )

    assert tracking.dataset_slug == "shell_game_shuffle_color_lamp_touch_vla_v0"
    assert (tracking.max_episode_steps, tracking.split, tracking.memory_type) == (
        60,
        "Short",
        "Tracking",
    )


def test_task_registry_is_read_only_and_rejects_unknown_tasks() -> None:
    with pytest.raises(TypeError):
        TASK_REGISTRY["Unknown-VLA-v0"] = get_task_spec("ShellGameTouch-VLA-v0")

    with pytest.raises(ValueError, match="Unsupported MIKASA task"):
        get_task_spec("Unknown-VLA-v0")


def test_env_adapter_uses_registered_touch_horizon_as_fallback() -> None:
    adapter = MikasaEnvAdapter(
        config=MikasaEnvConfig(env_id="ShellGameTouch-VLA-v0"),
        env=SimpleNamespace(),
    )

    assert adapter.max_episode_steps == 30
