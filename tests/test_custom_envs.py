"""Local variant registration checks that do not construct a GPU simulator."""

import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from dom_vpwem.config import ExperimentConfig
from dom_vpwem.tasks import (
    INTERCEPT_FAST_COVER2_ENV_ID,
    INTERCEPT_FAST_COVER_ENV_ID,
    REMEMBER_COLOR_SEQUENCE3_LONG_ENV_ID,
    SHELL_GAME_SHUFFLE_TOUCH_CUSTOM_ENV_ID,
    get_task_spec,
)


def test_custom_metadata_and_config_need_no_simulator_imports():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import dom_vpwem.custom_envs; import dom_vpwem.mikasa_env; "
            "assert not {'sapien', 'mani_skill', 'gymnasium', 'mikasa_robo_suite'} "
            ".intersection(sys.modules)",
        ],
        check=True,
    )
    for env_id, name in (
        (INTERCEPT_FAST_COVER_ENV_ID, "intercept_fast_cover"),
        (INTERCEPT_FAST_COVER2_ENV_ID, "intercept_fast_cover2"),
        (SHELL_GAME_SHUFFLE_TOUCH_CUSTOM_ENV_ID, "shell_game_shuffle_touch"),
        (REMEMBER_COLOR_SEQUENCE3_LONG_ENV_ID, "remember_color_sequence3_long"),
    ):
        config = ExperimentConfig.from_yaml(f"configs/{name}.yaml")
        task = get_task_spec(config.task.env_id)
        assert config.task.env_id == env_id
        assert config.task.max_episode_steps == task.max_episode_steps
        assert config.task.dataset_dir == task.default_dataset_dir
        assert config.train.output_dir == f"outputs/{name}"
        assert task.public_dataset is (env_id == SHELL_GAME_SHUFFLE_TOUCH_CUSTOM_ENV_ID)


def test_public_shell_alias_matches_upstream_settings():
    pytest.importorskip("mikasa_robo_suite")
    from dom_vpwem.custom_envs.shell_game_shuffle_touch import ShellGameShuffleTouch

    upstream = ShellGameShuffleTouch.__bases__[0]
    for name in ("CUE_PHASE_STEPS", "SHUFFLE_PHASE_STEPS", "NUM_SWAPS", "SWAP_ARC_HEIGHT"):
        assert getattr(ShellGameShuffleTouch, name) == getattr(upstream, name), (
            "Modified shell tasks must stop reusing the public dataset"
        )


def test_custom_registration_preserves_upstream_and_supports_wrappers():
    gym = pytest.importorskip("gymnasium")
    pytest.importorskip("mikasa_robo_suite")

    from mani_skill.utils.registration import REGISTERED_ENVS
    from mikasa_robo_suite.vla.utils.apply_wrappers import VLA_WRAPPER_CONFIGS

    from dom_vpwem.custom_envs import register_custom_envs
    from dom_vpwem.custom_envs.intercept_fast_cover import (
        InterceptCueNoopActionWrapper,
        InterceptFastCover,
    )
    from dom_vpwem.custom_envs.intercept_fast_cover2 import InterceptFastCover2
    from dom_vpwem.custom_envs.remember_color_sequence import (
        CuePhaseNoopActionWrapper,
        RememberColorSequence3Long,
    )
    from dom_vpwem.custom_envs.shell_game_shuffle_touch import ShellGameShuffleTouch
    from dom_vpwem.mikasa_env import _load_runtime

    variants = {
        INTERCEPT_FAST_COVER_ENV_ID: InterceptFastCover,
        INTERCEPT_FAST_COVER2_ENV_ID: InterceptFastCover2,
        SHELL_GAME_SHUFFLE_TOUCH_CUSTOM_ENV_ID: ShellGameShuffleTouch,
        REMEMBER_COLOR_SEQUENCE3_LONG_ENV_ID: RememberColorSequence3Long,
    }
    base_ids = {get_task_spec(env_id).base_env_id for env_id in variants}
    originals = {env_id: REGISTERED_ENVS[env_id] for env_id in base_ids}
    original_wrappers = {env_id: VLA_WRAPPER_CONFIGS[env_id] for env_id in base_ids}
    # The same path used by the evaluator must register local variants.
    _load_runtime()
    first = {env_id: REGISTERED_ENVS[env_id] for env_id in variants}
    register_custom_envs()
    for env_id, cls in variants.items():
        base_id = get_task_spec(env_id).base_env_id
        assert REGISTERED_ENVS[env_id] is first[env_id]
        assert first[env_id].cls is cls
        assert issubclass(cls, originals[base_id].cls)
        assert gym.spec(env_id).max_episode_steps == get_task_spec(env_id).max_episode_steps
        expected_wrapper = original_wrappers[base_id]
        if env_id in (INTERCEPT_FAST_COVER_ENV_ID, INTERCEPT_FAST_COVER2_ENV_ID):
            expected_wrapper = replace(
                expected_wrapper, curriculum_wrapper=InterceptCueNoopActionWrapper
            )
        elif env_id == REMEMBER_COLOR_SEQUENCE3_LONG_ENV_ID:
            expected_wrapper = replace(
                expected_wrapper, curriculum_wrapper=CuePhaseNoopActionWrapper
            )
        assert VLA_WRAPPER_CONFIGS[env_id] == expected_wrapper
        assert VLA_WRAPPER_CONFIGS[env_id] is not original_wrappers[base_id]
        assert first[env_id].asset_download_ids == originals[base_id].asset_download_ids
        assert first[env_id].asset_download_ids is not originals[base_id].asset_download_ids
    for base_id in base_ids:
        assert REGISTERED_ENVS[base_id] is originals[base_id]
        assert VLA_WRAPPER_CONFIGS[base_id] is original_wrappers[base_id]

    # Missing these assets or this wrapper would break scene creation or let
    # the robot interfere with the cue/shuffle before it should act.
    from mikasa_robo_suite.vla.utils.wrappers import CurriculumPhaseNoopActionWrapper

    assert first[SHELL_GAME_SHUFFLE_TOUCH_CUSTOM_ENV_ID].asset_download_ids == ["ycb"]
    assert (
        VLA_WRAPPER_CONFIGS[SHELL_GAME_SHUFFLE_TOUCH_CUSTOM_ENV_ID].curriculum_wrapper
        is CurriculumPhaseNoopActionWrapper
    )


@pytest.mark.parametrize(
    ("env_id", "centers", "half_size"),
    [
        (INTERCEPT_FAST_COVER_ENV_ID, [(-0.20, -0.30, 0.25)], [0.20, 0.20, 0.05]),
        (
            INTERCEPT_FAST_COVER2_ENV_ID,
            [(-0.20, -13 / 30, 0.25), (-0.20, -1 / 6, 0.25)],
            [0.20, 1 / 15, 0.05],
        ),
    ],
)
def test_cover_geometry_and_wrapped_rollout_on_gpu(env_id, centers, half_size):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("A working NVIDIA GPU and Vulkan renderer are required for scene validation")
    sapien = pytest.importorskip("sapien")
    pytest.importorskip("mikasa_robo_suite")

    from dom_vpwem.mikasa_env import MikasaEnvAdapter, MikasaEnvConfig

    with MikasaEnvAdapter(MikasaEnvConfig(env_id=env_id)) as adapter:
        for options in ({}, {"reconfigure": True}):
            obs, _ = adapter.reset(seed=42, options=options)
            env = adapter.env.unwrapped
            if env_id == INTERCEPT_FAST_COVER_ENV_ID:
                covers = (env.cover,)
            else:
                covers = env.covers
                # The unsplit actor must not remain over the opening.
                assert "cover" not in env.scene.actors
            assert len(covers) == len(centers)
            for cover, center in zip(covers, centers):
                np.testing.assert_allclose(cover.pose.p.cpu(), [center])
                for body in cover._bodies:
                    assert isinstance(body, sapien.physx.PhysxRigidStaticComponent)
                    (collision,) = body.collision_shapes
                    assert isinstance(collision, sapien.physx.PhysxCollisionShapeBox)
                    np.testing.assert_allclose(collision.half_size, half_size)
                for entity in cover._objs:
                    render_body = entity.find_component_by_type(sapien.render.RenderBodyComponent)
                    (shape,) = render_body.render_shapes
                    assert isinstance(shape, sapien.render.RenderShapeBox)
                    np.testing.assert_allclose(shape.half_size, half_size)
                    assert shape.material.base_color[3] == 1.0
            for _ in range(3):
                obs, reward, _, _, _ = adapter.step(np.zeros(7, dtype=np.float32))
                assert obs["rgb"].shape == (128, 128, 6)
                assert obs["rgb"].dtype == np.uint8
                assert obs["proprio"].shape == (7,)
                assert np.isfinite(reward)
                for cover, center in zip(covers, centers):
                    np.testing.assert_allclose(cover.pose.p.cpu(), [center])
