"""Repository-owned MIKASA variants; importing this package needs no simulator.

Call ``register_custom_envs()`` before using these tasks through Gymnasium.
The project's ``make_mikasa_env`` does this automatically. Add variants to
``dom_vpwem.tasks.TASK_SPECS`` with an entry point and upstream base env ID.
"""

from __future__ import annotations

from dataclasses import replace
from importlib import import_module

from dom_vpwem.tasks import TASK_SPECS


def register_custom_envs() -> None:
    """Register local classes with base assets and VLA wrappers, idempotently.

    Requires the ``eval`` extra. Registration creates no simulator instances.
    Each variant lives in its own module and subclasses an upstream task.
    A class can override ``CURRICULUM_WRAPPER`` for its own action timing.
    """
    from mani_skill.utils.registration import REGISTERED_ENVS, register_env
    from mikasa_robo_suite.vla.utils.apply_wrappers import VLA_WRAPPER_CONFIGS

    for task in TASK_SPECS.values():
        if task.entry_point is None:
            continue
        module_name, class_name = task.entry_point.split(":")
        env_class = getattr(import_module(module_name), class_name)
        wrapper_config = VLA_WRAPPER_CONFIGS[task.base_env_id]
        existing = REGISTERED_ENVS.get(task.env_id)
        if existing is not None and existing.cls is not env_class:
            raise ValueError(f"Environment {task.env_id!r} is already registered to another class.")
        if existing is None:
            # Shell-game variants need the base task's YCB mug assets. Copy
            # the list so local registration does not mutate upstream metadata.
            base_spec = REGISTERED_ENVS[task.base_env_id]
            register_env(
                task.env_id,
                max_episode_steps=task.max_episode_steps,
                asset_download_ids=list(base_spec.asset_download_ids or []),
            )(env_class)
        VLA_WRAPPER_CONFIGS[task.env_id] = replace(
            wrapper_config,
            curriculum_wrapper=getattr(
                env_class, "CURRICULUM_WRAPPER", wrapper_config.curriculum_wrapper
            ),
        )


__all__ = ["register_custom_envs"]
