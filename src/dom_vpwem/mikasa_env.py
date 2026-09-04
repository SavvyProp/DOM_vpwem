"""Dependency-light adapter for supported MIKASA-Robo-VLA shell-game tasks.

The module deliberately does not import Gymnasium, ManiSkill, Torch, or
MIKASA-Robo at import time.  Those packages are only needed when
``make_mikasa_env`` constructs a real simulator.  Tests and callers with an
already-created environment can therefore use the observation/action contract
without installing the simulator stack.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np

from .simulator_assets import require_mikasa_lamp_asset
from .tasks import (
    DEFAULT_ENV_ID,
    SHELL_GAME_SHUFFLE_COLOR_LAMP_TOUCH_ENV_ID,
    get_task_spec,
)

# Kept as a public legacy identifier; it is not part of the current two-task
# registry and therefore has no metadata fallback when a runtime exposes no
# horizon.
LONG_ENV_ID = "ShellGameShuffleColorLampTouch-Long-VLA-v0"
LANGUAGE_INSTRUCTION = get_task_spec(DEFAULT_ENV_ID).language_instruction

IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
RGB_CHANNELS = 6
PROPRIO_DIM = 7
ACTION_DIM = 7

EnvFactory = Callable[..., Any]
WrapperFactory = Callable[..., Any]


class MikasaDependencyError(ImportError):
    """Raised when a real environment is requested without its dependencies."""


class MikasaContractError(ValueError):
    """Raised when an observation or action violates the VLA contract."""


@dataclass(frozen=True)
class MikasaEnvConfig:
    """Configuration for the canonical single-environment evaluation setup."""

    env_id: str = DEFAULT_ENV_ID
    num_envs: int = 1
    obs_mode: str = "rgb"
    control_mode: str = "pd_ee_delta_pose"
    reward_mode: str = "normalized_dense"
    render_mode: str = "all"
    sim_backend: str | None = "gpu"
    include_overlays: bool = False
    clip_actions: bool = True
    make_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.num_envs != 1:
            raise ValueError(
                "MikasaEnvAdapter intentionally implements the canonical "
                f"single-env protocol; got num_envs={self.num_envs}."
            )
        if self.obs_mode != "rgb":
            raise ValueError(
                "Canonical VLA evaluation requires obs_mode='rgb', "
                f"got {self.obs_mode!r}."
            )
        if self.control_mode != "pd_ee_delta_pose":
            raise ValueError(
                "Canonical VLA evaluation requires control_mode='pd_ee_delta_pose', "
                f"got {self.control_mode!r}."
            )

        protected = {
            "num_envs",
            "obs_mode",
            "control_mode",
            "reward_mode",
            "render_mode",
            "sim_backend",
        }
        overlap = protected.intersection(self.make_kwargs)
        if overlap:
            names = ", ".join(sorted(overlap))
            raise ValueError(
                "Put canonical environment options on MikasaEnvConfig, "
                f"not make_kwargs: {names}."
            )

    def gym_make_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "num_envs": self.num_envs,
            "obs_mode": self.obs_mode,
            "control_mode": self.control_mode,
            "reward_mode": self.reward_mode,
            "render_mode": self.render_mode,
        }
        if self.sim_backend is not None:
            kwargs["sim_backend"] = self.sim_backend
        kwargs.update(self.make_kwargs)
        return kwargs


def _load_runtime() -> tuple[EnvFactory, WrapperFactory]:
    """Import and return the real Gym factory and canonical MIKASA wrapper."""

    try:
        gymnasium = importlib.import_module("gymnasium")
        # Importing this package registers the VLA environment IDs with Gym.
        importlib.import_module("mikasa_robo_suite.vla.memory_envs")
        wrapper_module = importlib.import_module("mikasa_robo_suite.vla.utils.apply_wrappers")
    except (ImportError, ModuleNotFoundError) as exc:
        raise MikasaDependencyError(
            "MIKASA evaluation requires gymnasium, ManiSkill, and "
            "mikasa-robo-suite. Install the current MIKASA-Robo-VLA runtime "
            "before constructing a simulator."
        ) from exc

    apply_wrappers = getattr(wrapper_module, "apply_mikasa_vla_wrappers", None)
    if apply_wrappers is None:
        raise MikasaDependencyError(
            "Installed mikasa-robo-suite does not expose "
            "apply_mikasa_vla_wrappers; use the current VLA release."
        )
    return gymnasium.make, apply_wrappers


def make_mikasa_env(
    config: MikasaEnvConfig | None = None,
    *,
    env_factory: EnvFactory | None = None,
    wrapper_factory: WrapperFactory | None = None,
) -> Any:
    """Create and canonically wrap a MIKASA environment.

    ``env_factory`` and ``wrapper_factory`` are primarily dependency-injection
    hooks for unit tests.  Supply both together or neither.
    """

    config = config or MikasaEnvConfig()
    if (env_factory is None) != (wrapper_factory is None):
        raise ValueError("env_factory and wrapper_factory must be supplied together.")
    using_installed_runtime = env_factory is None
    if using_installed_runtime:
        env_factory, wrapper_factory = _load_runtime()
        if config.env_id == SHELL_GAME_SHUFFLE_COLOR_LAMP_TOUCH_ENV_ID:
            require_mikasa_lamp_asset()

    assert wrapper_factory is not None
    env = env_factory(config.env_id, **config.gym_make_kwargs())
    try:
        return wrapper_factory(env, include_overlays=config.include_overlays)
    except Exception:
        close = getattr(env, "close", None)
        if callable(close):
            close()
        raise


def _to_numpy(value: Any) -> np.ndarray:
    """Convert NumPy-like or Torch-like values without importing Torch."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _first_scalar(value: Any, *, default: Any) -> Any:
    if value is None:
        return default
    array = _to_numpy(value)
    if array.size == 0:
        return default
    return array.reshape(-1)[0].item()


def canonicalize_observation(observation: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Return one unbatched, non-privileged MIKASA VLA observation.

    The real environment emits batch-one arrays.  A fake or replay environment
    may already emit unbatched arrays, which are accepted as well.  Extra keys
    such as ``oracle_info`` or ``task_cue`` are intentionally discarded.
    """

    if not isinstance(observation, Mapping):
        raise MikasaContractError(
            f"Expected a mapping with 'rgb' and 'proprio', got {type(observation).__name__}."
        )
    missing = {"rgb", "proprio"}.difference(observation)
    if missing:
        raise MikasaContractError(f"MIKASA observation is missing keys: {sorted(missing)}.")

    rgb = _to_numpy(observation["rgb"])
    proprio = _to_numpy(observation["proprio"])

    if rgb.shape == (1, IMAGE_HEIGHT, IMAGE_WIDTH, RGB_CHANNELS):
        rgb = rgb[0]
    if proprio.shape == (1, PROPRIO_DIM):
        proprio = proprio[0]

    expected_rgb = (IMAGE_HEIGHT, IMAGE_WIDTH, RGB_CHANNELS)
    if rgb.shape != expected_rgb:
        raise MikasaContractError(
            f"Expected rgb shape {expected_rgb} or (1, *shape), got {rgb.shape}."
        )
    if rgb.dtype != np.uint8:
        raise MikasaContractError(f"Expected uint8 RGB observations, got dtype {rgb.dtype}.")
    if proprio.shape != (PROPRIO_DIM,):
        raise MikasaContractError(
            f"Expected proprio shape ({PROPRIO_DIM},) or (1, {PROPRIO_DIM}), got {proprio.shape}."
        )
    if not np.issubdtype(proprio.dtype, np.number):
        raise MikasaContractError(f"Expected numeric proprioception, got dtype {proprio.dtype}.")
    if not np.isfinite(proprio).all():
        raise MikasaContractError("Proprioception contains NaN or infinity.")

    return {
        "rgb": np.ascontiguousarray(rgb),
        "proprio": np.ascontiguousarray(proprio, dtype=np.float32),
    }


def split_rgb(rgb: Any) -> tuple[np.ndarray, np.ndarray]:
    """Split canonical HWC6 RGB into base/top and wrist HWC3 images."""

    array = _to_numpy(rgb)
    if array.shape == (1, IMAGE_HEIGHT, IMAGE_WIDTH, RGB_CHANNELS):
        array = array[0]
    expected = (IMAGE_HEIGHT, IMAGE_WIDTH, RGB_CHANNELS)
    if array.shape != expected:
        raise MikasaContractError(f"Expected RGB shape {expected}, got {array.shape}.")
    return array[..., :3], array[..., 3:]


def _prepare_action(action: Any, *, clip: bool, target_device: Any = None) -> Any:
    """Validate one action and add MIKASA's batch dimension."""

    action_array = _to_numpy(action)
    if action_array.shape == (1, ACTION_DIM):
        single = action_array[0]
    elif action_array.shape == (ACTION_DIM,):
        single = action_array
    else:
        raise MikasaContractError(
            f"Expected one action with shape ({ACTION_DIM},) or "
            f"(1, {ACTION_DIM}), got {action_array.shape}."
        )
    if not np.issubdtype(single.dtype, np.number):
        raise MikasaContractError(f"Expected a numeric action, got dtype {single.dtype}.")
    if not np.isfinite(single).all():
        raise MikasaContractError("Action contains NaN or infinity.")
    if not clip and np.any(np.abs(single) > 1.0):
        raise MikasaContractError("Action is outside the normalized [-1, 1] controller range.")

    # Preserve Torch-like tensors so a GPU ManiSkill environment does not incur
    # an unnecessary CPU round trip.  Validation above uses a detached copy.
    is_tensor_like = all(hasattr(action, attr) for attr in ("detach", "shape", "unsqueeze"))
    if is_tensor_like:
        prepared = action
        if tuple(prepared.shape) == (ACTION_DIM,):
            prepared = prepared.unsqueeze(0)
        if hasattr(prepared, "float"):
            prepared = prepared.float()
        if clip and hasattr(prepared, "clamp"):
            prepared = prepared.clamp(-1.0, 1.0)
        if target_device is not None and hasattr(prepared, "to"):
            prepared = prepared.to(device=target_device)
        return prepared

    prepared_np = np.asarray(single, dtype=np.float32)
    if clip:
        prepared_np = np.clip(prepared_np, -1.0, 1.0)
    prepared_np = np.ascontiguousarray(prepared_np[None, :])
    if target_device is not None:
        # ManiSkill's GPU backend consumes Torch tensors.  This import remains
        # on the real-runtime path; NumPy-only tests and dataset tooling do not
        # need Torch merely to import the adapter.
        try:
            torch = importlib.import_module("torch")
        except (ImportError, ModuleNotFoundError) as exc:
            raise MikasaDependencyError(
                "A Torch-backed MIKASA environment requires torch actions."
            ) from exc
        return torch.as_tensor(prepared_np, dtype=torch.float32, device=target_device)
    return prepared_np


class MikasaEnvAdapter:
    """Canonical, single-env facade around current MIKASA-Robo-VLA."""

    def __init__(
        self,
        config: MikasaEnvConfig | None = None,
        *,
        env: Any | None = None,
        env_factory: EnvFactory | None = None,
        wrapper_factory: WrapperFactory | None = None,
    ) -> None:
        self.config = config or MikasaEnvConfig()
        self._owns_env = env is None
        self.env = env if env is not None else make_mikasa_env(
            self.config,
            env_factory=env_factory,
            wrapper_factory=wrapper_factory,
        )
        self._closed = False

    @property
    def max_episode_steps(self) -> int:
        value = getattr(self.env, "max_episode_steps", None)
        if value is None:
            spec = getattr(self.env, "spec", None)
            value = getattr(spec, "max_episode_steps", None)
        if value is None:
            try:
                value = get_task_spec(self.config.env_id).max_episode_steps
            except ValueError:
                value = None
        if value is None or int(value) <= 0:
            raise RuntimeError("The environment does not expose a positive episode horizon.")
        return int(value)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], Mapping[str, Any]]:
        kwargs: dict[str, Any] = {}
        if seed is not None:
            kwargs["seed"] = seed
        if options is not None:
            kwargs["options"] = dict(options)
        observation, info = self.env.reset(**kwargs)
        if not isinstance(info, Mapping):
            raise MikasaContractError(
                f"Expected reset info to be a mapping, got {type(info).__name__}."
            )
        return canonicalize_observation(observation), info

    def step(
        self,
        action: Any,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, Mapping[str, Any]]:
        unwrapped = getattr(self.env, "unwrapped", self.env)
        target_device = getattr(unwrapped, "device", None)
        prepared = _prepare_action(
            action,
            clip=self.config.clip_actions,
            target_device=target_device,
        )
        observation, reward, terminated, truncated, info = self.env.step(prepared)
        if not isinstance(info, Mapping):
            raise MikasaContractError(
                f"Expected step info to be a mapping, got {type(info).__name__}."
            )
        return (
            canonicalize_observation(observation),
            float(_first_scalar(reward, default=0.0)),
            bool(_first_scalar(terminated, default=False)),
            bool(_first_scalar(truncated, default=False)),
            info,
        )

    def close(self) -> None:
        if not self._closed:
            close = getattr(self.env, "close", None)
            if callable(close):
                close()
            self._closed = True

    def __enter__(self) -> "MikasaEnvAdapter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


__all__ = [
    "ACTION_DIM",
    "DEFAULT_ENV_ID",
    "LANGUAGE_INSTRUCTION",
    "LONG_ENV_ID",
    "MikasaContractError",
    "MikasaDependencyError",
    "MikasaEnvAdapter",
    "MikasaEnvConfig",
    "canonicalize_observation",
    "make_mikasa_env",
    "split_rgb",
]
