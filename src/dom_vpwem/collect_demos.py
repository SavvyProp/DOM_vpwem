"""Collect successful RGB/proprio/action episodes from a state PPO oracle."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .data import discover_npz_episodes, load_npz_episode
from .demo_env import DemoEnv
from .oracle_env import make_oracle_agent
from .tasks import (
    DEFAULT_DATASET_ROOT,
    INTERCEPT_FAST_COVER2_ENV_ID,
    INTERCEPT_FAST_COVER_ENV_ID,
    SHELL_GAME_SHUFFLE_TOUCH_CUSTOM_ENV_ID,
    get_task_spec,
)

MANIFEST = "collection.json"


@dataclass
class CollectConfig:
    env_id: str
    checkpoint: str | None = None
    data_root: str = DEFAULT_DATASET_ROOT
    episodes: int = 250
    num_envs: int = 4
    seed: int = 100_000
    max_attempts: int | None = None
    device: str = "cuda"
    sim_backend: str = "gpu"

    def validate(self):
        get_task_spec(self.env_id)
        for name in ("episodes", "num_envs"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if self.max_attempts is None:
            self.max_attempts = max(self.episodes * 10, self.num_envs)
        if type(self.max_attempts) is not int or self.max_attempts < self.num_envs:
            raise ValueError("max_attempts must allow at least one full batch")
        if self.seed + self.max_attempts >= 2**32:
            raise ValueError("seed + max_attempts must be below 2**32")
        if self.sim_backend not in ("gpu", "cpu"):
            raise ValueError("sim_backend must be gpu or cpu")
        if self.sim_backend == "cpu" and self.num_envs != 1:
            raise ValueError("CPU simulation requires num_envs=1")

    @property
    def destination(self):
        return Path(self.data_root).expanduser() / get_task_spec(self.env_id).dataset_slug


def _numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _save_json(path: Path, value):
    temporary = path.with_suffix(".json.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _checkpoint_digest(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def existing_episodes(config: CollectConfig) -> list[Path]:
    """Validate completed NPZs, including datasets placed here by the user."""
    if not config.destination.exists():
        return []
    task = get_task_spec(config.env_id)
    allowed_ids = {task.env_id}
    if task.env_id == SHELL_GAME_SHUFFLE_TOUCH_CUSTOM_ENV_ID:
        allowed_ids.add(task.base_env_id)
    manifest_path = config.destination / MANIFEST
    if manifest_path.exists() and json.loads(manifest_path.read_text())["env_id"] != task.env_id:
        raise ValueError(f"{manifest_path} belongs to another environment")
    files = discover_npz_episodes(config.destination)
    for path in files:
        episode = load_npz_episode(path)
        if episode.rgb.shape[1:] != (128, 128, 6) or episode.length > task.max_episode_steps:
            raise ValueError(f"{path}: unexpected image dimensions or episode horizon")
        if np.any(np.abs(episode.action) > 1.00001):
            raise ValueError(f"{path}: actions must be normalized to [-1, 1]")
        with np.load(path, allow_pickle=False) as data:
            if "env_id" in data and str(data["env_id"].item()) not in allowed_ids:
                raise ValueError(f"{path}: environment ID does not match {task.env_id}")
            if "success_once" in data and not bool(data["success_once"].item()):
                raise ValueError(f"{path}: unsuccessful episode")
            if "success" in data and not np.any(data["success"][: episode.length]):
                raise ValueError(f"{path}: no successful step in the valid episode")
    return files


def load_expert(env, config: CollectConfig):
    checkpoint = Path(config.checkpoint).expanduser()
    sidecar = checkpoint.with_suffix(".json")
    if sidecar.exists():
        metadata = json.loads(sidecar.read_text())
        if metadata.get("schema") != env.schema:
            raise ValueError(f"{sidecar}: oracle observation/action schema does not match")
        allowed_ids = {config.env_id}
        if config.env_id in (INTERCEPT_FAST_COVER_ENV_ID, INTERCEPT_FAST_COVER2_ENV_ID):
            allowed_ids.update(
                ("InterceptFast-VLA-v0", INTERCEPT_FAST_COVER_ENV_ID, INTERCEPT_FAST_COVER2_ENV_ID)
            )
        elif config.env_id == SHELL_GAME_SHUFFLE_TOUCH_CUSTOM_ENV_ID:
            allowed_ids.add("ShellGameShuffleTouch-VLA-v0")
        if metadata.get("env_id") not in allowed_ids:
            raise ValueError(f"{sidecar}: checkpoint is for an incompatible task")
    else:
        print(
            "Checkpoint has no schema sidecar; validating network dimensions and rollout success."
        )
    agent = make_oracle_agent(env, torch.device(config.device))
    weights = torch.load(checkpoint, map_location=config.device, weights_only=True)
    agent.load_state_dict(weights, strict=True)
    agent.eval()
    return agent


def _write_episode(path: Path, frames: list[dict], *, env_id: str, seed: int, digest: str):
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    arrays = {key: np.stack([frame[key] for frame in frames]) for key in frames[0]}
    arrays["done"] = np.zeros(len(frames), dtype=bool)
    arrays["done"][-1] = True
    temporary = path.with_suffix(".npz.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                **arrays,
                episode_length=np.asarray(len(frames), dtype=np.int64),
                episode_seed=np.asarray(seed, dtype=np.int64),
                success_once=np.asarray(True),
                env_id=np.asarray(env_id),
                oracle_sha256=np.asarray(digest),
            )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def collect(
    config: CollectConfig,
    *,
    env_factory: Callable = DemoEnv,
    expert_factory: Callable = load_expert,
) -> Path:
    config.validate()
    config.destination.mkdir(parents=True, exist_ok=True)
    with (config.destination / ".collection.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"Another collector is writing {config.destination}") from exc
        return _collect_locked(config, env_factory, expert_factory)


def _collect_locked(config, env_factory, expert_factory):
    files = existing_episodes(config)
    if len(files) >= config.episodes:
        print(f"Reusing {len(files)} validated episodes in {config.destination}", flush=True)
        return config.destination
    if config.checkpoint is None:
        raise ValueError("A checkpoint is required to collect missing episodes")
    checkpoint = Path(config.checkpoint).expanduser().resolve()
    digest = _checkpoint_digest(checkpoint)
    task = get_task_spec(config.env_id)
    identity = {
        "format_version": 1,
        "env_id": config.env_id,
        "oracle_sha256": digest,
        "seed": config.seed,
        "num_envs": config.num_envs,
        "max_episode_steps": task.max_episode_steps,
        "sim_backend": config.sim_backend,
    }
    manifest_path = config.destination / MANIFEST
    if manifest_path.exists():
        progress = json.loads(manifest_path.read_text())
        for key, expected in identity.items():
            if progress.get(key) != expected:
                raise ValueError(f"Cannot resume {manifest_path}: {key} differs")
        for path in files:
            with np.load(path, allow_pickle=False) as data:
                if "oracle_sha256" not in data or str(data["oracle_sha256"].item()) != digest:
                    raise ValueError(
                        f"Cannot resume: {path} has different or missing oracle provenance"
                    )
    else:
        if files:
            raise ValueError(
                "This dataset is incomplete and has no collection manifest. "
                "Use a separate data root to collect new episodes."
            )
        progress = {**identity, "attempted_episodes": 0, "checkpoint": str(checkpoint)}
    saved = len(files)
    next_id = max((int(path.stem.rsplit("_", 1)[1]) for path in files), default=-1) + 1
    with closing(
        env_factory(config.env_id, num_envs=config.num_envs, sim_backend=config.sim_backend)
    ) as env:
        agent = expert_factory(env, config)
        if "schema" in progress and progress["schema"] != env.schema:
            raise ValueError("Cannot resume: simulator observation/action schema changed")
        progress["schema"] = env.schema
        progress["requested_episodes"] = config.episodes
        progress["saved_episodes"] = saved
        _save_json(manifest_path, progress)
        while saved < config.episodes:
            attempted = progress["attempted_episodes"]
            if attempted + config.num_envs > config.max_attempts:
                raise RuntimeError(
                    f"Collected {saved}/{config.episodes} successful episodes "
                    f"after {attempted} attempts. Validate the oracle or increase "
                    "--max-attempts; completed episodes are retained."
                )
            seeds = list(range(config.seed + attempted, config.seed + attempted + config.num_envs))
            # Reserve seeds before the rollout, so an interrupted batch is not
            # replayed and saved twice when the command resumes.
            progress["attempted_episodes"] += config.num_envs
            _save_json(manifest_path, progress)
            obs, _ = env.reset(seed=seeds)
            buffers = [[] for _ in seeds]
            active = np.ones(config.num_envs, dtype=bool)
            for step in range(task.max_episode_steps):
                rgb = _numpy(obs["rgb"])
                proprio = _numpy(obs["proprio"]).astype(np.float32, copy=False)
                if rgb.shape != (config.num_envs, 128, 128, 6) or rgb.dtype != np.uint8:
                    raise ValueError(f"Unexpected RGB batch: {rgb.shape}, {rgb.dtype}")
                if proprio.shape != (config.num_envs, 7) or not np.isfinite(proprio).all():
                    raise ValueError("Expected finite batched 7-D proprioception")
                # Copy BEFORE stepping: observations may refer to reusable
                # simulator buffers. Labels must describe this observation.
                for row in np.flatnonzero(active):
                    buffers[row].append({"rgb": rgb[row].copy(), "proprio": proprio[row].copy()})
                state = {
                    key: value.to(config.device) for key, value in env.oracle_observation().items()
                }
                if any(not torch.isfinite(value).all() for value in state.values()):
                    raise ValueError("Simulator emitted nonfinite oracle state")
                with torch.no_grad():
                    action = agent.get_action(state, deterministic=True).clamp(-1, 1)
                if action.shape != (config.num_envs, 7) or not torch.isfinite(action).all():
                    raise ValueError("Oracle emitted invalid actions")
                action[torch.as_tensor(~active, device=action.device)] = 0
                obs, reward, terminated, truncated, info = env.step(action.to(env.device))
                executed = _numpy(env.executed_action).astype(np.float32, copy=False)
                reward = _numpy(reward).reshape(config.num_envs).astype(np.float32, copy=False)
                success = _numpy(info["success"]).reshape(config.num_envs).astype(bool)
                ended = (
                    _numpy(terminated).reshape(config.num_envs).astype(bool)
                    | _numpy(truncated).reshape(config.num_envs).astype(bool)
                    | success
                    | (step + 1 == task.max_episode_steps)
                )
                if executed.shape != (config.num_envs, 7) or not np.isfinite(executed).all():
                    raise ValueError("Invalid executed action recorded by simulator wrapper")
                if np.any(np.abs(executed) > 1.00001) or not np.isfinite(reward).all():
                    raise ValueError("Nonfinite reward or unnormalized executed action")
                for row in np.flatnonzero(active):
                    buffers[row][-1].update(
                        action=executed[row].copy(), reward=reward[row], success=success[row]
                    )
                    if ended[row]:
                        if success[row] and saved < config.episodes:
                            _write_episode(
                                config.destination / f"train_data_{next_id:06d}.npz",
                                buffers[row],
                                env_id=config.env_id,
                                seed=seeds[row],
                                digest=digest,
                            )
                            next_id += 1
                            saved += 1
                            progress["saved_episodes"] = saved
                            _save_json(manifest_path, progress)
                            print(
                                f"{config.env_id}: {saved}/{config.episodes} saved "
                                f"(seed {seeds[row]})",
                                flush=True,
                            )
                        buffers[row].clear()
                        active[row] = False
                if not active.any() or saved >= config.episodes:
                    break
        return config.destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--data-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--episodes", type=int, default=250)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=100_000)
    parser.add_argument("--max-attempts", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sim-backend", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument(
        "--status",
        action="store_true",
        help="Exit 0 if enough valid episodes exist, 1 if incomplete",
    )
    args = vars(parser.parse_args(argv))
    status = args.pop("status")
    config = CollectConfig(**args)
    try:
        config.validate()
        if status:
            count = len(existing_episodes(config))
            print(f"{config.env_id}: {count}/{config.episodes} existing episodes")
            return 0 if count >= config.episodes else 1
        collect(config)
        return 0
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        parser.exit(2, f"Collection failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
