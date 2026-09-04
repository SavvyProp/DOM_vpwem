"""Evaluate a chunk-producing memory policy on the current MIKASA shell game.

The programmatic API only requires a policy with these methods::

    policy.reset(batch_size=1, device=None, seed=episode_seed)
    policy.act({"rgb": uint8[128, 128, 6], "proprio": float32[7]})
        -> action[7] or action_chunk[K, 7]

Actions in a returned chunk are executed FIFO/open-loop, matching the official
MIKASA-Robo-VLA evaluation protocol.  A fresh queue and policy state are used
for every episode. Legacy ``reset(batch_size, device)`` and no-argument
``reset()`` methods are also accepted.
"""

from __future__ import annotations

import argparse
import json
import operator
import sys
from collections import deque
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Deque, Mapping, Protocol, Sequence

import numpy as np

from .mikasa_env import (
    ACTION_DIM,
    DEFAULT_ENV_ID,
    MikasaEnvAdapter,
    MikasaEnvConfig,
    _first_scalar,
)
from .rollout_video import (
    DEFAULT_VIDEO_FPS,
    Mp4RolloutRecorder,
    RolloutFrameSink,
)
from .tasks import get_task_spec

START_SEED = 4_242_424_242
NUM_EPISODES = 50


class MemoryPolicy(Protocol):
    """Structural policy interface consumed by :func:`evaluate_policy`."""

    def reset(
        self,
        batch_size: int = 1,
        device: str | None = None,
        seed: int | None = None,
    ) -> None: ...

    def act(self, observation: Mapping[str, np.ndarray]) -> Any: ...


@dataclass(frozen=True)
class EpisodeResult:
    seed: int
    success_once: bool
    episode_return: float
    n_steps: int
    action_chunk_size: int = 1


@dataclass(frozen=True)
class EvaluationResult:
    env_id: str
    start_seed: int
    episodes: tuple[EpisodeResult, ...]
    control_mode: str = "pd_ee_delta_pose"
    obs_mode: str = "rgb"
    split: str = "Short"
    memory_type: str = "Tracking"
    model: str = "dom-vpwem"
    model_config: Mapping[str, Any] | None = None
    benchmark_commit: str = "unknown"
    action_chunk_size: int = 1

    @property
    def success_rate(self) -> float:
        return float(np.mean([episode.success_once for episode in self.episodes]))

    @property
    def mean_return(self) -> float:
        return float(np.mean([episode.episode_return for episode in self.episodes]))

    def to_dict(self) -> dict[str, Any]:
        """Return the task-level JSON shape used by MIKASA's benchmark."""

        return {
            "env_id": self.env_id,
            "start_seed": self.start_seed,
            "n_episodes": len(self.episodes),
            "successes": [episode.success_once for episode in self.episodes],
            "returns": [episode.episode_return for episode in self.episodes],
            "sr": self.success_rate,
            "mean_return": self.mean_return,
            "episode_lengths": [episode.n_steps for episode in self.episodes],
            "episode_seeds": [episode.seed for episode in self.episodes],
            "control_mode": self.control_mode,
            "obs_mode": self.obs_mode,
            "split": self.split,
            "memory_type": self.memory_type,
            "model": {
                "name": self.model,
                "config": dict(self.model_config or {}),
            },
            "benchmark_commit": self.benchmark_commit,
            "action_chunk_size": self.action_chunk_size,
            "wrapper_chain": "apply_mikasa_vla_wrappers(include_overlays=False)",
            "episodes": [asdict(episode) for episode in self.episodes],
        }


def _reset_policy(policy: MemoryPolicy, *, device: str | None, seed: int) -> None:
    reset = getattr(policy, "reset", None)
    if not callable(reset):
        raise TypeError("Policy must provide reset(batch_size=1, device=...) or reset().")
    try:
        reset(batch_size=1, device=device, seed=seed)
    except TypeError as seeded_error:
        try:
            reset(batch_size=1, device=device)
        except TypeError:
            try:
                reset()
            except TypeError:
                raise seeded_error


def _action_chunk(actions: Any) -> list[Any]:
    """Validate and split ``[7]``, ``[K, 7]``, or ``[1, K, 7]`` output."""

    shape_value = getattr(actions, "shape", None)
    shape = tuple(shape_value) if shape_value is not None else tuple(np.asarray(actions).shape)
    if shape == (ACTION_DIM,):
        return [actions]
    if len(shape) == 2 and shape[1] == ACTION_DIM:
        if shape[0] == 0:
            raise ValueError("Policy returned an empty action chunk.")
        return [actions[index] for index in range(shape[0])]
    if len(shape) == 3 and shape[0] == 1 and shape[2] == ACTION_DIM:
        if shape[1] == 0:
            raise ValueError("Policy returned an empty action chunk.")
        return [actions[0, index] for index in range(shape[1])]
    raise ValueError(
        f"Policy.act must return ({ACTION_DIM},), (K, {ACTION_DIM}), or "
        f"(1, K, {ACTION_DIM}); got {shape}."
    )


def run_episode(
    adapter: MikasaEnvAdapter,
    policy: MemoryPolicy,
    *,
    seed: int,
    device: str | None = None,
    frame_sink: RolloutFrameSink | None = None,
) -> EpisodeResult:
    """Run one seeded episode with a fresh memory state and action queue."""

    _reset_policy(policy, device=device, seed=seed)
    observation, _ = adapter.reset(seed=seed)
    action_queue: Deque[Any] = deque()
    success_once = False
    episode_return = 0.0
    n_steps = 0
    action_chunk_size: int | None = None

    if frame_sink is not None:
        frame_sink.write_frame(observation, step=0, success=False)

    for _ in range(adapter.max_episode_steps):
        if not action_queue:
            action_chunk = _action_chunk(policy.act(observation))
            emitted_size = len(action_chunk)
            if action_chunk_size is None:
                action_chunk_size = emitted_size
            elif emitted_size != action_chunk_size:
                raise ValueError(
                    "Policy emitted inconsistent action chunk sizes within an "
                    f"episode: expected {action_chunk_size}, got {emitted_size}."
                )
            action_queue.extend(action_chunk)

        observation, reward, terminated, truncated, info = adapter.step(action_queue.popleft())
        n_steps += 1
        episode_return += reward
        success_once = success_once or bool(_first_scalar(info.get("success"), default=False))
        if frame_sink is not None:
            frame_sink.write_frame(observation, step=n_steps, success=success_once)
        if terminated or truncated:
            break

    return EpisodeResult(
        seed=seed,
        success_once=success_once,
        episode_return=episode_return,
        n_steps=n_steps,
        action_chunk_size=action_chunk_size or 1,
    )


def _validate_declared_chunk_size(policy: MemoryPolicy, actual_size: int) -> None:
    """Ensure optional policy metadata agrees with emitted action chunks."""

    for attribute in ("action_chunk_size", "chunk_size"):
        declared = getattr(policy, attribute, None)
        if declared is None:
            continue
        try:
            declared_size = operator.index(declared)
        except TypeError as exc:
            raise ValueError(
                f"policy.{attribute} must be a positive integer, got {declared!r}."
            ) from exc
        if declared_size <= 0:
            raise ValueError(f"policy.{attribute} must be positive, got {declared_size}.")
        if declared_size != actual_size:
            raise ValueError(
                f"policy.{attribute} declares {declared_size}, but policy.act "
                f"emitted chunks of size {actual_size}."
            )


def evaluate_policy(
    policy: MemoryPolicy,
    *,
    env_config: MikasaEnvConfig | None = None,
    adapter: MikasaEnvAdapter | None = None,
    n_episodes: int = NUM_EPISODES,
    start_seed: int = START_SEED,
    device: str | None = None,
    close_adapter: bool | None = None,
    model_name: str = "dom-vpwem",
    model_config: Mapping[str, Any] | None = None,
    benchmark_commit: str = "unknown",
    video_output: str | Path | None = None,
    video_episode: int = 0,
    video_fps: int = DEFAULT_VIDEO_FPS,
) -> EvaluationResult:
    """Evaluate one policy using canonical MIKASA seeds and success latching.

    If this function constructs the adapter it closes it automatically.  A
    caller-supplied adapter remains open unless ``close_adapter=True``.
    When ``video_output`` is set, only zero-based ``video_episode`` is recorded
    and all requested episodes still contribute to the returned metrics.
    """

    if n_episodes <= 0:
        raise ValueError(f"n_episodes must be positive, got {n_episodes}.")
    if start_seed < 0:
        raise ValueError(f"start_seed must be non-negative, got {start_seed}.")
    if adapter is not None and env_config is not None:
        raise ValueError("Pass either adapter or env_config, not both.")
    video_path = Path(video_output) if video_output is not None else None
    if video_path is not None:
        if video_path.suffix.lower() != ".mp4":
            raise ValueError(f"video_output must end in .mp4, got {video_path}.")
        if video_episode < 0 or video_episode >= n_episodes:
            raise ValueError(
                f"video_episode must be between 0 and {n_episodes - 1}, got {video_episode}."
            )
        if isinstance(video_fps, bool) or not isinstance(video_fps, int) or video_fps <= 0:
            raise ValueError(f"video_fps must be a positive integer, got {video_fps!r}.")

    owns_adapter = adapter is None
    if adapter is None:
        adapter = MikasaEnvAdapter(env_config or MikasaEnvConfig())
    should_close = owns_adapter if close_adapter is None else close_adapter

    episodes: list[EpisodeResult] = []
    action_chunk_size: int | None = None
    try:
        task_spec = get_task_spec(adapter.config.env_id)
        for episode_index in range(n_episodes):
            episode_seed = start_seed + episode_index
            recorder = None
            if video_path is not None and episode_index == video_episode:
                recorder = Mp4RolloutRecorder(
                    video_path,
                    fps=video_fps,
                    episode_index=episode_index,
                    seed=episode_seed,
                    horizon=adapter.max_episode_steps,
                )
            frame_sink_context = recorder if recorder is not None else nullcontext(None)
            with frame_sink_context as frame_sink:
                episode = run_episode(
                    adapter,
                    policy,
                    seed=episode_seed,
                    device=device,
                    frame_sink=frame_sink,
                )
            if action_chunk_size is None:
                action_chunk_size = episode.action_chunk_size
                _validate_declared_chunk_size(policy, action_chunk_size)
            elif episode.action_chunk_size != action_chunk_size:
                raise ValueError(
                    "Policy emitted inconsistent action chunk sizes across episodes: "
                    f"expected {action_chunk_size}, got {episode.action_chunk_size}."
                )
            episodes.append(episode)
    finally:
        if should_close:
            adapter.close()

    return EvaluationResult(
        env_id=adapter.config.env_id,
        start_seed=start_seed,
        episodes=tuple(episodes),
        control_mode=adapter.config.control_mode,
        obs_mode=adapter.config.obs_mode,
        split=task_spec.split,
        memory_type=task_spec.memory_type,
        model=model_name,
        model_config=dict(model_config or {}),
        benchmark_commit=benchmark_commit,
        action_chunk_size=action_chunk_size or 1,
    )


def _load_cli_policy(args: argparse.Namespace) -> MemoryPolicy:
    # Local policy integration is lazy so importing this evaluator never loads
    # Torch, VPWEM, or checkpoint code.
    from .policy import load_policy

    return load_policy(
        checkpoint=Path(args.checkpoint),
        device=args.device,
        num_inference_steps=args.num_inference_steps,
        action_chunk_size=args.action_chunk_size,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint consumed by dom_vpwem.policy.load_policy.",
    )
    parser.add_argument("--env-id", default=DEFAULT_ENV_ID)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sim-backend", default="gpu")
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES)
    parser.add_argument("--start-seed", type=int, default=START_SEED)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--action-chunk-size", type=int, default=None)
    parser.add_argument("--model-name", default="dom-vpwem")
    parser.add_argument(
        "--benchmark-commit",
        default="unknown",
        help="MIKASA-Robo git commit used for the rollout metadata.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path; stdout is always printed.",
    )
    parser.add_argument(
        "--video-output",
        type=Path,
        default=None,
        help="Optional .mp4 path for one side-by-side rollout video.",
    )
    parser.add_argument(
        "--video-episode",
        type=int,
        default=0,
        help="Zero-based episode index to record when --video-output is set (default: 0).",
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=DEFAULT_VIDEO_FPS,
        help=f"MP4 playback frame rate (default: {DEFAULT_VIDEO_FPS}, the control rate).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if (
        args.output is not None
        and args.video_output is not None
        and args.output.resolve() == args.video_output.resolve()
    ):
        parser.error("--output and --video-output must be different paths")
    policy = _load_cli_policy(args)
    model_config: dict[str, Any] = {"checkpoint": str(Path(args.checkpoint))}
    if args.num_inference_steps is not None:
        model_config["num_inference_steps"] = args.num_inference_steps
    if args.action_chunk_size is not None:
        model_config["requested_action_chunk_size"] = args.action_chunk_size
    result = evaluate_policy(
        policy,
        env_config=MikasaEnvConfig(env_id=args.env_id, sim_backend=args.sim_backend),
        n_episodes=args.episodes,
        start_seed=args.start_seed,
        device=args.device,
        model_name=args.model_name,
        model_config=model_config,
        benchmark_commit=args.benchmark_commit,
        video_output=args.video_output,
        video_episode=args.video_episode,
        video_fps=args.video_fps,
    )
    payload = result.to_dict()
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if args.video_output is not None:
        print(f"Saved rollout video: {args.video_output}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())


__all__ = [
    "EpisodeResult",
    "EvaluationResult",
    "MemoryPolicy",
    "NUM_EPISODES",
    "START_SEED",
    "evaluate_policy",
    "run_episode",
]
