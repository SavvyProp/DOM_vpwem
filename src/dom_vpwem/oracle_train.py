"""Train a privileged-state PPO oracle for registered MIKASA tasks and variants.

Uses MIKASA's AgentStateOnly network, controller, and state wrappers. The local
loop provides configurable clipped PPO, GAE, fixed-seed success evaluation,
plain state-dict exports for collection, and resumable optimizer checkpoints.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from contextlib import ExitStack
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import yaml
from torch import Tensor

from .oracle_env import make_oracle_agent, make_oracle_env, oracle_observation_schema
from .tasks import INTERCEPT_FAST_COVER_ENV_ID, get_task_spec


@dataclass
class OracleTrainConfig:
    env_id: str = INTERCEPT_FAST_COVER_ENV_ID
    output_dir: str | None = None
    seed: int = 123
    device: str = "cuda"
    sim_backend: str = "gpu"
    num_envs: int = 256
    num_steps: int | None = None
    total_timesteps: int = 150_000_000
    learning_rate: float = 3e-4
    anneal_lr: bool = False
    gamma: float = 0.99
    gae_lambda: float = 0.95
    finite_horizon_gae: bool = True
    num_minibatches: int = 16
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = False
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.2
    num_eval_envs: int = 16
    eval_episodes: int = 64
    eval_seed: int = 10_000_000
    eval_every: int = 25
    success_threshold: float = 0.95
    success_evals: int = 3
    checkpoint_every: int = 25
    log_every: int = 10
    checkpoint: str | None = None
    resume: str | None = None

    def validate(self) -> None:
        task = get_task_spec(self.env_id)
        if self.num_steps is None:
            self.num_steps = task.max_episode_steps
        if self.output_dir is None:
            self.output_dir = f"outputs/oracles/{task.dataset_slug}"
        if not isinstance(self.output_dir, str) or not self.output_dir.strip():
            raise ValueError("output_dir must be a nonempty path string")
        for name in (
            "num_envs",
            "num_steps",
            "total_timesteps",
            "num_minibatches",
            "update_epochs",
            "num_eval_envs",
            "eval_episodes",
            "eval_every",
            "success_evals",
            "checkpoint_every",
            "log_every",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("seed", "eval_seed"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("anneal_lr", "finite_horizon_gae", "norm_adv", "clip_vloss"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        for name in ("gamma", "gae_lambda", "success_threshold"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in ("learning_rate", "clip_coef", "max_grad_norm", "target_kl"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("ent_coef", "vf_coef"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.batch_size % self.num_minibatches or self.batch_size // self.num_minibatches < 2:
            raise ValueError("num_envs * num_steps must divide into minibatches of at least 2")
        if self.total_timesteps < self.batch_size:
            raise ValueError("total_timesteps must cover at least one num_envs * num_steps rollout")
        if self.sim_backend not in {"gpu", "cpu"}:
            raise ValueError("sim_backend must be 'gpu' or 'cpu'")
        if self.sim_backend == "cpu" and (self.num_envs != 1 or self.num_eval_envs != 1):
            raise ValueError("CPU simulation requires num_envs=1 and num_eval_envs=1")
        if self.checkpoint and self.resume:
            raise ValueError("Use checkpoint for a weight warm start or resume for training state")

    @property
    def batch_size(self) -> int:
        assert self.num_steps is not None
        return self.num_envs * self.num_steps


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Optional flat YAML; CLI flags override it.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved config and exit.")
    for field in fields(OracleTrainConfig):
        kwargs: dict[str, Any] = {"default": None}
        if isinstance(field.default, bool):
            kwargs["action"] = argparse.BooleanOptionalAction
        else:
            kwargs["type"] = (
                int
                if field.name == "num_steps"
                else (str if field.default is None else type(field.default))
            )
        parser.add_argument("--" + field.name.replace("_", "-"), **kwargs)
    return parser


def parse_config(argv: Sequence[str] | None = None) -> tuple[OracleTrainConfig, bool]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    raw = {}
    if args.config is not None:
        with args.config.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if not isinstance(raw, dict):
            parser.error("Oracle configuration must be a flat YAML mapping")
    names = {field.name for field in fields(OracleTrainConfig)}
    if unknown := raw.keys() - names:
        parser.error(f"Unknown oracle settings: {', '.join(sorted(unknown))}")
    raw.update({name: getattr(args, name) for name in names if getattr(args, name) is not None})
    config = OracleTrainConfig(**raw)
    try:
        config.validate()
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    return config, args.dry_run


def _observation(obs: Mapping[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {key: value.to(device=device, dtype=torch.float32) for key, value in obs.items()}


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def compute_advantages(
    rewards: Tensor,
    values: Tensor,
    next_values: Tensor,
    dones: Tensor,
    *,
    gamma: float,
    gae_lambda: float,
    finite_horizon: bool = True,
) -> tuple[Tensor, Tensor]:
    """GAE with next values already corrected for termination and auto-reset.

    Stop the trace at either kind of episode boundary. Timeout transitions may
    bootstrap from their final observation, but never from the next episode.
    The optional normalized finite-horizon variant follows MIKASA's PPO recipe.
    """
    advantages = torch.zeros_like(rewards)
    carry = torch.zeros_like(rewards[0])
    coefficient = torch.zeros_like(carry)
    reward_sum = torch.zeros_like(carry)
    value_sum = torch.zeros_like(carry)
    for step in reversed(range(len(rewards))):
        continues = (~dones[step]).float()
        if finite_horizon:
            coefficient = 1 + gae_lambda * coefficient * continues
            reward_sum = gae_lambda * gamma * reward_sum * continues + coefficient * rewards[step]
            value_sum = gae_lambda * gamma * value_sum * continues + gamma * next_values[step]
            advantages[step] = (reward_sum + value_sum) / coefficient - values[step]
        else:
            delta = rewards[step] + gamma * next_values[step] - values[step]
            carry = delta + gamma * gae_lambda * continues * carry
            advantages[step] = carry
    return advantages, advantages + values


@torch.no_grad()
def bootstrap_values(
    agent: Any,
    next_obs: Mapping[str, Tensor],
    terminated: Tensor,
    truncated: Tensor,
    info: Mapping[str, Any],
    device: torch.device,
) -> Tensor:
    values = agent.get_value(next_obs).flatten()
    timeout = truncated & ~terminated
    if timeout.any():
        if "final_observation" not in info:
            raise RuntimeError("Auto-reset timeout is missing final_observation for bootstrapping")
        final_obs = _observation(info["final_observation"], device)
        terminal_values = agent.get_value({key: value[timeout] for key, value in final_obs.items()})
        values[timeout] = terminal_values.flatten()
    return values.masked_fill(terminated, 0)


def ppo_update(
    agent: Any, optimizer: Any, batch: Mapping[str, Any], config: OracleTrainConfig
) -> dict[str, float]:
    """Optimize the sampled Gaussian actions (before controller clipping)."""
    agent.train()
    n = len(batch["actions"])
    minibatch_size = n // config.num_minibatches
    metrics: list[tuple[float, ...]] = []
    for _ in range(config.update_epochs):
        indices = torch.randperm(n, device=batch["actions"].device)
        stop = False
        for start in range(0, n, minibatch_size):
            idx = indices[start : start + minibatch_size]
            _, logprob, entropy, value = agent.get_action_and_value(
                {key: value[idx] for key, value in batch["obs"].items()},
                batch["actions"][idx],
            )
            logratio = logprob - batch["logprobs"][idx]
            ratio = logratio.exp()
            with torch.no_grad():
                approx_kl = ((ratio - 1) - logratio).mean()
            if approx_kl > config.target_kl:
                stop = True
                break
            advantages = batch["advantages"][idx]
            if config.norm_adv:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            policy_loss = torch.maximum(
                -advantages * ratio,
                -advantages * ratio.clamp(1 - config.clip_coef, 1 + config.clip_coef),
            ).mean()
            value = value.flatten()
            error = (value - batch["returns"][idx]).square()
            if config.clip_vloss:
                old_value = batch["values"][idx]
                clipped = old_value + (value - old_value).clamp(-config.clip_coef, config.clip_coef)
                error = torch.maximum(error, (clipped - batch["returns"][idx]).square())
            value_loss = 0.5 * error.mean()
            entropy_mean = entropy.mean()
            loss = policy_loss + config.vf_coef * value_loss - config.ent_coef * entropy_mean
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite PPO loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
            optimizer.step()
            metrics.append(
                (policy_loss.item(), value_loss.item(), entropy_mean.item(), approx_kl.item())
            )
        if stop:
            break
    if not metrics:
        raise RuntimeError(
            "PPO performed no updates; check checkpoint and observation compatibility"
        )
    return dict(
        zip(
            ("policy_loss", "value_loss", "entropy", "approx_kl"), np.mean(metrics, axis=0).tolist()
        )
    )


@torch.no_grad()
def evaluate_oracle(
    agent: Any, env: Any, config: OracleTrainConfig, device: torch.device
) -> dict[str, Any]:
    """Measure success_once over fixed, separate validation episode seeds."""
    rng = _rng_state()
    was_training = agent.training
    agent.eval()
    successes, returns = [], []
    try:
        for offset in range(0, config.eval_episodes, config.num_eval_envs):
            obs, _ = env.reset(
                seed=[config.eval_seed + offset + i for i in range(config.num_eval_envs)]
            )
            success = torch.zeros(config.num_eval_envs, dtype=torch.bool, device=device)
            episode_return = torch.zeros(config.num_eval_envs, device=device)
            for _ in range(get_task_spec(config.env_id).max_episode_steps):
                action = agent.get_action(_observation(obs, device), deterministic=True)
                obs, reward, _, _, info = env.step(action.clamp(-1, 1).to(env.device))
                success |= torch.as_tensor(info["success"], device=device).bool().reshape(-1)
                episode_return += torch.as_tensor(reward, device=device).reshape(-1)
            take = min(config.num_eval_envs, config.eval_episodes - offset)
            successes.extend(success[:take].cpu().tolist())
            returns.extend(episode_return[:take].cpu().tolist())
    finally:
        agent.train(was_training)
        _restore_rng(rng)
    return {
        "success_rate": float(np.mean(successes)),
        "mean_return": float(np.mean(returns)),
        "successes": successes,
        "returns": returns,
        "episode_seeds": list(range(config.eval_seed, config.eval_seed + config.eval_episodes)),
    }


def _save_torch(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _save_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def train_oracle(
    config: OracleTrainConfig,
    *,
    env_factory: Callable[..., Any] = make_oracle_env,
    agent_factory: Callable[..., Any] = make_oracle_agent,
) -> Path:
    """Train and export weights. Resuming restarts simulator episodes, not mid-episode state."""
    config.validate()
    device = torch.device(config.device)
    if (device.type == "cuda" or config.sim_backend == "gpu") and not torch.cuda.is_available():
        raise RuntimeError("PPO oracle training requires a working NVIDIA/CUDA GPU for this config")
    output = Path(config.output_dir)
    if output.exists() and any(output.iterdir()) and not config.resume:
        raise ValueError(
            f"Output directory {output} is not empty; use --resume or a new --output-dir"
        )
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    with ExitStack() as stack:
        env = env_factory(
            config.env_id, num_envs=config.num_envs, sim_backend=config.sim_backend, auto_reset=True
        )
        stack.callback(env.close)
        eval_env = env_factory(
            config.env_id,
            num_envs=config.num_eval_envs,
            sim_backend=config.sim_backend,
            auto_reset=False,
        )
        stack.callback(eval_env.close)
        schema = oracle_observation_schema(env)
        if schema != oracle_observation_schema(eval_env):
            raise ValueError("Training and validation observation/action schemas differ")
        if schema["action_shape"] != [7]:
            raise ValueError("MIKASA oracle collection requires a 7-D pd_ee_delta_pose action")
        agent = agent_factory(env, device)
        optimizer = torch.optim.Adam(agent.parameters(), lr=config.learning_rate, eps=1e-5)
        iteration, global_step, best_success, streak = 0, 0, -1.0, 0
        restored_rng = None
        if config.resume:
            saved = torch.load(config.resume, map_location=device, weights_only=False)
            # Configuration, input order, controller, and PPO settings must agree.
            mutable = {
                "output_dir",
                "resume",
                "checkpoint",
                "total_timesteps",
                "log_every",
                "checkpoint_every",
            }
            current = asdict(config)
            if saved["schema"] != schema or any(
                current[name] != value
                for name, value in saved["config"].items()
                if name not in mutable
            ):
                raise ValueError(
                    "Resume configuration or observation schema does not match checkpoint"
                )
            agent.load_state_dict(saved["agent"])
            optimizer.load_state_dict(saved["optimizer"])
            iteration, global_step = saved["iteration"], saved["global_step"]
            best_success, streak = saved["best_success"], saved["success_streak"]
            restored_rng = saved["rng"]
        elif config.checkpoint:
            sidecar = Path(config.checkpoint).with_suffix(".json")
            if sidecar.exists():
                metadata = json.loads(sidecar.read_text(encoding="utf-8"))
                if metadata.get("schema") != schema:
                    raise ValueError(
                        "Warm-start checkpoint observation/action schema does not match"
                    )
            agent.load_state_dict(
                torch.load(config.checkpoint, map_location=device, weights_only=True)
            )
        target_iterations = config.total_timesteps // config.batch_size
        if iteration >= target_iterations:
            raise ValueError(
                "Checkpoint already reached total_timesteps; increase the training budget"
            )
        output.mkdir(parents=True, exist_ok=True)
        _save_json(output / "config.json", asdict(config))
        _save_json(output / "observation_schema.json", schema)
        obs, _ = env.reset(seed=config.seed + global_step)
        obs = _observation(obs, device)
        if restored_rng is not None:
            _restore_rng(restored_rng)
        log = stack.enter_context((output / "metrics.jsonl").open("a", encoding="utf-8"))
        started, starting_step = time.monotonic(), global_step
        last_evaluation = None

        def export(name: str) -> None:
            _save_torch(output / name, agent.state_dict())
            _save_json(
                output / Path(name).with_suffix(".json"),
                {
                    "env_id": config.env_id,
                    "global_step": global_step,
                    "schema": schema,
                    "evaluation": last_evaluation,
                },
            )

        def save_training() -> None:
            _save_torch(
                output / "training_state.pt",
                {
                    "config": asdict(config),
                    "schema": schema,
                    "agent": agent.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "iteration": iteration,
                    "global_step": global_step,
                    "best_success": best_success,
                    "success_streak": streak,
                    "rng": _rng_state(),
                },
            )

        print(
            f"Training {config.env_id}: {config.num_envs} envs, {config.num_steps} steps/rollout, "
            f"{target_iterations * config.batch_size} total transitions; output={output}",
            flush=True,
        )
        for iteration in range(iteration + 1, target_iterations + 1):
            if config.anneal_lr:
                optimizer.param_groups[0]["lr"] = config.learning_rate * (
                    1 - (iteration - 1) / target_iterations
                )
            obs_buffer = {
                key: torch.empty((config.num_steps, *value.shape), device=device)
                for key, value in obs.items()
            }
            actions = torch.empty((config.num_steps, config.num_envs, 7), device=device)
            logprobs = torch.empty((config.num_steps, config.num_envs), device=device)
            rewards, values, next_values = [torch.empty_like(logprobs) for _ in range(3)]
            dones = torch.empty_like(logprobs, dtype=torch.bool)
            agent.eval()
            for step in range(config.num_steps):
                for key in obs:
                    obs_buffer[key][step].copy_(obs[key])
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(obs)
                actions[step], logprobs[step], values[step] = action, logprob, value.flatten()
                # Keep raw samples for the PPO ratio; only the controller sees clipping.
                next_obs, reward, term, trunc, info = env.step(action.clamp(-1, 1).to(env.device))
                obs = _observation(next_obs, device)
                term = torch.as_tensor(term, device=device).bool().reshape(-1)
                trunc = torch.as_tensor(trunc, device=device).bool().reshape(-1)
                next_values[step] = bootstrap_values(agent, obs, term, trunc, info, device)
                rewards[step] = torch.as_tensor(reward, device=device).reshape(-1)
                dones[step] = term | trunc
                global_step += config.num_envs
            advantages, returns = compute_advantages(
                rewards,
                values,
                next_values,
                dones,
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
                finite_horizon=config.finite_horizon_gae,
            )
            metrics = ppo_update(
                agent,
                optimizer,
                {
                    "obs": {key: value.flatten(0, 1) for key, value in obs_buffer.items()},
                    "actions": actions.flatten(0, 1),
                    "logprobs": logprobs.flatten(),
                    "values": values.flatten(),
                    "advantages": advantages.flatten(),
                    "returns": returns.flatten(),
                },
                config,
            )
            evaluate = (
                iteration == 1
                or iteration % config.eval_every == 0
                or (iteration == target_iterations)
            )
            reached_target = False
            if evaluate:
                last_evaluation = evaluate_oracle(agent, eval_env, config, device)
                metrics["evaluation"] = last_evaluation
                success_rate = last_evaluation["success_rate"]
                if success_rate > best_success:
                    best_success = success_rate
                    export("best_ckpt.pt")
                streak = streak + 1 if success_rate >= config.success_threshold else 0
                reached_target = streak >= config.success_evals
                if reached_target:
                    export("final_success_ckpt.pt")
            metrics.update(
                iteration=iteration,
                global_step=global_step,
                learning_rate=optimizer.param_groups[0]["lr"],
                mean_rollout_reward=rewards.mean().item(),
                steps_per_second=(global_step - starting_step) / (time.monotonic() - started),
            )
            log.write(json.dumps(metrics) + "\n")
            log.flush()
            if evaluate or iteration % config.log_every == 0:
                print(json.dumps(metrics), flush=True)
            if evaluate or iteration % config.checkpoint_every == 0:
                save_training()
            if reached_target:
                break
        export("final_ckpt.pt")
        save_training()
        print(f"Saved oracle weights to {output / 'final_ckpt.pt'}", flush=True)
    return output / "final_ckpt.pt"


def main(argv: Sequence[str] | None = None) -> int:
    config, dry_run = parse_config(argv)
    if dry_run:
        print(json.dumps(asdict(config), indent=2))
        return 0
    train_oracle(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
