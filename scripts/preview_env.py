"""Record a task's fixed and wrist camera views without a policy checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from dom_vpwem.evaluate import evaluate_policy
from dom_vpwem.mikasa_env import MikasaEnvConfig
from dom_vpwem.tasks import INTERCEPT_FAST_COVER_ENV_ID


class ZeroActionPolicy:
    """Let the scene evolve while commanding zero end-effector deltas."""

    def reset(self, **kwargs) -> None:
        pass

    def act(self, observation) -> np.ndarray:
        return np.zeros(7, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default=INTERCEPT_FAST_COVER_ENV_ID)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sim-backend", default="gpu")
    parser.add_argument("--output", type=Path, default=Path("eval_results/preview.mp4"))
    args = parser.parse_args()
    evaluate_policy(
        ZeroActionPolicy(),
        env_config=MikasaEnvConfig(env_id=args.env_id, sim_backend=args.sim_backend),
        n_episodes=1,
        start_seed=args.seed,
        video_output=args.output,
    )
    print(f"Saved camera preview: {args.output}")


if __name__ == "__main__":
    main()
