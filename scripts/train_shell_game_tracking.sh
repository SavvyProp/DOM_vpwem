#!/usr/bin/env bash
# Train the 1–2-shuffle shell-game oracle, collect demos, then train TrackingVPWEM.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$SCRIPT_DIR/.."

# 1. PPO state oracle: up to 150M transitions, with success-based early stopping.
printf '\n[1/3] Training the PPO oracle\n'
uv run --locked --extra eval --inexact python -m dom_vpwem.oracle_train \
  --config configs/oracles/shell_game_shuffle_touch.yaml

# 2. Save 250 successful RGB/action demos, including target-cup tracking labels.
printf '\n[2/3] Collecting tracking-labeled demonstrations\n'
uv run --locked --extra eval --inexact python -m dom_vpwem.collect_demos \
  --env-id ShellGameShuffleTouchCustom-VLA-v0 \
  --checkpoint outputs/oracles/shell_game_shuffle_touch/swaps1_2_tracking_v1/best_ckpt.pt \
  --data-root data_mikasa_robo/data_npz \
  --episodes 250 \
  --num-envs 4

# 3. Train TrackingVPWEM with diffusion/tracking losses and a predicted-position token.
# Default: 600K gradient steps; edit its YAML to change training settings.
printf '\n[3/3] Training TrackingVPWEM (FP32)\n'
uv run --locked --extra eval --inexact python -m dom_vpwem.train_tracking \
  --config configs/shell_game_shuffle_touch_tracking.yaml \
  --fp32
