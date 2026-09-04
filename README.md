# DOM-VPWEM

This repository contains a focused VPWEM training and evaluation port for
MIKASA-Robo-VLA shell-game tasks. The default task is
`ShellGameShuffleColorLampTouch-VLA-v0`: the policy is language-free, while
the episode-specific color mapping, cup shuffle, and lamp target are observed
visually. A separate config supports the stationary-cup
`ShellGameTouch-VLA-v0` task.

The implementation includes:

- two-frame visual/proprioceptive working memory;
- recursive Q-Former-style episodic memory with bounded caches;
- adjacent-similarity or FIFO cache compression;
- a transformer DDPM action model with VPWEM/DP-PTP action alignment;
- current MIKASA 7-D `proprio` and normalized 7-D `pd_ee_delta_pose` actions;
- episode-safe NPZ loading, training checkpoints with EMA and normalization;
- canonical 50-episode evaluation with memory resets and `success_once`
  latching;
- optional H.264 MP4 recording of evaluation rollouts.

The architecture follows the [VPWEM paper](https://arxiv.org/abs/2603.04910)
and was checked against the Apache-2.0
[reference implementation](https://github.com/HarryLui98/code_vpwem). The
simulator and dataset contract come from
[MIKASA-Robo-VLA](https://github.com/CognitiveAISystems/MIKASA-Robo).

## Supported tasks

| Task | Config | Horizon | Distinguishing behavior |
| --- | --- | ---: | --- |
| `ShellGameShuffleColorLampTouch-VLA-v0` | [`shell_game_shuffle_color_lamp_touch.yaml`](configs/shell_game_shuffle_color_lamp_touch.yaml) | 60 | Cups shuffle; the policy must track them and use the observed lamp/color cue. This remains the default. |
| `ShellGameTouch-VLA-v0` | [`shell_game_touch.yaml`](configs/shell_game_touch.yaml) | 30 | Cups remain stationary after covering the ball; there is no cup-shuffle phase. |

Train and evaluate the stationary-cup task with:

```bash
uv run --locked dom-vpwem-train --config configs/shell_game_touch.yaml

uv run --locked --extra eval dom-vpwem-eval \
  --checkpoint outputs/shell_game_touch/checkpoint_600000.pt \
  --env-id ShellGameTouch-VLA-v0 \
  --device cuda \
  --sim-backend gpu \
  --action-chunk-size 1 \
  --output eval_results/shell_game_touch.json
```

The VPWEM paper reports **91%** success for the legacy MIKASA environment
`ShellGameTouch-v0`. That environment predates the current VLA interface and
is not `ShellGameTouch-VLA-v0`; the published number is therefore not a
benchmark result for this repository's VLA rewrite.

## Installation

The project environment is managed by [uv](https://docs.astral.sh/uv/). Python
3.10 is pinned in `.python-version`, and `uv.lock` contains the complete,
hash-checked resolution. Install uv, then create the core training/test
environment with:

```bash
uv python install 3.10
uv sync --locked
```

Add MIKASA, ManiSkill, and the online evaluation/collection stack with:

```bash
uv sync --locked --extra eval
```

MIKASA-Robo 1.0.0 accidentally omitted the lamp mesh used by
`ShellGameShuffleColorLampTouch-VLA-v0` from both of its Python package
artifacts. After syncing the evaluation extra, install the exact file from the
immutable upstream release commit and verify its SHA-256 with:

```bash
uv run --locked --extra eval dom-vpwem-install-sim-assets
```

The command resolves the active uv environment dynamically and is an
idempotent no-op when the correct asset is already present. For an offline
install, download the file separately and pass
`--source /path/to/low_poly_light_bulb.glb`; the same checksum is required.
Unexpected existing content is preserved unless `--force` is supplied. This
extra step is needed for the shuffle/lamp task, not `ShellGameTouch-VLA-v0`.

If evaluation otherwise fails with a path ending in
`vla/utils/objects/low_poly_light_bulb.glb`, rerun the installer command above
in the same repository. Do not copy into a hard-coded `site-packages` path,
because the location depends on the uv environment and Python version.

The downloaded model is
[Low Poly Light Bulb](https://sketchfab.com/3d-models/low-poly-light-bulb-a7d27c2224d94c86a04083de8f9df7db)
by AleixoAlonso, used unmodified under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Add the lightweight public-dataset downloader/converter stack with:

```bash
uv sync --locked --extra data
```

Both extras can be installed together with:

```bash
uv sync --locked --extra data --extra eval
```

uv creates a local `.venv` and `uv run` uses it automatically; shell activation
is unnecessary. The lock is intentionally restricted to Linux x86-64, matching
the upstream MIKASA GPU environment, and aligns NumPy 1.23.5, PyTorch 2.2.1,
and torchvision 0.17.1 with MIKASA-Robo-VLA 1.0. The locked PyTorch wheel uses
the CUDA 12.1 runtime; GPU execution requires a compatible host NVIDIA driver,
and uv does not install system GPU or Vulkan drivers.

Verify the resolved runtime after syncing:

```bash
uv run --locked python -c \
  "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

## Dataset

Install both supported public datasets with one command:

```bash
uv run --locked --extra data dom-vpwem-install-datasets
```

The equivalent source-tree script is
[`scripts/install_datasets.py`](scripts/install_datasets.py):

```bash
uv run --locked --extra data python scripts/install_datasets.py
```

The command downloads only the two task folders from the official
[MIKASA LeRobot v3 release](https://huggingface.co/datasets/mikasa-robo/mikasa-robo-vla-lerobot)
at the immutable commit
`fa5417a266d1cb87ed7715c3dd2d0e4edc067b04`. It streams the Parquet state and
action rows alongside the top/wrist AV1 videos, then produces the exact NPZ
layout used by the trainer:

```text
data_mikasa_robo/data_npz/
├── shell_game_shuffle_color_lamp_touch_vla_v0/
│   ├── train_data_000000.npz
│   ├── ...
│   └── .dom_vpwem_dataset.json
└── shell_game_touch_vla_v0/
    ├── train_data_000000.npz
    ├── ...
    └── .dom_vpwem_dataset.json
```

Install just one task by environment ID, dataset slug, or short alias:

```bash
uv run --locked --extra data dom-vpwem-install-datasets --task touch
uv run --locked --extra data dom-vpwem-install-datasets --task shuffle
```

Conversion happens in a sibling staging directory. Every episode is validated
and recorded with its byte size and SHA-256 digest before the completed task is
promoted. A matching installation is an idempotent no-op and does not access
the network. Run a full integrity check later with:

```bash
uv run --locked dom-vpwem-install-datasets --verify-only
```

Task selection defaults to both for verification as well; append `--task
touch` or `--task shuffle` if only one was installed.

Use `--force` to build and verify a replacement before atomically swapping an
existing installer-managed task. Unmarked directories are never merged or
overwritten without that explicit flag. Downloads are resumable in
`<output-root>/.cache/huggingface`; `--cache-dir` selects another cache, and
`--offline` permits cached-only installation.

Allow roughly 2 GiB for the two decoded NPZ datasets in addition to the compact
Hugging Face cache and temporary staging space. The source videos are the
official release's AV1-encoded camera streams; use locally collected NPZ or the
larger lossless RLDS release if exact pre-video pixel values are required.

Padded and unpadded numeric filenames are both accepted. Each episode must
contain:

```text
rgb       [T, 128, 128, 6] uint8
proprio   [T, 7]            float32
action    [T, 7]            float32 in [-1, 1]
```

`done`, `success`, and `episode_length` are used when available. Language is
never deserialized or supplied to the policy.

The installer writes `done=True` only on the final row and records
`episode_length`; the public LeRobot export does not include stepwise success,
reward, or language fields, none of which are consumed by this trainer.

### Collect custom trajectories instead

To generate new source NPZ rather than installing the public release, run the
current MIKASA PPO collector with a separately trained oracle checkpoint:

```bash
uv run --locked --extra eval \
  python -m mikasa_robo_suite.vla.dataset_collectors.get_mikasa_robo_datasets \
  --env-id ShellGameShuffleColorLampTouch-VLA-v0 \
  --path-to-save-data data_mikasa_robo \
  --ckpt-dir /path/to/oracle/checkpoints \
  --num-train-data 250
```

For the stationary-cup dataset, use `--env-id ShellGameTouch-VLA-v0`; the
collector writes it under the task-specific directory configured above.
The collector requires the MIKASA GPU simulation stack and does not ship its
VLA oracle checkpoints as part of this repository.

## Train

Review `task.dataset_dir`, `train.output_dir`, batch size, and device in
[`configs/shell_game_shuffle_color_lamp_touch.yaml`](configs/shell_game_shuffle_color_lamp_touch.yaml),
then run:

```bash
uv run --locked dom-vpwem-train \
  --config configs/shell_game_shuffle_color_lamp_touch.yaml
```

Select the stationary-cup task instead with
`--config configs/shell_game_touch.yaml`; its checkpoints are written to
`outputs/shell_game_touch/` by default.

`train.preload_dataset: true` retains approximately 1.1 GiB of RGB data for
this single task and avoids reopening random NPZ files for every timestep.
Linux DataLoader workers normally share those read-only pages through `fork`.
Set it to `false` if host memory is constrained, at the cost of substantially
slower shuffled NPZ access.

Useful one-off overrides:

```bash
uv run --locked dom-vpwem-train \
  --dataset-dir data_mikasa_robo/data_npz/shell_game_shuffle_color_lamp_touch_vla_v0 \
  --output-dir outputs/shell_game_shuffle_color_lamp_touch \
  --device cuda \
  --batch-size 16 \
  --steps 600000
```

Resume with
`--resume outputs/shell_game_shuffle_color_lamp_touch/checkpoint_20000.pt`.
Checkpoints
contain the online model, EMA model, optimizer, scheduler, scaler, random-state
snapshots, complete configuration, and proprio/action normalization statistics.
Resume intentionally rejects changes to the task, architecture, optimization
schedule, device, or data-loader sampling settings.

The paper's reported experiments use a vision encoder frozen from a preliminary
short-context policy. This port can reproduce that initialization by setting
`model.freeze_vision_encoder: true` and
`train.vision_encoder_checkpoint: /path/to/short_context_checkpoint.pt` (or by
passing `--vision-encoder-checkpoint`). It prefers the checkpoint's EMA encoder
and refuses to freeze randomly initialized ResNets. The supplied YAML defaults
to single-stage raw-image training because no task-specific short-context
checkpoint is distributed here.

The model hyperparameters use the paper values: 256-D embeddings, two
working-memory frames, two episodic tokens, an eight-layer action transformer,
five-frame memory subsampling, an eight-entry FIFO recursive cache, eight
predicted future actions, 50 DDPM steps, and 600,000 MIKASA updates. The YAML
uses batch size 16 instead of the paper's precomputed-embedding batch size 64,
because this port consumes raw 128×128 images.

## Evaluate

The safest default for the shuffle-tracking task is one executed action per
inference call. That lets VPWEM observe every shuffle frame and update memory
at every simulator step. The same conservative setting is used in the
stationary-task command above.

```bash
uv run --locked --extra eval dom-vpwem-eval \
  --checkpoint outputs/shell_game_shuffle_color_lamp_touch/checkpoint_600000.pt \
  --env-id ShellGameShuffleColorLampTouch-VLA-v0 \
  --device cuda \
  --sim-backend gpu \
  --action-chunk-size 1 \
  --benchmark-commit 509b875f3d207c287497c0a897661062de928bb0 \
  --output eval_results/shell_game_shuffle_color_lamp_touch.json
```

By default evaluation uses 50 episodes with seeds `4242424242` through
`4242424291`. The environment is created with:

```python
gym.make(
    "ShellGameShuffleColorLampTouch-VLA-v0",
    num_envs=1,
    obs_mode="rgb",
    control_mode="pd_ee_delta_pose",
    reward_mode="normalized_dense",
    render_mode="all",
    sim_backend="gpu",
)
```

and immediately wrapped with `apply_mikasa_vla_wrappers(...,
include_overlays=False)`. Policy and action-queue state are reset for every
episode. `--num-inference-steps` can select a faster deterministic DDIM
schedule; omitting it uses the trained 50-step DDPM sampler.

The output JSON follows MIKASA's task-result fields, including the Short split,
the task-specific memory type (Tracking for shuffle-color or Spatial for
stationary touch), episode seeds, actual emitted action-chunk size, model
configuration, and benchmark commit.

### Record an evaluation rollout

Add `--video-output` to the normal evaluator to save one episode as an MP4. For
example, this records a single shuffle-task rollout while also writing its JSON
result:

```bash
uv run --locked --extra eval dom-vpwem-eval \
  --checkpoint outputs/shell_game_shuffle_color_lamp_touch/checkpoint_600000.pt \
  --env-id ShellGameShuffleColorLampTouch-VLA-v0 \
  --device cuda \
  --sim-backend gpu \
  --episodes 1 \
  --action-chunk-size 1 \
  --video-output eval_results/videos/shell_game_shuffle.mp4 \
  --output eval_results/shell_game_shuffle_color_lamp_touch.json
```

The video shows the overhead and wrist RGB observations side by side, with the
episode seed, step, and latched success state overlaid. It includes the reset
frame and every post-action frame (including the terminal frame), and defaults
to the simulator's 20 Hz control rate. During a longer evaluation,
`--video-episode N` records only zero-based episode `N`; metrics are still
computed over every requested episode. Use `--video-fps` to change playback
speed without changing the rollout itself. Video status is printed to stderr,
so stdout remains valid result JSON.

## Important shuffle-color data caveat

In MIKASA-Robo-VLA 1.0 source, the standard task's curriculum wrapper executes
zero robot actions during the color-reveal and shuffle prefix, while the PPO
collector appears to serialize the oracle's pre-wrapper actions. This is a
source-code inference, not a confirmed dataset issue. The wrapper also zeros
those actions during evaluation, so they cannot move the robot, but they may
add irrelevant behavior-cloning loss. Inspect your collected prefix labels or
recollect post-wrapper actions before treating results as definitive.

## Validation

Run:

```bash
uv run --locked pytest -q
uv run --locked ruff check src tests
```

Tests cover file discovery, malformed and variable-length episodes, temporal
window alignment, memory isolation across episodes, current environment/action
contracts, canonical success evaluation, normalization, and VPWEM
forward/backward/sampling shapes. A real MIKASA rollout still requires its GPU
simulator stack and is intentionally not mocked as evidence of task success.

## Attribution and license

Code in this repository is Apache-2.0. See [`NOTICE`](NOTICE) for upstream
attribution. MIKASA-Robo, ManiSkill, simulator assets, oracle checkpoints, and
trajectory datasets are separate works under their respective licenses.
