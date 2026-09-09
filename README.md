# DOM-VPWEM

This repository contains a focused VPWEM training and evaluation port for
MIKASA-Robo-VLA shell-game tasks. The default task is
`ShellGameShuffleColorLampTouch-VLA-v0`: the policy is language-free, while
the episode-specific color mapping, cup shuffle, and lamp target are observed
visually. A separate config supports the stationary-cup
`ShellGameTouch-VLA-v0` task.
Repository-owned MIKASA variants live in `src/dom_vpwem/custom_envs/`, including
`InterceptFastCover-VLA-v0`, its split-cover variant `InterceptFastCover2-VLA-v0`,
`ShellGameShuffleTouchCustom-VLA-v0`, and `RememberColorSequence3-Long-VLA-v0`.

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
| `ShellGameShuffleTouchCustom-VLA-v0` (local variant) | [`shell_game_shuffle_touch.yaml`](configs/shell_game_shuffle_touch.yaml) | 60 | Track the cup hiding the ball through a shuffle, then touch it. Editable subclass of MIKASA's short shuffle-and-touch task. |
| `RememberColorSequence3-Long-VLA-v0` (local variant) | [`remember_color_sequence3_long.yaml`](configs/remember_color_sequence3_long.yaml) | 600 | Observe a random-length color sequence with blank gaps, then select the second-to-last color from three choices. |
| `InterceptFastCover-VLA-v0` (local variant) | [`intercept_fast_cover.yaml`](configs/intercept_fast_cover.yaml) | 60 | InterceptFast with an opaque, collisionless box above the table. Requires locally collected training data. |
| `InterceptFastCover2-VLA-v0` (local variant) | [`intercept_fast_cover2.yaml`](configs/intercept_fast_cover2.yaml) | 60 | Two opaque, collisionless sections with the cover's middle third removed along the ball's travel direction. Requires locally collected training data. |

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

## Custom MIKASA environments

[`InterceptFastCover`](src/dom_vpwem/custom_envs/intercept_fast_cover.py)
subclasses upstream `InterceptFastVLAEnv` and adds a static, opaque gray box.
Its center is **(-0.20, -0.30, 0.25) m** and its full X/Y/Z dimensions are
**(0.40, 0.40, 0.10) m**. The bounds are X=[-0.40, 0.00], Y=[-0.50, -0.10],
Z=[0.20, 0.30]. Only visual geometry is added: the robot and ball pass through
the box. Launch randomization, speed (0.75–1.0 m/s along +Y), cameras, controls,
rewards, and the 60-step horizon are inherited from InterceptFast.

[`InterceptFastCover2`](src/dom_vpwem/custom_envs/intercept_fast_cover2.py)
keeps the same overall footprint and height, but removes the middle third
along Y, the ball's main travel direction. Each remaining section measures
**40 × 13.33 × 10 cm** (X/Y/Z), with a **13.33 cm gap** between them.
Both centers have X=-0.20 m and Z=0.25 m:

| Region, in travel order (+Y) | Y bounds (m) | Center Y (m) |
| --- | --- | --- |
| First cover section | [-0.5000, -0.3667] | -0.4333 |
| Open gap | [-0.3667, -0.2333] | -0.3000 |
| Second cover section | [-0.2333, -0.1000] | -0.1667 |

The gap is intended to let the policy observe the ball again between covered
portions of its travel. Actual visibility and its duration depend on camera
perspective and the ball's trajectory; a gap in world coordinates does not
guarantee a clear view from both cameras.

The cover's apparent occlusion depends on the camera angle. The wrist camera
can see beneath it. To inspect both policy camera views over an episode, run
the preview with the evaluation extra and working NVIDIA/Vulkan drivers:

```bash
uv run --locked --extra eval python scripts/preview_env.py \
  --env-id InterceptFastCover-VLA-v0 \
  --output eval_results/intercept_fast_cover_preview.mp4

uv run --locked --extra eval python scripts/preview_env.py \
  --env-id InterceptFastCover2-VLA-v0 \
  --output eval_results/intercept_fast_cover2_preview.mp4
```

The preview sends zero end-effector deltas and a neutral gripper command; it
needs no checkpoint and is for inspecting the scene, not assessing task success.
`--seed` selects a different ball launch.

The project adapter automatically registers local variants and applies the
upstream VLA observation wrappers:

```python
from dom_vpwem.mikasa_env import MikasaEnvAdapter, MikasaEnvConfig

with MikasaEnvAdapter(MikasaEnvConfig(env_id="InterceptFastCover-VLA-v0")) as env:
    obs, info = env.reset(seed=42)
    # obs["rgb"]: (128, 128, 6); obs["proprio"]: (7,)
```

For direct Gymnasium use (including parallel collection), call
`from dom_vpwem.custom_envs import register_custom_envs; register_custom_envs()`
before `gym.make("InterceptFastCover-VLA-v0", ...)`, then immediately apply
`apply_mikasa_vla_wrappers(env, include_overlays=False)` as usual. Repeated
registration is safe. No installed MIKASA files are modified.

The training config and evaluator accept both local IDs:

```bash
# After collecting NPZ trajectories in the covered environment:
uv run --locked dom-vpwem-train --config configs/intercept_fast_cover.yaml

# After collecting NPZ trajectories with the split cover:
uv run --locked dom-vpwem-train --config configs/intercept_fast_cover2.yaml

uv run --locked --extra eval dom-vpwem-eval \
  --checkpoint outputs/intercept_fast_cover/checkpoint_600000.pt \
  --env-id InterceptFastCover-VLA-v0 \
  --output eval_results/intercept_fast_cover.json
```

The dataset installer downloads only tasks marked `public_dataset=True`.
These cover variants have no public datasets; collect separate trajectories
for each cover layout in its configured dataset directory. Their distinct
environment IDs identify results as local variants, not official MIKASA
benchmark scores. Their Short/Spatial metadata follows the base task's
benchmark grouping.

[`ShellGameShuffleTouch`](src/dom_vpwem/custom_envs/shell_game_shuffle_touch.py)
is a repository-owned subclass of upstream `ShellGameShuffleTouchVLAEnv`,
registered as **`ShellGameShuffleTouchCustom-VLA-v0`**. The `Custom` suffix
avoids conflicting with MIKASA's existing `ShellGameShuffleTouch-VLA-v0` ID.
It starts with the upstream short-task behavior: a 1–5-step ball cue, a
20–35-step shuffle with 2–4 swaps, then touching the cup that hides the ball.
The episode horizon is 60 steps and the metadata is Short/Tracking.

Edit `CUE_PHASE_STEPS`, `SHUFFLE_PHASE_STEPS`, `NUM_SWAPS`, and
`SWAP_ARC_HEIGHT` in the local class to customize it. Scene construction,
observations, rewards, and controls are inherited. The registrar carries over
the YCB asset requirement and the curriculum wrapper that freezes robot
actions during cue and shuffle. If increasing phase durations, also increase
the task/config horizon to leave time for touching the cup.

Preview and train this custom task with:

```bash
uv run --locked --extra eval python scripts/preview_env.py \
  --env-id ShellGameShuffleTouchCustom-VLA-v0 \
  --output eval_results/shell_game_shuffle_touch_preview.mp4

uv run --locked --extra eval dom-vpwem-train-oracle \
  --config configs/oracles/shell_game_shuffle_touch.yaml

# After collecting custom-task NPZ trajectories:
uv run --locked dom-vpwem-train --config configs/shell_game_shuffle_touch.yaml
```

Its dataset directory is
`data_mikasa_robo/data_npz/shell_game_shuffle_touch_custom_vla_v0`.
The dataset installer excludes this local ID; collect trajectories for your
customized environment. Pass `--env-id ShellGameShuffleTouchCustom-VLA-v0`
to the evaluator when using its student checkpoint.

[`RememberColorSequence3Long`](src/dom_vpwem/custom_envs/remember_color_sequence.py)
subclasses `RememberColor3LongVLAEnv` as **`RememberColorSequence3-Long-VLA-v0`**.
It shows red, lime, or blue cubes one at a time at the same central location,
with a blank gap between items. Colors are sampled independently, allowing
repetitions. After the sequence and a final blank delay, all three answer
cubes appear in randomly assigned, separated slots. The correct color is
the **second-to-last sequence item** by default (`n=2`; the last item is `n=1`).
For example, red → blue → lime → red makes lime the answer.

The editable class settings are:

| Setting | Default | Meaning |
| --- | --- | --- |
| `TARGET_FROM_END` | 2 | Which item to recall, counting backward from the end. |
| `SEQUENCE_LENGTH_RANGE` | (3, 7) | Inclusive random sequence length per episode. |
| `COLOR_STEPS` | 10 | Control steps showing each item. |
| `GAP_STEPS` | 5 | Blank control steps between items, including repeated colors. |
| `EMPTY_PHASE_STEPS` | (50, 450) | Inclusive random blank delay after the final item. |

The last item is followed directly by the final delay. Thus, a length-L
sequence takes `L * COLOR_STEPS + (L - 1) * GAP_STEPS` steps before that delay.
The defaults leave at least 50 of the 600 episode steps for answering.
Keep the minimum length at least `TARGET_FROM_END`, and adjust the task and
training horizons if extending the timing. Direct Gymnasium construction also
accepts `target_from_end=n`; the supplied configs and task metadata describe
the default n=2 task. Use a separate registered ID/config for experiments with
a different rule so datasets and results remain identifiable.

Success and rewards are disabled until the answer cubes appear. Robot actions
remain enabled, as in the upstream long task. The state oracle receives the
target and phase timing as privileged information; the RGB student receives
only camera images and proprioception through the standard adapter. Its state
layout includes additional timing fields, so upstream RememberColor oracle
weights cannot be assumed to load unchanged. Resets support individual
episodes within a parallel batch without changing other episodes' sequences.
The local reporting labels are Long/TemporalOrder.

```bash
uv run --locked --extra eval python scripts/preview_env.py \
  --env-id RememberColorSequence3-Long-VLA-v0 \
  --output eval_results/remember_color_sequence3_long_preview.mp4

uv run --locked --extra eval dom-vpwem-train-oracle \
  --config configs/oracles/remember_color_sequence3_long.yaml

# After collecting trajectories for this custom task:
uv run --locked dom-vpwem-train --config configs/remember_color_sequence3_long.yaml
```

Its local dataset directory is
`data_mikasa_robo/data_npz/remember_color_sequence3_long_vla_v0`.
This variant is excluded from public dataset downloads. The existing
single-color cue recordings do not demonstrate this sequence task.

To add another variant:

1. Add a module under `src/dom_vpwem/custom_envs/` with a subclass of the
   upstream environment. Override scene construction or task behavior there.
2. Add a `TaskSpec` in [`tasks.py`](src/dom_vpwem/tasks.py) with a unique
   `env_id`, dataset slug, horizon, metadata, `public_dataset=False`,
   `base_env_id`, and `entry_point="module.path:ClassName"`.
3. Add a training YAML if needed. The shared registrar imports the entry point,
   registers it with ManiSkill/Gymnasium, and copies the base task's asset
   requirements and VLA wrapper configuration. No per-variant changes to the
   adapter are needed.

Choose a base wrapper configuration compatible with the new task's observation
and action semantics. Importing task metadata or `dom_vpwem.custom_envs` itself
does not load the optional simulator stack.

## Training PPO oracles

The state PPO trainer uses the same task registry as the student evaluator,
including local variants. Its environment factory is in
[`oracle_env.py`](src/dom_vpwem/oracle_env.py), and its training loop is in
[`oracle_train.py`](src/dom_vpwem/oracle_train.py). It trains MIKASA's
`AgentStateOnly` actor/critic from privileged simulator state using
`pd_ee_delta_pose` actions and normalized dense rewards. No camera images are
fed to the oracle, but the installed ManiSkill runtime still needs working
NVIDIA/Vulkan drivers to construct the scene.

Train the InterceptFastCover oracle with:

```bash
uv run --locked --extra eval dom-vpwem-train-oracle \
  --config configs/oracles/intercept_fast_cover.yaml
```

For InterceptFastCover2, use
[`configs/oracles/intercept_fast_cover2.yaml`](configs/oracles/intercept_fast_cover2.yaml)
in the same command. It uses the same PPO settings and writes to
`outputs/oracles/intercept_fast_cover2/`.

The equivalent module command is `python -m dom_vpwem.oracle_train`, and
`scripts/train_oracle.py` is a source-tree entry point. Inspect the resolved
settings without creating a simulator or starting training:

```bash
uv run --locked --extra eval dom-vpwem-train-oracle \
  --config configs/oracles/intercept_fast_cover.yaml --dry-run
```

The supplied config uses 256 parallel environments, 60 steps per rollout,
and a maximum of 150 million transitions. The budget rounds down to complete
rollouts. PPO uses clipped policy updates, normalized finite-horizon GAE,
gradient clipping, and a KL early-stop limit for each optimization pass.
Validation runs after the first update, every 25 updates, and at the end,
using 64 episodes from a fixed seed range starting at 10,000,000. The metric
is `success_once`, latched over the task's full horizon. Training stops early
after at least 95% success in three consecutive validation rounds. This is an
oracle selection metric, not a visuomotor benchmark result.

All config fields have CLI overrides; for example, a short run to check GPU
setup and checkpoint writing (not enough to train a useful oracle):

```bash
uv run --locked --extra eval dom-vpwem-train-oracle \
  --config configs/oracles/intercept_fast_cover.yaml \
  --num-envs 16 --num-eval-envs 4 --eval-episodes 4 \
  --total-timesteps 1920 --eval-every 1 \
  --output-dir outputs/oracles/intercept_fast_cover_smoke
```

Training writes local console/JSONL metrics; it does not contact an experiment
tracking service. The default output directory is
`outputs/oracles/intercept_fast_cover/`:

| File | Purpose |
| --- | --- |
| `best_ckpt.pt` | Raw MIKASA-compatible state dict from the highest validation success rate. |
| `final_ckpt.pt` | Raw state dict at the end of training, whether or not the success threshold was reached. |
| `final_success_ckpt.pt` | Written only when the consecutive validation threshold is reached. |
| `training_state.pt` | Model, optimizer, counters, config, schema, and process RNG states for continuation. Saved at evaluations, every 25 updates, and at completion. |
| `config.json`, `observation_schema.json` | Resolved settings and ordered state input/action layout. |
| `metrics.jsonl`, checkpoint `.json` sidecars | Losses, validation episodes/seeds, throughput, and export metadata. |

Continue an interrupted run with the same config:

```bash
uv run --locked --extra eval dom-vpwem-train-oracle \
  --config configs/oracles/intercept_fast_cover.yaml \
  --resume outputs/oracles/intercept_fast_cover/training_state.pt
```

`total_timesteps` is the total target including work already completed; increase
it to extend a completed run. Resume restores optimizer and training counters
but starts fresh simulator episodes, so it is not a bitwise continuation of a
mid-episode simulation. For a weight-only warm start, use `--checkpoint` with a
raw oracle state dict and choose a new `--output-dir`. New runs refuse to
overwrite an existing nonempty run directory.

For another variant, add its task entry and environment class as described
above, then supply its `env_id` in an oracle YAML. Omit `num_steps` to use that
task's registered horizon. The shared state factory applies the base task's
curriculum action wrapper where needed, independently of the RGB student
adapter. The `oracle=False` wrapper setting removes the additional oracle
field; `obs_mode="state"` still supplies privileged information.

The raw `.pt` exports load directly into MIKASA's collector network. Matching
the state layout and control mode is necessary when transferring an oracle.
The local demonstration collector uses this same network and privileged
observation contract, with support for the custom task registry.

## Generate custom training examples

[`scripts/generate_training_examples.sh`](scripts/generate_training_examples.sh)
generates training data for all four custom environments:

```bash
# Inspect the commands without loading Python, creating environments, or training.
bash scripts/generate_training_examples.sh --dry-run

# Train missing experts and collect 250 successful episodes per task.
bash scripts/generate_training_examples.sh
```

Both InterceptFast cover variants share one state expert. The shell-game and
color-sequence tasks each use their own expert, so at most three PPO runs are
started. These use the YAMLs in `configs/oracles/`, whose default budgets are
150 million transitions per expert with success-based early stopping. Existing
exports under `outputs/oracles/` are reused; interrupted PPO runs with a
`training_state.pt` are resumed. This command generates demonstration data;
visuomotor policy training remains a separate step.

To use checkpoints you already have, supply raw MIKASA `AgentStateOnly` state
dicts. The two covers can use a compatible upstream InterceptFast expert, and
the unchanged shell task can use a ShellGameShuffleTouch expert:

```bash
bash scripts/generate_training_examples.sh --collect-only \
  --intercept-checkpoint /path/to/intercept.pt \
  --shell-checkpoint /path/to/shell.pt \
  --sequence-checkpoint /path/to/sequence.pt
```

`--collect-only` prevents PPO training. Checkpoint sidecars, when present, must
match the task and observation/action schema. Checkpoints without sidecars
are checked for network compatibility, and only successful rollouts are saved.

The default output locations match the student YAMLs:

| Environment | Dataset directory under `data_mikasa_robo/data_npz/` |
| --- | --- |
| InterceptFastCover | `intercept_fast_cover_vla_v0/` |
| InterceptFastCover2 | `intercept_fast_cover2_vla_v0/` |
| ShellGameShuffleTouchCustom | `shell_game_shuffle_touch_custom_vla_v0/` |
| RememberColorSequence3-Long | `remember_color_sequence3_long_vla_v0/` |

Each successful episode is stored as a compressed `train_data_000000.npz`,
with the `rgb`, `proprio`, and `action` arrays described below, plus rewards,
success flags, episode length, seed, environment ID, and checkpoint hash.
The collector retains the cue, blank gaps, and waiting period before the
successful action. Images and privileged expert state come from the same
simulator instance. Action labels record what reaches the controller, including
the zero actions enforced during the shell-game cue and shuffle.

Use `--episodes N` to change the total successful-episode target per task,
`--num-envs N` to change the collection batch size (default 4), and
`--data-root DIR` to change the dataset root. Collection starts at seed 100,000,
separate from oracle validation seeds. The default attempt limit is ten times
the episode target, or one full batch if larger. If an expert produces too few
successes, collection stops with an error and retains completed episodes.
`--max-attempts N` raises this total limit, including attempts from prior runs.

Rerun the same command to resume: completed datasets are validated and reused,
and partial collections continue with fresh seeds using `collection.json`.
Keep the oracle weights, seed, batch size, and simulator backend unchanged.
Use a new data root when changing these settings. A complete matching upstream
ShellGameShuffleTouch NPZ dataset can also be placed in the custom shell task's
directory for reuse. Files without task/success metadata must already be known
to contain successful demonstrations of that task. Partial external datasets
without a collection manifest cannot be extended by this command.

The script requires the MIKASA GPU simulation/rendering stack and task assets.
It uses `uv run --locked --extra eval --inexact python` by default; set
`PYTHON_BIN=/path/to/python` to use an existing environment with the project
dependencies installed. All relative paths are resolved from the repository
root. See `--help` for checkpoint paths and PPO budget/environment overrides.

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

### Collect new trajectories for the upstream tasks

For the four custom variants, use the Bash entry point above. To generate new
NPZ for the upstream tasks instead of installing the public release, run the
MIKASA PPO collector with a separately trained oracle checkpoint:

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
The local `dom_vpwem.collect_demos` collector records post-wrapper actions.

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
Custom-environment tests check lazy imports, registration, wrapper compatibility,
and dataset selection. The cover scene test checks actual geometry, absence of
collisions, and reset/step observations when CUDA is available; it is skipped
on machines without a working NVIDIA GPU.
Oracle tests exercise PPO updates and checkpoint continuation on a small CPU
test environment, verify time-limit bootstrapping and advantage boundaries,
and check compatibility with MIKASA's collector network. A separate state-oracle
simulator test is skipped when no CUDA GPU is available. CPU test runs do not
establish robotic task success.
Collection tests verify observation/action alignment, individual success
filtering, resumable writes, bounded attempts, and orchestration of the Bash
entry point. GPU tests compare the collector's privileged observations with
the PPO environment for each custom task; these also skip without CUDA.

## Attribution and license

Code in this repository is Apache-2.0. See [`NOTICE`](NOTICE) for upstream
attribution. MIKASA-Robo, ManiSkill, simulator assets, oracle checkpoints, and
trajectory datasets are separate works under their respective licenses.
