#!/usr/bin/env bash
# Generate successful demonstrations for all four repository-owned tasks.
set -euo pipefail

usage() {
    cat <<'HELP'
Usage: bash scripts/generate_training_examples.sh [options]

Generate datasets for InterceptFastCover, InterceptFastCover2,
ShellGameShuffleTouchCustom, and RememberColorSequence3-Long.
One InterceptFast oracle serves both covers. Missing experts are trained
using the existing PPO configs; --collect-only requires existing experts.
This generates data only; it does not train visuomotor/student policies.

Options:
  --episodes N                 Target successful episodes per task (default: 250)
  --num-envs N                 Parallel collection environments (default: 4)
  --seed N                     First collection seed (default: 100000)
  --max-attempts N             Total attempts per task, including prior runs
                               (default: max(10 * episodes, num-envs))
  --data-root DIR              NPZ root (default: data_mikasa_robo/data_npz)
  --oracle-root DIR            Oracle root (default: outputs/oracles)
  --intercept-checkpoint FILE  Existing state oracle for BOTH cover variants
  --shell-checkpoint FILE      Existing ShellGameShuffleTouch state oracle
  --sequence-checkpoint FILE   Existing RememberColorSequence3-Long state oracle
  --oracle-timesteps N         Override the training budget for missing experts
  --oracle-num-envs N          Override parallel environments for PPO training
  --collect-only              Fail if a required oracle checkpoint is missing
  --dry-run                   Print the plan without starting Python or training
  -h, --help                  Show this help

Complete, validated datasets are reused. Interrupted collections resume with
new seeds and never overwrite saved episodes. Keep seed, num-envs, and oracle
weights unchanged when resuming. Use a new data root for a new experiment.
An existing complete upstream ShellGameShuffleTouch NPZ dataset may be placed
in the configured shell_game_shuffle_touch_custom_vla_v0 directory for reuse.

By default commands use: uv run --locked --extra eval --inexact python
Set PYTHON_BIN to a Python executable with the project dependencies installed
to use an existing environment instead. Relative paths are rooted at this repo.
HELP
}

die() { printf 'Error: %s\n' "$*" >&2; exit 2; }
need_value() { [[ $# -ge 2 && -n "$2" ]] || die "Missing value for $1"; }

EPISODES=250
NUM_ENVS=4
SEED=100000
MAX_ATTEMPTS=""
DATA_ROOT="data_mikasa_robo/data_npz"
ORACLE_ROOT="outputs/oracles"
INTERCEPT_CHECKPOINT=""
SHELL_CHECKPOINT=""
SEQUENCE_CHECKPOINT=""
ORACLE_TIMESTEPS=""
ORACLE_NUM_ENVS=""
COLLECT_ONLY=0
DRY_RUN=0

while (($#)); do
    case "$1" in
        --episodes) need_value "$@"; EPISODES="$2"; shift 2 ;;
        --num-envs) need_value "$@"; NUM_ENVS="$2"; shift 2 ;;
        --seed) need_value "$@"; SEED="$2"; shift 2 ;;
        --max-attempts) need_value "$@"; MAX_ATTEMPTS="$2"; shift 2 ;;
        --data-root) need_value "$@"; DATA_ROOT="$2"; shift 2 ;;
        --oracle-root) need_value "$@"; ORACLE_ROOT="$2"; shift 2 ;;
        --intercept-checkpoint) need_value "$@"; INTERCEPT_CHECKPOINT="$2"; shift 2 ;;
        --shell-checkpoint) need_value "$@"; SHELL_CHECKPOINT="$2"; shift 2 ;;
        --sequence-checkpoint) need_value "$@"; SEQUENCE_CHECKPOINT="$2"; shift 2 ;;
        --oracle-timesteps) need_value "$@"; ORACLE_TIMESTEPS="$2"; shift 2 ;;
        --oracle-num-envs) need_value "$@"; ORACLE_NUM_ENVS="$2"; shift 2 ;;
        --collect-only) COLLECT_ONLY=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "Unknown option: $1 (see --help)" ;;
    esac
done

for number in "$EPISODES" "$NUM_ENVS" "${MAX_ATTEMPTS:-1}" "${ORACLE_TIMESTEPS:-1}" "${ORACLE_NUM_ENVS:-1}"; do
    [[ "$number" =~ ^[1-9][0-9]*$ ]] || die "Counts must be positive integers: $number"
done
[[ "$SEED" =~ ^(0|[1-9][0-9]*)$ ]] || die "Seed must be a nonnegative integer"

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON=("$PYTHON_BIN")
else
    UV_BIN="${UV_BIN:-uv}"
    if ! command -v "$UV_BIN" >/dev/null 2>&1 && [[ -x "$HOME/.local/bin/uv" ]]; then
        UV_BIN="$HOME/.local/bin/uv"
    fi
    PYTHON=("$UV_BIN" run --locked --extra eval --inexact python)
fi

run() {
    printf '+ '
    printf '%q ' "$@"
    printf '\n'
    if (( ! DRY_RUN )); then "$@"; fi
}

complete_dataset() {
    if (( DRY_RUN )); then return 1; fi
    local status
    if "${PYTHON[@]}" -m dom_vpwem.collect_demos --status \
        --env-id "$1" --data-root "$DATA_ROOT" --episodes "$EPISODES"; then
        return 0
    else
        status=$?
    fi
    [[ "$status" == 1 ]] || die "Existing dataset validation failed for $1"
    return 1
}

find_checkpoint() {
    RESOLVED_CHECKPOINT=""
    local filename
    for filename in final_success_ckpt.pt best_ckpt.pt final_ckpt.pt; do
        if [[ -f "$1/$filename" ]]; then
            RESOLVED_CHECKPOINT="$1/$filename"
            return 0
        fi
    done
    return 1
}

prepare_oracle() {
    local config_name="$1" supplied="$2" output_dir="$ORACLE_ROOT/$1"
    if [[ -n "$supplied" ]]; then
        [[ -f "$supplied" ]] || die "Checkpoint does not exist: $supplied"
        RESOLVED_CHECKPOINT="$supplied"
        return
    fi
    # A best checkpoint can exist after just one PPO update. Resume an
    # interrupted run before using it unless collection-only was requested.
    local interrupted=0
    if [[ -f "$output_dir/training_state.pt" && ! -f "$output_dir/final_ckpt.pt" && ! -f "$output_dir/final_success_ckpt.pt" ]]; then
        interrupted=1
    fi
    if (( ! interrupted || COLLECT_ONLY )) && find_checkpoint "$output_dir"; then return; fi
    (( ! COLLECT_ONLY )) || die "No oracle in $output_dir; supply a checkpoint or allow training"
    local command=("${PYTHON[@]}" -m dom_vpwem.oracle_train
        --config "configs/oracles/$config_name.yaml" --output-dir "$output_dir")
    if [[ -f "$output_dir/training_state.pt" ]]; then
        command+=(--resume "$output_dir/training_state.pt")
    fi
    if [[ -n "$ORACLE_TIMESTEPS" ]]; then command+=(--total-timesteps "$ORACLE_TIMESTEPS"); fi
    if [[ -n "$ORACLE_NUM_ENVS" ]]; then command+=(--num-envs "$ORACLE_NUM_ENVS"); fi
    run "${command[@]}"
    if (( DRY_RUN )); then
        RESOLVED_CHECKPOINT="$output_dir/best_ckpt.pt"
    else
        find_checkpoint "$output_dir" || die "PPO did not produce an oracle in $output_dir"
    fi
}

ENV_IDS=(
    InterceptFastCover-VLA-v0
    InterceptFastCover2-VLA-v0
    ShellGameShuffleTouchCustom-VLA-v0
    RememberColorSequence3-Long-VLA-v0
)
ORACLE_CONFIGS=(intercept_fast_cover intercept_fast_cover shell_game_shuffle_touch remember_color_sequence3_long)
SUPPLIED=("$INTERCEPT_CHECKPOINT" "$INTERCEPT_CHECKPOINT" "$SHELL_CHECKPOINT" "$SEQUENCE_CHECKPOINT")
declare -A CHECKPOINT_CACHE=()

for index in "${!ENV_IDS[@]}"; do
    env_id="${ENV_IDS[$index]}"
    if complete_dataset "$env_id"; then continue; fi
    oracle="${ORACLE_CONFIGS[$index]}"
    if [[ -z "${CHECKPOINT_CACHE[$oracle]:-}" ]]; then
        prepare_oracle "$oracle" "${SUPPLIED[$index]}"
        CHECKPOINT_CACHE[$oracle]="$RESOLVED_CHECKPOINT"
    fi
    command=("${PYTHON[@]}" -m dom_vpwem.collect_demos
        --env-id "$env_id" --checkpoint "${CHECKPOINT_CACHE[$oracle]}"
        --data-root "$DATA_ROOT" --episodes "$EPISODES" --num-envs "$NUM_ENVS" --seed "$SEED")
    if [[ -n "$MAX_ATTEMPTS" ]]; then command+=(--max-attempts "$MAX_ATTEMPTS"); fi
    run "${command[@]}"
done

if (( DRY_RUN )); then
    printf 'Dry run complete. No training or collection was started.\n'
else
    printf 'All four datasets are ready under %s\n' "$DATA_ROOT"
fi
