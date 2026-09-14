import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/generate_training_examples.sh"


def commands(output):
    return [shlex.split(line[2:]) for line in output.splitlines() if line.startswith("+ ")]


@pytest.mark.parametrize("intercepts_only", [False, True])
def test_dry_run_plans_separate_experts_for_the_different_cover_start_poses(
    tmp_path, intercepts_only
):
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--dry-run",
            "--episodes",
            "17",
            "--oracle-root",
            str(tmp_path / "oracles"),
            "--data-root",
            str(tmp_path / "data with spaces"),
            *(["--intercepts-only"] if intercepts_only else []),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    plan = commands(result.stdout)
    training = [cmd for cmd in plan if "dom_vpwem.oracle_train" in cmd]
    collection = [cmd for cmd in plan if "dom_vpwem.collect_demos" in cmd]
    expected_count = 2 if intercepts_only else 3
    assert len(training) == expected_count
    assert len(collection) == expected_count
    downloads = [cmd for cmd in plan if "dom_vpwem.dataset_installer" in cmd]
    assert len(downloads) == (0 if intercepts_only else 1)
    if downloads:
        (cmd,) = downloads
        assert cmd[cmd.index("--task") + 1] == "ShellGameShuffleTouchCustom-VLA-v0"
        assert cmd[cmd.index("--output-root") + 1] == str(tmp_path / "data with spaces")
    assert all("ShellGameShuffleTouchCustom-VLA-v0" not in cmd for cmd in collection)
    paths = [cmd[cmd.index("--checkpoint") + 1] for cmd in collection]
    assert paths[0] != paths[1]
    assert paths[1] == str(
        tmp_path / "oracles/intercept_fast_cover2/collision_cue5_v1/best_ckpt.pt"
    )
    assert paths[0] == str(tmp_path / "oracles/intercept_fast_cover/collision_cue5_v1/best_ckpt.pt")
    if not intercepts_only:
        assert paths[2] == str(
            tmp_path / "oracles/remember_color_sequence3_long/wait_for_choices_v1/best_ckpt.pt"
        )
    # Direct PPO commands and the Bash entry point must use the same run layout.
    for cmd in training:
        config = SCRIPT.parent.parent / cmd[cmd.index("--config") + 1]
        default_output = Path(yaml.safe_load(config.read_text())["output_dir"])
        expected_output = tmp_path / "oracles" / default_output.relative_to("outputs/oracles")
        assert cmd[cmd.index("--output-dir") + 1] == str(expected_output)
    if intercepts_only:
        assert all("InterceptFastCover" in cmd[cmd.index("--env-id") + 1] for cmd in collection)
    for cmd in collection:
        assert cmd[cmd.index("--episodes") + 1] == "17"
        assert cmd[cmd.index("--data-root") + 1] == str(tmp_path / "data with spaces")
    assert not (tmp_path / "data with spaces").exists()
    assert not (tmp_path / "oracles").exists()


def test_supplied_experts_skip_training_and_missing_collect_only_fails(tmp_path):
    paths = [tmp_path / f"{name} expert.pt" for name in ("intercept", "shell", "sequence")]
    for path in paths:
        path.write_bytes(b"weights")
    args = ["bash", str(SCRIPT), "--dry-run", "--collect-only"]
    for flag, path in zip(
        ("--intercept-checkpoint", "--shell-checkpoint", "--sequence-checkpoint"), paths
    ):
        args.extend((flag, str(path)))
    result = subprocess.run(args, check=True, capture_output=True, text=True)
    assert len(commands(result.stdout)) == 4
    assert "dom_vpwem.oracle_train" not in result.stdout
    # An explicitly supplied shared expert remains supported for compatibility.
    plan = commands(result.stdout)
    assert plan[0][plan[0].index("--checkpoint") + 1] == str(paths[0])
    assert plan[1][plan[1].index("--checkpoint") + 1] == str(paths[0])
    cover2 = tmp_path / "cover2 expert.pt"
    cover2.write_bytes(b"different weights")
    result = subprocess.run(
        args + ["--intercept2-checkpoint", str(cover2)],
        check=True,
        capture_output=True,
        text=True,
    )
    plan = commands(result.stdout)
    assert plan[0][plan[0].index("--checkpoint") + 1] == str(paths[0])
    assert plan[1][plan[1].index("--checkpoint") + 1] == str(cover2)
    failed = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--dry-run",
            "--collect-only",
            "--oracle-root",
            str(tmp_path / "missing"),
        ],
        capture_output=True,
        text=True,
    )
    assert failed.returncode == 2
    assert "No oracle" in failed.stderr


@pytest.mark.parametrize("intercepts_only", [False, True])
def test_script_ignores_old_task_artifacts_and_reuses_new_completed_runs(
    tmp_path, intercepts_only
):
    runner = tmp_path / "fake python"
    log = tmp_path / "commands.jsonl"
    runner.write_text(
        "#!"
        + sys.executable
        + "\n"
        + """
import json, os, pathlib, sys
from dom_vpwem.tasks import get_task_spec
args = sys.argv[1:]
with open(os.environ["PIPELINE_TEST_LOG"], "a") as stream:
    stream.write(json.dumps(args) + "\\n")
def value(flag):
    return args[args.index(flag) + 1]
if "dom_vpwem.oracle_train" in args:
    output = pathlib.Path(value("--output-dir"))
    output.mkdir(parents=True, exist_ok=True)
    (output / "final_success_ckpt.pt").write_bytes(b"oracle")
else:
    downloading = "dom_vpwem.dataset_installer" in args
    output = pathlib.Path(value("--output-root" if downloading else "--data-root"))
    task = get_task_spec(value("--task" if downloading else "--env-id"))
    marker = output / task.dataset_slug / ".test_complete"
    if "--status" in args:
        sys.exit(0 if marker.exists() else 1)
    if not downloading:
        assert pathlib.Path(value("--checkpoint")).is_file()
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("complete")
"""
    )
    runner.chmod(0o755)
    args = [
        "bash",
        str(SCRIPT),
        "--oracle-root",
        str(tmp_path / "oracle outputs"),
        "--data-root",
        str(tmp_path / "demo outputs"),
        *(["--intercepts-only"] if intercepts_only else []),
    ]
    old_files = []
    for name, revisions in (
        ("intercept_fast_cover", ("", "fixed_start_v1")),
        ("intercept_fast_cover2", ("", "fixed_start_v1")),
        ("remember_color_sequence3_long", ("",)),
    ):
        for revision in revisions:
            for filename in ("final_success_ckpt.pt", "training_state.pt"):
                old_files.append(tmp_path / "oracle outputs" / name / revision / filename)
            suffix = f"_{revision}" if revision else ""
            old_files.append(
                tmp_path / "demo outputs" / f"{name}_vla_v0{suffix}" / ".test_complete"
            )
    for path in old_files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"old task artifact")
    env = {**os.environ, "PYTHON_BIN": str(runner), "PIPELINE_TEST_LOG": str(log)}
    subprocess.run(args, env=env, cwd=tmp_path, check=True, capture_output=True, text=True)
    first = [json.loads(line) for line in log.read_text().splitlines()]
    expected_count = 2 if intercepts_only else 3
    assert sum("dom_vpwem.oracle_train" in cmd for cmd in first) == expected_count
    assert (
        sum("dom_vpwem.collect_demos" in cmd and "--status" not in cmd for cmd in first)
        == expected_count
    )
    assert sum("dom_vpwem.dataset_installer" in cmd for cmd in first) == (
        0 if intercepts_only else 1
    )
    assert all("--resume" not in cmd for cmd in first)  # Never resume an earlier task version.
    for path in old_files:
        assert path.read_bytes() == b"old task artifact"
    subprocess.run(args, env=env, cwd=tmp_path, check=True, capture_output=True, text=True)
    all_calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(all_calls) == len(first) + (2 if intercepts_only else 4)
    assert all("--status" in cmd for cmd in all_calls[len(first) :])


def test_interrupted_ppo_run_resumes_even_when_best_checkpoint_exists(tmp_path):
    output = tmp_path / "oracles" / "intercept_fast_cover" / "collision_cue5_v1"
    output.mkdir(parents=True)
    (output / "best_ckpt.pt").write_bytes(b"early checkpoint")
    (output / "training_state.pt").write_bytes(b"resume")
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--dry-run",
            "--oracle-root",
            str(tmp_path / "oracles"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--resume" in commands(result.stdout)[0]


def test_collect_only_can_download_shell_without_a_shell_checkpoint(tmp_path):
    checkpoint = tmp_path / "expert.pt"
    checkpoint.write_bytes(b"expert")
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--dry-run",
            "--collect-only",
            "--intercept-checkpoint",
            str(checkpoint),
            "--sequence-checkpoint",
            str(checkpoint),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    plan = commands(result.stdout)
    assert len(plan) == 4
    assert sum("dom_vpwem.dataset_installer" in cmd for cmd in plan) == 1
    assert not any("dom_vpwem.oracle_train" in cmd for cmd in plan)


def test_shell_download_failure_never_falls_back_to_ppo(tmp_path):
    runner = tmp_path / "fake python"
    log = tmp_path / "calls.jsonl"
    runner.write_text(
        "#!"
        + sys.executable
        + "\n"
        + """
import json, os, sys
args = sys.argv[1:]
with open(os.environ["PIPELINE_TEST_LOG"], "a") as stream:
    stream.write(json.dumps(args) + "\\n")
if "--status" in args:
    sys.exit(1 if "ShellGameShuffleTouchCustom-VLA-v0" in args else 0)
if "dom_vpwem.dataset_installer" in args:
    sys.exit(29)
raise AssertionError("Must not collect or train a shell oracle after download failure")
"""
    )
    runner.chmod(0o755)
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        env={**os.environ, "PYTHON_BIN": str(runner), "PIPELINE_TEST_LOG": str(log)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 29
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert "dom_vpwem.dataset_installer" in calls[-1]
    assert not any("dom_vpwem.oracle_train" in cmd for cmd in calls)


def test_larger_shell_dataset_requires_explicit_local_collection(tmp_path):
    args = [
        "bash",
        str(SCRIPT),
        "--dry-run",
        "--episodes",
        "251",
        "--oracle-root",
        str(tmp_path / "oracles"),
    ]
    rejected = subprocess.run(args, capture_output=True, text=True)
    assert rejected.returncode == 2
    assert "public shell dataset has 250 episodes" in rejected.stderr
    assert not commands(rejected.stdout)  # Reject before any expensive training starts.
    subprocess.run(args + ["--intercepts-only"], check=True, capture_output=True, text=True)
    local = subprocess.run(
        args + ["--shell-source", "collect"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "configs/oracles/shell_game_shuffle_touch.yaml" in local.stdout
    assert "dom_vpwem.dataset_installer" not in local.stdout
