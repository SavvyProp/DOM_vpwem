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
    expected_count = 2 if intercepts_only else 4
    assert len(training) == expected_count
    assert len(collection) == expected_count
    paths = [cmd[cmd.index("--checkpoint") + 1] for cmd in collection]
    assert paths[0] != paths[1]
    assert paths[1] == str(tmp_path / "oracles/intercept_fast_cover2/fixed_start_v1/best_ckpt.pt")
    assert paths[0] == str(tmp_path / "oracles/intercept_fast_cover/fixed_start_v1/best_ckpt.pt")
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
def test_script_ignores_old_intercept_artifacts_and_reuses_new_completed_runs(
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
    output = pathlib.Path(value("--data-root"))
    marker = output / get_task_spec(value("--env-id")).dataset_slug / ".test_complete"
    if "--status" in args:
        sys.exit(0 if marker.exists() else 1)
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
    for name in ("intercept_fast_cover", "intercept_fast_cover2"):
        for filename in ("final_success_ckpt.pt", "training_state.pt"):
            old_files.append(tmp_path / "oracle outputs" / name / filename)
        old_files.append(tmp_path / "demo outputs" / f"{name}_vla_v0" / ".test_complete")
    for path in old_files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"old pose artifact")
    env = {**os.environ, "PYTHON_BIN": str(runner), "PIPELINE_TEST_LOG": str(log)}
    subprocess.run(args, env=env, cwd=tmp_path, check=True, capture_output=True, text=True)
    first = [json.loads(line) for line in log.read_text().splitlines()]
    expected_count = 2 if intercepts_only else 4
    assert sum("dom_vpwem.oracle_train" in cmd for cmd in first) == expected_count
    assert (
        sum("dom_vpwem.collect_demos" in cmd and "--status" not in cmd for cmd in first)
        == expected_count
    )
    assert all("--resume" not in cmd for cmd in first)  # Never resume the old arm-pose run.
    for path in old_files:
        assert path.read_bytes() == b"old pose artifact"
    subprocess.run(args, env=env, cwd=tmp_path, check=True, capture_output=True, text=True)
    all_calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(all_calls) == len(first) + expected_count
    assert all("--status" in cmd for cmd in all_calls[len(first) :])


def test_interrupted_ppo_run_resumes_even_when_best_checkpoint_exists(tmp_path):
    output = tmp_path / "oracles" / "intercept_fast_cover" / "fixed_start_v1"
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
