import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/generate_training_examples.sh"


def commands(output):
    return [shlex.split(line[2:]) for line in output.splitlines() if line.startswith("+ ")]


def test_dry_run_plans_three_experts_and_four_datasets_with_shared_intercept(tmp_path):
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
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    plan = commands(result.stdout)
    training = [cmd for cmd in plan if "dom_vpwem.oracle_train" in cmd]
    collection = [cmd for cmd in plan if "dom_vpwem.collect_demos" in cmd]
    assert len(training) == 3
    assert len(collection) == 4
    paths = [cmd[cmd.index("--checkpoint") + 1] for cmd in collection]
    assert paths[0] == paths[1]
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


def test_script_executes_pipeline_and_reuses_complete_datasets(tmp_path):
    runner = tmp_path / "fake python"
    log = tmp_path / "commands.jsonl"
    runner.write_text(
        "#!"
        + sys.executable
        + "\n"
        + """
import json, os, pathlib, sys
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
    marker = output / value("--env-id")
    if "--status" in args:
        sys.exit(0 if marker.exists() else 1)
    assert pathlib.Path(value("--checkpoint")).is_file()
    output.mkdir(parents=True, exist_ok=True)
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
    ]
    env = {**os.environ, "PYTHON_BIN": str(runner), "PIPELINE_TEST_LOG": str(log)}
    subprocess.run(args, env=env, cwd=tmp_path, check=True, capture_output=True, text=True)
    first = [json.loads(line) for line in log.read_text().splitlines()]
    assert sum("dom_vpwem.oracle_train" in cmd for cmd in first) == 3
    assert sum("dom_vpwem.collect_demos" in cmd and "--status" not in cmd for cmd in first) == 4
    subprocess.run(args, env=env, cwd=tmp_path, check=True, capture_output=True, text=True)
    all_calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(all_calls) == len(first) + 4
    assert all("--status" in cmd for cmd in all_calls[len(first) :])


def test_interrupted_ppo_run_resumes_even_when_best_checkpoint_exists(tmp_path):
    output = tmp_path / "oracles" / "intercept_fast_cover"
    output.mkdir(parents=True)
    (output / "best_ckpt.pt").write_bytes(b"early checkpoint")
    (output / "training_state.pt").write_bytes(b"resume")
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--dry-run",
            "--oracle-root",
            str(output.parent),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--resume" in commands(result.stdout)[0]
