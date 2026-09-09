from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dom_vpwem.dataset_installer import (
    MANIFEST_FILENAME,
    DatasetInstallError,
    dataset_path,
    install_datasets,
    resolve_tasks,
    verify_install,
)
from dom_vpwem.lerobot_import import (
    LEROBOT_REPO_ID,
    LEROBOT_REVISION,
    LeRobotImportError,
    LeRobotMetadata,
    LeRobotRow,
    write_npz_episodes,
)
from dom_vpwem.tasks import get_task_spec


def _fake_streams(value: int = 0):
    rows = [
        LeRobotRow(
            episode_index=0,
            frame_index=0,
            proprio=np.full(7, value / 100.0, dtype=np.float32),
            action=np.full(7, value / 100.0, dtype=np.float32),
        )
    ]
    top = [np.full((128, 128, 3), value, dtype=np.uint8)]
    wrist = [np.full((128, 128, 3), value + 1, dtype=np.uint8)]
    return rows, top, wrist


def _converter(value: int = 0):
    def convert(source_root, destination, task, *, progress=None):
        assert Path(source_root).name == task.dataset_slug
        rows, top, wrist = _fake_streams(value)
        metadata = LeRobotMetadata(
            env_id=task.env_id,
            codebase_version="v3.0",
            fps=10,
            total_episodes=1,
            total_frames=1,
        )
        return write_npz_episodes(
            rows,
            top,
            wrist,
            destination,
            metadata,
            progress=progress,
        )

    return convert


def _snapshot(tmp_path: Path, *tasks):
    snapshot = tmp_path / "cache" / "snapshots" / LEROBOT_REVISION
    for task in tasks:
        (snapshot / task.dataset_slug).mkdir(parents=True, exist_ok=True)
    return snapshot


def test_task_resolution_and_destination_contract(tmp_path: Path) -> None:
    published = [
        get_task_spec("ShellGameShuffleColorLampTouch-VLA-v0"),
        get_task_spec("ShellGameTouch-VLA-v0"),
    ]
    assert resolve_tasks(None) == published
    assert resolve_tasks(["all"]) == published
    assert [task.env_id for task in resolve_tasks(["touch", "shuffle", "touch"])] == [
        "ShellGameTouch-VLA-v0",
        "ShellGameShuffleColorLampTouch-VLA-v0",
    ]
    touch = get_task_spec("ShellGameTouch-VLA-v0")
    assert resolve_tasks([touch.dataset_slug]) == [touch]
    assert dataset_path(tmp_path, touch) == tmp_path / "shell_game_touch_vla_v0"

    with pytest.raises(DatasetInstallError, match="Unknown task"):
        resolve_tasks(["not-a-task"])
    with pytest.raises(DatasetInstallError, match="cannot be combined"):
        resolve_tasks(["all", "not-a-task"])


@pytest.mark.parametrize(
    "env_id",
    [
        "InterceptFastCover-VLA-v0",
        "InterceptFastCover2-VLA-v0",
        "ShellGameShuffleTouchCustom-VLA-v0",
        "RememberColorSequence3-Long-VLA-v0",
    ],
)
def test_custom_task_is_rejected_before_downloading_or_creating_directories(tmp_path, env_id):
    task = get_task_spec(env_id)
    for alias in (task.env_id, task.dataset_slug):
        with pytest.raises(DatasetInstallError, match="has no published dataset"):
            resolve_tasks([alias])
    destination = tmp_path / "custom"
    with pytest.raises(DatasetInstallError, match="has no published dataset"):
        install_datasets([task], output_root=destination)
    assert not destination.exists()


def test_install_fetches_converts_verifies_and_then_skips_without_network(
    tmp_path: Path,
) -> None:
    task = get_task_spec("ShellGameTouch-VLA-v0")
    snapshot = _snapshot(tmp_path, task)
    calls = []

    def fetch(tasks, **kwargs):
        calls.append((list(tasks), kwargs))
        return snapshot

    output_root = tmp_path / "data_npz"
    paths = install_datasets(
        [task],
        output_root=output_root,
        fetch_snapshot=fetch,
        converter=_converter(7),
        print_fn=lambda _: None,
    )

    target = output_root / task.dataset_slug
    assert paths == [target]
    assert (target / MANIFEST_FILENAME).is_file()
    manifest = verify_install(target, task, check_hashes=True)
    assert manifest["source"]["repo_id"] == LEROBOT_REPO_ID
    assert manifest["source"]["resolved_revision"] == LEROBOT_REVISION
    assert manifest["episode_count"] == 1
    assert len(calls) == 1
    assert calls[0][0] == [task]
    assert calls[0][1]["revision"] == LEROBOT_REVISION
    assert calls[0][1]["local_files_only"] is False
    assert calls[0][1]["cache_dir"] == output_root.resolve() / ".cache" / "huggingface"

    def unexpected_fetch(*args, **kwargs):
        raise AssertionError("a completed install should skip before network access")

    skipped = install_datasets(
        [task],
        output_root=output_root,
        fetch_snapshot=unexpected_fetch,
        converter=_converter(9),
        print_fn=lambda _: None,
    )
    assert skipped == [target]


def test_verify_detects_same_size_checksum_corruption(tmp_path: Path) -> None:
    task = get_task_spec("ShellGameTouch-VLA-v0")
    snapshot = _snapshot(tmp_path, task)
    target = install_datasets(
        [task],
        output_root=tmp_path / "data_npz",
        fetch_snapshot=lambda *args, **kwargs: snapshot,
        converter=_converter(3),
        print_fn=lambda _: None,
    )[0]
    episode_path = target / "train_data_000000.npz"
    contents = bytearray(episode_path.read_bytes())
    contents[len(contents) // 2] ^= 1
    episode_path.write_bytes(contents)

    with pytest.raises(DatasetInstallError, match="failed SHA-256"):
        verify_install(target, task, check_hashes=True)


def test_failed_forced_conversion_keeps_old_install_and_cleans_staging(
    tmp_path: Path,
) -> None:
    task = get_task_spec("ShellGameTouch-VLA-v0")
    snapshot = _snapshot(tmp_path, task)
    output_root = tmp_path / "data_npz"
    target = install_datasets(
        [task],
        output_root=output_root,
        fetch_snapshot=lambda *args, **kwargs: snapshot,
        converter=_converter(4),
        print_fn=lambda _: None,
    )[0]
    old_bytes = (target / "train_data_000000.npz").read_bytes()

    def fail_conversion(*args, **kwargs):
        raise LeRobotImportError("injected decoder failure")

    with pytest.raises(DatasetInstallError, match="injected decoder failure"):
        install_datasets(
            [task],
            output_root=output_root,
            force=True,
            fetch_snapshot=lambda *args, **kwargs: snapshot,
            converter=fail_conversion,
            print_fn=lambda _: None,
        )

    assert (target / "train_data_000000.npz").read_bytes() == old_bytes
    assert not list(output_root.glob(f".{task.dataset_slug}.partial-*"))
    verify_install(target, task, check_hashes=True)


def test_force_atomically_replaces_a_valid_install(tmp_path: Path) -> None:
    task = get_task_spec("ShellGameTouch-VLA-v0")
    snapshot = _snapshot(tmp_path, task)
    output_root = tmp_path / "data_npz"
    target = install_datasets(
        [task],
        output_root=output_root,
        fetch_snapshot=lambda *args, **kwargs: snapshot,
        converter=_converter(2),
        print_fn=lambda _: None,
    )[0]

    install_datasets(
        [task],
        output_root=output_root,
        force=True,
        fetch_snapshot=lambda *args, **kwargs: snapshot,
        converter=_converter(22),
        print_fn=lambda _: None,
    )

    with np.load(target / "train_data_000000.npz", allow_pickle=False) as archive:
        assert archive["rgb"][0, 0, 0].tolist() == [22, 22, 22, 23, 23, 23]
    assert not list(output_root.glob(f".{task.dataset_slug}.backup-*"))
    verify_install(target, task, check_hashes=True)


def test_unmanifested_destination_is_never_merged(tmp_path: Path) -> None:
    task = get_task_spec("ShellGameTouch-VLA-v0")
    output_root = tmp_path / "data_npz"
    target = output_root / task.dataset_slug
    target.mkdir(parents=True)
    (target / "user-file.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(DatasetInstallError, match="no .* completion marker"):
        install_datasets(
            [task],
            output_root=output_root,
            fetch_snapshot=lambda *args, **kwargs: pytest.fail("must not download"),
            converter=_converter(),
            print_fn=lambda _: None,
        )

    assert (target / "user-file.txt").read_text(encoding="utf-8") == "keep"


def test_dataset_symlink_is_refused(tmp_path: Path) -> None:
    task = get_task_spec("ShellGameTouch-VLA-v0")
    output_root = tmp_path / "data_npz"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    output_root.mkdir()
    (output_root / task.dataset_slug).symlink_to(elsewhere, target_is_directory=True)

    with pytest.raises(DatasetInstallError, match="symlink"):
        install_datasets(
            [task],
            output_root=output_root,
            fetch_snapshot=lambda *args, **kwargs: pytest.fail("must not download"),
            converter=_converter(),
            print_fn=lambda _: None,
        )


def test_output_root_symlink_is_refused_before_any_write(tmp_path: Path) -> None:
    task = get_task_spec("ShellGameTouch-VLA-v0")
    real_root = tmp_path / "real-root"
    real_root.mkdir()
    linked_root = tmp_path / "linked-root"
    linked_root.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(DatasetInstallError, match="output root cannot be a symlink"):
        install_datasets(
            [task],
            output_root=linked_root,
            fetch_snapshot=lambda *args, **kwargs: pytest.fail("must not download"),
            converter=_converter(),
            print_fn=lambda _: None,
        )

    assert not list(real_root.iterdir())
