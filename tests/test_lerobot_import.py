from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from dom_vpwem.data import load_npz_episode
from dom_vpwem.lerobot_import import (
    LEROBOT_REVISION,
    DatasetDependencyError,
    LeRobotImportError,
    LeRobotMetadata,
    LeRobotRow,
    download_lerobot_snapshot,
    load_lerobot_metadata,
    sha256_file,
    write_npz_episodes,
)
from dom_vpwem.tasks import get_task_spec


def _metadata(*lengths: int) -> LeRobotMetadata:
    return LeRobotMetadata(
        env_id="ShellGameTouch-VLA-v0",
        codebase_version="v3.0",
        fps=10,
        total_episodes=len(lengths),
        total_frames=sum(lengths),
    )


def _streams(*lengths: int, base: float = 0.0):
    rows = []
    top = []
    wrist = []
    global_index = 0
    for episode_index, length in enumerate(lengths):
        for frame_index in range(length):
            value = base + global_index
            rows.append(
                LeRobotRow(
                    episode_index=episode_index,
                    frame_index=frame_index,
                    proprio=np.full(7, value / 100.0, dtype=np.float32),
                    action=np.full(7, value / 100.0, dtype=np.float32),
                )
            )
            top.append(np.full((128, 128, 3), global_index + 1, dtype=np.uint8))
            wrist.append(np.full((128, 128, 3), global_index + 101, dtype=np.uint8))
            global_index += 1
    return rows, top, wrist


def _write_source_metadata(root: Path, *, env_id: str = "ShellGameTouch-VLA-v0") -> None:
    features = {
        "observation.images.top": {"dtype": "video", "shape": [128, 128, 3]},
        "observation.images.wrist": {"dtype": "video", "shape": [128, 128, 3]},
        "observation.state": {"dtype": "float32", "shape": [7]},
        "action": {"dtype": "float32", "shape": [7]},
    }
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "robot_type": "mikasa_robo",
                "total_episodes": 2,
                "total_frames": 3,
                "fps": 10,
                "features": features,
            }
        ),
        encoding="utf-8",
    )
    (root / "source_rlds_metadata.json").write_text(
        json.dumps(
            {
                "env_id": env_id,
                "num_episodes": 2,
                "episode_lengths": [2, 1],
            }
        ),
        encoding="utf-8",
    )


def test_source_metadata_is_validated_against_the_registered_task(tmp_path: Path) -> None:
    task = get_task_spec("ShellGameTouch-VLA-v0")
    _write_source_metadata(tmp_path)

    metadata = load_lerobot_metadata(tmp_path, task)

    assert metadata == _metadata(2, 1)


def test_source_metadata_rejects_a_different_environment(tmp_path: Path) -> None:
    task = get_task_spec("ShellGameTouch-VLA-v0")
    _write_source_metadata(tmp_path, env_id="ShellGamePush-VLA-v0")

    with pytest.raises(LeRobotImportError, match="expected env_id"):
        load_lerobot_metadata(tmp_path, task)


def test_source_length_order_does_not_override_lerobot_episode_indices(
    tmp_path: Path,
) -> None:
    task = get_task_spec("ShellGameTouch-VLA-v0")
    source = tmp_path / "source"
    _write_source_metadata(source)  # Source lengths are [2, 1].
    metadata = load_lerobot_metadata(source, task)
    rows, top, wrist = _streams(1, 2)  # LeRobot order is [1, 2].

    result = write_npz_episodes(rows, top, wrist, tmp_path / "converted", metadata)

    assert [episode.length for episode in result.episodes] == [1, 2]


def test_snapshot_download_uses_exact_task_pattern_and_standard_http(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = get_task_spec("ShellGameTouch-VLA-v0")
    snapshot = tmp_path / LEROBOT_REVISION
    snapshot.mkdir()
    calls = []

    class FakeHub:
        @staticmethod
        def snapshot_download(**kwargs):
            calls.append(kwargs)
            return str(snapshot)

    def fake_import(name: str):
        return FakeHub if name == "huggingface_hub" else object()

    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)
    monkeypatch.setattr("dom_vpwem.lerobot_import.importlib.import_module", fake_import)

    result = download_lerobot_snapshot([task], cache_dir=tmp_path / "cache")

    assert result == snapshot
    assert calls[0]["revision"] == LEROBOT_REVISION
    assert calls[0]["allow_patterns"] == ["shell_game_touch_vla_v0/**"]
    assert calls[0]["cache_dir"] == str(tmp_path / "cache")
    assert calls[0]["repo_type"] == "dataset"
    assert os.environ["HF_HUB_DISABLE_XET"] == "1"


def test_snapshot_download_fails_before_network_when_data_extra_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = get_task_spec("ShellGameTouch-VLA-v0")

    def fake_import(name: str):
        if name in {"pyarrow.parquet", "av"}:
            raise ImportError(name)
        return type("FakeHub", (), {"snapshot_download": pytest.fail})

    monkeypatch.setattr("dom_vpwem.lerobot_import.importlib.import_module", fake_import)

    with pytest.raises(DatasetDependencyError, match="pyarrow.parquet, av"):
        download_lerobot_snapshot([task])


def test_streaming_conversion_writes_loader_compatible_episodes(tmp_path: Path) -> None:
    rows, top, wrist = _streams(2, 1)
    progress = []

    result = write_npz_episodes(
        rows,
        top,
        wrist,
        tmp_path / "converted",
        _metadata(2, 1),
        progress=lambda completed, total: progress.append((completed, total)),
    )

    assert progress == [(1, 2), (2, 2)]
    assert [episode.filename for episode in result.episodes] == [
        "train_data_000000.npz",
        "train_data_000001.npz",
    ]
    assert result.total_frames == 3

    first_path = tmp_path / "converted" / "train_data_000000.npz"
    first = load_npz_episode(first_path)
    assert first.rgb.shape == (2, 128, 128, 6)
    assert first.rgb.dtype == np.uint8
    assert first.proprio.dtype == np.float32
    assert first.action.dtype == np.float32
    assert first.rgb[:, 0, 0, :3].tolist() == [[1, 1, 1], [2, 2, 2]]
    assert first.rgb[:, 0, 0, 3:].tolist() == [[101, 101, 101], [102, 102, 102]]
    with np.load(first_path, allow_pickle=False) as archive:
        assert archive["done"].tolist() == [False, True]
        assert archive["episode_length"].item() == 2

    assert result.episodes[0].sha256 == sha256_file(first_path)
    assert result.episodes[0].size_bytes == first_path.stat().st_size


@pytest.mark.parametrize("missing_camera", ["top", "wrist"])
def test_conversion_rejects_a_short_video_stream(
    tmp_path: Path, missing_camera: str
) -> None:
    rows, top, wrist = _streams(2)
    if missing_camera == "top":
        top.pop()
    else:
        wrist.pop()

    with pytest.raises(LeRobotImportError, match=rf"{missing_camera} video stream ended"):
        write_npz_episodes(rows, top, wrist, tmp_path / missing_camera, _metadata(2))


@pytest.mark.parametrize("extra_camera", ["top", "wrist"])
def test_conversion_rejects_an_extra_video_frame(
    tmp_path: Path, extra_camera: str
) -> None:
    rows, top, wrist = _streams(1)
    extra = np.zeros((128, 128, 3), dtype=np.uint8)
    if extra_camera == "top":
        top.append(extra)
    else:
        wrist.append(extra)

    with pytest.raises(LeRobotImportError, match=rf"{extra_camera} video stream contains more"):
        write_npz_episodes(rows, top, wrist, tmp_path / extra_camera, _metadata(1))


@pytest.mark.parametrize(
    ("rows", "lengths", "message"),
    [
        (
            [
                LeRobotRow(0, 1, np.zeros(7), np.zeros(7)),
            ],
            (1,),
            "expected frame_index 0",
        ),
        (
            [
                LeRobotRow(1, 0, np.zeros(7), np.zeros(7)),
            ],
            (1,),
            "Expected episode_index 0",
        ),
        (
            [
                LeRobotRow(0, 0, np.zeros(7), np.full(7, 1.1)),
            ],
            (1,),
            r"outside the normalized \[-1, 1\]",
        ),
        (
            [
                LeRobotRow(0, 0, np.full(7, np.nan), np.zeros(7)),
            ],
            (1,),
            "NaN or infinity",
        ),
    ],
)
def test_conversion_rejects_misaligned_or_invalid_rows(
    tmp_path: Path,
    rows: list[LeRobotRow],
    lengths: tuple[int, ...],
    message: str,
) -> None:
    frame = np.zeros((128, 128, 3), dtype=np.uint8)
    frames = [frame for _ in rows]

    with pytest.raises(LeRobotImportError, match=message):
        write_npz_episodes(
            rows,
            frames,
            frames,
            tmp_path / f"bad-{message[:4]}",
            _metadata(*lengths),
        )


def test_conversion_rejects_declared_frame_total_mismatch(tmp_path: Path) -> None:
    rows, top, wrist = _streams(2)
    metadata = LeRobotMetadata(
        env_id="ShellGameTouch-VLA-v0",
        codebase_version="v3.0",
        fps=10,
        total_episodes=1,
        total_frames=1,
    )

    with pytest.raises(LeRobotImportError, match="Converted 2 frames, expected 1"):
        write_npz_episodes(rows, top, wrist, tmp_path / "bad-length", metadata)


def test_conversion_rejects_an_episode_beyond_metadata(tmp_path: Path) -> None:
    rows, top, wrist = _streams(1, 1)

    with pytest.raises(LeRobotImportError, match="exceeds the declared 1 episodes"):
        write_npz_episodes(rows, top, wrist, tmp_path / "extra-episode", _metadata(1))


def test_conversion_rejects_internally_inconsistent_metadata(tmp_path: Path) -> None:
    rows, top, wrist = _streams(1)
    metadata = LeRobotMetadata(
        env_id="ShellGameTouch-VLA-v0",
        codebase_version="v3.0",
        fps=10,
        total_episodes=2,
        total_frames=1,
    )

    with pytest.raises(LeRobotImportError, match="Converted 1 episodes, expected 2"):
        write_npz_episodes(rows, top, wrist, tmp_path / "bad-metadata", metadata)
    download_lerobot_snapshot,
