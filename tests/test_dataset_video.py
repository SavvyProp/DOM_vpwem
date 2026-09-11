from pathlib import Path

import numpy as np
import pytest

from dom_vpwem import dataset_video


def _episode(root: Path, episode_id: int, **metadata) -> Path:
    path = root / f"train_data_{episode_id}.npz"
    np.savez(
        path,
        rgb=np.full((5, 128, 128, 6), 30, dtype=np.uint8),
        proprio=np.zeros((5, 7), dtype=np.float32),
        action=np.zeros((5, 7), dtype=np.float32),
        # Language objects must never be deserialized.
        language_instruction=np.asarray(["task"], dtype=object),
        **metadata,
    )
    return path


def test_selection_ignores_failures_and_success_after_the_episode_boundary(tmp_path):
    _episode(tmp_path, 1, success_once=False)
    _episode(
        tmp_path, 2, success_once=True, success=[0, 0, 0, 0, 1], done=[0, 0, 1, 0, 0]
    )
    _episode(tmp_path, 3)
    expected = _episode(tmp_path, 10, success=[0, 0, 0, 1, 0])
    _episode(tmp_path, 11, success_once=True)

    assert dataset_video.select_episode(tmp_path) == expected
    # Even when unknown outcomes are allowed, verified success takes priority.
    assert dataset_video.select_episode(tmp_path, allow_unknown_success=True) == expected
    assert dataset_video.select_episode(tmp_path, episode_id=10) == expected
    with pytest.raises(ValueError, match="No confirmed successful"):
        dataset_video.select_episode(tmp_path, episode_id=2, allow_unknown_success=True)


def test_missing_success_needs_explicit_fallback_and_done_is_not_success(tmp_path):
    expected = _episode(tmp_path, 0, done=[0, 0, 0, 0, 1])
    _episode(tmp_path, 1, success_once=False)

    with pytest.raises(ValueError, match="--allow-unknown-success"):
        dataset_video.select_episode(tmp_path)
    assert dataset_video.select_episode(tmp_path, allow_unknown_success=True) == expected
    with pytest.raises(ValueError, match="No confirmed successful"):
        dataset_video.select_episode(tmp_path, episode_id=1, allow_unknown_success=True)
    with pytest.raises(ValueError, match="with ID 99"):
        dataset_video.select_episode(tmp_path, episode_id=99)


@pytest.mark.parametrize(
    "metadata",
    [
        {"success": [0, 0, 0, 0, np.nan]},
        {"success": True},
        {"success": [0, 1]},
        {"success_once": [True, False]},
        {"success_once": "false"},
    ],
)
def test_invalid_success_metadata_cannot_qualify_an_episode(tmp_path, metadata):
    _episode(tmp_path, 0, **metadata)
    with pytest.raises(ValueError, match="success"):
        dataset_video.select_episode(tmp_path, allow_unknown_success=True)


@pytest.mark.parametrize("status_kind", ["stepwise", "episode", "unknown"])
def test_export_preserves_trimmed_frames_seed_and_known_or_unknown_status(
    tmp_path, monkeypatch, status_kind
):
    metadata = {"episode_length": 5, "done": [0, 0, 1, 0, 0]}
    if status_kind == "stepwise":
        metadata.update(success=[0, 1, 0, 0, 0], episode_seed=100003)
    elif status_kind == "episode":
        metadata.update(success_once=True)
    _episode(tmp_path, 7, **metadata)
    frames = []
    recorder_options = {}

    class Recorder:
        def __init__(self, output, **kwargs):
            recorder_options.update(output=output, **kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def write_frame(self, observation, *, step, success):
            frames.append((observation["rgb"], step, success))

    monkeypatch.setattr(dataset_video, "Mp4RolloutRecorder", Recorder)
    output = tmp_path / "video.mp4"
    assert dataset_video.export_dataset_episode(
        tmp_path, output, fps=10, allow_unknown_success=(status_kind == "unknown")
    ) == output
    assert recorder_options == {
        "output": output, "fps": 10, "episode_index": 7, "horizon": 3,
        "seed": 100003 if status_kind == "stepwise" else None,
    }
    expected_status = {
        "stepwise": [False, True, True], "episode": [True, True, True],
        "unknown": [None, None, None],
    }[status_kind]
    assert [frame[1] for frame in frames] == [0, 1, 2]
    assert [frame[2] for frame in frames] == expected_status
    for rgb, _, _ in frames:
        np.testing.assert_array_equal(rgb, np.full((128, 128, 6), 30, dtype=np.uint8))
