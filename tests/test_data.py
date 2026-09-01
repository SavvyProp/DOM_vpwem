from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from dom_vpwem.data import (
    DatasetFormatError,
    MikasaNpzDataset,
    discover_npz_episodes,
    load_npz_episode,
)


def _episode_arrays(length: int, *, base: int = 0, height: int = 2, width: int = 3):
    rgb = np.empty((length, height, width, 6), dtype=np.uint8)
    proprio = np.empty((length, 7), dtype=np.float32)
    action = np.empty((length, 7), dtype=np.float32)
    for timestep in range(length):
        rgb[timestep].fill(base + timestep)
        proprio[timestep].fill(base + timestep)
        action[timestep].fill(base + timestep)
    return {"rgb": rgb, "proprio": proprio, "action": action}


def _write_episode(
    root: Path,
    name: str,
    *,
    length: int = 6,
    base: int = 0,
    **metadata,
) -> Path:
    path = root / name
    arrays = _episode_arrays(length, base=base)
    # Object-typed language data would fail if the loader tried to deserialize
    # it under allow_pickle=False.  Its presence verifies that language is
    # genuinely ignored rather than merely dropped after loading.
    arrays["language_instruction"] = np.asarray(["track the cups"], dtype=object)
    arrays.update(metadata)
    np.savez(path, **arrays)
    return path


def test_discovery_sorts_padded_and_unpadded_indices(tmp_path: Path) -> None:
    np.savez(tmp_path / "train_data_10.npz", marker=10)
    np.savez(tmp_path / "train_data_000002.npz", marker=2)
    np.savez(tmp_path / "train_data_1.npz", marker=1)
    np.savez(tmp_path / "not_an_episode.npz", marker=-1)
    (tmp_path / "train_data_bad.npz").write_bytes(b"ignored")

    discovered = discover_npz_episodes(tmp_path)

    assert [path.name for path in discovered] == [
        "train_data_1.npz",
        "train_data_000002.npz",
        "train_data_10.npz",
    ]


def test_discovery_rejects_duplicate_numeric_indices(tmp_path: Path) -> None:
    np.savez(tmp_path / "train_data_1.npz", marker=1)
    np.savez(tmp_path / "train_data_0001.npz", marker=1)

    with pytest.raises(DatasetFormatError, match="Duplicate numeric episode index 1"):
        discover_npz_episodes(tmp_path)


def test_safe_length_precedence_and_language_is_ignored(tmp_path: Path) -> None:
    done_early = np.zeros(6, dtype=bool)
    done_early[3] = True
    success_earlier = np.zeros(6, dtype=bool)
    success_earlier[1] = True
    explicit_path = _write_episode(
        tmp_path,
        "train_data_0.npz",
        episode_length=np.asarray(6),
        done=done_early,
        success=success_earlier,
    )

    done = np.zeros(6, dtype=np.int8)
    done[4] = 1
    done_path = _write_episode(tmp_path, "train_data_1.npz", done=done)

    success = np.zeros((6, 1), dtype=np.float32)
    success[2, 0] = 1.0
    success_path = _write_episode(tmp_path, "train_data_2.npz", success=success)
    full_path = _write_episode(tmp_path, "train_data_3.npz")

    assert load_npz_episode(explicit_path).length == 4
    assert load_npz_episode(done_path).length == 5
    assert load_npz_episode(success_path).length == 3
    assert load_npz_episode(full_path).length == 6


@pytest.mark.parametrize(
    ("key", "replacement", "message"),
    [
        ("rgb", np.zeros((4, 2, 3, 3), dtype=np.uint8), r"rgb.*\[T,H,W,6\]"),
        ("proprio", np.zeros((4, 8), dtype=np.float32), r"proprio.*\[4,7\]"),
        ("action", np.zeros((3, 7), dtype=np.float32), r"action.*\[4,7\]"),
    ],
)
def test_shape_validation(
    tmp_path: Path,
    key: str,
    replacement: np.ndarray,
    message: str,
) -> None:
    arrays = _episode_arrays(4)
    arrays[key] = replacement
    path = tmp_path / "train_data_0.npz"
    np.savez(path, **arrays)

    with pytest.raises(DatasetFormatError, match=message):
        load_npz_episode(path)


def test_invalid_terminal_metadata_is_rejected(tmp_path: Path) -> None:
    path = _write_episode(
        tmp_path,
        "train_data_0.npz",
        done=np.zeros(5, dtype=bool),
    )

    with pytest.raises(DatasetFormatError, match=r"done.*leading length 6"):
        load_npz_episode(path)


def test_windows_padding_and_memory_never_cross_episodes(tmp_path: Path) -> None:
    _write_episode(tmp_path, "train_data_000000.npz", length=5, base=10)
    _write_episode(tmp_path, "train_data_1.npz", length=4, base=100)
    dataset = MikasaNpzDataset(
        tmp_path,
        obs_steps=3,
        horizon=5,
        memory_steps=2,
        cache_size=1,
    )

    assert len(dataset) == 9

    late_first_episode = dataset[3]
    assert late_first_episode["obs"]["rgb"].shape == (3, 2, 3, 6)
    assert late_first_episode["obs"]["proprio"].shape == (3, 7)
    assert late_first_episode["action"].shape == (5, 7)
    assert late_first_episode["memory_rgb"].shape == (2, 2, 3, 6)
    assert late_first_episode["memory_proprio"].shape == (2, 7)
    assert late_first_episode["obs"]["rgb"].dtype == torch.uint8
    assert late_first_episode["obs"]["proprio"].dtype == torch.float32
    assert late_first_episode["action"].dtype == torch.float32

    # t=3, obs_steps=3 -> observations 1,2,3 and memory strictly before
    # that short window -> only timestep 0, right-aligned behind padding.
    assert late_first_episode["obs"]["rgb"][:, 0, 0, 0].tolist() == [11, 12, 13]
    # The action horizon starts with the observation window, so offset
    # obs_steps-1 (2) is the current action at t=3.
    assert late_first_episode["action"][:, 0].tolist() == [11.0, 12.0, 13.0, 14.0, 14.0]
    assert late_first_episode["action_mask"].tolist() == [True, True, True, True, False]
    assert late_first_episode["memory_timestep"].tolist() == [-1, 0]
    assert late_first_episode["memory_mask"].tolist() == [False, True]
    assert late_first_episode["memory_rgb"][:, 0, 0, 0].tolist() == [0, 10]

    # Dataset index 5 is episode two's t=0, so index 6 is t=1.  Left padding
    # repeats only episode two's first frame; it cannot pull episode one's tail.
    early_second_episode = dataset[6]
    assert early_second_episode["episode_index"].item() == 1
    assert early_second_episode["episode_id"].item() == 1
    assert early_second_episode["timestep"].item() == 1
    assert early_second_episode["obs"]["rgb"][:, 0, 0, 0].tolist() == [100, 100, 101]
    assert early_second_episode["obs_mask"].tolist() == [False, True, True]
    assert early_second_episode["memory_timestep"].tolist() == [-1, -1]
    assert not early_second_episode["memory_mask"].any()
    assert early_second_episode["action"][:, 0].tolist() == [100.0, 100.0, 101.0, 102.0, 103.0]
    assert early_second_episode["action_mask"].tolist() == [False, True, True, True, True]

    # Standard negative indexing remains useful for quick dataset inspection.
    assert dataset[-1]["episode_index"].item() == 1
    assert dataset[-1]["timestep"].item() == 3


def test_float_rgb_is_canonicalized_to_uint8(tmp_path: Path) -> None:
    arrays = _episode_arrays(2)
    arrays["rgb"] = arrays["rgb"].astype(np.float32) / 255.0
    path = tmp_path / "train_data_0.npz"
    np.savez(path, **arrays)

    episode = load_npz_episode(path)

    assert episode.rgb.dtype == np.uint8
    np.testing.assert_array_equal(episode.rgb, _episode_arrays(2)["rgb"])


def test_memory_subsamples_one_frame_per_segment(tmp_path: Path) -> None:
    _write_episode(tmp_path, "train_data_0.npz", length=12)
    dataset = MikasaNpzDataset(
        tmp_path,
        obs_steps=2,
        horizon=9,
        memory_steps=4,
        memory_subsample_ratio=3,
        random_memory_sampling=False,
    )

    # At t=11 the short window starts at 10, so history is [0, 10).
    # Taking each segment's final frame gives 2, 5, 8, 9.
    sample = dataset[11]
    assert sample["memory_timestep"].tolist() == [2, 5, 8, 9]
    assert sample["memory_mask"].tolist() == [True, True, True, True]


def test_empty_dataset_directory_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No train_data"):
        MikasaNpzDataset(tmp_path)
