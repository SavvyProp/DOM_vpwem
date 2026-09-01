"""Episode-safe NPZ loading for the MIKASA-Robo-VLA shell-game task.

The source collector writes one ``train_data_<index>.npz`` file per episode.
This module deliberately consumes only the canonical, non-language fields:

``rgb``
    ``[T, H, W, 6]`` uint8 frames.  The top and wrist cameras are concatenated
    on the channel axis and are intentionally left that way here.
``proprio``
    ``[T, 7]`` end-effector pose and gripper state.
``action``
    ``[T, 7]`` normalized ``pd_ee_delta_pose`` actions.

Language fields may be present in an archive, including object arrays, but are
never read.  This keeps the single-task adapter language-free and permits
``allow_pickle=False`` throughout.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

_EPISODE_FILE_RE = re.compile(r"^train_data_(\d+)\.npz$")


class DatasetFormatError(ValueError):
    """Raised when a MIKASA NPZ archive violates the expected contract."""


@dataclass(frozen=True)
class EpisodeRecord:
    """Lightweight metadata retained in the dataset index."""

    path: Path
    episode_id: int
    length: int
    full_length: int
    height: int
    width: int


@dataclass(frozen=True)
class EpisodeData:
    """Validated arrays cropped to the safe episode prefix."""

    path: Path
    episode_id: int
    rgb: np.ndarray
    proprio: np.ndarray
    action: np.ndarray
    full_length: int

    @property
    def length(self) -> int:
        return int(self.action.shape[0])


def _episode_id(path: Path) -> int:
    match = _EPISODE_FILE_RE.fullmatch(path.name)
    if match is None:
        raise DatasetFormatError(
            f"Expected a filename like train_data_0.npz or "
            f"train_data_000000.npz, got {path.name!r}."
        )
    return int(match.group(1))


def discover_npz_episodes(root: str | Path) -> list[Path]:
    """Discover padded and unpadded episode files in numeric index order.

    Unrelated files are ignored.  Two spellings of the same numeric index, for
    example ``train_data_1.npz`` and ``train_data_0001.npz``, are rejected
    because silently choosing one would make an experiment non-reproducible.
    """

    root_path = Path(root).expanduser()
    if root_path.is_file():
        return [root_path] if _EPISODE_FILE_RE.fullmatch(root_path.name) else []
    if not root_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {root_path}")
    if not root_path.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {root_path}")

    indexed: list[tuple[int, Path]] = []
    for path in root_path.iterdir():
        if not path.is_file():
            continue
        match = _EPISODE_FILE_RE.fullmatch(path.name)
        if match is not None:
            indexed.append((int(match.group(1)), path))
    indexed.sort(key=lambda item: (item[0], item[1].name))

    for (previous_id, previous_path), (episode_id, path) in zip(indexed, indexed[1:]):
        if previous_id == episode_id:
            raise DatasetFormatError(
                "Duplicate numeric episode index "
                f"{episode_id}: {previous_path.name!r} and {path.name!r}."
            )
    return [path for _, path in indexed]


def _read_npz_array(archive: Mapping[str, np.ndarray], key: str, path: Path) -> np.ndarray:
    if key not in archive:
        raise DatasetFormatError(f"{path}: required array {key!r} is missing.")
    try:
        return np.asarray(archive[key])
    except ValueError as exc:
        raise DatasetFormatError(f"{path}: could not safely read array {key!r}: {exc}") from exc


def _validate_terminal_array(
    value: np.ndarray,
    key: str,
    full_length: int,
    path: Path | None,
) -> int | None:
    array = np.asarray(value)
    location = f"{path}: " if path is not None else ""
    if array.ndim == 0 or array.shape[0] != full_length:
        raise DatasetFormatError(
            f"{location}{key!r} must have leading length {full_length}, got shape {array.shape}."
        )
    if array.dtype.kind not in "biuf":
        raise DatasetFormatError(
            f"{location}{key!r} must be boolean or numeric, got {array.dtype}."
        )
    if array.dtype.kind == "f" and not np.isfinite(array).all():
        raise DatasetFormatError(f"{location}{key!r} contains NaN or infinity.")
    per_step = array.reshape(full_length, -1).astype(bool, copy=False).any(axis=1)
    true_indices = np.flatnonzero(per_step)
    return int(true_indices[0]) + 1 if true_indices.size else None


def derive_safe_episode_length(
    arrays: Mapping[str, np.ndarray],
    full_length: int,
    *,
    path: str | Path | None = None,
) -> int:
    """Choose a safe, terminal-inclusive episode prefix.

    The explicit ``episode_length`` is preferred and clipped to the physical
    array length.  An earlier ``done`` marker shortens that prefix.  If neither
    supplies an endpoint, the first ``success`` marker is used; otherwise the
    full array length is valid.  ``success`` is only a fallback because some
    environments remain active for useful post-success steps.
    """

    source_path = Path(path) if path is not None else None
    location = f"{source_path}: " if source_path is not None else ""
    if full_length <= 0:
        raise DatasetFormatError(f"{location}episode arrays must contain at least one timestep.")

    explicit_length: int | None = None
    if "episode_length" in arrays:
        try:
            raw_length = np.asarray(arrays["episode_length"])
        except ValueError as exc:
            raise DatasetFormatError(
                f"{location}could not safely read 'episode_length': {exc}"
            ) from exc
        if raw_length.size != 1:
            raise DatasetFormatError(
                f"{location}'episode_length' must be scalar, got shape {raw_length.shape}."
            )
        try:
            numeric_length = float(raw_length.reshape(-1)[0])
        except (TypeError, ValueError) as exc:
            raise DatasetFormatError(f"{location}'episode_length' must be numeric.") from exc
        if not np.isfinite(numeric_length) or not numeric_length.is_integer():
            raise DatasetFormatError(
                f"{location}'episode_length' must be a finite integer, got {numeric_length!r}."
            )
        explicit_length = int(numeric_length)
        if explicit_length <= 0:
            raise DatasetFormatError(
                f"{location}'episode_length' must be positive, got {explicit_length}."
            )
        explicit_length = min(explicit_length, full_length)

    done_end: int | None = None
    if "done" in arrays:
        try:
            done_end = _validate_terminal_array(arrays["done"], "done", full_length, source_path)
        except ValueError as exc:
            if isinstance(exc, DatasetFormatError):
                raise
            raise DatasetFormatError(f"{location}could not safely read 'done': {exc}") from exc

    if explicit_length is not None:
        return min(explicit_length, done_end) if done_end is not None else explicit_length
    if done_end is not None:
        return done_end

    if "success" in arrays:
        try:
            success_end = _validate_terminal_array(
                arrays["success"], "success", full_length, source_path
            )
        except ValueError as exc:
            if isinstance(exc, DatasetFormatError):
                raise
            raise DatasetFormatError(f"{location}could not safely read 'success': {exc}") from exc
        if success_end is not None:
            return success_end
    return full_length


def _canonical_uint8_rgb(rgb: np.ndarray, path: Path) -> np.ndarray:
    if rgb.dtype == np.uint8:
        return np.ascontiguousarray(rgb)
    if rgb.dtype.kind not in "iuf":
        raise DatasetFormatError(f"{path}: 'rgb' must be numeric, got dtype {rgb.dtype}.")
    if not np.isfinite(rgb).all():
        raise DatasetFormatError(f"{path}: 'rgb' contains NaN or infinity.")
    minimum = float(rgb.min())
    maximum = float(rgb.max())
    if minimum < 0.0 or maximum > 255.0:
        raise DatasetFormatError(
            f"{path}: 'rgb' values must lie in [0, 255] (or [0, 1]), got [{minimum}, {maximum}]."
        )
    scaled = rgb.astype(np.float32, copy=False)
    if rgb.dtype.kind == "f" and maximum <= 1.0:
        scaled = scaled * 255.0
    return np.ascontiguousarray(np.rint(scaled).astype(np.uint8))


def _canonical_float32(array: np.ndarray, key: str, path: Path) -> np.ndarray:
    if array.dtype.kind not in "iuf":
        raise DatasetFormatError(
            f"{path}: {key!r} must be real numeric data, got dtype {array.dtype}."
        )
    if not np.isfinite(array).all():
        raise DatasetFormatError(f"{path}: {key!r} contains NaN or infinity.")
    return np.ascontiguousarray(array, dtype=np.float32)


def load_npz_episode(path: str | Path) -> EpisodeData:
    """Load, validate, canonicalize, and trim one episode archive."""

    source_path = Path(path).expanduser()
    episode_id = _episode_id(source_path)
    try:
        with np.load(source_path, allow_pickle=False) as archive:
            rgb = _read_npz_array(archive, "rgb", source_path)
            proprio = _read_npz_array(archive, "proprio", source_path)
            action = _read_npz_array(archive, "action", source_path)

            if rgb.ndim != 4 or rgb.shape[-1] != 6:
                raise DatasetFormatError(
                    f"{source_path}: 'rgb' must have shape [T,H,W,6], got {rgb.shape}."
                )
            full_length = int(rgb.shape[0])
            if full_length <= 0 or rgb.shape[1] <= 0 or rgb.shape[2] <= 0:
                raise DatasetFormatError(
                    f"{source_path}: 'rgb' has an empty dimension: {rgb.shape}."
                )
            if proprio.shape != (full_length, 7):
                raise DatasetFormatError(
                    f"{source_path}: 'proprio' must have shape [{full_length},7], "
                    f"got {proprio.shape}."
                )
            if action.shape != (full_length, 7):
                raise DatasetFormatError(
                    f"{source_path}: 'action' must have shape [{full_length},7], "
                    f"got {action.shape}."
                )

            length = derive_safe_episode_length(archive, full_length, path=source_path)
            rgb = _canonical_uint8_rgb(rgb[:length], source_path)
            proprio = _canonical_float32(proprio[:length], "proprio", source_path)
            action = _canonical_float32(action[:length], "action", source_path)
    except DatasetFormatError:
        raise
    except (OSError, ValueError, EOFError) as exc:
        raise DatasetFormatError(f"Could not load NPZ episode {source_path}: {exc}") from exc

    return EpisodeData(
        path=source_path,
        episode_id=episode_id,
        rgb=rgb,
        proprio=proprio,
        action=action,
        full_length=full_length,
    )


class MikasaNpzDataset(Dataset):
    """Timestep-indexed, episode-safe windows for VPWEM training.

    Each index is one real timestep.  The observation window ends at that
    timestep and is left-padded with the episode's first observation.  The
    action horizon begins at the same logical position as the observation
    window, matching Diffusion Policy / VPWEM alignment: action index
    ``obs_steps - 1`` is the action at the current timestep.  The action window
    is padded at either episode boundary and ``action_mask`` distinguishes that
    padding. Memory contains at most ``memory_steps`` frames strictly before
    the short observation window. One frame is selected from each
    ``memory_subsample_ratio`` segment, following VPWEM training; entries are
    right-aligned behind zero padding and accompanied by a mask.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        obs_steps: int = 1,
        horizon: int = 1,
        memory_steps: int = 0,
        memory_subsample_ratio: int = 1,
        random_memory_sampling: bool = True,
        preload: bool = False,
        cache_size: int = 1,
    ) -> None:
        if obs_steps <= 0:
            raise ValueError(f"obs_steps must be positive, got {obs_steps}.")
        if horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}.")
        if memory_steps < 0:
            raise ValueError(f"memory_steps must be non-negative, got {memory_steps}.")
        if memory_subsample_ratio <= 0:
            raise ValueError(
                "memory_subsample_ratio must be positive, got "
                f"{memory_subsample_ratio}."
            )
        if cache_size < 0:
            raise ValueError(f"cache_size must be non-negative, got {cache_size}.")

        paths = discover_npz_episodes(root)
        if not paths:
            raise FileNotFoundError(f"No train_data_<index>.npz episodes found under {Path(root)}.")

        records: list[EpisodeRecord] = []
        preloaded: list[EpisodeData] | None = [] if preload else None
        spatial_shape: tuple[int, int] | None = None
        for path in paths:
            episode = load_npz_episode(path)
            height, width = map(int, episode.rgb.shape[1:3])
            if spatial_shape is None:
                spatial_shape = (height, width)
            elif (height, width) != spatial_shape:
                raise DatasetFormatError(
                    f"{path}: spatial shape {(height, width)} differs from earlier "
                    f"episodes {spatial_shape}."
                )
            records.append(
                EpisodeRecord(
                    path=path,
                    episode_id=episode.episode_id,
                    length=episode.length,
                    full_length=episode.full_length,
                    height=height,
                    width=width,
                )
            )
            if preloaded is not None:
                preloaded.append(episode)

        self.root = Path(root).expanduser()
        self.obs_steps = int(obs_steps)
        self.horizon = int(horizon)
        self.memory_steps = int(memory_steps)
        self.memory_subsample_ratio = int(memory_subsample_ratio)
        self.random_memory_sampling = bool(random_memory_sampling)
        self.cache_size = int(cache_size)
        self.episodes = tuple(records)
        self._preloaded = tuple(preloaded) if preloaded is not None else None
        self._sample_index = tuple(
            (episode_index, timestep)
            for episode_index, record in enumerate(self.episodes)
            for timestep in range(record.length)
        )
        self._cache: OrderedDict[int, EpisodeData] = OrderedDict()

    def __len__(self) -> int:
        return len(self._sample_index)

    def _load_episode(self, episode_index: int) -> EpisodeData:
        if self._preloaded is not None:
            return self._preloaded[episode_index]
        cached = self._cache.get(episode_index)
        if cached is not None:
            self._cache.move_to_end(episode_index)
            return cached

        record = self.episodes[episode_index]
        episode = load_npz_episode(record.path)
        if (
            episode.length != record.length
            or episode.full_length != record.full_length
            or tuple(episode.rgb.shape[1:3]) != (record.height, record.width)
        ):
            raise DatasetFormatError(
                f"Episode changed after indexing: {record.path}. Recreate the dataset index."
            )
        if self.cache_size > 0:
            self._cache[episode_index] = episode
            self._cache.move_to_end(episode_index)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return episode

    @staticmethod
    def _tensor(array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(array))

    def __getitem__(self, index: int) -> dict[str, object]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        episode_index, timestep = self._sample_index[index]
        record = self.episodes[episode_index]
        episode = self._load_episode(episode_index)
        length = record.length

        raw_obs_indices = np.arange(timestep - self.obs_steps + 1, timestep + 1, dtype=np.int64)
        obs_mask = raw_obs_indices >= 0
        obs_indices = np.clip(raw_obs_indices, 0, length - 1)

        sequence_start = timestep - self.obs_steps + 1
        raw_action_indices = np.arange(
            sequence_start,
            sequence_start + self.horizon,
            dtype=np.int64,
        )
        action_mask = (raw_action_indices >= 0) & (raw_action_indices < length)
        action_indices = np.clip(raw_action_indices, 0, length - 1)

        # Memory excludes the current short observation window.  This prevents
        # duplicate "long" and "short" context while keeping all history local
        # to the same episode.
        obs_start = max(0, sequence_start)
        memory_end = obs_start
        if self.memory_steps and memory_end:
            segment_starts = np.arange(
                0, memory_end, self.memory_subsample_ratio, dtype=np.int64
            )
            segment_ends = np.minimum(
                segment_starts + self.memory_subsample_ratio, memory_end
            )
            if self.random_memory_sampling:
                offsets = np.array(
                    [
                        np.random.randint(0, end - start)
                        for start, end in zip(segment_starts, segment_ends)
                    ],
                    dtype=np.int64,
                )
                memory_indices = segment_starts + offsets
            else:
                memory_indices = segment_ends - 1
            memory_indices = memory_indices[-self.memory_steps :]
        else:
            memory_indices = np.empty((0,), dtype=np.int64)
        n_memory = int(memory_indices.size)
        memory_offset = self.memory_steps - n_memory

        memory_rgb = np.zeros(
            (self.memory_steps, record.height, record.width, 6), dtype=np.uint8
        )
        memory_proprio = np.zeros((self.memory_steps, 7), dtype=np.float32)
        memory_timestep = np.full((self.memory_steps,), -1, dtype=np.int64)
        memory_mask = np.zeros((self.memory_steps,), dtype=bool)
        if n_memory:
            memory_rgb[memory_offset:] = episode.rgb[memory_indices]
            memory_proprio[memory_offset:] = episode.proprio[memory_indices]
            memory_timestep[memory_offset:] = memory_indices
            memory_mask[memory_offset:] = True

        return {
            "obs": {
                "rgb": self._tensor(episode.rgb[obs_indices]),
                "proprio": self._tensor(episode.proprio[obs_indices]),
            },
            "obs_mask": self._tensor(obs_mask),
            "action": self._tensor(episode.action[action_indices]),
            "action_mask": self._tensor(action_mask),
            "memory_rgb": self._tensor(memory_rgb),
            "memory_proprio": self._tensor(memory_proprio),
            "memory_timestep": self._tensor(memory_timestep),
            "memory_mask": self._tensor(memory_mask),
            "episode_index": torch.tensor(episode_index, dtype=torch.int64),
            "episode_id": torch.tensor(record.episode_id, dtype=torch.int64),
            "timestep": torch.tensor(timestep, dtype=torch.int64),
            "episode_length": torch.tensor(length, dtype=torch.int64),
            "source_path": str(record.path),
        }


__all__: Sequence[str] = (
    "DatasetFormatError",
    "EpisodeData",
    "EpisodeRecord",
    "MikasaNpzDataset",
    "derive_safe_episode_length",
    "discover_npz_episodes",
    "load_npz_episode",
)
