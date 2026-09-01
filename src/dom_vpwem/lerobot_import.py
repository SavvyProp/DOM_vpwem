"""Convert the public MIKASA LeRobot v3 release to this project's NPZ format.

The Hugging Face release stores proprioception/actions in Parquet and the two
camera streams in AV1 videos.  The training loader in :mod:`dom_vpwem.data`
expects one NumPy archive per episode, so this module performs a streaming
conversion without retaining an entire task's decoded video in memory.

Heavy, dataset-only dependencies are imported lazily.  Training, evaluation,
``--help``, and integrity verification therefore do not require PyArrow,
PyAV, or ``huggingface_hub``.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np

from .tasks import TaskSpec

LEROBOT_REPO_ID = "mikasa-robo/mikasa-robo-vla-lerobot"
LEROBOT_REVISION = "fa5417a266d1cb87ed7715c3dd2d0e4edc067b04"

TOP_CAMERA_KEY = "observation.images.top"
WRIST_CAMERA_KEY = "observation.images.wrist"
STATE_KEY = "observation.state"
ACTION_KEY = "action"
EPISODE_INDEX_KEY = "episode_index"
FRAME_INDEX_KEY = "frame_index"

EXPECTED_IMAGE_SHAPE = (128, 128, 3)
EXPECTED_VECTOR_SHAPE = (7,)


class LeRobotImportError(RuntimeError):
    """Raised when a downloaded release cannot be converted safely."""


class DatasetDependencyError(LeRobotImportError):
    """Raised when the optional download/conversion dependencies are absent."""


@dataclass(frozen=True)
class LeRobotMetadata:
    """Validated task-level metadata from a LeRobot v3 directory."""

    env_id: str
    codebase_version: str
    fps: int
    total_episodes: int
    total_frames: int


@dataclass(frozen=True)
class LeRobotRow:
    """The non-image fields for one timestep."""

    episode_index: int
    frame_index: int
    proprio: np.ndarray
    action: np.ndarray


@dataclass(frozen=True)
class EpisodeArtifact:
    """Integrity information for one generated episode file."""

    filename: str
    episode_index: int
    length: int
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class ConversionResult:
    """Summary of a completed LeRobot-to-NPZ conversion."""

    metadata: LeRobotMetadata
    episodes: tuple[EpisodeArtifact, ...]

    @property
    def total_frames(self) -> int:
        return sum(episode.length for episode in self.episodes)


def _optional_dependency_message() -> str:
    return (
        "Dataset installation requires the optional 'data' dependencies. "
        "Run `uv run --locked --extra data dom-vpwem-install-datasets` "
        "or install them with `uv sync --locked --extra data`."
    )


def download_lerobot_snapshot(
    tasks: Sequence[TaskSpec],
    *,
    repo_id: str = LEROBOT_REPO_ID,
    revision: str = LEROBOT_REVISION,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    max_workers: int = 8,
) -> Path:
    """Download only the selected task folders at an immutable revision."""

    # The files needed here are small enough for the standard HTTP/LFS path.
    # Disabling Xet by default also makes ``cache_dir`` authoritative instead
    # of consulting a separately configured HF_XET_CACHE. Respect an explicit
    # user setting if they intentionally enabled Xet.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    modules: dict[str, Any] = {}
    missing: list[str] = []
    for module_name in ("huggingface_hub", "pyarrow.parquet", "av"):
        try:
            modules[module_name] = importlib.import_module(module_name)
        except (ImportError, OSError):  # pragma: no cover - depends on optional env
            missing.append(module_name)
    if missing:
        raise DatasetDependencyError(
            f"Missing dataset dependencies: {', '.join(missing)}. "
            + _optional_dependency_message()
        )
    snapshot_download = modules["huggingface_hub"].snapshot_download

    if not tasks:
        raise ValueError("At least one task is required for a snapshot download.")
    if max_workers <= 0:
        raise ValueError(f"max_workers must be positive, got {max_workers}.")

    patterns = [f"{task.dataset_slug}/**" for task in tasks]
    try:
        snapshot = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            allow_patterns=patterns,
            cache_dir=str(Path(cache_dir).expanduser()) if cache_dir is not None else None,
            local_files_only=local_files_only,
            max_workers=max_workers,
        )
    except Exception as exc:
        mode = "the local Hugging Face cache" if local_files_only else "Hugging Face"
        raise LeRobotImportError(
            f"Could not obtain MIKASA data from {mode} at revision {revision}: {exc}"
        ) from exc
    return Path(snapshot)


def _require_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise LeRobotImportError(f"Downloaded dataset is missing {path}.") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LeRobotImportError(f"Could not read dataset metadata {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise LeRobotImportError(f"Dataset metadata {path} must contain a JSON object.")
    return value


def _require_positive_int(value: Any, name: str, path: Path) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise LeRobotImportError(f"{path}: {name} must be a positive integer, got {value!r}.")
    return value


def _validate_feature(
    features: Mapping[str, Any],
    key: str,
    *,
    dtype: str,
    shape: tuple[int, ...],
    path: Path,
) -> None:
    feature = features.get(key)
    if not isinstance(feature, Mapping):
        raise LeRobotImportError(f"{path}: required feature {key!r} is missing.")
    actual_shape = feature.get("shape")
    if feature.get("dtype") != dtype or actual_shape != list(shape):
        raise LeRobotImportError(
            f"{path}: feature {key!r} must be {dtype}{list(shape)}, got "
            f"{feature.get('dtype')!r}{actual_shape!r}."
        )


def load_lerobot_metadata(task_root: str | Path, task: TaskSpec) -> LeRobotMetadata:
    """Read and validate the official task and signal metadata."""

    root = Path(task_root)
    info_path = root / "meta" / "info.json"
    source_path = root / "source_rlds_metadata.json"
    info = _require_json_object(info_path)
    source = _require_json_object(source_path)

    codebase_version = info.get("codebase_version")
    if codebase_version != "v3.0":
        raise LeRobotImportError(
            f"{info_path}: expected LeRobot codebase_version 'v3.0', "
            f"got {codebase_version!r}."
        )
    if info.get("robot_type") != "mikasa_robo":
        raise LeRobotImportError(
            f"{info_path}: expected robot_type 'mikasa_robo', got {info.get('robot_type')!r}."
        )
    if source.get("env_id") != task.env_id:
        raise LeRobotImportError(
            f"{source_path}: expected env_id {task.env_id!r}, got {source.get('env_id')!r}."
        )

    total_episodes = _require_positive_int(info.get("total_episodes"), "total_episodes", info_path)
    total_frames = _require_positive_int(info.get("total_frames"), "total_frames", info_path)
    fps = _require_positive_int(info.get("fps"), "fps", info_path)
    if source.get("num_episodes") != total_episodes:
        raise LeRobotImportError(
            f"{source_path}: num_episodes does not match {info_path}: "
            f"{source.get('num_episodes')!r} != {total_episodes}."
        )

    raw_lengths = source.get("episode_lengths")
    if not isinstance(raw_lengths, list) or len(raw_lengths) != total_episodes:
        raise LeRobotImportError(
            f"{source_path}: episode_lengths must contain {total_episodes} entries."
        )
    # This file describes the pre-export RLDS/NPZ order. The LeRobot exporter
    # may renumber episodes in a different order, so the list is authoritative
    # only for its count and total, not for episode_index-to-length alignment.
    source_episode_lengths = tuple(
        _require_positive_int(value, f"episode_lengths[{index}]", source_path)
        for index, value in enumerate(raw_lengths)
    )
    if sum(source_episode_lengths) != total_frames:
        raise LeRobotImportError(
            f"{source_path}: episode lengths sum to {sum(source_episode_lengths)}, "
            f"but info.json declares {total_frames} frames."
        )

    features = info.get("features")
    if not isinstance(features, Mapping):
        raise LeRobotImportError(f"{info_path}: features must be a JSON object.")
    _validate_feature(
        features,
        TOP_CAMERA_KEY,
        dtype="video",
        shape=EXPECTED_IMAGE_SHAPE,
        path=info_path,
    )
    _validate_feature(
        features,
        WRIST_CAMERA_KEY,
        dtype="video",
        shape=EXPECTED_IMAGE_SHAPE,
        path=info_path,
    )
    _validate_feature(
        features,
        STATE_KEY,
        dtype="float32",
        shape=EXPECTED_VECTOR_SHAPE,
        path=info_path,
    )
    _validate_feature(
        features,
        ACTION_KEY,
        dtype="float32",
        shape=EXPECTED_VECTOR_SHAPE,
        path=info_path,
    )

    return LeRobotMetadata(
        env_id=task.env_id,
        codebase_version=codebase_version,
        fps=fps,
        total_episodes=total_episodes,
        total_frames=total_frames,
    )


def _list_data_files(task_root: Path) -> list[Path]:
    paths = sorted((task_root / "data").glob("chunk-*/file-*.parquet"))
    if not paths:
        raise LeRobotImportError(f"No Parquet data shards were found under {task_root / 'data'}.")
    return paths


def _list_video_files(task_root: Path, camera_key: str) -> list[Path]:
    root = task_root / "videos" / camera_key
    paths = sorted(root.glob("chunk-*/file-*.mp4"))
    if not paths:
        raise LeRobotImportError(f"No video shards were found under {root}.")
    return paths


def _fixed_vector_column(batch: Any, key: str, width: int) -> np.ndarray:
    """Convert a PyArrow list column to a dense float32 matrix."""

    try:
        index = batch.schema.get_field_index(key)
        if index < 0:
            raise KeyError(key)
        array = np.asarray(batch.column(index).to_pylist(), dtype=np.float32)
    except Exception as exc:
        raise LeRobotImportError(f"Could not decode Parquet column {key!r}: {exc}") from exc
    if array.shape != (batch.num_rows, width):
        raise LeRobotImportError(
            f"Parquet column {key!r} must have shape [{batch.num_rows},{width}], "
            f"got {array.shape}."
        )
    if not np.isfinite(array).all():
        raise LeRobotImportError(f"Parquet column {key!r} contains NaN or infinity.")
    return array


def _integer_column(batch: Any, key: str) -> np.ndarray:
    try:
        index = batch.schema.get_field_index(key)
        if index < 0:
            raise KeyError(key)
        values = np.asarray(batch.column(index).to_pylist(), dtype=np.int64)
    except Exception as exc:
        raise LeRobotImportError(f"Could not decode Parquet column {key!r}: {exc}") from exc
    if values.shape != (batch.num_rows,):
        raise LeRobotImportError(
            f"Parquet column {key!r} must be scalar per row, got {values.shape}."
        )
    return values


def iter_parquet_rows(paths: Sequence[Path], *, batch_size: int = 1024) -> Iterator[LeRobotRow]:
    """Stream canonical rows from ordered LeRobot Parquet shards."""

    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:  # pragma: no cover - depends on optional env
        raise DatasetDependencyError(_optional_dependency_message()) from exc

    columns = [STATE_KEY, ACTION_KEY, EPISODE_INDEX_KEY, FRAME_INDEX_KEY]
    for path in paths:
        try:
            parquet_file = parquet.ParquetFile(path)
            batches = parquet_file.iter_batches(batch_size=batch_size, columns=columns)
            for batch in batches:
                proprio = _fixed_vector_column(batch, STATE_KEY, EXPECTED_VECTOR_SHAPE[0])
                action = _fixed_vector_column(batch, ACTION_KEY, EXPECTED_VECTOR_SHAPE[0])
                episode_indices = _integer_column(batch, EPISODE_INDEX_KEY)
                frame_indices = _integer_column(batch, FRAME_INDEX_KEY)
                for index in range(batch.num_rows):
                    yield LeRobotRow(
                        episode_index=int(episode_indices[index]),
                        frame_index=int(frame_indices[index]),
                        proprio=proprio[index],
                        action=action[index],
                    )
        except LeRobotImportError:
            raise
        except Exception as exc:
            raise LeRobotImportError(f"Could not stream Parquet shard {path}: {exc}") from exc


def iter_video_frames(paths: Sequence[Path]) -> Iterator[np.ndarray]:
    """Decode ordered video shards as RGB uint8 frames."""

    try:
        import av
    except ImportError as exc:  # pragma: no cover - depends on optional env
        raise DatasetDependencyError(_optional_dependency_message()) from exc

    for path in paths:
        try:
            with av.open(str(path), mode="r") as container:
                if len(container.streams.video) != 1:
                    raise LeRobotImportError(
                        f"{path}: expected exactly one video stream, "
                        f"found {len(container.streams.video)}."
                    )
                stream = container.streams.video[0]
                stream.thread_type = "AUTO"
                for frame in container.decode(stream):
                    image = np.asarray(frame.to_ndarray(format="rgb24"))
                    if image.dtype != np.uint8 or image.shape != EXPECTED_IMAGE_SHAPE:
                        raise LeRobotImportError(
                            f"{path}: decoded frame must be uint8{EXPECTED_IMAGE_SHAPE}, "
                            f"got {image.dtype}{image.shape}."
                        )
                    yield image
        except LeRobotImportError:
            raise
        except Exception as exc:
            raise LeRobotImportError(f"Could not decode video shard {path}: {exc}") from exc


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without loading it all at once."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_row(row: LeRobotRow) -> tuple[np.ndarray, np.ndarray]:
    try:
        proprio = np.asarray(row.proprio, dtype=np.float32)
        action = np.asarray(row.action, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise LeRobotImportError(
            f"Episode {row.episode_index}, frame {row.frame_index}: "
            f"state/action is not numeric: {exc}"
        ) from exc
    if proprio.shape != EXPECTED_VECTOR_SHAPE:
        raise LeRobotImportError(
            f"Episode {row.episode_index}, frame {row.frame_index}: proprio must have "
            f"shape {EXPECTED_VECTOR_SHAPE}, got {proprio.shape}."
        )
    if action.shape != EXPECTED_VECTOR_SHAPE:
        raise LeRobotImportError(
            f"Episode {row.episode_index}, frame {row.frame_index}: action must have "
            f"shape {EXPECTED_VECTOR_SHAPE}, got {action.shape}."
        )
    if not np.isfinite(proprio).all() or not np.isfinite(action).all():
        raise LeRobotImportError(
            f"Episode {row.episode_index}, frame {row.frame_index}: state/action contains "
            "NaN or infinity."
        )
    if np.any(action < -1.001) or np.any(action > 1.001):
        raise LeRobotImportError(
            f"Episode {row.episode_index}, frame {row.frame_index}: action lies outside "
            "the normalized [-1, 1] contract."
        )
    return np.ascontiguousarray(proprio), np.ascontiguousarray(action)


def _validate_frame(frame: np.ndarray, camera: str, row: LeRobotRow) -> np.ndarray:
    image = np.asarray(frame)
    if image.dtype != np.uint8 or image.shape != EXPECTED_IMAGE_SHAPE:
        raise LeRobotImportError(
            f"Episode {row.episode_index}, frame {row.frame_index}: {camera} image must be "
            f"uint8{EXPECTED_IMAGE_SHAPE}, got {image.dtype}{image.shape}."
        )
    return np.ascontiguousarray(image)


def _write_episode(
    destination: Path,
    episode_index: int,
    top_frames: list[np.ndarray],
    wrist_frames: list[np.ndarray],
    proprio_rows: list[np.ndarray],
    action_rows: list[np.ndarray],
) -> EpisodeArtifact:
    length = len(action_rows)
    if length == 0:
        raise LeRobotImportError(f"Episode {episode_index} contains no frames.")
    top = np.stack(top_frames, axis=0)
    wrist = np.stack(wrist_frames, axis=0)
    rgb = np.ascontiguousarray(np.concatenate((top, wrist), axis=-1), dtype=np.uint8)
    proprio = np.ascontiguousarray(np.stack(proprio_rows, axis=0), dtype=np.float32)
    action = np.ascontiguousarray(np.stack(action_rows, axis=0), dtype=np.float32)
    done = np.zeros(length, dtype=bool)
    done[-1] = True

    filename = f"train_data_{episode_index:06d}.npz"
    path = destination / filename
    try:
        np.savez_compressed(
            path,
            rgb=rgb,
            proprio=proprio,
            action=action,
            done=done,
            episode_length=np.asarray(length, dtype=np.int64),
        )
    except OSError as exc:
        raise LeRobotImportError(f"Could not write converted episode {path}: {exc}") from exc
    return EpisodeArtifact(
        filename=filename,
        episode_index=episode_index,
        length=length,
        size_bytes=path.stat().st_size,
        sha256=sha256_file(path),
    )


def write_npz_episodes(
    rows: Iterable[LeRobotRow],
    top_frames: Iterable[np.ndarray],
    wrist_frames: Iterable[np.ndarray],
    destination: str | Path,
    metadata: LeRobotMetadata,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> ConversionResult:
    """Join three ordered streams and write validated per-episode archives.

    This is deliberately dependency-injectable: tests can exercise the exact
    episode grouping and NPZ output without downloading data or importing the
    optional Parquet/video libraries.
    """

    output = Path(destination)
    if metadata.total_episodes <= 0 or metadata.total_frames <= 0:
        raise LeRobotImportError(
            "LeRobot metadata must declare positive episode and frame totals."
        )
    output.mkdir(parents=True, exist_ok=False)
    row_iterator = iter(rows)
    top_iterator = iter(top_frames)
    wrist_iterator = iter(wrist_frames)

    current_episode: int | None = None
    expected_episode = 0
    expected_frame = 0
    observed_rows = 0
    artifacts: list[EpisodeArtifact] = []
    top_buffer: list[np.ndarray] = []
    wrist_buffer: list[np.ndarray] = []
    proprio_buffer: list[np.ndarray] = []
    action_buffer: list[np.ndarray] = []

    def finish_episode() -> None:
        nonlocal expected_episode, expected_frame
        if current_episode is None:
            return
        if current_episode < 0 or current_episode >= metadata.total_episodes:
            raise LeRobotImportError(
                f"Episode index {current_episode} exceeds the declared "
                f"{metadata.total_episodes} episodes."
            )
        artifact = _write_episode(
            output,
            current_episode,
            top_buffer,
            wrist_buffer,
            proprio_buffer,
            action_buffer,
        )
        artifacts.append(artifact)
        expected_episode += 1
        expected_frame = 0
        if progress is not None:
            progress(len(artifacts), metadata.total_episodes)

    sentinel = object()
    try:
        for row in row_iterator:
            if row.episode_index != current_episode:
                finish_episode()
                if row.episode_index != expected_episode:
                    raise LeRobotImportError(
                        f"Expected episode_index {expected_episode}, got {row.episode_index}. "
                        "Episode indices must be contiguous and ordered."
                    )
                current_episode = row.episode_index
                top_buffer = []
                wrist_buffer = []
                proprio_buffer = []
                action_buffer = []

            if row.frame_index != expected_frame:
                raise LeRobotImportError(
                    f"Episode {row.episode_index}: expected frame_index {expected_frame}, "
                    f"got {row.frame_index}."
                )
            top_frame = next(top_iterator, sentinel)
            wrist_frame = next(wrist_iterator, sentinel)
            if top_frame is sentinel or wrist_frame is sentinel:
                camera = "top" if top_frame is sentinel else "wrist"
                raise LeRobotImportError(
                    f"The {camera} video stream ended before data row {observed_rows}."
                )

            proprio, action = _validate_row(row)
            top_buffer.append(_validate_frame(top_frame, "top", row))
            wrist_buffer.append(_validate_frame(wrist_frame, "wrist", row))
            proprio_buffer.append(proprio)
            action_buffer.append(action)
            expected_frame += 1
            observed_rows += 1

        finish_episode()

        if next(top_iterator, sentinel) is not sentinel:
            raise LeRobotImportError("The top video stream contains more frames than Parquet data.")
        if next(wrist_iterator, sentinel) is not sentinel:
            raise LeRobotImportError(
                "The wrist video stream contains more frames than Parquet data."
            )
    finally:
        for iterator in (row_iterator, top_iterator, wrist_iterator):
            close = getattr(iterator, "close", None)
            if close is not None:
                close()

    if len(artifacts) != metadata.total_episodes:
        raise LeRobotImportError(
            f"Converted {len(artifacts)} episodes, expected {metadata.total_episodes}."
        )
    if observed_rows != metadata.total_frames:
        raise LeRobotImportError(
            f"Converted {observed_rows} frames, expected {metadata.total_frames}."
        )
    return ConversionResult(metadata=metadata, episodes=tuple(artifacts))


def convert_lerobot_task(
    task_root: str | Path,
    destination: str | Path,
    task: TaskSpec,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> ConversionResult:
    """Convert one downloaded MIKASA LeRobot v3 task directory."""

    root = Path(task_root)
    metadata = load_lerobot_metadata(root, task)
    return write_npz_episodes(
        iter_parquet_rows(_list_data_files(root)),
        iter_video_frames(_list_video_files(root, TOP_CAMERA_KEY)),
        iter_video_frames(_list_video_files(root, WRIST_CAMERA_KEY)),
        destination,
        metadata,
        progress=progress,
    )


__all__ = [
    "ConversionResult",
    "DatasetDependencyError",
    "EpisodeArtifact",
    "LEROBOT_REPO_ID",
    "LEROBOT_REVISION",
    "LeRobotImportError",
    "LeRobotMetadata",
    "LeRobotRow",
    "convert_lerobot_task",
    "download_lerobot_snapshot",
    "load_lerobot_metadata",
    "sha256_file",
    "write_npz_episodes",
]
