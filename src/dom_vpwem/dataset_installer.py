"""Safe installer for the public MIKASA datasets used by this repository."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shutil
import tempfile
import uuid
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np

from . import __version__
from .data import DatasetFormatError, discover_npz_episodes, load_npz_episode
from .lerobot_import import (
    LEROBOT_REPO_ID,
    LEROBOT_REVISION,
    ConversionResult,
    LeRobotImportError,
    convert_lerobot_task,
    download_lerobot_snapshot,
    sha256_file,
)
from .tasks import DEFAULT_DATASET_ROOT, TASK_SPECS, TaskSpec

MANIFEST_FILENAME = ".dom_vpwem_dataset.json"
MANIFEST_SCHEMA_VERSION = 1
DATASET_FORMAT = "dom-vpwem-npz-v1"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class DatasetInstallError(RuntimeError):
    """Raised when an installation is unsafe, incomplete, or corrupt."""


def _task_aliases() -> dict[str, str]:
    aliases: dict[str, str] = {
        "touch": "ShellGameTouch-VLA-v0",
        "shell-game-touch": "ShellGameTouch-VLA-v0",
        "shuffle": "ShellGameShuffleColorLampTouch-VLA-v0",
        "shuffle-color": "ShellGameShuffleColorLampTouch-VLA-v0",
        "shell-game-shuffle-color-lamp-touch": (
            "ShellGameShuffleColorLampTouch-VLA-v0"
        ),
    }
    for env_id, task in TASK_SPECS.items():
        aliases[env_id] = env_id
        aliases[env_id.lower()] = env_id
        aliases[task.dataset_slug] = env_id
    return aliases


TASK_ALIASES = _task_aliases()


def resolve_tasks(values: Sequence[str] | None) -> list[TaskSpec]:
    """Resolve CLI task aliases, defaulting to all supported tasks."""

    if not values:
        return list(TASK_SPECS.values())
    if any(value.lower() == "all" for value in values):
        if len(values) != 1:
            raise DatasetInstallError(
                "Task alias 'all' cannot be combined with other --task values."
            )
        return list(TASK_SPECS.values())

    resolved: list[TaskSpec] = []
    seen: set[str] = set()
    for value in values:
        env_id = TASK_ALIASES.get(value, TASK_ALIASES.get(value.lower()))
        if env_id is None:
            aliases = "touch, shuffle, all"
            supported = ", ".join(TASK_SPECS)
            raise DatasetInstallError(
                f"Unknown task {value!r}. Use one of {supported}, or an alias: {aliases}."
            )
        if env_id not in seen:
            resolved.append(TASK_SPECS[env_id])
            seen.add(env_id)
    return resolved


def dataset_path(output_root: str | Path, task: TaskSpec) -> Path:
    """Return the final NPZ path for a registered task."""

    return Path(output_root).expanduser() / task.dataset_slug


def _read_manifest(path: Path) -> dict[str, Any]:
    manifest_path = path / MANIFEST_FILENAME
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DatasetInstallError(
            f"{path} exists but has no {MANIFEST_FILENAME} completion marker. "
            "It may be a collector dataset or an interrupted install; it will not be modified."
        ) from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DatasetInstallError(
            f"Could not read install manifest {manifest_path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise DatasetInstallError(f"Install manifest {manifest_path} must be a JSON object.")
    return value


def _manifest_int(manifest: Mapping[str, Any], key: str, manifest_path: Path) -> int:
    value = manifest.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DatasetInstallError(
            f"Install manifest {manifest_path}: {key!r} must be a non-negative integer."
        )
    return value


def _validate_manifest_header(
    manifest: Mapping[str, Any],
    path: Path,
    task: TaskSpec,
    *,
    expected_repo_id: str | None,
    expected_requested_revision: str | None,
) -> tuple[list[Mapping[str, Any]], int, int]:
    manifest_path = path / MANIFEST_FILENAME
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise DatasetInstallError(
            f"Install manifest {manifest_path} has unsupported schema_version "
            f"{manifest.get('schema_version')!r}."
        )
    if manifest.get("dataset_format") != DATASET_FORMAT:
        raise DatasetInstallError(
            f"Install manifest {manifest_path} has unexpected dataset_format "
            f"{manifest.get('dataset_format')!r}."
        )

    task_data = manifest.get("task")
    if not isinstance(task_data, Mapping):
        raise DatasetInstallError(f"Install manifest {manifest_path}: 'task' must be an object.")
    if task_data.get("env_id") != task.env_id or task_data.get("dataset_slug") != task.dataset_slug:
        raise DatasetInstallError(
            f"Install manifest {manifest_path} belongs to "
            f"{task_data.get('env_id')!r}/{task_data.get('dataset_slug')!r}, not {task.env_id!r}."
        )

    source = manifest.get("source")
    if not isinstance(source, Mapping):
        raise DatasetInstallError(f"Install manifest {manifest_path}: 'source' must be an object.")
    resolved_revision = source.get("resolved_revision")
    if not isinstance(resolved_revision, str) or not _COMMIT_RE.fullmatch(resolved_revision):
        raise DatasetInstallError(
            f"Install manifest {manifest_path}: source.resolved_revision must be a 40-digit commit."
        )
    if expected_repo_id is not None and source.get("repo_id") != expected_repo_id:
        raise DatasetInstallError(
            f"Installed source repo {source.get('repo_id')!r} does not match "
            f"requested repo {expected_repo_id!r}."
        )
    if (
        expected_requested_revision is not None
        and source.get("requested_revision") != expected_requested_revision
    ):
        raise DatasetInstallError(
            f"Installed source revision {source.get('requested_revision')!r} does not match "
            f"requested revision {expected_requested_revision!r}."
        )

    episode_count = _manifest_int(manifest, "episode_count", manifest_path)
    frame_count = _manifest_int(manifest, "frame_count", manifest_path)
    raw_episodes = manifest.get("episodes")
    if not isinstance(raw_episodes, list) or not all(
        isinstance(episode, Mapping) for episode in raw_episodes
    ):
        raise DatasetInstallError(
            f"Install manifest {manifest_path}: 'episodes' must be a list of objects."
        )
    episodes: list[Mapping[str, Any]] = list(raw_episodes)
    if len(episodes) != episode_count:
        raise DatasetInstallError(
            f"Install manifest {manifest_path}: episode_count is {episode_count}, "
            f"but {len(episodes)} episode records are present."
        )
    return episodes, episode_count, frame_count


def verify_install(
    path: str | Path,
    task: TaskSpec,
    *,
    check_hashes: bool,
    expected_repo_id: str | None = None,
    expected_requested_revision: str | None = None,
) -> dict[str, Any]:
    """Validate an installed dataset, optionally reading every array and hash."""

    root = Path(path)
    if root.is_symlink():
        raise DatasetInstallError(f"Refusing dataset symlink: {root}")
    if not root.exists():
        raise DatasetInstallError(f"Dataset is not installed: {root}")
    if not root.is_dir():
        raise DatasetInstallError(f"Dataset target is not a directory: {root}")

    manifest = _read_manifest(root)
    records, episode_count, frame_count = _validate_manifest_header(
        manifest,
        root,
        task,
        expected_repo_id=expected_repo_id,
        expected_requested_revision=expected_requested_revision,
    )
    try:
        discovered = discover_npz_episodes(root)
    except (OSError, DatasetFormatError) as exc:
        raise DatasetInstallError(f"Could not index installed dataset {root}: {exc}") from exc

    expected_names: list[str] = []
    expected_frames = 0
    for position, record in enumerate(records):
        filename = record.get("filename")
        episode_index = record.get("episode_index")
        length = record.get("length")
        size_bytes = record.get("size_bytes")
        digest = record.get("sha256")
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or filename != f"train_data_{position:06d}.npz"
        ):
            raise DatasetInstallError(
                f"Install manifest {root / MANIFEST_FILENAME}: invalid episode filename "
                f"at position {position}: {filename!r}."
            )
        if episode_index != position:
            raise DatasetInstallError(
                f"Install manifest {root / MANIFEST_FILENAME}: episode_index "
                f"{episode_index!r} is not contiguous at position {position}."
            )
        if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
            raise DatasetInstallError(f"Manifest episode {position} has invalid length {length!r}.")
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes <= 0:
            raise DatasetInstallError(
                f"Manifest episode {position} has invalid size_bytes {size_bytes!r}."
            )
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise DatasetInstallError(f"Manifest episode {position} has an invalid SHA-256 digest.")

        episode_path = root / filename
        try:
            actual_size = episode_path.stat().st_size
        except OSError as exc:
            raise DatasetInstallError(
                f"Could not stat installed episode {episode_path}: {exc}"
            ) from exc
        if actual_size != size_bytes:
            raise DatasetInstallError(
                f"Installed episode {episode_path} has size {actual_size}, expected {size_bytes}."
            )
        if check_hashes:
            try:
                actual_digest = sha256_file(episode_path)
            except OSError as exc:
                raise DatasetInstallError(f"Could not checksum {episode_path}: {exc}") from exc
            if actual_digest != digest:
                raise DatasetInstallError(
                    f"Installed episode {episode_path} failed SHA-256 verification."
                )
            try:
                episode = load_npz_episode(episode_path)
            except (OSError, DatasetFormatError) as exc:
                raise DatasetInstallError(
                    f"Installed episode {episode_path} is invalid: {exc}"
                ) from exc
            if episode.episode_id != position or episode.length != length:
                raise DatasetInstallError(
                    f"Installed episode {episode_path} does not match its manifest record."
                )
            if episode.rgb.shape[1:] != (128, 128, 6):
                raise DatasetInstallError(
                    f"Installed episode {episode_path} has RGB shape {episode.rgb.shape}; "
                    "expected [T,128,128,6]."
                )
            if np.any(episode.action < -1.001) or np.any(episode.action > 1.001):
                raise DatasetInstallError(
                    f"Installed episode {episode_path} contains actions outside [-1, 1]."
                )

        expected_names.append(filename)
        expected_frames += length

    actual_names = [episode.name for episode in discovered]
    if actual_names != expected_names:
        raise DatasetInstallError(
            f"Installed dataset {root} has an unexpected episode file set. "
            f"Expected {episode_count} canonical files, found {len(actual_names)}."
        )
    if expected_frames != frame_count:
        raise DatasetInstallError(
            f"Install manifest {root / MANIFEST_FILENAME}: frame_count is {frame_count}, "
            f"but episode lengths sum to {expected_frames}."
        )
    return manifest


def _resolved_revision(snapshot_root: Path, requested_revision: str) -> str:
    if _COMMIT_RE.fullmatch(snapshot_root.name):
        return snapshot_root.name
    if _COMMIT_RE.fullmatch(requested_revision):
        return requested_revision
    raise DatasetInstallError(
        f"Could not determine the immutable commit for downloaded snapshot {snapshot_root}."
    )


def build_manifest(
    task: TaskSpec,
    result: ConversionResult,
    *,
    repo_id: str,
    requested_revision: str,
    resolved_revision: str,
) -> dict[str, Any]:
    """Build the completion marker written after all episode files."""

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset_format": DATASET_FORMAT,
        "task": {
            "env_id": task.env_id,
            "dataset_slug": task.dataset_slug,
        },
        "source": {
            "format": "LeRobotDataset-v3",
            "repo_id": repo_id,
            "requested_revision": requested_revision,
            "resolved_revision": resolved_revision,
            "codebase_version": result.metadata.codebase_version,
            "fps": result.metadata.fps,
        },
        "camera_channel_order": ["top.rgb", "wrist.rgb"],
        "arrays": {
            "rgb": {"dtype": "uint8", "shape": ["T", 128, 128, 6]},
            "proprio": {"dtype": "float32", "shape": ["T", 7]},
            "action": {"dtype": "float32", "shape": ["T", 7]},
            "done": {"dtype": "bool", "shape": ["T"]},
        },
        "episode_count": len(result.episodes),
        "frame_count": result.total_frames,
        "episodes": [
            {
                "filename": episode.filename,
                "episode_index": episode.episode_index,
                "length": episode.length,
                "size_bytes": episode.size_bytes,
                "sha256": episode.sha256,
            }
            for episode in result.episodes
        ],
    }


def _write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    manifest_path = path / MANIFEST_FILENAME
    temporary_path = path / f".{MANIFEST_FILENAME}.tmp"
    try:
        with temporary_path.open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, manifest_path)
    except OSError as exc:
        raise DatasetInstallError(
            f"Could not write install manifest {manifest_path}: {exc}"
        ) from exc


@contextmanager
def _installer_lock(output_root: Path) -> Iterator[None]:
    lock_path = output_root / ".dom-vpwem-install.lock"
    try:
        stream = lock_path.open("a+")
    except OSError as exc:
        raise DatasetInstallError(f"Could not open installer lock {lock_path}: {exc}") from exc
    try:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise DatasetInstallError(
                f"Another dataset installer is already using {output_root}."
            ) from exc
        yield
    finally:
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def _promote(staging: Path, target: Path, *, replace: bool) -> None:
    """Promote a complete sibling directory, preserving old data on failure."""

    if target.is_symlink():
        raise DatasetInstallError(f"Refusing to replace dataset symlink: {target}")
    if target.exists() and not target.is_dir():
        raise DatasetInstallError(f"Refusing to replace non-directory dataset target: {target}")
    if target.exists() and not replace:
        raise DatasetInstallError(f"Dataset target appeared during installation: {target}")
    if not target.exists():
        os.replace(staging, target)
        return

    backup = target.with_name(f".{target.name}.backup-{uuid.uuid4().hex}")
    os.replace(target, backup)
    try:
        os.replace(staging, target)
    except BaseException:
        try:
            os.replace(backup, target)
        except OSError as restore_exc:
            raise DatasetInstallError(
                f"Could not promote {staging} and could not restore the previous dataset. "
                f"The recoverable backup remains at {backup}: {restore_exc}"
            ) from restore_exc
        raise
    try:
        shutil.rmtree(backup)
    except OSError as exc:
        warnings.warn(
            f"Installed {target}, but could not remove recoverable backup {backup}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )


SnapshotFetcher = Callable[..., Path]
TaskConverter = Callable[..., ConversionResult]


def install_datasets(
    tasks: Sequence[TaskSpec],
    *,
    output_root: str | Path = DEFAULT_DATASET_ROOT,
    repo_id: str = LEROBOT_REPO_ID,
    revision: str = LEROBOT_REVISION,
    cache_dir: str | Path | None = None,
    offline: bool = False,
    max_workers: int = 8,
    force: bool = False,
    verify_only: bool = False,
    fetch_snapshot: SnapshotFetcher = download_lerobot_snapshot,
    converter: TaskConverter = convert_lerobot_task,
    print_fn: Callable[[str], None] = print,
) -> list[Path]:
    """Install or verify selected datasets and return their final paths."""

    selected = list(tasks)
    if not selected:
        raise DatasetInstallError("No datasets were selected.")
    requested_root = Path(output_root).expanduser()
    if requested_root.is_symlink():
        raise DatasetInstallError(f"Dataset output root cannot be a symlink: {requested_root}")
    root = requested_root.resolve()

    if verify_only:
        verified: list[Path] = []
        for task in selected:
            target = dataset_path(root, task)
            manifest = verify_install(target, task, check_hashes=True)
            print_fn(
                f"Verified {task.env_id}: {manifest['episode_count']} episodes, "
                f"{manifest['frame_count']} frames at {target}"
            )
            verified.append(target)
        return verified

    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise DatasetInstallError(f"Could not create dataset output root {root}: {exc}") from exc
    if root.is_symlink() or not root.is_dir():
        raise DatasetInstallError(f"Dataset output root must be a real directory: {root}")
    effective_cache_dir = (
        Path(cache_dir).expanduser().resolve()
        if cache_dir is not None
        else root / ".cache" / "huggingface"
    )

    installed: list[Path] = []
    with _installer_lock(root):
        pending: list[TaskSpec] = []
        for task in selected:
            target = dataset_path(root, task)
            if target.is_symlink():
                raise DatasetInstallError(f"Refusing dataset symlink: {target}")
            if target.exists() and not force:
                try:
                    manifest = verify_install(
                        target,
                        task,
                        check_hashes=False,
                        expected_repo_id=repo_id,
                        expected_requested_revision=revision,
                    )
                except DatasetInstallError as exc:
                    raise DatasetInstallError(
                        f"Existing destination {target} is not a matching completed install: {exc} "
                        "Pass --force to build a replacement while preserving the old data until "
                        "the new copy is complete."
                    ) from exc
                print_fn(
                    f"Already installed {task.env_id}: {manifest['episode_count']} episodes at "
                    f"{target} (use --verify-only for full checksums)"
                )
                installed.append(target)
            else:
                if target.exists() and not target.is_dir():
                    raise DatasetInstallError(
                        f"Refusing to replace non-directory dataset target: {target}"
                    )
                pending.append(task)

        if not pending:
            return installed

        print_fn(
            "Downloading official MIKASA LeRobot v3 data for: "
            + ", ".join(task.env_id for task in pending)
        )
        print_fn(f"Hugging Face cache: {effective_cache_dir}")
        snapshot_root = Path(
            fetch_snapshot(
                pending,
                repo_id=repo_id,
                revision=revision,
                cache_dir=effective_cache_dir,
                local_files_only=offline,
                max_workers=max_workers,
            )
        )
        resolved_revision = _resolved_revision(snapshot_root, revision)

        for task in pending:
            source_root = snapshot_root / task.dataset_slug
            if not source_root.is_dir():
                raise DatasetInstallError(
                    f"Downloaded snapshot does not contain task directory {source_root}."
                )
            target = dataset_path(root, task)
            staging_parent = Path(
                tempfile.mkdtemp(prefix=f".{task.dataset_slug}.partial-", dir=root)
            )
            staging = staging_parent / "dataset"

            def report_progress(completed: int, total: int) -> None:
                if completed == 1 or completed % 25 == 0 or completed == total:
                    print_fn(f"  {task.env_id}: converted {completed}/{total} episodes")

            try:
                print_fn(f"Converting {task.env_id} to {target}")
                result = converter(
                    source_root,
                    staging,
                    task,
                    progress=report_progress,
                )
                manifest = build_manifest(
                    task,
                    result,
                    repo_id=repo_id,
                    requested_revision=revision,
                    resolved_revision=resolved_revision,
                )
                _write_manifest(staging, manifest)
                print_fn(f"Validating {task.env_id} checksums and NPZ schema")
                verify_install(
                    staging,
                    task,
                    check_hashes=True,
                    expected_repo_id=repo_id,
                    expected_requested_revision=revision,
                )
                _promote(staging, target, replace=force)
            except (OSError, DatasetFormatError, LeRobotImportError) as exc:
                raise DatasetInstallError(f"Could not install {task.env_id}: {exc}") from exc
            finally:
                if staging_parent.exists():
                    shutil.rmtree(staging_parent, ignore_errors=True)

            print_fn(
                f"Installed {task.env_id}: {len(result.episodes)} episodes, "
                f"{result.total_frames} frames at {target}"
            )
            installed.append(target)

    return installed


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser without importing optional dependencies."""

    parser = argparse.ArgumentParser(
        prog="dom-vpwem-install-datasets",
        description=(
            "Download the pinned public MIKASA LeRobot v3 release and convert it "
            "to the per-episode NPZ files consumed by DOM-VPWEM."
        ),
    )
    parser.add_argument(
        "--task",
        action="append",
        metavar="TASK",
        help=(
            "Task environment ID, dataset slug, or alias ('touch'/'shuffle'). "
            "Repeat to select multiple tasks. Default: both supported tasks."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(DEFAULT_DATASET_ROOT),
        help=f"Directory containing task NPZ folders (default: {DEFAULT_DATASET_ROOT}).",
    )
    parser.add_argument("--repo-id", default=LEROBOT_REPO_ID, help=argparse.SUPPRESS)
    parser.add_argument(
        "--revision",
        default=LEROBOT_REVISION,
        help="Hugging Face revision; the default is an immutable official commit.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Optional Hugging Face cache directory (downloads are resumable).",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use only files already present in the Hugging Face cache.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Concurrent Hugging Face download workers (default: 8).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Build and verify a replacement before atomically swapping existing data.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Do not download; validate all NPZ files against their SHA-256 manifest.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Dataset installer command-line entry point."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.force and args.verify_only:
        parser.error("--force and --verify-only cannot be used together")
    if args.max_workers <= 0:
        parser.error("--max-workers must be positive")
    try:
        tasks = resolve_tasks(args.task)
        install_datasets(
            tasks,
            output_root=args.output_root,
            repo_id=args.repo_id,
            revision=args.revision,
            cache_dir=args.cache_dir,
            offline=args.offline,
            max_workers=args.max_workers,
            force=args.force,
            verify_only=args.verify_only,
        )
    except (DatasetInstallError, LeRobotImportError) as exc:
        parser.exit(1, f"error: {exc}\n")
    except KeyboardInterrupt:
        parser.exit(130, "error: interrupted; no partial dataset was promoted\n")
    return 0


__all__ = [
    "DATASET_FORMAT",
    "DatasetInstallError",
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA_VERSION",
    "build_manifest",
    "build_parser",
    "dataset_path",
    "install_datasets",
    "main",
    "resolve_tasks",
    "verify_install",
]
