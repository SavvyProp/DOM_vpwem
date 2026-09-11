"""Export a successful saved NPZ episode without running the simulator."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping

import numpy as np

from .data import derive_safe_episode_length, discover_npz_episodes, load_npz_episode
from .rollout_video import DEFAULT_VIDEO_FPS, Mp4RolloutRecorder


def _boolean_flags(value: np.ndarray, *, key: str, path: Path) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "biuf" or not np.isin(array, (0, 1)).all():
        raise ValueError(f"{path}: {key!r} must contain boolean or numeric 0/1 flags.")
    return array.astype(bool, copy=False)


def _success_metadata(
    archive: Mapping[str, np.ndarray], path: Path
) -> tuple[bool | None, np.ndarray | None]:
    """Read success only within the valid prefix, without decompressing RGB."""

    outcome = None
    if "success_once" in archive:
        flag = _boolean_flags(archive["success_once"], key="success_once", path=path)
        if flag.size != 1:
            raise ValueError(f"{path}: 'success_once' must be scalar.")
        outcome = bool(flag.item())

    timeline = None
    if "success" in archive:
        if "action" not in archive or archive["action"].ndim != 2:
            raise ValueError(f"{path}: 'action' must have shape [T, 7].")
        full_length = archive["action"].shape[0]
        flags = _boolean_flags(archive["success"], key="success", path=path)
        if flags.ndim == 0 or flags.shape[0] != full_length or flags.size == 0:
            raise ValueError(f"{path}: 'success' must have leading length {full_length}.")
        length = derive_safe_episode_length(archive, full_length, path=path)
        timeline = np.maximum.accumulate(flags.reshape(full_length, -1).any(axis=1)[:length])
        # An explicit failure cannot be overridden, and success after the safe
        # endpoint cannot qualify an episode even when success_once is true.
        outcome = bool(timeline[-1]) and outcome is not False
    return outcome, timeline


def select_episode(
    dataset_dir: str | Path,
    *,
    episode_id: int | None = None,
    allow_unknown_success: bool = False,
) -> Path:
    """Choose the first verified success, or an explicitly allowed unknown."""

    if episode_id is not None and episode_id < 0:
        raise ValueError("episode_id must be non-negative.")
    paths = discover_npz_episodes(dataset_dir)
    if episode_id is not None:
        paths = [path for path in paths if int(path.stem.rsplit("_", 1)[1]) == episode_id]
    if not paths:
        requested = "" if episode_id is None else f" with ID {episode_id}"
        raise ValueError(f"No train_data_*.npz episodes{requested} found in {dataset_dir}.")

    first_unknown = None
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            outcome, _ = _success_metadata(archive, path)
        if outcome is True:
            return path
        if outcome is None and first_unknown is None:
            first_unknown = path

    if first_unknown is not None and allow_unknown_success:
        return first_unknown
    hint = ""
    if first_unknown is not None:
        hint = (
            " Some episodes have no success metadata; use --allow-unknown-success "
            "to export one with an UNKNOWN success label."
        )
    raise ValueError(f"No confirmed successful episode found in {dataset_dir}.{hint}")


def export_dataset_episode(
    dataset_dir: str | Path,
    output: str | Path | None = None,
    *,
    episode_id: int | None = None,
    fps: int = DEFAULT_VIDEO_FPS,
    allow_unknown_success: bool = False,
) -> Path:
    """Encode the selected episode's valid RGB frames and recorded status."""

    source = select_episode(
        dataset_dir, episode_id=episode_id, allow_unknown_success=allow_unknown_success
    )
    episode = load_npz_episode(source)
    with np.load(source, allow_pickle=False) as archive:
        outcome, timeline = _success_metadata(archive, source)
        seed = None
        if "episode_seed" in archive:
            value = np.asarray(archive["episode_seed"])
            if value.size != 1 or value.dtype.kind not in "iu":
                raise ValueError(f"{source}: 'episode_seed' must be an integer scalar.")
            seed = int(value.item())

    destination = (
        Path(output).expanduser()
        if output is not None
        else Path("eval_results/videos") / f"{source.parent.name}_{source.stem}.mp4"
    )
    with Mp4RolloutRecorder(
        destination,
        fps=fps,
        episode_index=episode.episode_id,
        seed=seed,
        horizon=episode.length,
    ) as recorder:
        for step, rgb in enumerate(episode.rgb):
            # With only episode-level metadata, report the overall outcome
            # throughout rather than inventing a success timestep.
            success = bool(timeline[step]) if timeline is not None else outcome
            recorder.write_frame({"rgb": rgb}, step=step, success=success)

    status = "confirmed successful" if outcome else "success UNKNOWN (metadata absent)"
    print(f"Selected {source}: {status}")
    print(
        f"Saved {destination}: {episode.length} frames at {fps} FPS "
        f"({episode.length / fps:.2f} seconds)"
    )
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True, help="NPZ episode directory.")
    parser.add_argument(
        "--output",
        type=Path,
        help="MP4 destination; defaults to eval_results/videos/<dataset>_<episode>.mp4.",
    )
    parser.add_argument(
        "--episode-id",
        type=int,
        help="Numeric train_data_<ID>.npz ID; otherwise select the first successful episode.",
    )
    parser.add_argument(
        "--fps", type=int, default=DEFAULT_VIDEO_FPS, help="Playback FPS (default: 20)."
    )
    parser.add_argument(
        "--allow-unknown-success",
        action="store_true",
        help="If no verified success exists, allow missing success metadata; never allow failures.",
    )
    args = parser.parse_args(argv)
    if args.fps <= 0:
        parser.error("--fps must be positive")
    try:
        export_dataset_episode(**vars(args))
    except (OSError, ValueError, ImportError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
