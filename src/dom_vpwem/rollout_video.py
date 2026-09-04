"""Optional MP4 recording for canonical MIKASA policy observations.

The encoder is deliberately loaded only when a video is requested.  Ordinary
training, dataset, and evaluation imports therefore do not require ImageIO or
start an FFmpeg subprocess.
"""

from __future__ import annotations

import importlib
import os
import uuid
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np

from .mikasa_env import MikasaContractError, split_rgb

DEFAULT_VIDEO_FPS = 20
VIDEO_SCALE = 2
STATUS_BAR_HEIGHT = 48


class RolloutVideoError(RuntimeError):
    """Raised when an MP4 rollout cannot be encoded or finalized."""


class RolloutVideoDependencyError(ImportError):
    """Raised when optional MP4 encoding dependencies are unavailable."""


class RolloutFrameSink(Protocol):
    """Consumer for observations captured during one rollout."""

    def write_frame(
        self,
        observation: Mapping[str, np.ndarray],
        *,
        step: int,
        success: bool,
    ) -> None: ...


def compose_policy_views(rgb: Any) -> np.ndarray:
    """Return the top/base and wrist policy views side by side as RGB."""

    top, wrist = split_rgb(rgb)
    if top.dtype != np.uint8:
        raise MikasaContractError(
            f"Rollout video requires uint8 RGB observations, got dtype {top.dtype}."
        )
    return np.ascontiguousarray(np.concatenate((top, wrist), axis=1))


def _load_video_dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        imageio = importlib.import_module("imageio.v2")
        image_module = importlib.import_module("PIL.Image")
        image_draw = importlib.import_module("PIL.ImageDraw")
        image_font = importlib.import_module("PIL.ImageFont")
    except (ImportError, ModuleNotFoundError) as exc:
        raise RolloutVideoDependencyError(
            "MP4 recording requires ImageIO, Pillow, and ImageIO's FFmpeg backend. "
            "Install them with `uv sync --locked --extra eval`."
        ) from exc
    return imageio, image_module, image_draw, image_font


def _default_font(image_font: Any, *, size: int) -> Any:
    try:
        return image_font.load_default(size=size)
    except TypeError:  # Pillow before the scalable default font.
        return image_font.load_default()


def _annotate_frame(
    camera_frame: np.ndarray,
    *,
    episode_index: int,
    seed: int,
    step: int,
    horizon: int,
    success: bool,
    image_module: Any,
    image_draw: Any,
    image_font: Any,
) -> np.ndarray:
    """Upscale and label one side-by-side RGB frame."""

    height, width, _ = camera_frame.shape
    output_width = width * VIDEO_SCALE
    camera_height = height * VIDEO_SCALE
    output_height = camera_height + STATUS_BAR_HEIGHT

    resampling = getattr(image_module, "Resampling", image_module)
    camera_image = image_module.fromarray(camera_frame).resize(
        (output_width, camera_height),
        resample=resampling.BILINEAR,
    )
    canvas = image_module.new("RGB", (output_width, output_height), (18, 20, 24))
    canvas.paste(camera_image, (0, STATUS_BAR_HEIGHT))

    draw = image_draw.Draw(canvas)
    primary_font = _default_font(image_font, size=15)
    secondary_font = _default_font(image_font, size=13)
    draw.text(
        (8, 3),
        f"Episode {episode_index + 1}  |  seed {seed}  |  step {step}/{horizon}",
        fill=(245, 245, 245),
        font=primary_font,
    )
    draw.text((8, 28), "Overhead camera", fill=(205, 215, 230), font=secondary_font)
    draw.text((width + 8, 28), "Wrist camera", fill=(205, 215, 230), font=secondary_font)

    status_text = "SUCCESS: YES" if success else "SUCCESS: NO"
    status_fill = (40, 130, 75) if success else (120, 75, 35)
    badge_left = output_width - 112
    draw.rounded_rectangle((badge_left, 25, output_width - 7, 44), radius=4, fill=status_fill)
    draw.text(
        (badge_left + 7, 28),
        status_text,
        fill=(255, 255, 255),
        font=secondary_font,
    )
    draw.line(
        (width, STATUS_BAR_HEIGHT, width, output_height - 1),
        fill=(255, 255, 255),
        width=1,
    )
    return np.ascontiguousarray(np.asarray(canvas, dtype=np.uint8))


class Mp4RolloutRecorder:
    """Stream one rollout to an atomically finalized H.264 MP4 file."""

    def __init__(
        self,
        output: str | Path,
        *,
        fps: int = DEFAULT_VIDEO_FPS,
        episode_index: int,
        seed: int,
        horizon: int,
    ) -> None:
        self.output = Path(output)
        if self.output.suffix.lower() != ".mp4":
            raise ValueError(f"Video output must end in .mp4, got {self.output}.")
        if isinstance(fps, bool) or not isinstance(fps, int) or fps <= 0:
            raise ValueError(f"video fps must be a positive integer, got {fps!r}.")
        if episode_index < 0:
            raise ValueError(f"episode_index must be non-negative, got {episode_index}.")
        if horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}.")

        self.fps = fps
        self.episode_index = episode_index
        self.seed = seed
        self.horizon = horizon
        self.frames_written = 0
        self._writer: Any | None = None
        self._temporary: Path | None = None
        self._entered = False
        self._closed = False

    def __enter__(self) -> "Mp4RolloutRecorder":
        if self._entered or self._closed:
            raise RuntimeError("A rollout recorder cannot be reused.")
        self.output.parent.mkdir(parents=True, exist_ok=True)
        token = uuid.uuid4().hex
        self._temporary = self.output.with_name(f".{self.output.stem}.{token}.partial.mp4")
        self._entered = True
        return self

    def _open_writer(self, imageio: Any) -> Any:
        assert self._temporary is not None
        try:
            return imageio.get_writer(
                str(self._temporary),
                format="FFMPEG",
                mode="I",
                fps=self.fps,
                codec="libx264",
                pixelformat="yuv420p",
                macro_block_size=16,
                quality=8,
                ffmpeg_log_level="error",
            )
        except Exception as exc:
            raise RolloutVideoError(
                f"Could not start the MP4 encoder for {self.output}. "
                "Check that the eval extra and its FFmpeg backend are installed."
            ) from exc

    def write_frame(
        self,
        observation: Mapping[str, np.ndarray],
        *,
        step: int,
        success: bool,
    ) -> None:
        if not self._entered or self._closed:
            raise RuntimeError("Use Mp4RolloutRecorder as a context manager before writing.")
        if "rgb" not in observation:
            raise MikasaContractError("Rollout observation is missing the 'rgb' key.")

        try:
            imageio, image_module, image_draw, image_font = _load_video_dependencies()
            camera_frame = compose_policy_views(observation["rgb"])
            frame = _annotate_frame(
                camera_frame,
                episode_index=self.episode_index,
                seed=self.seed,
                step=step,
                horizon=self.horizon,
                success=success,
                image_module=image_module,
                image_draw=image_draw,
                image_font=image_font,
            )
            if self._writer is None:
                self._writer = self._open_writer(imageio)
            self._writer.append_data(frame)
        except (MikasaContractError, RolloutVideoDependencyError):
            self.abort()
            raise
        except RolloutVideoError:
            self.abort()
            raise
        except Exception as exc:
            self.abort()
            raise RolloutVideoError(
                f"Could not encode frame {step} for {self.output}: {exc}"
            ) from exc
        self.frames_written += 1

    def _remove_temporary(self) -> None:
        if self._temporary is not None:
            self._temporary.unlink(missing_ok=True)

    def abort(self) -> None:
        """Close the encoder and remove an incomplete output."""

        if self._closed:
            return
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:
                pass
            self._writer = None
        self._remove_temporary()
        self._closed = True

    def close(self) -> None:
        """Flush the encoder and atomically publish the completed MP4."""

        if self._closed:
            return
        if not self._entered:
            raise RuntimeError("Use Mp4RolloutRecorder as a context manager before closing.")
        if self._writer is None or self.frames_written == 0:
            self.abort()
            raise RolloutVideoError(f"No rollout frames were written for {self.output}.")

        try:
            self._writer.close()
            self._writer = None
        except Exception as exc:
            self._writer = None
            self._remove_temporary()
            self._closed = True
            raise RolloutVideoError(f"Could not finalize MP4 video {self.output}: {exc}") from exc

        assert self._temporary is not None
        if not self._temporary.is_file() or self._temporary.stat().st_size == 0:
            self._remove_temporary()
            self._closed = True
            raise RolloutVideoError(f"The MP4 encoder produced no output for {self.output}.")
        try:
            os.replace(self._temporary, self.output)
        except OSError as exc:
            self._remove_temporary()
            self._closed = True
            raise RolloutVideoError(f"Could not save MP4 video to {self.output}: {exc}") from exc
        self._closed = True

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()


__all__ = [
    "DEFAULT_VIDEO_FPS",
    "Mp4RolloutRecorder",
    "RolloutFrameSink",
    "RolloutVideoDependencyError",
    "RolloutVideoError",
    "compose_policy_views",
]
