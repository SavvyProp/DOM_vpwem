from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

from dom_vpwem import rollout_video
from dom_vpwem.mikasa_env import MikasaContractError
from dom_vpwem.rollout_video import (
    Mp4RolloutRecorder,
    RolloutVideoError,
    compose_policy_views,
)


def _rgb(top=(10, 20, 30), wrist=(100, 110, 120)) -> np.ndarray:
    rgb = np.empty((128, 128, 6), dtype=np.uint8)
    rgb[..., :3] = top
    rgb[..., 3:] = wrist
    return rgb


class _FakeWriter:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.frames: list[np.ndarray] = []
        self.closed = False

    def append_data(self, frame: np.ndarray) -> None:
        self.frames.append(frame.copy())

    def close(self) -> None:
        self.path.write_bytes(b"fake-mp4")
        self.closed = True


class _FakeImageIO:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.writer: _FakeWriter | None = None

    def get_writer(self, path: str, **kwargs):
        self.calls.append((path, kwargs))
        self.writer = _FakeWriter(path)
        return self.writer


def _patch_video_dependencies(monkeypatch) -> _FakeImageIO:
    fake_imageio = _FakeImageIO()
    image_module = importlib.import_module("PIL.Image")
    image_draw = importlib.import_module("PIL.ImageDraw")
    image_font = importlib.import_module("PIL.ImageFont")
    monkeypatch.setattr(
        rollout_video,
        "_load_video_dependencies",
        lambda: (fake_imageio, image_module, image_draw, image_font),
    )
    return fake_imageio


def test_compose_policy_views_places_top_then_wrist() -> None:
    frame = compose_policy_views(_rgb())

    assert frame.shape == (128, 256, 3)
    assert frame.dtype == np.uint8
    assert frame.flags.c_contiguous
    np.testing.assert_array_equal(frame[:, :128], np.full((128, 128, 3), (10, 20, 30)))
    np.testing.assert_array_equal(frame[:, 128:], np.full((128, 128, 3), (100, 110, 120)))

    with pytest.raises(MikasaContractError, match="uint8"):
        compose_policy_views(_rgb().astype(np.float32))


def test_recorder_encodes_annotated_rgb_and_atomically_replaces_output(
    monkeypatch, tmp_path: Path
) -> None:
    fake_imageio = _patch_video_dependencies(monkeypatch)
    output = tmp_path / "nested" / "rollout.mp4"
    output.parent.mkdir()
    output.write_bytes(b"old")

    recorder = Mp4RolloutRecorder(
        output,
        fps=20,
        episode_index=1,
        seed=123,
        horizon=4,
    )
    with recorder:
        recorder.write_frame({"rgb": _rgb()}, step=0, success=False)
        recorder.write_frame({"rgb": _rgb()}, step=1, success=True)

    assert recorder.frames_written == 2
    assert output.read_bytes() == b"fake-mp4"
    assert list(output.parent.glob(".*.partial.mp4")) == []
    assert len(fake_imageio.calls) == 1
    _, writer_kwargs = fake_imageio.calls[0]
    assert writer_kwargs == {
        "format": "FFMPEG",
        "mode": "I",
        "fps": 20,
        "codec": "libx264",
        "pixelformat": "yuv420p",
        "macro_block_size": 16,
        "quality": 8,
        "ffmpeg_log_level": "error",
    }

    assert fake_imageio.writer is not None
    assert fake_imageio.writer.closed
    assert len(fake_imageio.writer.frames) == 2
    frame = fake_imageio.writer.frames[0]
    assert frame.shape == (304, 512, 3)
    assert frame.dtype == np.uint8
    assert frame.flags.c_contiguous
    np.testing.assert_array_equal(frame[100, 100], np.array((10, 20, 30)))
    np.testing.assert_array_equal(frame[100, 400], np.array((100, 110, 120)))


def test_recorder_removes_partial_video_when_rollout_fails(monkeypatch, tmp_path: Path) -> None:
    _patch_video_dependencies(monkeypatch)
    output = tmp_path / "rollout.mp4"
    recorder = Mp4RolloutRecorder(
        output,
        episode_index=0,
        seed=123,
        horizon=4,
    )

    with pytest.raises(RuntimeError, match="policy failed"):
        with recorder:
            recorder.write_frame({"rgb": _rgb()}, step=0, success=False)
            raise RuntimeError("policy failed")

    assert not output.exists()
    assert list(tmp_path.glob(".*.partial.mp4")) == []


def test_recorder_reports_encoder_failure_and_removes_partial(monkeypatch, tmp_path: Path) -> None:
    fake_imageio = _patch_video_dependencies(monkeypatch)

    class FailingWriter(_FakeWriter):
        def append_data(self, frame: np.ndarray) -> None:
            raise OSError("encoder stopped")

    def get_failing_writer(path: str, **kwargs):
        del kwargs
        fake_imageio.writer = FailingWriter(path)
        return fake_imageio.writer

    monkeypatch.setattr(fake_imageio, "get_writer", get_failing_writer)
    output = tmp_path / "rollout.mp4"
    output.write_bytes(b"old-video")

    with pytest.raises(RolloutVideoError, match="frame 0.*encoder stopped"):
        with Mp4RolloutRecorder(
            output,
            episode_index=0,
            seed=123,
            horizon=4,
        ) as recorder:
            recorder.write_frame({"rgb": _rgb()}, step=0, success=False)

    assert output.read_bytes() == b"old-video"
    assert list(tmp_path.glob(".*.partial.mp4")) == []


@pytest.mark.parametrize(
    ("output", "fps", "message"),
    [
        ("rollout.mov", 20, "end in .mp4"),
        ("rollout.mp4", 0, "positive integer"),
        ("rollout.mp4", 2.5, "positive integer"),
    ],
)
def test_recorder_validates_output_and_fps(output, fps, message) -> None:
    with pytest.raises(ValueError, match=message):
        Mp4RolloutRecorder(output, fps=fps, episode_index=0, seed=1, horizon=4)
