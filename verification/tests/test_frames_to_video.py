"""Tests for frames_to_video.py.

docs/plans/2026-07-22-003-feat-frames-to-video-plan.md U1 test scenarios:
- happy path: a directory of known frames produces an MP4 with matching
  frame count and dimensions
- edge case: empty directory raises a clear error
- edge case: non-matching files in the directory are ignored
- edge case: inconsistent frame dimensions raise a clear error naming the
  offending file
- error path: a VideoWriter that fails to open raises a clear RuntimeError
- integration: the CLI entrypoint produces a video file
"""

import sys
from pathlib import Path
from unittest import mock

import cv2
import matplotlib.image as mplimg
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import frames_to_video as f2v


def _write_frame(path: Path, size: tuple[int, int], fill: int) -> None:
    """Write a size=(H, W) uint8 RGB PNG frame filled with a constant value,
    matching render.py's frame_NNNNN.png naming/content conventions closely
    enough for frames_to_video's own loader."""
    h, w = size
    img = np.full((h, w, 3), fill, dtype=np.uint8)
    mplimg.imsave(str(path), img)


class TestFramesToVideoHappyPath:
    def test_frame_count_and_dimensions_match_source(self, tmp_path):
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        for i in range(3):
            _write_frame(frames_dir / f"frame_{i:05d}.png", (32, 48), fill=i * 50)

        output_path = tmp_path / "out.mp4"
        f2v.frames_to_video(frames_dir, output_path, fps=10.0)

        cap = cv2.VideoCapture(str(output_path))
        assert cap.get(cv2.CAP_PROP_FRAME_COUNT) == 3
        assert cap.get(cv2.CAP_PROP_FRAME_WIDTH) == 48
        assert cap.get(cv2.CAP_PROP_FRAME_HEIGHT) == 32
        cap.release()


class TestFramesToVideoEdgeCases:
    def test_empty_directory_raises_clear_error(self, tmp_path):
        frames_dir = tmp_path / "empty"
        frames_dir.mkdir()

        with pytest.raises(ValueError, match="No frame_\\*.png files found"):
            f2v.frames_to_video(frames_dir, tmp_path / "out.mp4")

    def test_non_matching_files_are_ignored(self, tmp_path):
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        _write_frame(frames_dir / "frame_00000.png", (16, 16), fill=100)
        _write_frame(frames_dir / "frame_00001.png", (16, 16), fill=200)
        (frames_dir / "notes.txt").write_text("not a frame")
        (frames_dir / "other_00002.png").write_bytes(b"not a real png either")

        output_path = tmp_path / "out.mp4"
        f2v.frames_to_video(frames_dir, output_path, fps=5.0)

        cap = cv2.VideoCapture(str(output_path))
        assert cap.get(cv2.CAP_PROP_FRAME_COUNT) == 2
        cap.release()

    def test_inconsistent_dimensions_raise_clear_error_naming_file(self, tmp_path):
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        _write_frame(frames_dir / "frame_00000.png", (16, 16), fill=100)
        bad_path = frames_dir / "frame_00001.png"
        _write_frame(bad_path, (24, 24), fill=100)

        with pytest.raises(ValueError, match="frame_00001.png"):
            f2v.frames_to_video(frames_dir, tmp_path / "out.mp4")


class TestFramesToVideoWriterOpenFailure:
    def test_writer_open_failure_raises_runtime_error(self, tmp_path):
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        _write_frame(frames_dir / "frame_00000.png", (16, 16), fill=100)

        fake_writer = mock.MagicMock()
        fake_writer.isOpened.return_value = False
        with mock.patch.object(f2v.cv2, "VideoWriter", return_value=fake_writer):
            with pytest.raises(RuntimeError, match="failed to open"):
                f2v.frames_to_video(frames_dir, tmp_path / "out.mp4")


class TestFramesToVideoCli:
    def test_cli_produces_video_at_output_path(self, tmp_path, monkeypatch):
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        _write_frame(frames_dir / "frame_00000.png", (16, 16), fill=100)
        _write_frame(frames_dir / "frame_00001.png", (16, 16), fill=150)
        output_path = tmp_path / "cli_out.mp4"

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "frames_to_video.py",
                "--frames",
                str(frames_dir),
                "--output",
                str(output_path),
                "--fps",
                "8",
            ],
        )
        f2v.main()

        assert output_path.exists()
        assert output_path.stat().st_size > 0
