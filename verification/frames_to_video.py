#!/usr/bin/env python3
"""Stitch a directory of rendered frame PNGs into an MP4.

Mirrors particle-tracking/track.py's cv2.VideoWriter pattern (mp4v fourcc,
RGB->BGR conversion before write) rather than shelling out to a system
ffmpeg binary. Usable both from render.py's --video flag and standalone
against any existing frame directory. See docs/plans/2026-07-22-003-feat-
frames-to-video-plan.md.

Usage:
    uv run python frames_to_video.py --frames verification_output/synthetic_frames/ \
        --output verification_output/preview.mp4 [--fps 10]
"""

import argparse
import sys
from pathlib import Path

import cv2

from benchmark import _load_frame_rgb


def frames_to_video(frames_dir: Path, output_path: Path, fps: float = 10.0) -> None:
    """Glob, sort, and encode frames_dir's frame_*.png files into an MP4 at
    output_path.

    Raises ValueError if frames_dir has no matching frames, or if a frame's
    dimensions don't match the first frame's (cv2.VideoWriter.write() would
    otherwise silently drop a mismatched frame with no error). Raises
    RuntimeError if the VideoWriter fails to open (e.g. an unsupported
    codec) -- cv2 does not raise on this itself, it just makes every
    subsequent .write() a silent no-op.
    """
    frames_dir = Path(frames_dir)
    frame_paths = sorted(frames_dir.glob("frame_*.png"))
    if not frame_paths:
        raise ValueError(f"No frame_*.png files found in {frames_dir}")

    first = _load_frame_rgb(frame_paths[0])
    h, w = first.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(
            f"cv2.VideoWriter failed to open for '{output_path}' (fourcc mp4v, {w}x{h} @ "
            f"{fps}fps) -- the installed opencv-python build may lack mp4v codec support."
        )

    try:
        writer.write(cv2.cvtColor(first, cv2.COLOR_RGB2BGR))
        for frame_path in frame_paths[1:]:
            frame = _load_frame_rgb(frame_path)
            if frame.shape[:2] != (h, w):
                raise ValueError(
                    f"Frame '{frame_path}' has shape {frame.shape[:2]}, expected {(h, w)} "
                    f"(from the first frame, '{frame_paths[0]}') -- all frames in a directory "
                    "must share the same dimensions."
                )
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def main():
    parser = argparse.ArgumentParser(description="Stitch a directory of frame PNGs into an MP4")
    parser.add_argument("--frames", required=True, help="Directory of frame_*.png files")
    parser.add_argument("--output", required=True, help="Output .mp4 path")
    parser.add_argument("--fps", type=float, default=10.0, help="Output video frame rate")
    args = parser.parse_args()

    try:
        frames_to_video(Path(args.frames), Path(args.output), fps=args.fps)
    except (ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Video → {args.output}")


if __name__ == "__main__":
    main()
