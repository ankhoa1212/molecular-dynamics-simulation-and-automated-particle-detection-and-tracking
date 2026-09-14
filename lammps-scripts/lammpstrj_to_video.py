#!/usr/bin/env python3
"""
Convert LAMMPS trajectory files (.lammpstrj) to AVI videos using OVITO.
Accepts either a single file or a folder containing multiple trajectory files.
"""

import argparse
import sys
from pathlib import Path
from ovito.io import import_file
from ovito.vis import Viewport, TachyonRenderer


def convert_trajectory_to_video(input_file, output_file=None, width=1920, height=1080, fps=30):
    """
    Convert a LAMMPS trajectory file to an AVI video.

    Args:
        input_file: Path to the .lammpstrj file
        output_file: Path to the output .avi file (optional)
        width: Video width in pixels
        height: Video height in pixels
        fps: Frames per second
    """
    input_path = Path(input_file)

    if not input_path.exists():
        print(f"Error: File '{input_file}' not found.")
        return False

    if output_file is None:
        output_file = input_path.with_suffix(".avi")
    else:
        output_file = Path(output_file)

    print(f"Converting: {input_path} -> {output_file}")

    try:
        pipeline = import_file(str(input_path))
        pipeline.add_to_scene()
        pipeline.compute()

        viewport = Viewport()
        viewport.type = Viewport.Type.Perspective
        viewport.fov = 35.0
        viewport.zoom_all()

        viewport.render_anim(
            filename=str(output_file), size=(width, height), renderer=TachyonRenderer(), fps=fps
        )

        print(f"Successfully created: {output_file}")
        return True

    except Exception as e:  # pylint: disable=broad-except
        print(f"Error converting {input_file}: {str(e)}")
        return False


def process_folder(folder_path, output_dir=None, width=1920, height=1080, fps=30):
    """
    Process all .lammpstrj files in a folder.

    Args:
        folder_path: Path to the folder containing trajectory files
        output_dir: Optional output directory for videos
        width: Video width in pixels
        height: Video height in pixels
        fps: Frames per second
    """
    folder = Path(folder_path)

    if not folder.exists():
        print(f"Error: Folder '{folder_path}' not found.")
        return

    if not folder.is_dir():
        print(f"Error: '{folder_path}' is not a directory.")
        return

    trajectory_files = list(folder.glob("*.lammpstrj"))

    if not trajectory_files:
        print(f"No .lammpstrj files found in '{folder_path}'")
        return

    print(f"Found {len(trajectory_files)} trajectory file(s)")

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = folder

    success_count = 0
    for traj_file in trajectory_files:
        output_file = output_path / traj_file.with_suffix(".avi").name
        if convert_trajectory_to_video(traj_file, output_file, width, height, fps):
            success_count += 1

    print(
        f"\nConversion complete: {success_count}/{len(trajectory_files)} "
        "files successfully converted"
    )


def main():
    """Main entry point for command-line execution."""
    parser = argparse.ArgumentParser(
        description="Convert LAMMPS trajectory files (.lammpstrj) to AVI videos using OVITO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert a single file
  %(prog)s trajectory.lammpstrj
  
  # Convert a single file with custom output
  %(prog)s trajectory.lammpstrj -o output.avi
  
  # Convert all files in a folder
  %(prog)s /path/to/folder/
  
  # Convert with custom resolution and fps
  %(prog)s trajectory.lammpstrj --width 3840 --height 2160 --fps 60
        """,
    )

    parser.add_argument(
        "input", help="Path to a .lammpstrj file or folder containing trajectory files"
    )

    parser.add_argument(
        "-o", "--output", help="Output file path (for single file) or output directory (for folder)"
    )

    parser.add_argument(
        "--width", type=int, default=1920, help="Video width in pixels (default: 1920)"
    )

    parser.add_argument(
        "--height", type=int, default=1080, help="Video height in pixels (default: 1080)"
    )

    parser.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30)")

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: '{args.input}' not found.")
        sys.exit(1)

    if input_path.is_file():
        convert_trajectory_to_video(input_path, args.output, args.width, args.height, args.fps)
    elif input_path.is_dir():
        process_folder(input_path, args.output, args.width, args.height, args.fps)
    else:
        print(f"Error: '{args.input}' is neither a file nor a directory.")
        sys.exit(1)


if __name__ == "__main__":
    main()
