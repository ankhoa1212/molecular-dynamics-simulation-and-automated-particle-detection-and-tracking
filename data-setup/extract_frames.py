"""Convert multi-page TIFF to image frames."""

import argparse
import os

import cv2
import numpy as np
import tifffile
from tqdm import tqdm


def convert_jpg_to_frames(input_path, output_folder, image_format="png"):
    """Convert a JPG image to different image format."""
    if os.path.isdir(input_path):
        jpg_files = sorted(
            [f for f in os.listdir(input_path) if f.lower().endswith((".jpg", ".jpeg"))]
        )
        if not jpg_files:
            print(f"No .jpg files found in directory: {input_path}")
            return
        for fname in tqdm(jpg_files, desc="Converting JPGs"):
            base = os.path.splitext(fname)[0]
            convert_jpg_to_frames(
                os.path.join(input_path, fname), output_folder, image_format=image_format
            )
        return

    os.makedirs(output_folder, exist_ok=True)

    try:
        print(input_path)
        img = cv2.imread(input_path)
        base = os.path.splitext(os.path.basename(input_path))[0]
        cv2.imwrite(os.path.join(output_folder, f"{base}.{image_format}"), img)
    except cv2.error as e:
        print(f"Error converting JPG '{input_path}': {e}")


def convert_tif_to_frames(input_path, output_folder, image_format="png", nth=10):
    """Convert multi-page TIFF files to individual image frames saved in output_folder."""

    # If input_path is a directory, find .tif files and process each
    if os.path.isdir(input_path):
        tif_files = sorted(
            [f for f in os.listdir(input_path) if f.lower().endswith((".tif", ".tiff"))]
        )
        if not tif_files:
            print(f"No .tif/.tiff files found in directory: {input_path}")
            return
        for fname in tqdm(tif_files, desc="Converting TIFs"):
            base = os.path.splitext(fname)[0]
            out_subdir = os.path.join(output_folder, f"{base}_frames")
            os.makedirs(out_subdir, exist_ok=True)
            convert_tif_to_frames(
                os.path.join(input_path, fname), out_subdir, image_format=image_format, nth=nth
            )
        return

    # 1. Create output directory (for single file)
    os.makedirs(output_folder, exist_ok=True)

    # 2. Read the TIF file
    try:
        tiff_stack = tifffile.imread(input_path)
        print(
            f"Data shape: {tiff_stack.shape} - reading {input_path}"
        )  # Usually (Frames, Height, Width)
    except (OSError, ValueError) as e:
        print(f"Error reading TIF '{input_path}': {e}")
        return

    # 3. Iterate through frames and save every nth frame
    saved_count = 0
    frame_indices = range(0, len(tiff_stack), nth)
    for i in tqdm(frame_indices, desc="Saving frames", unit="frame"):
        frame = tiff_stack[i]

        # 4. Normalization
        if frame.dtype != np.uint8:
            frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

        # 5. Handle Color Channels
        if len(frame.shape) == 2:
            frame = cv2.merge([frame, frame, frame])

        # 6. Save Frame
        cv2.imwrite(os.path.join(output_folder, f"frame_{saved_count:05d}.{image_format}"), frame)
        saved_count += 1

    print(f"Successfully converted {saved_count} frames (every {nth}th) to {output_folder}")


INPUT_PATH = (
    "/mnt/c/Users/ankho/git/molecular-dynamics-simulation/raw_data/2024.07.02/"
    "Trial 1 Au Citrate Best Trials/"
    "Au Cit+1% of 2um PS+NaCl 20% Light Intensity Test Video 300 ms Trial 17_1"
)


if __name__ == "__main__":

    PARSER = argparse.ArgumentParser(description="Convert multi-page TIFF to image frames.")
    PARSER.add_argument(
        "input_path",
        nargs="?",
        default=INPUT_PATH,
        help="Path to input .tif file or a directory containing .tif files",
    )
    PARSER.add_argument("output_dir", nargs="?", default=None, help="Directory to save frames")
    PARSER.add_argument(
        "--jpg", action="store_true", dest="convert_jpg", help="Convert JPG images instead of TIFF"
    )
    PARSER.add_argument(
        "-n", "--nth", dest="nth", type=int, default=10, help="Save every nth frame (default: 10)"
    )
    PARSER.add_argument(
        "-f",
        "--format",
        dest="image_format",
        default="png",
        help="Output image format (default: png)",
    )
    ARGS = PARSER.parse_args()

    if ARGS.output_dir is None:
        BASE_NAME = os.path.splitext(os.path.basename(ARGS.input_path))[0]
        ARGS.output_dir = os.path.join(os.getcwd(), f"{BASE_NAME}_frames")
    os.makedirs(ARGS.output_dir, exist_ok=True)

    if ARGS.convert_jpg:
        convert_jpg_to_frames(ARGS.input_path, ARGS.output_dir, ARGS.image_format)
    else:
        convert_tif_to_frames(ARGS.input_path, ARGS.output_dir, ARGS.image_format, ARGS.nth)
