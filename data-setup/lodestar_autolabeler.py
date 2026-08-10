"""
Use LodeSTAR for generating YOLO labels on TIFF files.
Recursively searches for .tif/.tiff files, extracts nth frames, and generates YOLO labels.
Output structure mimics RoboFlow: <filename>_dataset/{images, labels}/
"""

import argparse
import json
import os
import glob
import logging
from typing import NamedTuple

import numpy as np
import torch
import cv2  # pylint: disable=no-member
import tifffile
from tqdm import tqdm

# box_size/nms_distance resolution mirrors particle-tracking/track.py's
# equivalent LodeSTAR wiring: an explicit value (CLI, or JSON via the
# --config merge below) always wins; otherwise, when --dataset-profile is
# supplied, both derive from it; otherwise each falls back to today's
# existing hardcoded default (40 / 0.0), applied in _resolve_scale_params
# below rather than baked into argparse -- see that function's docstring.
from detectors_common.scale_derivation import resolve_box_size, resolve_nms_distance
from detectors_common.dataset_profile import load_dataset_profile

# Reuse shared utilities from lodestar_utils
from lodestar_utils import _nms, _load_model, _SaveConfig, _write_frame

# Suppress pint logging
logging.getLogger("pint").setLevel(logging.ERROR)


class _FrameCtx(NamedTuple):
    """Bundles inference state passed through the frame-processing pipeline."""

    model: object
    args: object
    device: str


def parse_args():
    """Parse command-line arguments for the LodeSTAR autolabeler."""
    parser = argparse.ArgumentParser(description="LodeSTAR Autolabeler for TIFF files.")
    parser.add_argument(
        "--model", type=str, required=False, help="Path to the saved LodeSTAR .pt weights file."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Root directory to search for .tif/.tiff files recursively.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a JSON configuration file containing these arguments.",
    )
    parser.add_argument(
        "--dataset-profile",
        type=str,
        default=None,
        help=(
            "Path to a dataset scale profile YAML (size_px/spacing_px). When set, "
            "--box-size/--nms-distance derive from it via detectors_common.scale_derivation "
            "unless explicitly passed (directly, or via --config)."
        ),
    )
    parser.add_argument("--nth", type=int, default=5, help="Save every nth frame (default: 5).")
    parser.add_argument(
        "--box-size",
        type=int,
        default=None,
        help=(
            "Fixed bounding box size in pixels. Default (40) is applied after "
            "--config/--dataset-profile resolution, not here -- see main()."
        ),
    )
    parser.add_argument(
        "--detect-batch-size", type=int, default=4, help="Batch size for detection to prevent OOM."
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Alpha parameter for LodeSTAR detect (default: 0.5).",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=0.5,
        help="Cutoff parameter for LodeSTAR detect (default: 0.5).",
    )
    parser.add_argument(
        "--nms-distance",
        type=float,
        default=None,
        help=(
            "Minimum pixel distance between detections (NMS). 0 disables NMS. "
            "Default (0.0) is applied after --config/--dataset-profile resolution, "
            "not here -- see main()."
        ),
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Overlay detections on the input image and save as <n>_overlay.png.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory to write YOLO label files (and overlays). "
            "Defaults to <n>_dataset/labels/ next to the input. "
            "For TIFF mode with multiple files, each TIFF gets a sub-folder."
        ),
    )
    parser.add_argument(
        "--use-radius",
        action="store_true",
        help=(
            "Use the model's per-detection radius as box size instead of --box-size. "
            "Requires num_outputs >= 3 in the saved model."
        ),
    )
    parser.add_argument(
        "--radius-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to the raw radius output to convert it to pixels.",
    )
    parser.add_argument(
        "--min-box-size",
        type=float,
        default=0.0,
        help=(
            "Minimum box size in pixels when --use-radius is active. "
            "0 = use --box-size as the floor."
        ),
    )
    parser.add_argument(
        "--png-frames",
        type=str,
        default=None,
        help=(
            "PNG file or directory of PNG frames to label "
            "(alternative to --input for TIFFs). "
            "Pass a single file path to label just that image."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader worker processes for prefetching (default: 4).",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use 16-bit mixed precision inference (faster on modern GPUs).",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile for kernel optimization (requires PyTorch 2.0+).",
    )
    return parser.parse_args()


class AutolabelDataset(torch.utils.data.Dataset):
    """Dataset for LodeSTAR autolabeling: supports TIFF stacks (lazy) and PNG files."""

    def __init__(self, items, tif_path=None):
        self.items = items  # list of indices (for TIFF) or paths (for PNG)
        self.tif_path = tif_path
        self._tif = None

    def __len__(self):
        return len(self.items)

    def _get_tif(self):
        if self._tif is None:
            self._tif = tifffile.TiffFile(self.tif_path)
        return self._tif

    def __getitem__(self, idx):
        item = self.items[idx]
        if self.tif_path:
            # Lazy read from TIFF
            tif = self._get_tif()
            frame = tif.pages[item].asarray()
        else:
            # Read from PNG
            frame = cv2.imread(item, cv2.IMREAD_UNCHANGED)
            if frame is None:
                # Provide a blank fallback if image is corrupt
                frame = np.zeros((64, 64), dtype=np.uint8)

        # Normalization (performed on CPU workers)
        frame_f = frame.astype(np.float32)
        f_min = frame_f.min()
        f_ptp = np.ptp(frame_f)
        if f_ptp == 0:
            f_ptp = 1.0
        frame_norm = (frame_f - f_min) / f_ptp

        # Return (normalized tensor, raw frame, item identifier)
        return torch.from_numpy(frame_norm).unsqueeze(0), frame, item

    def __del__(self):
        if self._tif is not None:
            self._tif.close()


def _detect_batch(batch_norm, ctx):
    """Run inference on a pre-normalized batch of frames and return detections.

    batch_norm is (B, 1, H, W) float32 tensor already moved to ctx.device.
    """
    # Use FP16 Mixed Precision if enabled and on CUDA
    use_fp16 = ctx.args.fp16 and ctx.device == "cuda"

    with torch.inference_mode():
        with torch.amp.autocast("cuda", enabled=use_fp16):
            try:
                detections = ctx.model.detect(
                    batch_norm,
                    alpha=ctx.args.alpha,
                    beta=1.0 - ctx.args.alpha,
                    cutoff=ctx.args.cutoff,
                    mode="ratio",
                )
            except torch.OutOfMemoryError:
                if ctx.device == "cuda":
                    print("\nGPU OOM. Falling back to CPU for this batch...")
                    torch.cuda.empty_cache()
                    ctx.model.to("cpu")
                    # No autocast on CPU for detect
                    detections = ctx.model.detect(
                        batch_norm.to("cpu"),
                        alpha=ctx.args.alpha,
                        beta=1.0 - ctx.args.alpha,
                        cutoff=ctx.args.cutoff,
                        mode="ratio",
                    )
                    ctx.model.to(ctx.device)  # Move back for next batches
                else:
                    raise

    # LodeSTAR returns a list of arrays if batch size > 1, or a single array if batch size == 1
    if not isinstance(detections, list):
        detections = [detections]

    # Post-processing: NMS for each frame in batch
    processed_dets = []
    for frame_dets in detections:
        if frame_dets is None or np.isnan(frame_dets).any() or np.isinf(frame_dets).any():
            processed_dets.append(None)
            continue

        if ctx.args.nms_distance > 0:
            frame_dets = _nms(list(frame_dets), ctx.args.nms_distance)
        processed_dets.append(frame_dets)

    return processed_dets


def _make_cfg(ctx, output_dir, frame_shape):
    """Build a _SaveConfig from autolabeler args."""
    min_box_px = ctx.args.min_box_size if ctx.args.min_box_size > 0 else float(ctx.args.box_size)
    return _SaveConfig(
        output_dir=output_dir,
        frame_shape=frame_shape,
        use_radius=ctx.args.use_radius,
        radius_scale=ctx.args.radius_scale,
        min_box_px=min_box_px,
        box_size=ctx.args.box_size,
        do_plot=ctx.args.plot,
    )


def _normalize_frame(frame_raw):
    """Normalize a raw frame to uint8, returning the normalized array."""
    if frame_raw.dtype != np.uint8:
        return cv2.normalize(  # pylint: disable=no-member
            frame_raw, None, 0, 255, cv2.NORM_MINMAX  # pylint: disable=no-member
        ).astype("uint8")
    return frame_raw


def _save_image(img_path, frame_norm):
    """Write a normalized frame to disk in BGR format."""
    if len(frame_norm.shape) == 2:
        cv2.imwrite(img_path, frame_norm)  # pylint: disable=no-member
    else:
        bgr = cv2.cvtColor(frame_norm, cv2.COLOR_RGB2BGR)  # pylint: disable=no-member
        cv2.imwrite(img_path, bgr)  # pylint: disable=no-member


def extract_and_labels(tif_path, model, args):
    """Process a single TIFF file: extract frames and generate YOLO labels."""
    base_name = os.path.splitext(os.path.basename(tif_path))[0]
    tif_dir = os.path.dirname(tif_path)

    # Unified output structure
    if args.output_dir:
        img_dir = os.path.join(args.output_dir, "images")
        lbl_dir = os.path.join(args.output_dir, "labels")
        # Prefix filename with base_name when merging into a single output_dir
        file_prefix = f"{base_name}_"
    else:
        img_dir = os.path.join(tif_dir, f"{base_name}_dataset", "images")
        lbl_dir = os.path.join(tif_dir, f"{base_name}_dataset", "labels")
        file_prefix = ""

    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    try:
        with tifffile.TiffFile(tif_path) as tif:
            num_frames = len(tif.pages)
    except OSError as e:
        print(f"Error reading {tif_path}: {e}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ctx = _FrameCtx(model=model.to(device), args=args, device=device)

    # Identify frames to process
    all_indices = list(range(0, num_frames, args.nth))
    to_process = []
    for idx in all_indices:
        lbl_path = os.path.join(lbl_dir, f"{file_prefix}frame_{idx:05d}.txt")
        if not os.path.exists(lbl_path):
            to_process.append(idx)

    if not to_process:
        print(f"All frames in {base_name} already labeled. Skipping.")
        return

    print(f"Processing {base_name}: {len(to_process)}/{len(all_indices)} frames...")

    # Set up DataLoader for Async Prefetching
    dataset = AutolabelDataset(to_process, tif_path=tif_path)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.detect_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    for batch_norm, batch_frames, batch_indices in tqdm(
        dataloader, desc=f"Batches from {base_name}"
    ):
        # Batch detection
        all_dets = _detect_batch(batch_norm.to(device), ctx)

        # Save results
        for j in range(len(batch_indices)):
            idx = int(batch_indices[j])
            frame_dets = all_dets[j]
            # Convert raw frame to normalized uint8 for saving
            frame_raw = batch_frames[j].numpy()
            frame_norm = _normalize_frame(frame_raw)

            img_out_path = os.path.join(img_dir, f"{file_prefix}frame_{idx:05d}.png")
            lbl_path = os.path.join(lbl_dir, f"{file_prefix}frame_{idx:05d}.txt")

            _save_image(img_out_path, frame_norm)

            if frame_dets is None:
                continue

            cfg = _make_cfg(ctx, lbl_dir, frame_norm.shape[:2])
            _write_frame(img_out_path, frame_norm, frame_dets, cfg)


def process_png_frames(png_files, model, args, png_dir):
    """Process a list of PNG files: run detection and generate YOLO labels."""
    base_name = os.path.basename(os.path.normpath(png_dir))

    # Unified output structure
    if args.output_dir:
        img_dir = os.path.join(args.output_dir, "images")
        lbl_dir = os.path.join(args.output_dir, "labels")
        # Prefix filename with base_name when merging into a single output_dir
        file_prefix = f"{base_name}_"
    else:
        img_dir = os.path.join(os.path.dirname(png_dir), f"{base_name}_dataset", "images")
        lbl_dir = os.path.join(os.path.dirname(png_dir), f"{base_name}_dataset", "labels")
        file_prefix = ""

    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ctx = _FrameCtx(model=model.to(device), args=args, device=device)

    # Identify frames to process
    to_process = []
    for idx, png_path in enumerate(png_files):
        lbl_path = os.path.join(lbl_dir, f"{file_prefix}frame_{idx:05d}.txt")
        if not os.path.exists(lbl_path):
            to_process.append((idx, png_path))

    if not to_process:
        print(f"All frames in {png_dir} already labeled. Skipping.")
        return

    print(f"Processing {png_dir}: {len(to_process)}/{len(png_files)} frames...")

    # Set up DataLoader for Async Prefetching
    # to_process is list of (idx, path), we just need the paths for the dataset
    paths_to_process = [p for _, p in to_process]
    dataset = AutolabelDataset(paths_to_process)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.detect_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    # We also need the original indices for filename generation
    indices_to_process = [i for i, _ in to_process]

    for b_idx, (batch_norm, batch_frames, batch_paths) in enumerate(
        tqdm(dataloader, desc=f"Batches from {base_name}")
    ):
        # Batch detection
        all_dets = _detect_batch(batch_norm.to(device), ctx)

        # Save results
        for j in range(len(batch_paths)):
            # Recover the original index
            global_idx = b_idx * args.detect_batch_size + j
            idx = indices_to_process[global_idx]

            frame_dets = all_dets[j]
            frame_raw = batch_frames[j].numpy()
            frame_norm = _normalize_frame(frame_raw)

            img_out_path = os.path.join(img_dir, f"{file_prefix}frame_{idx:05d}.png")
            lbl_path = os.path.join(lbl_dir, f"{file_prefix}frame_{idx:05d}.txt")

            _save_image(img_out_path, frame_norm)

            if frame_dets is None:
                continue

            cfg = _make_cfg(ctx, lbl_dir, frame_norm.shape[:2])
            _write_frame(img_out_path, frame_norm, frame_dets, cfg)


def _resolve_scale_params(args):
    """Resolve args.box_size/args.nms_distance in place via the shared precedence chain.

    Chain (mirrors particle-tracking/track.py's LodeSTAR box_size/nms_distance
    wiring, via detectors_common.scale_derivation):

        1. CLI-explicit value -- always wins.
        2. --config JSON value -- merged into args.box_size/args.nms_distance
           by the JSON-merge block in main() *before* this function runs, so
           by this point args.box_size/args.nms_distance already reflect
           "CLI-explicit-or-JSON" (CLI still wins over JSON, since the merge
           only overwrites a still-None value).
        3. --dataset-profile-derived value, if --dataset-profile was supplied.
        4. Today's existing hardcoded default (40 / 0.0), unchanged from
           pre-plan behavior when no profile is referenced (regression
           requirement).

    Must run after the --config JSON merge and before any code reads
    args.box_size/args.nms_distance as a plain number (both start out as
    None from argparse now -- see parse_args).
    """
    profile = load_dataset_profile(args.dataset_profile) if args.dataset_profile else None
    args.box_size = resolve_box_size(args.box_size, profile, hardcoded_default=40)
    args.nms_distance = resolve_nms_distance(args.nms_distance, profile, hardcoded_default=0.0)


def _apply_config_file(args, config_path):
    """Merge a JSON config file's values into `args` in place.

    CLI-explicit values always win: a key is only overwritten from the JSON
    file when the current value is still `None` (never passed on the CLI) or
    equals its known argparse default (the pre-existing "was this explicitly
    passed or just sitting at its default" heuristic -- see the module
    docstring note near --box-size/--nms-distance's argparse definitions for
    why those two keys are deliberately excluded from this heuristic).
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    defaults = {
        "nth": 5,
        "detect_batch_size": 4,
        "alpha": 0.5,
        "cutoff": 0.5,
        "radius_scale": 1.0,
        "min_box_size": 0.0,
        "plot": False,
        "use_radius": False,
        "fp16": False,
        "compile": False,
    }
    # box_size/nms_distance are deliberately absent from `defaults` above:
    # their argparse defaults are now None (not 40/0.0, see parse_args),
    # so "current_val is None" alone already distinguishes "never passed"
    # from "explicitly passed" for these two keys -- no need for (and no
    # risk from) the equality-with-default heuristic the other keys above
    # still rely on. See _resolve_scale_params for where 40/0.0 finally
    # apply.

    for key, value in config_data.items():
        current_val = getattr(args, key, None)
        # If the current value is the default or None, overwrite it with JSON value
        if current_val is None or (key in defaults and current_val == defaults[key]):
            setattr(args, key, value)


def main():
    """Entry point: load model and dispatch to TIFF or PNG processing."""
    args = parse_args()

    # Load from config file if provided
    if args.config:
        _apply_config_file(args, args.config)

    _resolve_scale_params(args)

    if not args.model:
        raise ValueError("--model (or 'model' in config) must be provided.")
    if not args.input and not args.png_frames:
        raise ValueError("Either --input or --png-frames must be provided.")

    # Bridge: _load_model() from lodestar_utils expects args.model_path
    args.model_path = args.model
    args.detect_mode = "ratio"
    # _load_model sets args.num_outputs from the companion JSON
    args.num_outputs = 3  # default; overwritten by _load_model if JSON exists

    print(f"Loading model: {args.model}")
    lodestar = _load_model(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lodestar.to(device)
    lodestar.eval()

    # Apply torch.compile if requested
    if args.compile:
        try:
            if hasattr(torch, "compile"):
                print("Compiling model for optimized inference (this may take a minute)...")
                lodestar = torch.compile(lodestar)
            else:
                print("Warning: torch.compile not available (requires PyTorch 2.0+). Skipping.")
        except Exception as e:
            print(f"Warning: Model compilation failed: {e}. Falling back to standard mode.")

    if args.png_frames:
        png_path = args.png_frames
        if os.path.isfile(png_path):
            # Single-image mode: wrap in a list, use the parent directory as the base
            png_files = [png_path]
            png_dir = os.path.dirname(os.path.abspath(png_path))
            print(f"Single-image mode: {png_path}")
        else:
            png_dir = png_path
            png_files = sorted(glob.glob(os.path.join(png_dir, "*.png")))
            if not png_files:
                print(f"No PNG files found in {png_dir}")
                return
            print(f"Found {len(png_files)} PNG files in {png_dir}.")
        process_png_frames(png_files, lodestar, args, png_dir)
        print("Done.")
        return

    # Default: process TIFFs (file or directory)
    if os.path.isfile(args.input):
        tif_files = [args.input]
    else:
        search_pattern = os.path.join(args.input, "**", "*.tif*")
        tif_files = glob.glob(search_pattern, recursive=True)

    if not tif_files:
        print(f"No TIFF files found for input: {args.input}")
        return
    print(f"Found {len(tif_files)} TIFF file(s).")
    for tif_path in tif_files:
        extract_and_labels(tif_path, lodestar, args)
    print("Done.")


if __name__ == "__main__":
    main()
