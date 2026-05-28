import argparse
import re
import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageSequence

SCRIPT_DIR = Path(__file__).parent

# RF-DETR variant name → class name in the rfdetr package
RFDETR_VARIANTS = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
    "base": "RFDETRBase",  # kept for backward compatibility
}


def load_config(config_path):
    config_path = Path(config_path)
    if not config_path.exists():
        return {}
    try:
        import yaml
    except ImportError:
        print("Warning: 'pyyaml' not installed — config file ignored. Run 'pip install pyyaml'.")
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def cfg_get(cfg, *keys, default=None):
    """Walk a nested dict by keys, returning default if any key is missing."""
    node = cfg
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node


def resolve_path(p):
    """Resolve a path relative to particle-tracking/ if not already absolute."""
    p = Path(p)
    return p if p.is_absolute() else SCRIPT_DIR / p


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------


def _normalize_device(device):
    """Map shorthand device strings to torch-style strings rfdetr accepts.

    rfdetr validates device via torch.device(), so bare integers like "0"
    are invalid. Map them to "cuda:N" so users can write device: "0" in config.
    """
    if device is None:
        return None
    s = str(device).strip()
    if s.lstrip("-").isdigit():
        return f"cuda:{s}"
    return s


def get_rfdetr_model(variant, checkpoint, device, num_classes=None, num_queries=None):
    """Load RF-DETR from the venv in rf-detr/."""
    rf_detr_venv = SCRIPT_DIR / ".." / "rf-detr" / ".venv"
    site_packages = list(rf_detr_venv.glob("lib/python*/site-packages"))
    if site_packages and str(site_packages[0]) not in sys.path:
        sys.path.insert(0, str(site_packages[0]))
    # If torch was already imported from the particle-tracking venv before this
    # path injection, torchvision from rf-detr's venv will conflict.  Evict the
    # stale torch/torchvision entries from sys.modules so they reload from
    # rf-detr's site-packages (which are now at position 0).
    for mod in list(sys.modules):
        if (
            mod == "torch"
            or mod.startswith("torch.")
            or mod == "torchvision"
            or mod.startswith("torchvision.")
        ):
            del sys.modules[mod]

    try:
        import rfdetr as _rfdetr

        cls_name = RFDETR_VARIANTS.get(variant)
        if cls_name is None:
            print(
                f"Error: unknown RF-DETR variant '{variant}'. Choose from: {', '.join(RFDETR_VARIANTS)}"
            )
            sys.exit(1)

        cls = getattr(_rfdetr, cls_name, None)
        if cls is None:
            print(f"Error: '{cls_name}' not found in installed rfdetr package.")
            sys.exit(1)

        # rfdetr manages device internally — pass normalized device string so
        # shorthand "0" becomes "cuda:0". Omit when None to let rfdetr auto-detect.
        kwargs = {"pretrain_weights": str(checkpoint)}
        normalized = _normalize_device(device)
        if normalized is not None:
            kwargs["device"] = normalized
        if num_classes is not None:
            kwargs["num_classes"] = num_classes
        if num_queries is not None:
            kwargs["num_queries"] = num_queries
        model = cls(**kwargs)
        if hasattr(model, "optimize_for_inference"):
            print("Optimizing RF-DETR model for inference...")
            model.optimize_for_inference()
        return model
    except ImportError:
        print("Error: 'rfdetr' not found. Run 'uv sync' inside rf-detr/.")
        sys.exit(1)


def get_yolo_model(checkpoint):
    try:
        from ultralytics import YOLO

        return YOLO(str(checkpoint))
    except ImportError:
        print("Error: 'ultralytics' not found. Run 'pip install ultralytics'.")
        sys.exit(1)


def get_lodestar_model(checkpoint, device, fp16=False):
    try:
        import deeplay as dl
        import torch
        import json

        config_path = Path(checkpoint).with_suffix(".json")
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            n_transforms = config.get("n_transforms", 8)
            num_outputs = config.get("num_outputs", 3)
        else:
            n_transforms, num_outputs = 8, 3

        model = dl.LodeSTAR(n_transforms=n_transforms, num_outputs=num_outputs).build()
        model.load_state_dict(torch.load(str(checkpoint), map_location=device, weights_only=False))
        model.to(device)
        if fp16:
            model.half()
        model.eval()
        return model
    except ImportError:
        print("Error: 'deeplay' or 'torch' not found. Run 'pip install deeplay deeptrack'.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


def detect_lodestar(model, frame, threshold, device, alpha=0.5, nms_distance=None, box_size=40):
    import torch
    import supervision as sv

    frame_f = frame.astype(np.float32)
    if frame.ndim == 3:
        frame_f = np.mean(frame_f, axis=2)

    f_min, f_ptp = frame_f.min(), np.ptp(frame_f)
    frame_norm = (frame_f - f_min) / f_ptp if f_ptp != 0 else frame_f - f_min

    tensor = torch.from_numpy(frame_norm).unsqueeze(0).unsqueeze(0).to(device)
    # Match model dtype (e.g. float16 when fp16=True)
    tensor = tensor.to(next(model.parameters()).dtype)

    with torch.inference_mode():
        detections_raw = model.detect(tensor, alpha=alpha, beta=0.5, cutoff=threshold, mode="ratio")

    if isinstance(detections_raw, list):
        detections_raw = detections_raw[0]

    if detections_raw is None or len(detections_raw) == 0:
        return sv.Detections.empty()

    xyxy, confidences = [], []
    for det in detections_raw:
        y, x = det[0], det[1]
        r = abs(det[2]) if len(det) >= 3 else box_size / 2
        xyxy.append([x - r, y - r, x + r, y + r])
        confidences.append(float(det[2]) if len(det) >= 3 else 1.0)

    result = sv.Detections(
        xyxy=np.array(xyxy, dtype=np.float32),
        confidence=np.array(confidences, dtype=np.float32),
        class_id=np.zeros(len(xyxy), dtype=int),
    )

    if nms_distance and nms_distance > 0 and len(result) > 1:
        centers = (result.xyxy[:, :2] + result.xyxy[:, 2:]) / 2
        order = np.argsort(-result.confidence)
        processed = np.zeros(len(result), dtype=bool)
        keep = []
        for idx in order:
            if processed[idx]:
                continue
            keep.append(idx)
            dists = np.sqrt(((centers - centers[idx]) ** 2).sum(axis=1))
            processed[dists < nms_distance] = True
        keep = np.array(keep)
        result = sv.Detections(
            xyxy=result.xyxy[keep],
            confidence=result.confidence[keep],
            class_id=result.class_id[keep],
        )

    return result


# ---------------------------------------------------------------------------
# Frame loading
# ---------------------------------------------------------------------------


def _natural_sort_key(path):
    """Sort key for filenames with embedded numbers (frame_2.png < frame_10.png)."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", path.name)]


def _to_rgb_uint8(frame):
    """Convert a single frame (any dtype, grayscale or color) to uint8 RGB (H, W, 3).

    Microscopy TIFFs are typically 16-bit grayscale. PIL silently converts them to
    all-white when calling .convert("RGB"). This function normalises the pixel range
    to [0, 255] before promoting to RGB so the content is actually visible.
    """
    # CHW → HWC (e.g. tifffile sometimes returns C×H×W for colour TIFFs)
    if frame.ndim == 3 and frame.shape[0] in (1, 3, 4) and frame.shape[0] < frame.shape[1]:
        frame = frame.transpose(1, 2, 0)
    # Drop alpha / extra channels
    if frame.ndim == 3 and frame.shape[2] == 4:
        frame = frame[:, :, :3]
    if frame.ndim == 3 and frame.shape[2] == 1:
        frame = frame[:, :, 0]

    # Normalise non-uint8 dtypes to [0, 255]
    if frame.dtype != np.uint8:
        f = frame.astype(np.float32)
        f_min, f_max = f.min(), f.max()
        if f_max > f_min:
            f = (f - f_min) / (f_max - f_min) * 255.0
        frame = f.clip(0, 255).astype(np.uint8)

    # Grayscale → RGB
    if frame.ndim == 2:
        frame = np.stack([frame, frame, frame], axis=-1)

    return frame


def load_lammpstrj(path):
    """Parse a LAMMPS trajectory file into a list of per-timestep DataFrames.

    Each DataFrame has at minimum columns: id, x, y (real or unwrapped coordinates).
    Scaled coordinates (xs, ys) are converted to real coordinates using box bounds.
    """
    frames = []
    with open(path) as f:
        while True:
            line = f.readline()
            if not line:
                break
            if "ITEM: TIMESTEP" not in line:
                continue

            timestep = int(f.readline().strip())
            f.readline()  # ITEM: NUMBER OF ATOMS
            n_atoms = int(f.readline().strip())

            f.readline()  # ITEM: BOX BOUNDS ...
            x_lo, x_hi = map(float, f.readline().split())
            y_lo, y_hi = map(float, f.readline().split())
            f.readline()  # z bounds (ignored for 2-D)

            atoms_header = f.readline().strip()  # ITEM: ATOMS id type x y ...
            columns = atoms_header.replace("ITEM: ATOMS", "").split()

            rows = []
            for _ in range(n_atoms):
                values = f.readline().split()
                rows.append(dict(zip(columns, values)))

            df = pd.DataFrame(rows)
            # Cast numeric columns
            for col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col])
                except ValueError:
                    pass

            # Resolve x coordinate: prefer unwrapped (xu) > real (x) > scaled (xs)
            if "xu" in df.columns and "yu" in df.columns:
                df = df.rename(columns={"xu": "x", "yu": "y"})
            elif "xs" in df.columns and "ys" in df.columns:
                df["x"] = df["xs"] * (x_hi - x_lo) + x_lo
                df["y"] = df["ys"] * (y_hi - y_lo) + y_lo

            if "x" not in df.columns or "y" not in df.columns:
                raise ValueError(
                    f"Timestep {timestep}: no recognised x/y columns. " f"Found: {list(df.columns)}"
                )

            df["timestep"] = timestep
            frames.append(df)

    return frames


def load_frames(input_path):
    """Load frames from a video file, image directory, or multi-page TIFF."""
    import tifffile

    input_path = Path(input_path)
    frames = []

    if input_path.is_dir():
        valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        files = sorted(
            [f for f in input_path.glob("*.*") if f.suffix.lower() in valid_exts],
            key=_natural_sort_key,
        )
        for f in files:
            if f.suffix.lower() in {".tif", ".tiff"}:
                frames.append(_to_rgb_uint8(tifffile.imread(str(f))))
            else:
                frames.append(np.array(Image.open(f).convert("RGB")))
    elif input_path.suffix.lower() in {".tif", ".tiff"}:
        # Use tifffile so that 16-bit microscopy stacks are read correctly.
        # PIL silently converts I;16 mode to all-white on .convert("RGB").
        data = tifffile.imread(str(input_path))
        # data shape is typically (n_frames, H, W) for a grayscale stack,
        # but may have extra axes for time/z/channel in OME-TIFF.
        # Squeeze any leading size-1 axes (e.g. Z=1, C=1) while keeping ≥3-D.
        while data.ndim > 3 and data.shape[0] == 1:
            data = data[0]
        if data.ndim == 2:
            # Single-frame TIFF
            frames.append(_to_rgb_uint8(data))
        else:
            # First axis is the frame/time axis
            for raw in data:
                frames.append(_to_rgb_uint8(raw))
    elif input_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
        frames.append(np.array(Image.open(input_path).convert("RGB")))
    else:
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            print(f"Error: Could not open video file {input_path}")
            return []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

    return frames


# ---------------------------------------------------------------------------
# Trajectory image
# ---------------------------------------------------------------------------


def _save_trajectory_image(df_tracked, background_frame, output_path, colormap="plasma"):
    """Render all complete trajectories onto the first frame with a start→end colour gradient.

    Each trajectory is drawn as a polyline whose colour shifts from the start of the
    colourmap (start of track) to the end of the colourmap (end of track), making it
    easy to see where particles came from and where they went.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.collections import LineCollection

    fig, ax = plt.subplots(
        figsize=(background_frame.shape[1] / 100, background_frame.shape[0] / 100), dpi=100
    )
    ax.imshow(background_frame)
    ax.set_axis_off()

    cmap = cm.get_cmap(colormap)

    for tid, grp in df_tracked.groupby("track_id"):
        grp = grp.sort_values("frame")
        xs = grp["x"].to_numpy()
        ys = grp["y"].to_numpy()
        if len(xs) < 2:
            continue

        # Build segments and per-segment progress values (0 = start, 1 = end)
        points = np.stack([xs, ys], axis=1)
        segments = np.stack([points[:-1], points[1:]], axis=1)
        progress = np.linspace(0, 1, len(segments))

        lc = LineCollection(segments, cmap=cmap, linewidth=1.0, alpha=0.8)
        lc.set_array(progress)
        lc.set_clim(0, 1)
        ax.add_collection(lc)

    plt.colorbar(
        cm.ScalarMappable(cmap=cmap),
        ax=ax,
        orientation="vertical",
        fraction=0.02,
        pad=0.01,
        label="Track progress  (start → end)",
    )
    plt.tight_layout(pad=0)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Particle Tracking with RF-DETR, YOLOv12, or LodeSTAR",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=str(SCRIPT_DIR / "config.yaml"),
        help="Path to YAML config file",
    )
    # Model
    parser.add_argument("--model-type", choices=["rf-detr", "yolo", "lodestar"])
    parser.add_argument("--checkpoint", help="Path to model weights (.pth, .pt, or .ckpt)")
    parser.add_argument("--variant", choices=list(RFDETR_VARIANTS), help="RF-DETR model size")
    parser.add_argument("--device", help="Inference device (e.g. 0 or cpu)")
    parser.add_argument("--threshold", type=float, help="Detection confidence threshold")
    # I/O
    parser.add_argument("--input", help="Path to video, image folder, or TIFF stack")
    parser.add_argument("--output-dir", help="Directory to save results")
    # Tracking
    parser.add_argument("--tracker", choices=["trackpy", "bytetrack"])
    parser.add_argument("--search-range", type=float, help="Trackpy: max pixel distance per frame")
    parser.add_argument("--memory", type=int, help="Trackpy: frames a particle may be missing")
    parser.add_argument("--stub-filter", type=int, help="Trackpy: min track length to keep")
    parser.add_argument(
        "--adaptive-stop",
        type=float,
        help="Trackpy: min search_range for adaptive linking (omit to disable)",
    )
    parser.add_argument(
        "--adaptive-step", type=float, help="Trackpy: search_range shrink factor per adaptive step"
    )
    parser.add_argument(
        "--lost-track-buffer", type=int, help="ByteTrack: frames to keep a lost track alive"
    )
    parser.add_argument(
        "--minimum-consecutive-frames",
        type=int,
        help="ByteTrack: frames before a track is confirmed",
    )
    parser.add_argument(
        "--track-activation-threshold",
        type=float,
        help="ByteTrack: min confidence to start a new track",
    )
    # LodeSTAR-specific detection
    parser.add_argument(
        "--lodestar-alpha", type=float, help="LodeSTAR: weight score exponent (default 0.5)"
    )
    parser.add_argument(
        "--lodestar-nms-distance",
        type=float,
        help="LodeSTAR: suppress detections within this pixel distance",
    )
    parser.add_argument(
        "--lodestar-fp16", action="store_true", help="LodeSTAR: run model in float16"
    )
    # Video output
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--fps", type=int, help="FPS for output video")
    parser.add_argument(
        "--trace-length", type=int, help="Frames of trajectory history shown in output video"
    )
    parser.add_argument(
        "--save-trajectory-image",
        action="store_true",
        help="Save a static PNG of all trajectories with start→end gradient",
    )
    parser.add_argument(
        "--trajectory-colormap",
        default=None,
        help="Matplotlib colormap for trajectory image (default: plasma)",
    )
    parser.add_argument(
        "--hexatic-order",
        action="store_true",
        help="Compute and save hexatic order parameter plot after tracking",
    )

    args = parser.parse_args()
    cfg = load_config(args.config)

    # Resolve final values: CLI arg → config → built-in default
    model_type = args.model_type or cfg_get(cfg, "model", "type", default="rf-detr")
    checkpoint = args.checkpoint or cfg_get(
        cfg, "model", "checkpoint", default="../rf-detr/checkpoints/checkpoint_best_ema.pth"
    )
    variant = args.variant or cfg_get(cfg, "model", "variant", default="large")
    num_classes = cfg_get(cfg, "model", "num_classes")
    num_queries = cfg_get(cfg, "model", "num_queries")
    device = _normalize_device(args.device or cfg_get(cfg, "model", "device", default="0")) or "cpu"
    threshold = args.threshold or cfg_get(cfg, "detection", "threshold", default=0.25)
    input_path = args.input or cfg_get(cfg, "input")
    output_dir = Path(
        args.output_dir
        or cfg_get(cfg, "output", "dir", default="evaluation/results/tracking_output")
    )
    tracker = args.tracker or cfg_get(cfg, "tracking", "tracker", default="trackpy")
    search_range = (
        args.search_range
        if args.search_range is not None
        else cfg_get(cfg, "tracking", "search_range", default=10.0)
    )
    memory = (
        args.memory if args.memory is not None else cfg_get(cfg, "tracking", "memory", default=3)
    )
    stub_filter = (
        args.stub_filter
        if args.stub_filter is not None
        else cfg_get(cfg, "tracking", "stub_filter", default=5)
    )
    adaptive_stop = (
        args.adaptive_stop
        if args.adaptive_stop is not None
        else cfg_get(cfg, "tracking", "adaptive_stop", default=None)
    )
    adaptive_step = (
        args.adaptive_step
        if args.adaptive_step is not None
        else cfg_get(cfg, "tracking", "adaptive_step", default=0.95)
    )
    lost_track_buffer = (
        args.lost_track_buffer
        if args.lost_track_buffer is not None
        else cfg_get(cfg, "tracking", "lost_track_buffer", default=30)
    )
    minimum_consecutive_frames = (
        args.minimum_consecutive_frames
        if args.minimum_consecutive_frames is not None
        else cfg_get(cfg, "tracking", "minimum_consecutive_frames", default=1)
    )
    track_activation_threshold = (
        args.track_activation_threshold
        if args.track_activation_threshold is not None
        else cfg_get(cfg, "tracking", "track_activation_threshold", default=0.25)
    )
    lodestar_alpha = (
        args.lodestar_alpha
        if args.lodestar_alpha is not None
        else cfg_get(cfg, "detection", "alpha", default=0.5)
    )
    lodestar_nms_distance = (
        args.lodestar_nms_distance
        if args.lodestar_nms_distance is not None
        else cfg_get(cfg, "detection", "nms_distance", default=None)
    )
    lodestar_fp16 = args.lodestar_fp16 or cfg_get(cfg, "detection", "fp16", default=False)
    save_trajectory_image = args.save_trajectory_image or cfg_get(
        cfg, "output", "save_trajectory_image", default=False
    )
    trajectory_colormap = args.trajectory_colormap or cfg_get(
        cfg, "output", "trajectory_colormap", default="plasma"
    )
    save_hexatic_order = args.hexatic_order or cfg_get(
        cfg, "analysis", "hexatic_order", default=False
    )
    save_video = args.save_video or cfg_get(cfg, "output", "save_video", default=False)
    fps = args.fps or cfg_get(cfg, "output", "fps", default=30)
    trace_length = (
        args.trace_length
        if args.trace_length is not None
        else cfg_get(cfg, "output", "trace_length", default=30)
    )

    if input_path is None:
        parser.error("--input is required (or set 'input' in config.yaml)")

    checkpoint = resolve_path(checkpoint)
    output_dir = resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(input_path)
    is_lammpstrj = input_path.suffix.lower() == ".lammpstrj"

    print(f"Config:    {args.config}")
    if not is_lammpstrj:
        print(f"Model:     {model_type} ({checkpoint})")
        print(f"Tracker:   {tracker}")
    print(f"Input:     {input_path}")
    print(f"Output:    {output_dir}")

    # -----------------------------------------------------------------------
    # LAMMPS trajectory: positions are known — skip detection entirely
    # -----------------------------------------------------------------------
    if is_lammpstrj:
        print(f"\nParsing LAMMPS trajectory: {input_path}")
        lammps_frames = load_lammpstrj(input_path)
        print(f"Found {len(lammps_frames)} timesteps.")

        tracking_data = []
        for frame_idx, df_frame in enumerate(lammps_frames):
            for _, atom in df_frame.iterrows():
                tracking_data.append(
                    {
                        "frame": frame_idx,
                        "timestep": int(atom["timestep"]),
                        "track_id": int(atom["id"]),
                        "x": atom["x"],
                        "y": atom["y"],
                    }
                )

        if save_video:
            print("Warning: --save-video is not supported for .lammpstrj input.")

        df_final = pd.DataFrame(tracking_data)
        csv_path = output_dir / "tracks.csv"
        df_final.to_csv(csv_path, index=False)
        print(f"Saved tracking data to {csv_path}")
        return

    # -----------------------------------------------------------------------
    # Image / video pipeline
    # -----------------------------------------------------------------------
    print(f"\nInitializing {model_type} model...")

    try:
        import supervision as sv
    except ImportError:
        print("Error: 'supervision' not found. Run 'pip install supervision'.")
        sys.exit(1)

    if tracker == "trackpy":
        try:
            import trackpy as tp
        except ImportError:
            print("Error: 'trackpy' not found. Run 'pip install trackpy'.")
            sys.exit(1)

    # Initialize detection model before loading frames so import/checkpoint
    # errors surface immediately rather than after a potentially long load.
    if model_type == "rf-detr":
        model = get_rfdetr_model(
            variant, checkpoint, device, num_classes=num_classes, num_queries=num_queries
        )
    elif model_type == "yolo":
        model = get_yolo_model(checkpoint)
    elif model_type == "lodestar":
        model = get_lodestar_model(checkpoint, device, fp16=lodestar_fp16)

    print(f"\nLoading frames from {input_path}...")
    frames = load_frames(input_path)
    if not frames:
        print("No frames found. Exiting.")
        return
    print(f"Found {len(frames)} frames.")

    # Resolve crop region from config (uses first frame dimensions)
    crop_cfg = cfg_get(cfg, "crop") or {}
    crop_x = crop_y = crop_w = crop_h = None
    if crop_cfg:
        fh, fw = frames[0].shape[:2]
        raw_w = crop_cfg.get("width")
        raw_h = crop_cfg.get("height")
        crop_w = int(raw_w * fw if isinstance(raw_w, float) and raw_w <= 1.0 else (raw_w or fw))
        crop_h = int(raw_h * fh if isinstance(raw_h, float) and raw_h <= 1.0 else (raw_h or fh))
        if crop_cfg.get("center", False):
            crop_x = (fw - crop_w) // 2
            crop_y = (fh - crop_h) // 2
        else:
            crop_x = int(crop_cfg.get("x", 0))
            crop_y = int(crop_cfg.get("y", 0))
        print(f"Crop:      x={crop_x} y={crop_y} w={crop_w} h={crop_h} (frame {fw}×{fh})")

    # 1. Detection phase
    all_detections = []
    raw_tracking_data = []

    for i, frame in enumerate(tqdm(frames, desc="Detecting")):
        detect_frame = (
            frame[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]
            if crop_x is not None
            else frame
        )

        if model_type == "rf-detr":
            detections = model.predict(detect_frame, threshold=threshold)
        elif model_type == "yolo":
            results = model.predict(detect_frame, conf=threshold, device=device, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
        elif model_type == "lodestar":
            detections = detect_lodestar(
                model,
                detect_frame,
                threshold,
                device,
                alpha=lodestar_alpha,
                nms_distance=lodestar_nms_distance,
            )

        # Shift bounding boxes back to full-frame coordinates
        if crop_x is not None and len(detections) > 0:
            detections.xyxy[:, [0, 2]] += crop_x
            detections.xyxy[:, [1, 3]] += crop_y

        all_detections.append(detections)

        for j in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[j]
            raw_tracking_data.append(
                {
                    "frame": i,
                    "x": (x1 + x2) / 2,
                    "y": (y1 + y2) / 2,
                    "w": x2 - x1,
                    "h": y2 - y1,
                    "conf": detections.confidence[j] if detections.confidence is not None else 1.0,
                }
            )

    # Detection summary — helps diagnose whether low track count is a detector problem
    det_counts = [len(d) for d in all_detections]
    if det_counts:
        total = sum(det_counts)
        avg = total / len(det_counts)
        print(
            f"\nDetection summary: {total} total detections across {len(det_counts)} frames "
            f"(avg {avg:.1f}/frame, min {min(det_counts)}, max {max(det_counts)})"
        )
        if avg < 5:
            print(
                "  Warning: very few detections per frame. "
                "Low track count is likely a detector issue, not a tracker issue. "
                "Consider lowering --threshold or retraining/fine-tuning the model."
            )

    # 2. Tracking phase
    df = pd.DataFrame(raw_tracking_data)
    tracking_data = []

    if df.empty and model_type == "rf-detr":
        # Run one probe frame at threshold=0 to show the actual score range.
        probe = model.predict(frames[0], threshold=0.0)
        if len(probe) > 0 and probe.confidence is not None:
            max_conf = float(probe.confidence.max())
            print(
                f"Warning: 0 detections with threshold={threshold}. "
                f"Max confidence seen on frame 0 was {max_conf:.4f}. "
                f"Try lowering --threshold (e.g. {max_conf * 0.8:.4f})."
            )
        else:
            print(f"Warning: 0 detections. The model may not be compatible with this input.")

    if tracker == "trackpy":
        print("Applying Trackpy (offline)...")
        if not df.empty:
            link_kwargs = {"search_range": search_range, "memory": memory}
            if adaptive_stop is not None:
                link_kwargs["adaptive_stop"] = adaptive_stop
                link_kwargs["adaptive_step"] = adaptive_step
            df = tp.link_df(df, **link_kwargs)
            if stub_filter > 0:
                df = tp.filter_stubs(df, stub_filter)
            df = df.rename(columns={"particle": "track_id"})
            tracking_data = df.to_dict("records")
        else:
            print("No detections to track.")

    elif tracker == "bytetrack":
        print("Applying ByteTrack (online)...")
        byte_tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_consecutive_frames=minimum_consecutive_frames,
        )
        tracked_frames_detections = []

        for i, detections in enumerate(tqdm(all_detections, desc="Tracking")):
            detections = byte_tracker.update_with_detections(detections)
            tracked_frames_detections.append(detections)

            if detections.tracker_id is not None:
                for j in range(len(detections.tracker_id)):
                    x1, y1, x2, y2 = detections.xyxy[j]
                    tracking_data.append(
                        {
                            "frame": i,
                            "track_id": int(detections.tracker_id[j]),
                            "x": (x1 + x2) / 2,
                            "y": (y1 + y2) / 2,
                            "w": x2 - x1,
                            "h": y2 - y1,
                            "conf": (
                                detections.confidence[j]
                                if detections.confidence is not None
                                else 1.0
                            ),
                        }
                    )
            else:
                tracked_frames_detections[-1] = sv.Detections.empty()

    # 3. Visualization phase
    if save_video:
        print("Annotating video...")
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        trace_annotator = sv.TraceAnnotator(trace_length=trace_length)
        df_tracked = pd.DataFrame(tracking_data)
        annotated_frames = []

        for i, frame in enumerate(tqdm(frames, desc="Visualizing")):
            if not df_tracked.empty and "track_id" in df_tracked.columns:
                frame_df = df_tracked[df_tracked["frame"] == i]
                if not frame_df.empty:
                    xyxy, tracker_ids = [], []
                    for _, row in frame_df.iterrows():
                        x, y, w, h = row["x"], row["y"], row["w"], row["h"]
                        xyxy.append([x - w / 2, y - h / 2, x + w / 2, y + h / 2])
                        tracker_ids.append(int(row["track_id"]))
                    detections = sv.Detections(
                        xyxy=np.array(xyxy, dtype=np.float32),
                        tracker_id=np.array(tracker_ids, dtype=int),
                        class_id=np.zeros(len(xyxy), dtype=int),
                    )
                else:
                    detections = sv.Detections.empty()
            else:
                detections = sv.Detections.empty()

            annotated_frame = frame.copy()
            if detections.tracker_id is not None and len(detections.tracker_id) > 0:
                labels = [f"#{tid}" for tid in detections.tracker_id]
                annotated_frame = trace_annotator.annotate(
                    scene=annotated_frame, detections=detections
                )
                annotated_frame = box_annotator.annotate(
                    scene=annotated_frame, detections=detections
                )
                annotated_frame = label_annotator.annotate(
                    scene=annotated_frame, detections=detections, labels=labels
                )
            annotated_frames.append(annotated_frame)

        video_path = output_dir / "tracking_visualization.mp4"
        h, w = annotated_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
        for f in annotated_frames:
            out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        out.release()
        print(f"Saved annotated video to {video_path}")

    # 4. Save results
    df_final = pd.DataFrame(tracking_data)
    csv_path = output_dir / "tracks.csv"
    df_final.to_csv(csv_path, index=False)
    print(f"Saved tracking data to {csv_path}")

    if save_trajectory_image and not df_final.empty and "track_id" in df_final.columns:
        print("Rendering trajectory image...")
        img_path = output_dir / "trajectories.png"
        _save_trajectory_image(df_final, frames[-1], img_path, colormap=trajectory_colormap)
        print(f"Saved trajectory image to {img_path}")

    if save_hexatic_order and not df_final.empty:
        print("Computing hexatic order parameter...")
        lammps_scripts_dir = SCRIPT_DIR / ".." / "lammps-scripts"
        lammps_venv_site = list((lammps_scripts_dir / ".venv").glob("lib/python*/site-packages"))
        if lammps_venv_site and str(lammps_venv_site[0]) not in sys.path:
            sys.path.insert(0, str(lammps_venv_site[0]))
        try:
            import matplotlib.pyplot as plt
            from hexatic_order_analysis import calc_hexatic_from_tracks

            fh, fw = frames[0].shape[:2]
            frame_nums, psi6 = calc_hexatic_from_tracks(df_final, fw, fh, verbose=0)
            if frame_nums:
                plt.figure(figsize=(10, 6))
                plt.plot(frame_nums, psi6, alpha=0.7)
                plt.xlabel("Frame")
                plt.ylabel(r"Global Hexatic Order $|\Psi_6|$")
                plt.title("Hexatic Order Parameter — Particle Tracking")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                hexatic_path = output_dir / "hexatic_order.png"
                plt.savefig(str(hexatic_path), dpi=300)
                plt.close()
                print(f"Saved hexatic order plot to {hexatic_path}")
        except ImportError:
            print(
                "Warning: could not import hexatic_order_analysis — ensure freud is installed in lammps-scripts/.venv"
            )


if __name__ == "__main__":
    main()
