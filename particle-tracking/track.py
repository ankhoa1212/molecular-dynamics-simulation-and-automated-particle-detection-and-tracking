import argparse
import re
import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageSequence

# Re-exported from detectors_common — edit there, not here.
# particle-tracking/.venv has detectors-common installed natively, so this is
# a plain module-scope re-export (unlike verification/benchmark.py, whose venv
# never installs detectors-common — see that file for the lazy-wrapper variant).
from detectors_common.rfdetr_loader import (
    RFDETR_VARIANTS,
    _normalize_device,
    get_rfdetr_model as _shared_get_rfdetr_model,
)
from detectors_common.lodestar_loader import get_lodestar_model, detect_lodestar
from detectors_common.tiling import detect_with_tiling
from detectors_common.defaults import load_detector_config
from detectors_common.scale_derivation import (
    resolve_box_size,
    resolve_nms_distance,
    resolve_tile_size,
)
from detectors_common.dataset_profile import load_dataset_profile as load_detection_profile

# Re-exported from trackers_common — edit there, not here. particle-tracking/.venv
# has trackers-common installed natively (module-scope import, same as
# detectors_common above — trackers-common has no CUDA-sensitive deps, so unlike
# detectors_common, verification/benchmark.py imports it the same way, no lazy
# wrapper needed there either — see that package's README).
from trackers_common.linking import bridge_track_gaps, link_and_filter_tracks
from trackers_common.defaults import load_tracking_config, DEFAULT_KEY_PATH_MAP
from trackers_common.scale_derivation import resolve_search_range, resolve_memory
from trackers_common.dataset_profile import load_dataset_profile as load_tracking_profile

# Maps canonical detector_defaults.yaml keys to this config's own dotted path.
# particle-tracking nests detector params by pipeline concern (detection.*),
# unlike verification's benchmark.lodestar.* — see defaults.py.
_LODESTAR_KEY_MAP = {
    "nms_distance": "detection.nms_distance",
    "alpha": "detection.alpha",
    "box_size": "detection.box_size",
}

SCRIPT_DIR = Path(__file__).parent


def load_config(config_path):
    """Load a config file, auto-resolving a top-level `extends:` key.

    An override file (e.g. lodestar_config.yaml) sets `extends: config.yaml`
    so `--config lodestar_config.yaml` alone stays a complete, correct
    invocation -- the base's keys are pulled in automatically rather than
    silently falling back to main()'s hardcoded defaults for anything the
    override doesn't restate. `extends` is resolved relative to the config
    file's own directory and is popped before merging, so it never appears
    in the returned dict.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        return {}
    try:
        import yaml
    except ImportError:
        print("Warning: 'pyyaml' not installed — config file ignored. Run 'pip install pyyaml'.")
        return {}
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    extends = cfg.pop("extends", None)
    if extends:
        base = load_config(config_path.parent / extends)
        cfg = merge_config(base, cfg)
    return cfg


def merge_config(base, override):
    """Recursively merge override onto base; override wins at any nesting depth.

    Used to layer a scenario override (e.g. lodestar_config.yaml) onto the
    shared base (config.yaml) without either file needing to restate the
    other's keys. Distinct from detectors_common.defaults.load_detector_config,
    which does a flat per-key dotted-path lookup against an explicit map —
    this merges two full nested config trees instead.

    Only the top-level dict and any dict nodes actually replaced are copied;
    unchanged nested values are shared by reference with base/override. Fine
    for this module's read-only-after-merge usage -- don't mutate a merged
    config's nested dicts/lists in place.
    """
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


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


def get_rfdetr_model(variant, checkpoint, device, num_classes=None, num_queries=None):
    """Load RF-DETR, supplying particle-tracking's own rf-detr/.venv path to
    the shared loader (detectors_common.rfdetr_loader.get_rfdetr_model takes
    the venv directory as a required parameter rather than hardcoding it,
    since verification/benchmark.py resolves a different path)."""
    rf_detr_venv = SCRIPT_DIR / ".." / "rf-detr" / ".venv"
    return _shared_get_rfdetr_model(
        variant, checkpoint, device, rf_detr_venv, num_classes=num_classes, num_queries=num_queries
    )


def get_yolo_model(checkpoint):
    try:
        from ultralytics import YOLO

        return YOLO(str(checkpoint))
    except ImportError:
        print("Error: 'ultralytics' not found. Run 'pip install ultralytics'.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


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
# Probe utilities
# ---------------------------------------------------------------------------


def _sample_frames(frames, n_samples):
    """Return up to n_samples evenly-spaced frames from the list."""
    if len(frames) <= n_samples:
        return frames
    indices = [int(i * len(frames) / n_samples) for i in range(n_samples)]
    return [frames[i] for i in indices]


def _run_detector(
    model,
    frame,
    model_type,
    threshold,
    device="cpu",
    lodestar_alpha=0.5,
    lodestar_nms_distance=None,
    lodestar_box_size=40,
):
    """Run the active detector on a single frame and return a detections object."""
    if model_type == "rf-detr":
        return model.predict(frame, threshold=threshold)
    elif model_type == "yolo":
        import supervision as sv

        results = model.predict(frame, conf=threshold, device=device, verbose=False)[0]
        return sv.Detections.from_ultralytics(results)
    elif model_type == "lodestar":
        return detect_lodestar(
            model,
            frame,
            threshold,
            device,
            alpha=lodestar_alpha,
            nms_distance=lodestar_nms_distance,
            box_size=lodestar_box_size,
        )
    return []


def run_density_probe(
    frames,
    model,
    model_type,
    threshold,
    n_samples=10,
    lodestar_alpha=0.5,
    lodestar_nms_distance=None,
    lodestar_box_size=40,
    device="cpu",
):
    """Sample N frames, run detector, return (p95_count, frame_w, frame_h)."""
    if not frames:
        return 0.0, 0, 0
    fh, fw = frames[0].shape[:2]
    sample = _sample_frames(frames, n_samples)
    counts = [
        len(
            _run_detector(
                model,
                f,
                model_type,
                threshold,
                device=device,
                lodestar_alpha=lodestar_alpha,
                lodestar_nms_distance=lodestar_nms_distance,
                lodestar_box_size=lodestar_box_size,
            )
        )
        for f in sample
    ]
    p95 = float(np.percentile(counts, 95)) if counts else 0.0
    return p95, fw, fh


def suggest_crop_size(p95_count, fw, fh, target=250):
    """Return (crop_w, crop_h) for a square center crop targeting at most `target` detections."""
    if p95_count <= 0:
        return fw, fh
    density = p95_count / (fw * fh)
    raw_size = int(np.ceil(np.sqrt(target / density)))
    clamped = max(512, min(raw_size, min(fw, fh)))
    return clamped, clamped


def probe_threshold(
    frames,
    model,
    model_type,
    n_samples=10,
    lodestar_alpha=0.5,
    lodestar_nms_distance=None,
    lodestar_box_size=40,
    device="cpu",
):
    """Run detector at threshold=0 on sampled frames; suggest a threshold from the score distribution.

    Returns (suggested_threshold, method) where method is 'valley' or 'percentile'.
    """
    sample = _sample_frames(frames, n_samples)
    all_scores = []
    for frame in sample:
        dets = _run_detector(
            model,
            frame,
            model_type,
            0.0,
            device=device,
            lodestar_alpha=lodestar_alpha,
            lodestar_nms_distance=lodestar_nms_distance,
            lodestar_box_size=lodestar_box_size,
        )
        if hasattr(dets, "confidence") and dets.confidence is not None and len(dets) > 0:
            all_scores.extend(dets.confidence.tolist())

    if not all_scores:
        return 0.25, "fallback"

    scores = np.array(all_scores)
    hist, bin_edges = np.histogram(scores, bins=20, range=(0.0, 1.0))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Look for a valley between two confidence peaks (signal vs. noise separation)
    suggested = None
    method = "percentile"
    mid = len(hist) // 2
    if mid > 0 and mid < len(hist):
        left_max_idx = int(np.argmax(hist[:mid]))
        right_max_idx = mid + int(np.argmax(hist[mid:]))
        if right_max_idx > left_max_idx + 1:
            valley_region = hist[left_max_idx + 1 : right_max_idx]
            if len(valley_region) > 0:
                valley_idx = left_max_idx + 1 + int(np.argmin(valley_region))
                left_peak = hist[left_max_idx]
                right_peak = hist[right_max_idx]
                valley_val = hist[valley_idx]
                valley_pos = float(bin_centers[valley_idx])
                if valley_pos - float(
                    bin_centers[left_max_idx]
                ) >= 0.05 and valley_val < 0.20 * min(left_peak, right_peak):
                    suggested = valley_pos
                    method = "valley"

    if suggested is None:
        suggested = float(np.percentile(scores, 85))
        method = "percentile"

    return suggested, method


# ---------------------------------------------------------------------------
# Track post-processing
# ---------------------------------------------------------------------------
#
# bridge_track_gaps and the core trackpy link+filter sequence now live in
# trackers_common.linking (re-exported above) — shared with
# verification/benchmark.py's tracking-metrics computation so the two can't
# drift on linking behavior. See trackers-common/README.md.


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_and_save_metrics(df_final, det_counts, run_meta, output_dir):
    """Compute tracking quality metrics and save to metrics.json alongside tracks.csv."""
    import json

    metrics: dict = {}

    if not df_final.empty and "track_id" in df_final.columns:
        lengths = df_final.groupby("track_id")["frame"].count()
        metrics["n_tracks"] = int(lengths.shape[0])
        metrics["track_length_mean"] = round(float(lengths.mean()), 2)
        metrics["track_length_median"] = round(float(lengths.median()), 2)
        metrics["track_length_max"] = int(lengths.max())
        metrics["track_length_min"] = int(lengths.min())
        if "conf" in df_final.columns:
            metrics["mean_confidence"] = round(float(df_final["conf"].mean()), 4)
        else:
            metrics["mean_confidence"] = None
    else:
        metrics.update(
            {
                "n_tracks": 0,
                "track_length_mean": 0.0,
                "track_length_median": 0.0,
                "track_length_max": 0,
                "track_length_min": 0,
                "mean_confidence": None,
            }
        )

    if det_counts:
        n_frames = len(det_counts)
        metrics["n_frames"] = n_frames
        metrics["detection_rate"] = round(sum(1 for c in det_counts if c > 0) / n_frames, 4)
        metrics["detections_per_frame_mean"] = round(sum(det_counts) / n_frames, 2)
        metrics["detections_per_frame_max"] = int(max(det_counts))
        metrics["frames_with_zero_detections"] = int(sum(1 for c in det_counts if c == 0))

    metrics.update(run_meta)

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nTracking summary:")
    print(f"  Tracks:           {metrics.get('n_tracks', 0)}")
    if metrics.get("n_tracks", 0) > 0:
        print(
            f"  Track length:     mean={metrics['track_length_mean']}, "
            f"median={metrics['track_length_median']}, max={metrics['track_length_max']}"
        )
    print(f"  Detection rate:   {metrics.get('detection_rate', 0):.1%} of frames")
    print(f"  Avg dets/frame:   {metrics.get('detections_per_frame_mean', 0):.1f}")
    if metrics.get("mean_confidence") is not None:
        print(f"  Mean confidence:  {metrics['mean_confidence']:.4f}")
    print(f"  Metrics saved to: {metrics_path}")


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
# Preview mode helpers
# ---------------------------------------------------------------------------


def resolve_preview_max_frames(test_flag, preview_n, max_frames_arg):
    """Resolve the effective max_frames from --test/--preview/--max-frames.

    Precedence: --test > --preview > --max-frames (mirrors the existing
    --test-over-max-frames relationship; --preview slots in above --max-frames).
    """
    if test_flag:
        return 1
    if preview_n is not None:
        return preview_n
    return max_frames_arg


def resolve_preview_stub_filter(stub_filter, n_frames):
    """Cap stub_filter so a short preview run doesn't misreport zero tracks.

    Directional relaxation only: caps stub_filter to roughly half the preview's
    frame count (minimum 1) so trackpy.filter_stubs doesn't discard every track
    purely because the full-run stub_filter (tuned for long runs) can't be
    reached within a short preview. Returns stub_filter unchanged if it already
    fits, or if stub_filter/n_frames don't call for capping.
    """
    if stub_filter is None or stub_filter <= 0 or n_frames is None or n_frames <= 0:
        return stub_filter
    return min(stub_filter, max(1, n_frames // 2))


def count_tracks_at_stub_filter(df, stub_filter):
    """Return the number of unique tracks that survive the given stub_filter.

    Log-only helper: used by preview mode to report what the full run's
    un-relaxed stub_filter would have produced on the same frames, without
    affecting the tracks actually reported by the current run.

    Expects an already-linked df with a 'track_id' column (trackers_common.
    linking.link_and_filter_tracks's output shape) -- implemented with plain
    pandas rather than tp.filter_stubs (which requires a column literally
    named 'particle') so it doesn't need to know about trackpy's own naming.
    """
    if df is None or df.empty:
        return 0
    if stub_filter is None or stub_filter <= 0:
        return df["track_id"].nunique()
    counts = df.groupby("track_id").size()
    return int((counts >= stub_filter).sum())


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
        help=(
            "Path to YAML config file. If it has a top-level 'extends:' key, that file "
            "is auto-merged underneath first."
        ),
    )
    parser.add_argument(
        "--override",
        default=None,
        help=(
            "Path to a second YAML config merged onto --config (override wins at any "
            "nesting depth). Rarely needed -- 'extends' inside a config file covers the "
            "common case of a scenario config layered onto the base."
        ),
    )
    # Model
    parser.add_argument("--model-type", choices=["rf-detr", "yolo", "lodestar"])
    parser.add_argument("--checkpoint", help="Path to model weights (.pth, .pt, or .ckpt)")
    parser.add_argument("--variant", choices=list(RFDETR_VARIANTS), help="RF-DETR model size")
    parser.add_argument("--device", help="Inference device (e.g. 0 or cpu)")
    parser.add_argument("--threshold", type=float, help="Detection confidence threshold")
    # I/O
    parser.add_argument(
        "--input", nargs="+", help="One or more paths to video, image folder, or TIFF stack"
    )
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
        "--lodestar-box-size",
        type=float,
        help="LodeSTAR: detection box side length in pixels (default 40) — manual estimate "
        "until real PSF calibration exists for this dataset",
    )
    parser.add_argument(
        "--lodestar-fp16", action="store_true", help="LodeSTAR: run model in float16"
    )
    # Video output
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument(
        "--video-labels",
        action="store_true",
        help="Show track ID labels in output video (off by default)",
    )
    parser.add_argument(
        "--no-video-labels",
        action="store_true",
        help="Omit track ID labels from output video (default)",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None, help="Only process the first N frames"
    )
    parser.add_argument(
        "--test", action="store_true", help="Test mode: process only the first frame"
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Preview mode: run the full pipeline on the first N frames with stub_filter "
            "relaxed to fit, and skip hexatic-order/trajectory-image by default (pass "
            "--hexatic-order/--save-trajectory-image explicitly to force them on). "
            "Precedence: --test > --preview > --max-frames."
        ),
    )
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
    # Model overrides (also settable via config)
    parser.add_argument("--num-classes", type=int, help="RF-DETR: number of output classes")
    parser.add_argument("--num-queries", type=int, help="RF-DETR: max detections per frame")
    # Probe mode
    parser.add_argument(
        "--probe",
        action="store_true",
        help=(
            "Probe mode: load model, sample frames, print PROBE_RESULT crop_w=N crop_h=N, exit. "
            "Use --probe-threshold to also print a suggested detection threshold."
        ),
    )
    parser.add_argument(
        "--probe-threshold",
        action="store_true",
        help="With --probe: also run threshold analysis and print PROBE_THRESHOLD suggested=X method=Y",
    )
    parser.add_argument(
        "--probe-n-samples",
        type=int,
        default=10,
        help="Number of frames to sample during probe (default: 10)",
    )
    # Gap-closing
    parser.add_argument(
        "--bridge-gap",
        type=int,
        default=None,
        help="Trackpy: reconnect track fragments with a gap of at most N frames (disabled by default)",
    )
    parser.add_argument(
        "--bridge-radius",
        type=float,
        default=None,
        help="Trackpy: spatial search radius for gap-closing in pixels (default: 2 × search_range)",
    )

    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.override:
        cfg = merge_config(cfg, load_config(args.override))

    # Dataset scale profile (size_px/spacing_px): when referenced, box_size/
    # nms_distance/tile_size/search_range/memory each derive from it via
    # detectors_common/trackers_common's shared scale_derivation modules,
    # sitting between an explicit config value (still always wins) and
    # today's hardcoded defaults (still applied unchanged when no profile is
    # referenced at all). Loaded once per run, via each package's own loader
    # (duplicated by design — see dataset-profiles/README.md).
    dataset_profile_path = cfg_get(cfg, "dataset_profile", default=None)
    detection_profile = None
    tracking_profile = None
    if dataset_profile_path is not None:
        profile_path = resolve_path(dataset_profile_path)
        detection_profile = load_detection_profile(profile_path)
        tracking_profile = load_tracking_profile(profile_path)

    # --test wins over --preview (mirrors --test's existing precedence over --max-frames)
    preview_active = args.preview is not None and not args.test

    # Resolve final values: CLI arg → config → built-in default
    model_type = args.model_type or cfg_get(cfg, "model", "type", default="rf-detr")
    checkpoint = args.checkpoint or cfg_get(
        cfg, "model", "checkpoint", default="../rf-detr/checkpoints/checkpoint_best_regular.pth"
    )
    variant = args.variant or cfg_get(cfg, "model", "variant", default="large")
    num_classes = (
        args.num_classes if args.num_classes is not None else cfg_get(cfg, "model", "num_classes")
    )
    num_queries = (
        args.num_queries if args.num_queries is not None else cfg_get(cfg, "model", "num_queries")
    )
    device = _normalize_device(args.device or cfg_get(cfg, "model", "device", default="0")) or "cpu"
    threshold_from_cli = args.threshold is not None
    threshold_from_cfg = cfg_get(cfg, "detection", "threshold", default=None) is not None
    threshold = args.threshold or cfg_get(cfg, "detection", "threshold", default=0.25)
    # Use the LodeSTAR autolabel cutoff as the default threshold when none was explicitly passed
    if model_type == "lodestar" and not threshold_from_cli and not threshold_from_cfg:
        from tracker_configs import read_lodestar_cutoff

        prior_threshold = read_lodestar_cutoff(SCRIPT_DIR)
        if prior_threshold is not None:
            threshold = prior_threshold
            print(f"Using LodeSTAR prior threshold: {threshold} (from autolabel config)")

    raw_input = args.input or cfg_get(cfg, "input")
    if raw_input is None:
        parser.error("--input is required (or set 'input' in config.yaml)")
    input_paths = [raw_input] if isinstance(raw_input, str) else list(raw_input)
    output_dir = Path(
        args.output_dir
        or cfg_get(cfg, "output", "dir", default="evaluation/results/tracking_output")
    )
    tracker = args.tracker or cfg_get(cfg, "tracking", "tracker", default="trackpy")
    # search_range: explicit config value -> dataset-profile-derived (spacing_px * 0.5)
    # -> the per-model canonical tuning (unchanged fallback behavior when no
    # profile is referenced) -- config.yaml's own prior default was 25.0
    # (rf-detr's canonical value) but lodestar_config.yaml's was 20.0
    # (lodestar's canonical value); a single shared literal here would
    # silently regress whichever config's default didn't match, so this
    # mirrors verification/benchmark.py's own per-model canonical_search_range
    # resolution rather than baking in one fixed number.
    canonical_search_range = load_tracking_config(model_type, {}, DEFAULT_KEY_PATH_MAP).get(
        "search_range", 25.0
    )
    search_range = (
        args.search_range
        if args.search_range is not None
        else resolve_search_range(
            cfg_get(cfg, "tracking", "search_range", default=None),
            tracking_profile,
            hardcoded_default=canonical_search_range,
        )
    )
    # memory: explicit config value -> trackers_common's per-model canonical
    # tuning (tracker_defaults.yaml) -- never derived from the profile itself
    # (R9: occlusion/blinking tolerance has no spatial grounding), but still
    # resolved through the same profile-aware call shape for consistency.
    memory = (
        args.memory
        if args.memory is not None
        else resolve_memory(
            cfg_get(cfg, "tracking", "memory", default=None), tracking_profile, model_type
        )
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
    bridge_gap = (
        args.bridge_gap
        if args.bridge_gap is not None
        else cfg_get(cfg, "tracking", "bridge_gap", default=None)
    )
    bridge_radius = (
        args.bridge_radius
        if args.bridge_radius is not None
        else cfg_get(cfg, "tracking", "bridge_radius", default=None)
    )
    probe_mode = args.probe
    probe_threshold_mode = args.probe_threshold
    probe_n_samples = args.probe_n_samples
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
    # alpha falls back through detector_defaults.yaml's canonical value (via
    # the shared key-path-mapped merge) before this file's own None default —
    # CLI arg still wins over everything. alpha is not part of scale
    # derivation (R6 only covers box_size/nms_distance/tile_size).
    _lodestar_defaults = load_detector_config("lodestar", cfg, _LODESTAR_KEY_MAP)
    lodestar_alpha = (
        args.lodestar_alpha
        if args.lodestar_alpha is not None
        else _lodestar_defaults.get("alpha", 0.5)
    )
    # nms_distance/box_size: explicit config value -> dataset-profile-derived
    # -> detectors_common's own hardcoded default (30/40, matching
    # detector_defaults.yaml's canonical lodestar values) when neither applies.
    lodestar_nms_distance = (
        args.lodestar_nms_distance
        if args.lodestar_nms_distance is not None
        else resolve_nms_distance(
            cfg_get(cfg, "detection", "nms_distance", default=None), detection_profile
        )
    )
    lodestar_box_size = (
        args.lodestar_box_size
        if args.lodestar_box_size is not None
        else resolve_box_size(
            cfg_get(cfg, "detection", "box_size", default=None), detection_profile
        )
    )
    if model_type == "lodestar" and detection_profile is not None:
        print(
            f"LodeSTAR:  box_size={lodestar_box_size}, nms_distance={lodestar_nms_distance} "
            "(derived from dataset_profile)"
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
    if preview_active:
        # Preview default: skip both unless explicitly requested on the CLI (same
        # explicit-flag-wins posture as video_labels/no_video_labels below).
        if not args.save_trajectory_image:
            save_trajectory_image = False
        if not args.hexatic_order:
            save_hexatic_order = False
    tiling_enabled = cfg_get(cfg, "tiling", "enabled", default=False)
    # tile_size's final value needs the source frame's own dimensions (the
    # profile-derived tier's clamp ceiling), so only the explicit config value
    # is captured here — full resolution happens per-input, once frame
    # dimensions are known (see resolve_tile_size call below).
    tiling_explicit_tile_size = cfg_get(cfg, "tiling", "tile_size", default=None)
    tiling_overlap = cfg_get(cfg, "tiling", "overlap", default=100)
    tiling_nms_threshold = cfg_get(cfg, "tiling", "nms_threshold", default=0.3)
    max_frames = resolve_preview_max_frames(args.test, args.preview, args.max_frames)
    if preview_active:
        print(
            f"Preview mode: capping to the first {max_frames} frame(s) "
            f"(a full run would process all available frames)."
        )
    save_video = args.save_video or cfg_get(cfg, "output", "save_video", default=False)
    if args.video_labels:
        video_labels = True
    elif args.no_video_labels:
        video_labels = False
    else:
        video_labels = cfg_get(cfg, "output", "video_labels", default=False)
    fps = args.fps or cfg_get(cfg, "output", "fps", default=30)
    trace_length = (
        args.trace_length
        if args.trace_length is not None
        else cfg_get(cfg, "output", "trace_length", default=30)
    )

    checkpoint = resolve_path(checkpoint)
    output_dir = resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Config:    {args.config}")

    # -----------------------------------------------------------------------
    # Pre-loop: imports and model initialisation (once for all inputs)
    # -----------------------------------------------------------------------
    needs_model = any(Path(p).suffix.lower() != ".lammpstrj" for p in input_paths)

    if needs_model:
        print(f"Model:     {model_type} ({checkpoint})")
        print(f"Tracker:   {tracker}")
        try:
            import supervision as sv
        except ImportError:
            print("Error: 'supervision' not found. Run 'pip install supervision'.")
            sys.exit(1)

        if tracker == "trackpy":
            # Fail fast with a friendly message before model loading, even though
            # link_and_filter_tracks (trackers_common.linking) does its own
            # trackpy import when actually called.
            try:
                import trackpy  # noqa: F401
            except ImportError:
                print("Error: 'trackpy' not found. Run 'pip install trackpy'.")
                sys.exit(1)

        print(f"\nInitializing {model_type} model...")
        if model_type == "rf-detr":
            model = get_rfdetr_model(
                variant, checkpoint, device, num_classes=num_classes, num_queries=num_queries
            )
        elif model_type == "yolo":
            model = get_yolo_model(checkpoint)
        elif model_type == "lodestar":
            model = get_lodestar_model(checkpoint, device, fp16=lodestar_fp16)
        else:
            model = None
    else:
        model = None

    used_stems: set = set()

    for raw_path in input_paths:
        input_path = Path(raw_path)
        stem = input_path.stem
        if stem in used_stems:
            counter = 2
            while f"{stem}_{counter}" in used_stems:
                counter += 1
            stem = f"{stem}_{counter}"
        used_stems.add(stem)
        per_input_output_dir = output_dir / stem
        per_input_output_dir.mkdir(parents=True, exist_ok=True)

        is_lammpstrj = input_path.suffix.lower() == ".lammpstrj"
        print(f"\nInput:     {input_path}")
        print(f"Output:    {per_input_output_dir}")

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
            csv_path = per_input_output_dir / "tracks.csv"
            df_final.to_csv(csv_path, index=False)
            print(f"Saved tracking data to {csv_path}")
            continue

        # -----------------------------------------------------------------------
        # Image / video pipeline
        # -----------------------------------------------------------------------
        print(f"\nLoading frames from {input_path}...")
        frames = load_frames(input_path)
        if not frames:
            print("No frames found. Skipping.")
            continue
        if max_frames is not None:
            frames = frames[:max_frames]
        print(f"Found {len(frames)} frames.")

        # Probe mode: compute crop size and optionally suggest threshold, then continue
        if probe_mode:
            print(f"Probing density on {probe_n_samples} sampled frames...")
            p95, fw, fh = run_density_probe(
                frames,
                model,
                model_type,
                threshold,
                n_samples=probe_n_samples,
                lodestar_alpha=lodestar_alpha,
                lodestar_nms_distance=lodestar_nms_distance,
                lodestar_box_size=lodestar_box_size,
                device=device,
            )
            crop_w, crop_h = suggest_crop_size(p95, fw, fh)
            print(f"  p95 detections/frame: {p95:.1f}, frame: {fw}×{fh}")
            print(f"PROBE_RESULT crop_w={crop_w} crop_h={crop_h}")
            if probe_threshold_mode:
                print(f"Probing threshold distribution on {probe_n_samples} sampled frames...")
                suggested, method = probe_threshold(
                    frames,
                    model,
                    model_type,
                    n_samples=probe_n_samples,
                    lodestar_alpha=lodestar_alpha,
                    lodestar_nms_distance=lodestar_nms_distance,
                    lodestar_box_size=lodestar_box_size,
                    device=device,
                )
                print(f"PROBE_THRESHOLD suggested={suggested:.4f} method={method}")
            continue

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
        if tiling_enabled:
            fh, fw = frames[0].shape[:2]
            # tile_size: explicit config value -> dataset-profile-derived
            # (clamped to this input's own frame dimensions) -> this file's
            # own hardcoded default (1024, matching config.yaml's long-
            # standing literal) when neither applies.
            tiling_tile_size = int(
                resolve_tile_size(
                    tiling_explicit_tile_size, detection_profile, fw, fh, hardcoded_default=1024
                )
            )
            stride = tiling_tile_size - tiling_overlap
            nx = len(list(range(0, fw - tiling_tile_size, stride))) + 1
            ny = len(list(range(0, fh - tiling_tile_size, stride))) + 1
            print(
                f"Tiling:    {nx}×{ny} tiles, tile_size={tiling_tile_size}, overlap={tiling_overlap} (frame {fw}×{fh})"
            )
            if tiling_explicit_tile_size is None and detection_profile is None:
                print(
                    "Warning:   tiling has no explicit tile_size and no dataset_profile -- "
                    f"running at the {tiling_tile_size}px hardcoded fallback, which may be "
                    "larger than this frame and silently cap detections at num_queries. "
                    "Set dataset_profile (or tile_size) to derive a value for this dataset."
                )

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
                if tiling_enabled:
                    detections = detect_with_tiling(
                        model,
                        detect_frame,
                        threshold,
                        tiling_tile_size,
                        tiling_overlap,
                        tiling_nms_threshold,
                    )
                else:
                    detections = model.predict(detect_frame, threshold=threshold)
            elif model_type == "yolo":
                results = model.predict(detect_frame, conf=threshold, device=device, verbose=False)[
                    0
                ]
                detections = sv.Detections.from_ultralytics(results)
            elif model_type == "lodestar":
                detections = detect_lodestar(
                    model,
                    detect_frame,
                    threshold,
                    device,
                    alpha=lodestar_alpha,
                    nms_distance=lodestar_nms_distance,
                    box_size=lodestar_box_size,
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
                        "conf": (
                            detections.confidence[j] if detections.confidence is not None else 1.0
                        ),
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
                effective_stub_filter = stub_filter
                if preview_active and stub_filter is not None and stub_filter > 0:
                    # Preview-only informational count: link once here (separately
                    # from the real linking pass below, via the same shared
                    # implementation) purely to report what the full-run
                    # stub_filter would produce on these same preview frames.
                    # Does not affect this run's reported tracks.
                    preview_linked = link_and_filter_tracks(
                        df,
                        search_range=search_range,
                        memory=memory,
                        adaptive_stop=adaptive_stop,
                        adaptive_step=adaptive_step,
                    )
                    effective_stub_filter = resolve_preview_stub_filter(stub_filter, len(frames))
                    full_run_track_count = count_tracks_at_stub_filter(preview_linked, stub_filter)
                    if effective_stub_filter < stub_filter:
                        print(
                            f"Preview mode: relaxed stub_filter {stub_filter} -> "
                            f"{effective_stub_filter} to fit the {len(frames)}-frame preview "
                            f"(the full-run stub_filter would likely under-report tracks this short)."
                        )
                    print(
                        f"Preview mode (log-only): the full-run stub_filter={stub_filter} would "
                        f"produce {full_run_track_count} track(s) on these same {len(frames)} "
                        f"preview frames — informational only, does not affect this run's "
                        f"reported tracks."
                    )

                if bridge_gap is not None:
                    radius = bridge_radius if bridge_radius is not None else 2.0 * search_range
                    print(f"Bridging track gaps (max_gap={bridge_gap}, radius={radius:.1f}px)...")
                linked = link_and_filter_tracks(
                    df,
                    search_range=search_range,
                    memory=memory,
                    stub_filter=effective_stub_filter,
                    adaptive_stop=adaptive_stop,
                    adaptive_step=adaptive_step,
                    bridge_gap=bridge_gap,
                    bridge_radius=bridge_radius,
                )
                tracking_data = linked.to_dict("records")
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
                    if video_labels:
                        annotated_frame = label_annotator.annotate(
                            scene=annotated_frame, detections=detections, labels=labels
                        )
                annotated_frames.append(annotated_frame)

            video_path = per_input_output_dir / "tracking_visualization.mp4"
            h, w = annotated_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
            for f in annotated_frames:
                out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            out.release()
            print(f"Saved annotated video to {video_path}")

        # 4. Save results
        df_final = pd.DataFrame(tracking_data)
        csv_path = per_input_output_dir / "tracks.csv"
        df_final.to_csv(csv_path, index=False)
        print(f"Saved tracking data to {csv_path}")

        run_meta: dict = {"model_type": model_type, "threshold": threshold, "tracker": tracker}
        if crop_x is not None:
            run_meta["crop"] = {"x": crop_x, "y": crop_y, "w": crop_w, "h": crop_h}
        if bridge_gap is not None:
            run_meta["bridge_gap"] = bridge_gap
        compute_and_save_metrics(df_final, det_counts, run_meta, per_input_output_dir)

        if save_trajectory_image and not df_final.empty and "track_id" in df_final.columns:
            print("Rendering trajectory image...")
            img_path = per_input_output_dir / "trajectories.png"
            _save_trajectory_image(df_final, frames[-1], img_path, colormap=trajectory_colormap)
            print(f"Saved trajectory image to {img_path}")

        if save_hexatic_order and not df_final.empty:
            print("Computing hexatic order parameter...")
            lammps_scripts_dir = SCRIPT_DIR / ".." / "lammps-scripts"
            lammps_venv_site = list(
                (lammps_scripts_dir / ".venv").glob("lib/python*/site-packages")
            )
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
                    hexatic_path = per_input_output_dir / "hexatic_order.png"
                    plt.savefig(str(hexatic_path), dpi=300)
                    plt.close()
                    print(f"Saved hexatic order plot to {hexatic_path}")
            except ImportError:
                print(
                    "Warning: could not import hexatic_order_analysis — ensure freud is installed in lammps-scripts/.venv"
                )


if __name__ == "__main__":
    main()
