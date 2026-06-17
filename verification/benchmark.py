#!/usr/bin/env python3
"""Benchmark RF-DETR detection accuracy on synthetic frames from render.py.

Loads synthetic TIFF frames and their ground_truth.json, runs RF-DETR
(with optional tiling), matches detections to known particle positions,
and reports per-frame precision/recall/F1 and mean position error.

Optionally computes MOTA/IDF1/fragmentation via py-motmetrics when
--ground-truth-tracks is supplied (CSV from render.py U1).

Note: the tracking metrics here use a standalone trackpy pass configured
via the tracking: section in config.yaml.  This is NOT the production
particle-tracking/track.py linker.  Run a separate comparison against
production tracker output before using MOTA/IDF1 for model selection.

Usage:
    uv run python benchmark.py \\
        --frames verification_output/synthetic_frames/ \\
        --ground-truth verification_output/ground_truth.json \\
        --ground-truth-tracks verification_output/ground_truth_tracks.csv
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import tifffile
import yaml
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# RF-DETR loading and tiling — inlined from particle-tracking/track.py to
# avoid importing the full script (which may execute module-level setup code).
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent

RFDETR_VARIANTS = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
    "base": "RFDETRBase",
}


def _normalize_device(device):
    if device is None:
        return None
    s = str(device).strip()
    if s.lstrip("-").isdigit():
        return f"cuda:{s}"
    return s


def get_rfdetr_model(variant, checkpoint, device, num_queries=None):
    """Load RF-DETR from rf-detr/.venv via site-packages injection."""
    rf_detr_venv = SCRIPT_DIR / ".." / "rf-detr" / ".venv"
    site_packages = list(rf_detr_venv.glob("lib/python*/site-packages"))
    if site_packages and str(site_packages[0]) not in sys.path:
        sys.path.insert(0, str(site_packages[0]))
    for mod in list(sys.modules):
        if mod in ("torch", "torchvision") or mod.startswith(("torch.", "torchvision.")):
            del sys.modules[mod]
    try:
        import rfdetr as _rfdetr

        cls_name = RFDETR_VARIANTS.get(variant)
        if cls_name is None:
            print(
                f"Error: unknown RF-DETR variant '{variant}'. Choose from: {', '.join(RFDETR_VARIANTS)}"
            )
            sys.exit(1)
        cls = getattr(_rfdetr, cls_name)
        kwargs = {"pretrain_weights": str(checkpoint)}
        normalized = _normalize_device(device)
        if normalized is not None:
            kwargs["device"] = normalized
        if num_queries is not None:
            kwargs["num_queries"] = num_queries
        model = cls(**kwargs)
        if hasattr(model, "optimize_for_inference"):
            model.optimize_for_inference()
        return model
    except ImportError:
        print("Error: 'rfdetr' not found. Run 'uv sync' inside rf-detr/.")
        sys.exit(1)


def detect_with_tiling(model, frame, threshold, tile_size, overlap, nms_threshold):
    """Run RF-DETR on overlapping tiles, merge with NMS. Falls back to
    a single predict() call when the frame fits within tile_size."""
    import supervision as sv

    H, W = frame.shape[:2]
    if H <= tile_size and W <= tile_size:
        return model.predict(frame, threshold=threshold)

    stride = tile_size - overlap

    def tile_starts(length):
        starts = list(range(0, length - tile_size, stride))
        starts.append(length - tile_size)
        return starts

    all_xyxy, all_conf, all_class_id = [], [], []
    for y0 in tile_starts(H):
        for x0 in tile_starts(W):
            tile = frame[y0 : y0 + tile_size, x0 : x0 + tile_size]
            dets = model.predict(tile, threshold=threshold)
            if len(dets) > 0:
                boxes = dets.xyxy.copy()
                boxes[:, [0, 2]] += x0
                boxes[:, [1, 3]] += y0
                all_xyxy.append(boxes)
                all_conf.append(dets.confidence)
                all_class_id.append(dets.class_id)

    if not all_xyxy:
        return sv.Detections.empty()

    merged = sv.Detections(
        xyxy=np.concatenate(all_xyxy),
        confidence=np.concatenate(all_conf),
        class_id=np.concatenate(all_class_id),
    )
    return merged.with_nms(threshold=nms_threshold)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def _cfg_get(cfg, *keys, default=None):
    node = cfg
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node


# ---------------------------------------------------------------------------
# Detection matching
# ---------------------------------------------------------------------------


def _match_detections(pred_centers, gt_centers, match_distance):
    """Greedy GT-centric nearest-neighbour matching (each GT matched at most once).

    Returns (n_tp, n_fp, n_fn, matched_distances).
    """
    if len(gt_centers) == 0:
        return 0, len(pred_centers), 0, []
    if len(pred_centers) == 0:
        return 0, 0, len(gt_centers), []

    tree = cKDTree(pred_centers)
    dists, pred_indices = tree.query(gt_centers, k=1, distance_upper_bound=match_distance)

    matched_pred = set()
    matched_dists = []
    tp = 0
    for dist, pred_idx in zip(dists, pred_indices):
        if dist <= match_distance and pred_idx not in matched_pred:
            tp += 1
            matched_pred.add(int(pred_idx))
            matched_dists.append(float(dist))

    fp = len(pred_centers) - tp
    fn = len(gt_centers) - tp
    return tp, fp, fn, matched_dists


# ---------------------------------------------------------------------------
# Frame loading
# ---------------------------------------------------------------------------


def _load_frame_rgb(tiff_path):
    """Load a (potentially 16-bit) TIFF and return uint8 RGB for RF-DETR."""
    img = tifffile.imread(str(tiff_path))
    if img.dtype != np.uint8:
        f = img.astype(np.float32)
        lo, hi = f.min(), f.max()
        if hi > lo:
            f = (f - lo) / (hi - lo) * 255.0
        img = f.clip(0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    return img


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _run_tracking_metrics(all_detections_by_frame, gt_tracks_path, cfg):
    """Run trackpy linking + motmetrics evaluation.

    Args:
        all_detections_by_frame: dict frame_idx → (N, 2) float array of pred (x, y)
        gt_tracks_path: path to ground_truth_tracks.csv
        cfg: full config dict

    Returns:
        dict of tracking metric values, or None if prerequisites missing.
    """
    tracking_enabled = _cfg_get(cfg, "tracking", "enabled", default=True)
    if not tracking_enabled:
        return None

    try:
        import motmetrics as mm
        import trackpy as tp
    except ImportError as e:
        missing = "motmetrics" if "motmetrics" in str(e) else "trackpy"
        print(
            f"Warning: {missing} not installed — skipping tracking metrics. "
            f"Run: uv add {missing}"
        )
        return None

    gt_path = Path(gt_tracks_path)
    if not gt_path.exists():
        print(f"Warning: {gt_path} not found — skipping tracking metrics (run render.py first).")
        return None

    import pandas as pd

    gt_df = pd.read_csv(gt_path)
    required = {"frame", "particle_id", "x", "y"}
    if not required.issubset(gt_df.columns):
        print(
            f"Warning: {gt_path} missing columns {required - set(gt_df.columns)} — skipping tracking metrics."
        )
        return None

    search_range = _cfg_get(cfg, "tracking", "search_range", default=15)
    memory = _cfg_get(cfg, "tracking", "memory", default=3)
    threshold_radii = _cfg_get(cfg, "tracking", "matching_threshold_radii", default=0.5)
    # psf_sigma_px can be under synthetic.psf_sigma (procedural) or synthetic.psf.sigma_px (deeptrack)
    psf_sigma_px = _cfg_get(cfg, "synthetic", "psf_sigma", default=None)
    if psf_sigma_px is None:
        psf_sigma_px = _cfg_get(cfg, "synthetic", "psf", "sigma_px", default=5.0)
    match_threshold = threshold_radii * psf_sigma_px

    # Build trackpy DataFrame from accumulated detections
    rows = []
    for frame_idx, centers in sorted(all_detections_by_frame.items()):
        for cx, cy in centers:
            rows.append({"frame": frame_idx, "x": cx, "y": cy})

    if not rows:
        print("Warning: no detections accumulated — skipping tracking metrics.")
        return None

    det_df = pd.DataFrame(rows)
    tp.quiet()
    linked = tp.link_df(det_df, search_range=search_range, memory=memory)
    linked = linked.rename(columns={"particle": "track_id"})

    # Build motmetrics accumulator frame-by-frame
    acc = mm.MOTAccumulator(auto_id=True)
    for frame_idx in sorted(gt_df["frame"].unique()):
        gt_frame = gt_df[gt_df["frame"] == frame_idx]
        pred_frame = (
            linked[linked["frame"] == frame_idx] if "frame" in linked.columns else pd.DataFrame()
        )

        gt_ids = gt_frame["particle_id"].tolist()
        gt_xy = gt_frame[["x", "y"]].to_numpy()

        pred_ids = pred_frame["track_id"].tolist() if not pred_frame.empty else []
        pred_xy = pred_frame[["x", "y"]].to_numpy() if not pred_frame.empty else np.zeros((0, 2))

        if len(gt_ids) == 0 and len(pred_ids) == 0:
            acc.update([], [], [])
            continue

        if len(gt_ids) > 0 and len(pred_ids) > 0:
            from scipy.spatial.distance import cdist

            dist_matrix = cdist(gt_xy, pred_xy) / psf_sigma_px
        elif len(gt_ids) > 0:
            dist_matrix = np.full((len(gt_ids), 0), np.nan)
        else:
            dist_matrix = np.full((0, len(pred_ids)), np.nan)

        acc.update(gt_ids, pred_ids, dist_matrix)

    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=[
            "mota",
            "idf1",
            "num_fragmentations",
            "num_switches",
            "num_misses",
            "num_false_positives",
        ],
        name="tracking",
    )

    result = {
        "mota": float(summary["mota"].iloc[0]),
        "idf1": float(summary["idf1"].iloc[0]),
        "num_fragmentations": int(summary["num_fragmentations"].iloc[0]),
        "num_switches": int(summary["num_switches"].iloc[0]),
        "num_misses": int(summary["num_misses"].iloc[0]),
        "num_false_positives": int(summary["num_false_positives"].iloc[0]),
        "matching_threshold_radii": threshold_radii,
        "psf_sigma_px": psf_sigma_px,
        "match_threshold_px": match_threshold,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Benchmark RF-DETR on synthetic frames")
    parser.add_argument(
        "--frames", required=True, help="Directory of synthetic TIFF frames (from render.py)"
    )
    parser.add_argument(
        "--ground-truth", required=True, help="Path to ground_truth.json (from render.py)"
    )
    parser.add_argument(
        "--ground-truth-tracks",
        default=None,
        help="Path to ground_truth_tracks.csv (from render.py) — enables MOTA/IDF1 tracking metrics",
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--device", default="0", help="CUDA device index or 'cpu'")
    args = parser.parse_args()

    cfg = _load_config(args.config).get("benchmark", {})
    checkpoint = Path(
        _cfg_get(cfg, "checkpoint", default="../rf-detr/checkpoints/checkpoint_best_ema.pth")
    )
    variant = _cfg_get(cfg, "variant", default="large")
    num_queries = _cfg_get(cfg, "num_queries", default=300)
    threshold = _cfg_get(cfg, "threshold", default=0.3)
    match_distance = _cfg_get(cfg, "match_distance", default=10)
    tiling_enabled = _cfg_get(cfg, "tiling", "enabled", default=True)
    tile_size = _cfg_get(cfg, "tiling", "tile_size", default=512)
    overlap = _cfg_get(cfg, "tiling", "overlap", default=50)
    nms_threshold = _cfg_get(cfg, "tiling", "nms_threshold", default=0.3)

    if not checkpoint.exists():
        print(f"Error: checkpoint not found at {checkpoint}")
        sys.exit(1)

    with open(args.ground_truth) as f:
        ground_truth = json.load(f)
    gt_by_frame = {entry["frame"]: entry for entry in ground_truth}

    frames_dir = Path(args.frames)
    tiff_files = sorted(frames_dir.glob("frame_*.tif"))
    if not tiff_files:
        print(f"Error: no frame_*.tif files in {frames_dir}")
        sys.exit(1)

    print(f"Checkpoint: {checkpoint}")
    print(f"Frames:     {len(tiff_files)}")
    print(f"Tiling:     {'enabled' if tiling_enabled else 'disabled'} (tile_size={tile_size})")

    model = get_rfdetr_model(variant, checkpoint, args.device, num_queries=num_queries)

    rows = []
    all_tp = all_fp = all_fn = 0
    all_dists = []
    all_detections_by_frame = {}  # frame_idx → (N, 2) array of (x, y) centroids

    for tiff_path in tiff_files:
        frame_idx = int(tiff_path.stem.replace("frame_", ""))
        if frame_idx not in gt_by_frame:
            continue

        gt_entry = gt_by_frame[frame_idx]
        gt_pos = gt_entry["positions"]
        gt_centers = np.array(gt_pos, dtype=np.float64) if gt_pos else np.zeros((0, 2))

        img_rgb = _load_frame_rgb(tiff_path)

        if tiling_enabled:
            dets = detect_with_tiling(model, img_rgb, threshold, tile_size, overlap, nms_threshold)
        else:
            dets = model.predict(img_rgb, threshold=threshold)

        if len(dets) > 0:
            pred_centers = ((dets.xyxy[:, :2] + dets.xyxy[:, 2:]) / 2).astype(np.float64)
        else:
            pred_centers = np.zeros((0, 2))

        tp, fp, fn, dists = _match_detections(pred_centers, gt_centers, match_distance)
        all_tp += tp
        all_fp += fp
        all_fn += fn
        all_dists.extend(dists)
        all_detections_by_frame[frame_idx] = pred_centers

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        mean_err = float(np.mean(dists)) if dists else float("nan")

        rows.append(
            {
                "frame": frame_idx,
                "n_gt": len(gt_centers),
                "n_det": len(pred_centers),
                "n_tp": tp,
                "n_fp": fp,
                "n_fn": fn,
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "mean_pos_error_px": "" if np.isnan(mean_err) else round(mean_err, 2),
            }
        )

    # Write per-frame CSV
    output_dir = Path("verification_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "accuracy_metrics.csv"
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    # Overall summary
    overall_prec = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    overall_rec = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
    overall_f1 = (
        2 * overall_prec * overall_rec / (overall_prec + overall_rec)
        if (overall_prec + overall_rec) > 0
        else 0.0
    )
    overall_err = float(np.mean(all_dists)) if all_dists else float("nan")

    print(f"\n=== Benchmark Summary ({len(rows)} frames) ===")
    print(f"Precision:           {overall_prec:.4f}")
    print(f"Recall:              {overall_rec:.4f}")
    print(f"F1:                  {overall_f1:.4f}")
    if not np.isnan(overall_err):
        print(f"Mean position error: {overall_err:.2f} px")
    print(f"Per-frame metrics:   {csv_path}")

    # --- Tracking metrics (optional) ---
    if args.ground_truth_tracks:
        full_cfg = _load_config(args.config)
        tracking_metrics = _run_tracking_metrics(
            all_detections_by_frame, args.ground_truth_tracks, full_cfg
        )
        if tracking_metrics:
            tracking_csv_path = output_dir / "tracking_metrics.csv"
            with open(tracking_csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(tracking_metrics.keys()))
                writer.writeheader()
                writer.writerow(tracking_metrics)

            print(f"\n=== Tracking Metrics ===")
            print(f"MOTA:              {tracking_metrics['mota']:.4f}")
            print(f"IDF1:              {tracking_metrics['idf1']:.4f}")
            print(f"Fragmentations:    {tracking_metrics['num_fragmentations']}")
            print(f"ID switches:       {tracking_metrics['num_switches']}")
            print(
                f"Match threshold:   {tracking_metrics['matching_threshold_radii']} × "
                f"{tracking_metrics['psf_sigma_px']} px = {tracking_metrics['match_threshold_px']:.2f} px"
            )
            print(f"Tracking metrics:  {tracking_csv_path}")
    else:
        print("\n(Tracking metrics skipped — pass --ground-truth-tracks to enable)")


if __name__ == "__main__":
    main()
