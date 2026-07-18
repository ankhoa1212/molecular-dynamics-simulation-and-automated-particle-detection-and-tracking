#!/usr/bin/env python3
"""Benchmark detection accuracy on synthetic frames from render.py.

Loads synthetic PNG frames and their ground_truth.json, runs RF-DETR (with
optional tiling) or LodeSTAR (full-frame, no tiling) via --model-type,
matches detections to known particle positions, and reports per-frame
precision/recall/F1 and mean position error.

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
        --ground-truth-tracks verification_output/ground_truth_tracks.csv \\
        [--model-type rf-detr|lodestar]
"""

import os
import re
import sys
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).parent

# Venv holding each model type's compiled dependencies (torch, torchvision, and
# either `rfdetr` or `deeplay`/`supervision`). Both run a different Python minor
# version than this script's own venv, so C extensions compiled there won't
# load under a mismatched interpreter without a matching-version re-exec.
# The keys here are the single source of truth for valid --model-type values —
# argparse's `choices` and this dict's default fallback both read from it.
_MODEL_VENV_DIRS = {
    "rf-detr": SCRIPT_DIR / ".." / "rf-detr" / ".venv",
    "lodestar": SCRIPT_DIR / ".." / "particle-tracking" / ".venv",
}
_DEFAULT_MODEL_TYPE = "rf-detr"


def _resolve_model_type(argv):
    """Pre-parse --model-type (or config.yaml's benchmark.model_type) ahead of
    the real argparse.ArgumentParser, so the correct venv can be selected
    before any heavy import happens. Must stay consistent with main()'s
    --model-type default/choices, both sourced from _MODEL_VENV_DIRS /
    _DEFAULT_MODEL_TYPE."""
    for i, arg in enumerate(argv):
        if arg == "--model-type" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--model-type="):
            return arg.split("=", 1)[1]

    config_path = "config.yaml"
    for i, arg in enumerate(argv):
        if arg == "--config" and i + 1 < len(argv):
            config_path = argv[i + 1]
        elif arg.startswith("--config="):
            config_path = arg.split("=", 1)[1]

    # Resolve relative to cwd, matching _load_config()'s resolution in main() —
    # both must agree on which file they're reading, or the pre-parse can pick
    # the wrong venv while main() loads a config naming a different model_type.
    config_path = Path(config_path)
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        model_type = (cfg.get("benchmark") or {}).get("model_type")
        if model_type:
            return model_type

    return _DEFAULT_MODEL_TYPE


def _reexec_for_model_venv(model_type):
    """Re-exec with the resolved model's venv Python if the current interpreter
    is a different minor version. Only called when running as __main__ —
    importing this module (e.g. from tests) must never re-exec the process,
    since re-exec blindly reuses sys.argv, which is only a valid `benchmark.py`
    invocation when this script is actually the one being run."""
    venv_dir = _MODEL_VENV_DIRS.get(model_type, _MODEL_VENV_DIRS["rf-detr"])
    venv_python = (venv_dir / "bin" / "python").absolute()
    if not venv_python.exists():
        return
    site_pkgs = list(venv_dir.glob("lib/python*/site-packages"))
    if not site_pkgs:
        return
    m = re.search(r"python(\d+\.\d+)", str(site_pkgs[0]))
    if m and m.group(1) != f"{sys.version_info.major}.{sys.version_info.minor}":
        os.execv(str(venv_python), [str(venv_python)] + sys.argv)


if __name__ == "__main__":
    _reexec_for_model_venv(_resolve_model_type(sys.argv[1:]))

import argparse
import csv
import json

import matplotlib.image as mplimg
import numpy as np
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# Re-exported from detectors_common — edit there, not here.
#
# verification/.venv never installs detectors_common (only rf-detr/.venv and
# particle-tracking/.venv do), and it only becomes reachable once this
# script's own re-exec above has landed in one of those venvs. Every wrapper
# below therefore imports detectors_common lazily inside its own body rather
# than at module scope — a module-scope import would make `import benchmark`
# (used directly by this file's own test suite, which never re-execs) fail
# with ModuleNotFoundError before a single test runs.
# ---------------------------------------------------------------------------


def _normalize_device(device):
    """Deliberately NOT re-exported from detectors_common, unlike everything
    else in this section: this helper has zero external dependencies (pure
    string mapping, no torch/rfdetr/deeplay), but main() calls it
    unconditionally on every invocation regardless of --model-type. Lazily
    importing it from a package verification/.venv never installs would force
    every test that exercises main() to mock this one dependency-free
    function — friction with no corresponding drift risk, since there's
    nothing here that can drift the way venv-injection or model-loading
    logic can. Kept identical to detectors_common.rfdetr_loader's copy."""
    if device is None:
        return None
    s = str(device).strip()
    if s.lstrip("-").isdigit():
        return f"cuda:{s}"
    return s


def get_rfdetr_model(variant, checkpoint, device, num_classes=None, num_queries=None):
    """Load RF-DETR, injecting rf-detr/.venv's site-packages via the shared loader."""
    from detectors_common.rfdetr_loader import get_rfdetr_model as _impl

    return _impl(
        variant,
        checkpoint,
        device,
        _MODEL_VENV_DIRS["rf-detr"],
        num_classes=num_classes,
        num_queries=num_queries,
    )


def get_lodestar_model(checkpoint, device, fp16=False):
    """Load LodeSTAR, injecting particle-tracking/.venv's site-packages via the shared loader."""
    from detectors_common.lodestar_loader import get_lodestar_model as _impl

    return _impl(
        checkpoint,
        device,
        inject_venv_site_packages=_MODEL_VENV_DIRS["lodestar"],
        fp16=fp16,
    )


def detect_lodestar(model, frame, threshold, device, alpha=0.5, nms_distance=None, box_size=40):
    from detectors_common.lodestar_loader import detect_lodestar as _impl

    return _impl(
        model, frame, threshold, device, alpha=alpha, nms_distance=nms_distance, box_size=box_size
    )


def detect_with_tiling(model, frame, threshold, tile_size, overlap, nms_threshold):
    from detectors_common.tiling import detect_with_tiling as _impl

    return _impl(model, frame, threshold, tile_size, overlap, nms_threshold)


def _load_lodestar_defaults(cfg):
    """Canonical nms_distance/alpha, merged over this config's own
    benchmark.lodestar.* values — see detectors_common.defaults."""
    from detectors_common.defaults import load_detector_config

    return load_detector_config(
        "lodestar", cfg, {"nms_distance": "lodestar.nms_distance", "alpha": "lodestar.alpha"}
    )


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


def _load_frame_rgb(png_path):
    """Load a PNG frame and return uint8 RGB for RF-DETR."""
    img = mplimg.imread(str(png_path))  # float32 [0, 1]; shape (H,W), (H,W,3), or (H,W,4)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    return (img * 255).clip(0, 255).astype(np.uint8)


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
    parser = argparse.ArgumentParser(description="Benchmark RF-DETR/LodeSTAR on synthetic frames")
    parser.add_argument(
        "--frames", required=True, help="Directory of synthetic PNG frames (from render.py)"
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
    parser.add_argument(
        "--model-type",
        choices=list(_MODEL_VENV_DIRS),
        default=None,
        help=f"Detector to benchmark (default: {_DEFAULT_MODEL_TYPE}, or "
        "benchmark.model_type from --config)",
    )
    parser.add_argument("--device", default=None, help="CUDA device index or 'cpu' (default: '0')")
    args = parser.parse_args()

    cfg = _load_config(args.config).get("benchmark", {})
    # Sourced from the same _MODEL_VENV_DIRS/_DEFAULT_MODEL_TYPE as the
    # module-level _resolve_model_type pre-parse, so the two can't drift.
    model_type = args.model_type or _cfg_get(cfg, "model_type", default=_DEFAULT_MODEL_TYPE)
    match_distance = _cfg_get(cfg, "match_distance", default=10)

    if model_type == "lodestar":
        checkpoint = Path(
            _cfg_get(
                cfg,
                "lodestar",
                "checkpoint",
                default="../data-setup/models/lodestar_model_15/model.pt",
            )
        )
        threshold = _cfg_get(cfg, "lodestar", "threshold", default=0.1)
        _lodestar_defaults = _load_lodestar_defaults(cfg)
        alpha = _lodestar_defaults.get("alpha", 0.5)
        nms_distance = _lodestar_defaults.get("nms_distance")
        box_size = _cfg_get(cfg, "lodestar", "box_size", default=40)
        fp16 = _cfg_get(cfg, "lodestar", "fp16", default=False)
        device_raw = args.device or _cfg_get(cfg, "lodestar", "device", default=None)
        # variant/num_queries/tiling_* are RF-DETR-only — the branches below that
        # read them (print, get_rfdetr_model, detect_with_tiling) are all gated
        # behind `model_type != "lodestar"`, so no placeholder values are needed here.
    else:
        checkpoint = Path(
            _cfg_get(cfg, "checkpoint", default="../rf-detr/checkpoints/checkpoint_best_ema.pth")
        )
        variant = _cfg_get(cfg, "variant", default="large")
        num_queries = _cfg_get(cfg, "num_queries", default=300)
        threshold = _cfg_get(cfg, "threshold", default=0.3)
        tiling_enabled = _cfg_get(cfg, "tiling", "enabled", default=True)
        tile_size = _cfg_get(cfg, "tiling", "tile_size", default=512)
        overlap = _cfg_get(cfg, "tiling", "overlap", default=50)
        nms_threshold = _cfg_get(cfg, "tiling", "nms_threshold", default=0.3)
        device_raw = args.device

    # Apply the "0" default before normalizing (not after) — _normalize_device(None)
    # returns None, so normalizing-then-defaulting would leave the raw, un-normalized
    # "0" in place. get_rfdetr_model() re-normalizes internally as a second safety net,
    # but get_lodestar_model()/detect_lodestar() trust this value as-is.
    device = _normalize_device(device_raw or "0")

    if not checkpoint.exists():
        print(f"Error: checkpoint not found at {checkpoint}")
        sys.exit(1)

    with open(args.ground_truth) as f:
        ground_truth = json.load(f)
    gt_by_frame = {entry["frame"]: entry for entry in ground_truth}

    frames_dir = Path(args.frames)
    tiff_files = sorted(frames_dir.glob("frame_*.png"))
    if not tiff_files:
        print(f"Error: no frame_*.png files in {frames_dir}")
        sys.exit(1)

    print(f"Model type: {model_type}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Frames:     {len(tiff_files)}")
    if model_type == "lodestar":
        print("Tiling:     n/a (LodeSTAR runs full-frame, no query cap to tile around)")
    else:
        print(f"Tiling:     {'enabled' if tiling_enabled else 'disabled'} (tile_size={tile_size})")

    if model_type == "lodestar":
        model = get_lodestar_model(checkpoint, device, fp16=fp16)
    else:
        model = get_rfdetr_model(variant, checkpoint, device, num_queries=num_queries)

    rows = []
    all_tp = all_fp = all_fn = 0
    all_dists = []
    all_detections_by_frame = {}  # frame_idx → (N, 2) array of (x, y) centroids

    for png_path in tiff_files:
        frame_idx = int(png_path.stem.replace("frame_", ""))
        if frame_idx not in gt_by_frame:
            continue

        gt_entry = gt_by_frame[frame_idx]
        gt_pos = gt_entry["positions"]
        gt_centers = np.array(gt_pos, dtype=np.float64) if gt_pos else np.zeros((0, 2))

        img_rgb = _load_frame_rgb(png_path)

        if model_type == "lodestar":
            dets = detect_lodestar(
                model,
                img_rgb,
                threshold,
                device,
                alpha=alpha,
                nms_distance=nms_distance,
                box_size=box_size,
            )
        elif tiling_enabled:
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

    # Write per-frame CSV. Named per model_type — a fixed filename would let a
    # later run of the other model type silently overwrite these results.
    output_dir = Path("verification_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"accuracy_metrics_{model_type}.csv"
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
            tracking_csv_path = output_dir / f"tracking_metrics_{model_type}.csv"
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
