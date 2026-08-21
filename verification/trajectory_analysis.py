#!/usr/bin/env python3
"""Three-way sim-to-real trajectory validation: GT-synthetic vs. tracked-synthetic
vs. tracked-real, for two detector pipelines (RF-DETR+trackpy, YOLOv12+trackpy).

Decomposes the total sim-vs-real gap into two named legs:
  tracking-induced error: GT-synthetic vs. tracked-synthetic (same synthetic
    frame timebase, same render-pixel space -- isolates detector/tracker noise
    from physics, since both sides describe the identical rendered video).
  domain gap: tracked-synthetic vs. tracked-real (same detector/tracker
    pipeline on both sides -- isolates real-vs-simulated physics, since
    pipeline noise is present on both sides).

Primary metric is the MSD log-log scaling exponent (alpha), reused from
compare.py's compute_msd -- alpha is scale-invariant, so it's directly
comparable across the synthetic (LAMMPS-timestep-per-frame) and real
(video-frame) timebases without inventing a physical time calibration
that doesn't exist anywhere in this repo. Raw MSD-slope-derived D and mean
velocity are reported per leg as secondary, explicitly-caveated numbers in
physical units (um), using two different pixel_scale values:
  - synthetic legs (GT-synthetic, tracked-synthetic): the render's own
    px-per-LJ-unit mapping (W / box_width), combined with lj_to_um -- see
    _synth_pixel_scale().
  - real legs (tracked-real): the confirmed real-video pixel_scale (shared
    optics with the 2um dataset -- see docs/plans/2026-08-21-001-feat-
    trajectory-analysis-sim-real-validation-plan.md U4).
GT-synthetic's velocity is derived from position-based finite differences on
ground_truth_tracks.csv (mirroring _track_velocity_magnitudes), NOT from the
raw LAMMPS vx/vy columns (compare.py's _sim_velocity_magnitudes) -- the
latter returns unscaled LJ velocity units that aren't comparable to the
other four legs' px/frame-derived velocities.

Usage:
    uv run python trajectory_analysis.py \
        --gt-synthetic ../particle-tracking/../verification/verification_output/trajectory_analysis/synth_N200/ground_truth_tracks.csv \
        --rfdetr-synthetic ../particle-tracking/output/trajectory_analysis/synth_rfdetr/frames/tracks.csv \
        --yolo-synthetic ../particle-tracking/output/trajectory_analysis/synth_yolo/frames/tracks.csv \
        --rfdetr-real ../particle-tracking/output/trajectory_analysis/real_rfdetr/frames_000-150/tracks.csv \
        --yolo-real ../particle-tracking/output/trajectory_analysis/real_yolo/frames_000-150/tracks.csv \
        --lammps ../lammps-scripts/results/density_ablation/continuous_force_200_5.0.lammpstrj \
        --image-width 512 --lj-to-um 5.0 --pixel-scale-real 0.108 \
        --output-dir verification_output/trajectory_analysis
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from compare import compute_msd, _track_velocity_magnitudes  # noqa: E402

MIN_MSD_POINTS = 2
MIN_FIT_QUALITY = 0.90  # R^2 floor below which alpha is unreliable (see
# docs/plans/2026-08-21-001-feat-trajectory-analysis-
# sim-real-validation-plan.md's doc-review findings
# on fit_quality gating and lag-regime mismatch).

LEG_COLORS = {
    "gt_synthetic": "black",
    "rfdetr_synthetic": "steelblue",
    "rfdetr_real": "darkorange",
    "yolo_synthetic": "mediumseagreen",
    "yolo_real": "firebrick",
}
LEG_LABELS = {
    "gt_synthetic": "GT-synthetic",
    "rfdetr_synthetic": "Tracked-synthetic (RF-DETR)",
    "rfdetr_real": "Tracked-real (RF-DETR)",
    "yolo_synthetic": "Tracked-synthetic (YOLOv12)",
    "yolo_real": "Tracked-real (YOLOv12)",
}


def _lammps_box_width(lammps_path):
    """Parse the first frame's x box-bounds width from a .lammpstrj file.

    Mirrors render.py's own box-to-pixel mapping (cx = (x - x_lo) /
    (x_hi - x_lo) * W) so the derived px-per-LJ-unit scale matches what
    render.py actually used to produce the synthetic frames.
    """
    with open(lammps_path) as f:
        for line in f:
            if "ITEM: BOX BOUNDS" in line:
                x_lo, x_hi = map(float, f.readline().split())
                return x_hi - x_lo
    raise ValueError(f"no BOX BOUNDS found in {lammps_path}")


def synth_pixel_scale(lammps_path, image_width, lj_to_um):
    """um per render-pixel, derived from render.py's own box-to-pixel mapping."""
    box_width = _lammps_box_width(lammps_path)
    px_per_lj_unit = image_width / box_width
    return lj_to_um / px_per_lj_unit


def load_gt_synthetic(path):
    df = pd.read_csv(path)
    return df.rename(columns={"particle_id": "track_id"})


def load_tracked(path):
    return pd.read_csv(path)


def fit_msd_exponent(lags, msd):
    """Log-log linear fit of MSD vs lag time -> (alpha, fit_quality).

    fit_quality is the fit's R^2 -- a low value signals the fitted lag
    window isn't well described by a single power law (e.g. it straddles
    a ballistic-to-diffusive crossover), which would make a reported alpha
    misleading rather than a genuine domain-gap signal. Returns (None, None)
    when fewer than MIN_MSD_POINTS valid points are available.
    """
    mask = ~np.isnan(msd) & (msd > 0) & (lags > 0)
    if mask.sum() < MIN_MSD_POINTS:
        return None, None
    log_lags = np.log(lags[mask])
    log_msd = np.log(msd[mask])
    slope, intercept = np.polyfit(log_lags, log_msd, 1)
    pred = slope * log_lags + intercept
    ss_res = np.sum((log_msd - pred) ** 2)
    ss_tot = np.sum((log_msd - np.mean(log_msd)) ** 2)
    fit_quality = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return float(slope), float(fit_quality)


def _velocity_stats(df, id_col, scale):
    """Position-based finite-difference velocity magnitude, scaled to um/frame."""
    speeds = _track_velocity_magnitudes(df, x_col="x", y_col="y", frame_col="frame", id_col=id_col)
    speeds = speeds * scale
    if len(speeds) == 0:
        return None, None
    return float(np.mean(speeds)), float(np.std(speeds))


def analyze_leg(df, id_col, scale, max_lag):
    lags, msd = compute_msd(
        df, id_col=id_col, time_col="frame", x_col="x", y_col="y", max_lag=max_lag, scale=scale
    )
    alpha, fit_quality = fit_msd_exponent(lags, msd)
    raw_slope_D = None
    mask = ~np.isnan(msd)
    if mask.sum() >= MIN_MSD_POINTS:
        raw_slope_D = float(np.polyfit(lags[mask], msd[mask], 1)[0] / 4.0)
    v_mean, v_std = _velocity_stats(df, id_col, scale)
    reliable = fit_quality is not None and fit_quality >= MIN_FIT_QUALITY
    return (
        {
            "alpha": alpha,
            "fit_quality": fit_quality,
            "alpha_reliable": reliable,
            "raw_slope_D": raw_slope_D,
            "velocity_mean": v_mean,
            "velocity_std": v_std,
            "n_msd_points": int(mask.sum()),
        },
        lags,
        msd,
    )


def plot_msd(results, output_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    plotted = False
    for key, (lags, msd) in results.items():
        mask = ~np.isnan(msd) & (msd > 0) & (lags > 0)
        if mask.sum() < MIN_MSD_POINTS:
            continue
        style = "--" if key == "gt_synthetic" else "-"
        ax.loglog(lags[mask], msd[mask], style, color=LEG_COLORS[key], lw=2, label=LEG_LABELS[key])
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("Lag (frames)")
    ax.set_ylabel("MSD (µm²)")
    ax.set_title("Mean Squared Displacement: sim-to-real trajectory decomposition")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_velocity(summary, output_path):
    keys = [k for k in LEG_LABELS if summary.get(k, {}).get("velocity_mean") is not None]
    if not keys:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    means = [summary[k]["velocity_mean"] for k in keys]
    stds = [summary[k]["velocity_std"] for k in keys]
    colors = [LEG_COLORS[k] for k in keys]
    labels = [LEG_LABELS[k] for k in keys]
    ax.bar(range(len(keys)), means, yerr=stds, color=colors, alpha=0.8)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Mean |v| (µm/frame)")
    ax.set_title("Velocity magnitude: sim-to-real trajectory decomposition")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")


def run(inputs, lammps_path, image_width, lj_to_um, pixel_scale_real, max_lag, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scale_synth = synth_pixel_scale(lammps_path, image_width, lj_to_um)

    leg_scale = {
        "gt_synthetic": scale_synth,
        "rfdetr_synthetic": scale_synth,
        "yolo_synthetic": scale_synth,
        "rfdetr_real": pixel_scale_real,
        "yolo_real": pixel_scale_real,
    }

    summary = {}
    msd_curves = {}
    for key, path in inputs.items():
        if path is None or not Path(path).exists():
            print(f"Skipping {key}: no input file")
            continue
        df = load_gt_synthetic(path) if key == "gt_synthetic" else load_tracked(path)
        stats, lags, msd = analyze_leg(df, id_col="track_id", scale=leg_scale[key], max_lag=max_lag)
        summary[key] = stats
        msd_curves[key] = (lags, msd)
        print(
            f"{LEG_LABELS[key]}: alpha={stats['alpha']}, fit_quality={stats['fit_quality']}, "
            f"reliable={stats['alpha_reliable']}, D={stats['raw_slope_D']}, "
            f"v_mean={stats['velocity_mean']}, v_std={stats['velocity_std']}"
        )

    plot_msd(msd_curves, output_dir / "msd_comparison.png")
    plot_velocity(summary, output_dir / "velocity_comparison.png")

    summary_meta = {
        "scale_synth_um_per_px": scale_synth,
        "scale_real_um_per_px": pixel_scale_real,
        "lj_to_um": lj_to_um,
        "max_lag_frames": max_lag,
        "min_fit_quality": MIN_FIT_QUALITY,
    }
    out = {"legs": summary, "meta": summary_meta}
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {summary_path}")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-synthetic", default=None)
    parser.add_argument("--rfdetr-synthetic", default=None)
    parser.add_argument("--yolo-synthetic", default=None)
    parser.add_argument("--rfdetr-real", default=None)
    parser.add_argument("--yolo-real", default=None)
    parser.add_argument(
        "--lammps", required=True, help="N=200 trajectory (for box-width scale derivation)"
    )
    parser.add_argument("--image-width", type=int, default=512)
    parser.add_argument("--lj-to-um", type=float, default=5.0)
    parser.add_argument("--pixel-scale-real", type=float, default=0.108)
    parser.add_argument("--max-lag", type=int, default=50)
    parser.add_argument("--output-dir", default="verification_output/trajectory_analysis")
    args = parser.parse_args()

    inputs = {
        "gt_synthetic": args.gt_synthetic,
        "rfdetr_synthetic": args.rfdetr_synthetic,
        "rfdetr_real": args.rfdetr_real,
        "yolo_synthetic": args.yolo_synthetic,
        "yolo_real": args.yolo_real,
    }
    run(
        inputs,
        args.lammps,
        args.image_width,
        args.lj_to_um,
        args.pixel_scale_real,
        args.max_lag,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
