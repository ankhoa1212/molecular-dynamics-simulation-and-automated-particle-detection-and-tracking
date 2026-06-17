#!/usr/bin/env python3
"""Compare physics observables between LAMMPS simulation and RF-DETR particle tracks.

Produces three side-by-side comparison plots saved to output_dir:
  hexatic_order.png  — |Ψ₆| (hexatic order parameter) vs. time
  msd.png            — Mean Squared Displacement vs. lag time
  velocity_dist.png  — velocity magnitude distributions

Usage:
    uv run python compare.py --lammps sim.lammpstrj --tracks output/tracks.csv
    uv run python compare.py --lammps sim.lammpstrj          # simulation only
    uv run python compare.py --tracks output/tracks.csv       # tracks only
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Dependency injections: freud (lammps-scripts/.venv) + hexatic_order_analysis
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).parent
_LAMMPS_DIR = _SCRIPT_DIR / ".." / "lammps-scripts"

sys.path.insert(0, str(_LAMMPS_DIR))

_freud_site = list((_LAMMPS_DIR / ".venv").glob("lib/python*/site-packages"))
if _freud_site and str(_freud_site[0]) not in sys.path:
    sys.path.insert(0, str(_freud_site[0]))

try:
    from hexatic_order_analysis import calc_hexatic_from_tracks, parse_and_calc_hexatic

    _HEXATIC_AVAILABLE = True
except ImportError:
    _HEXATIC_AVAILABLE = False
    print(
        "Warning: hexatic_order_analysis not importable — freud not found.\n"
        "  Run: cd lammps-scripts && pip install -r requirements.txt\n"
        "  Hexatic order plot will be skipped."
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
# LAMMPS trajectory loading (inlined from particle-tracking/track.py:310-365)
# ---------------------------------------------------------------------------


def load_lammpstrj(path):
    """Parse .lammpstrj → list of per-timestep DataFrames.

    Each DataFrame has columns: id, x, y (and vx, vy when present in the dump).
    Scaled coords (xs, ys) are converted to real coords using box bounds.
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
            for col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col])
                except ValueError:
                    pass

            if "xu" in df.columns and "yu" in df.columns:
                df = df.rename(columns={"xu": "x", "yu": "y"})
            elif "xs" in df.columns and "ys" in df.columns:
                df["x"] = df["xs"] * (x_hi - x_lo) + x_lo
                df["y"] = df["ys"] * (y_hi - y_lo) + y_lo

            df["timestep"] = timestep
            frames.append(df)

    return frames


# ---------------------------------------------------------------------------
# MSD computation
# ---------------------------------------------------------------------------


def compute_msd(df, id_col, time_col, x_col, y_col, max_lag, scale=1.0):
    """Ensemble- and time-averaged MSD.

    Args:
        df: DataFrame with at least (id_col, time_col, x_col, y_col) columns
        scale: multiply coordinates by this factor before squaring (unit conversion to µm)
        max_lag: maximum lag τ in time units

    Returns:
        (lags, msd_values) — both float arrays; msd in units of scale²
    """
    lags = np.arange(1, max_lag + 1, dtype=float)
    msd_vals = np.full(len(lags), np.nan)

    # Pre-build per-particle lookup tables for speed
    particles = {}
    for pid, group in df.groupby(id_col):
        group = group.sort_values(time_col)
        times = group[time_col].values
        xs = group[x_col].values * scale
        ys = group[y_col].values * scale
        t_map = {int(t): j for j, t in enumerate(times)}
        particles[pid] = (times, xs, ys, t_map)

    for li, lag in enumerate(lags):
        lag = int(lag)
        sq_disps = []
        for times, xs, ys, t_map in particles.values():
            for j, t0 in enumerate(times):
                target = int(t0) + lag
                if target in t_map:
                    k = t_map[target]
                    sq_disps.append((xs[k] - xs[j]) ** 2 + (ys[k] - ys[j]) ** 2)
        if sq_disps:
            msd_vals[li] = float(np.mean(sq_disps))

    return lags, msd_vals


# ---------------------------------------------------------------------------
# Velocity magnitude extraction
# ---------------------------------------------------------------------------


def _sim_velocity_magnitudes(sim_frames):
    """Extract |v| directly from vx, vy columns in LAMMPS frames (LJ units)."""
    parts = []
    for df in sim_frames:
        if "vx" in df.columns and "vy" in df.columns:
            v = np.sqrt(df["vx"].values ** 2 + df["vy"].values ** 2)
            parts.append(v)
    return np.concatenate(parts) if parts else np.array([])


def _track_velocity_magnitudes(
    df_tracks, x_col="x", y_col="y", frame_col="frame", id_col="track_id"
):
    """Estimate |v| from tracking data using central finite differences (px/frame)."""
    speeds = []
    for _, group in df_tracks.groupby(id_col):
        group = group.sort_values(frame_col).reset_index(drop=True)
        xs = group[x_col].values.astype(float)
        ys = group[y_col].values.astype(float)
        if len(xs) < 3:
            continue
        vx = (xs[2:] - xs[:-2]) / 2.0
        vy = (ys[2:] - ys[:-2]) / 2.0
        speeds.append(np.sqrt(vx**2 + vy**2))
    return np.concatenate(speeds) if speeds else np.array([])


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

_SIM_COLOR = "steelblue"
_TRK_COLOR = "darkorange"


def _plot_hexatic(lammps_path, df_tracks, output_path):
    n_panels = (lammps_path is not None) + (df_tracks is not None)
    if n_panels == 0:
        return
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4), squeeze=False)
    ax_idx = 0

    if lammps_path is not None and _HEXATIC_AVAILABLE:
        steps, psi6 = parse_and_calc_hexatic(str(lammps_path), verbose=0)
        ax = axes[0][ax_idx]
        ax.plot(steps, psi6, color=_SIM_COLOR, lw=1.5)
        ax.set_title("Simulation |Ψ₆|")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("|Ψ₆|")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax_idx += 1

    if df_tracks is not None and _HEXATIC_AVAILABLE and "track_id" in df_tracks.columns:
        fw = int(df_tracks["x"].max() * 1.1)
        fh = int(df_tracks["y"].max() * 1.1)
        frame_nums, psi6 = calc_hexatic_from_tracks(df_tracks, fw, fh, verbose=0)
        ax = axes[0][ax_idx]
        ax.plot(frame_nums, psi6, color=_TRK_COLOR, lw=1.5)
        ax.set_title("Tracked particles |Ψ₆|")
        ax.set_xlabel("Frame")
        ax.set_ylabel("|Ψ₆|")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close()
    print(f"Saved {output_path}")


def _plot_msd(df_sim_long, df_tracks, pixel_scale, lj_to_um, max_lag, output_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    plotted = False

    if df_sim_long is not None:
        lags, msd = compute_msd(
            df_sim_long,
            id_col="id",
            time_col="frame",
            x_col="x",
            y_col="y",
            max_lag=max_lag,
            scale=lj_to_um,
        )
        mask = ~np.isnan(msd)
        if mask.sum() >= 2:
            ax.plot(lags[mask], msd[mask], color=_SIM_COLOR, lw=2, label="Simulation (LJ→µm)")
            slope = float(np.polyfit(lags[mask], msd[mask], 1)[0])
            print(f"Simulation D ≈ {slope / 4:.5f} µm²/timestep (from MSD slope)")
            plotted = True

    if df_tracks is not None and "track_id" in df_tracks.columns:
        lags, msd = compute_msd(
            df_tracks,
            id_col="track_id",
            time_col="frame",
            x_col="x",
            y_col="y",
            max_lag=max_lag,
            scale=pixel_scale,
        )
        mask = ~np.isnan(msd)
        if mask.sum() >= 2:
            ax.plot(lags[mask], msd[mask], color=_TRK_COLOR, lw=2, label="Tracks (px→µm)")
            slope = float(np.polyfit(lags[mask], msd[mask], 1)[0])
            print(f"Tracks D ≈ {slope / 4:.5f} µm²/frame (from MSD slope)")
            plotted = True

    if not plotted:
        plt.close()
        return

    ax.set_xlabel("Lag time (frames / timesteps)")
    ax.set_ylabel("MSD (µm²)")
    ax.set_title("Mean Squared Displacement")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close()
    print(f"Saved {output_path}")


def _plot_velocity_dist(sim_frames, df_tracks, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    if sim_frames is not None:
        v_sim = _sim_velocity_magnitudes(sim_frames)
        ax = axes[0]
        if len(v_sim) > 0:
            ax.hist(v_sim, bins=60, density=True, color=_SIM_COLOR, alpha=0.75, edgecolor="none")
            ax.axvline(
                float(np.mean(v_sim)),
                color="navy",
                ls="--",
                lw=1.5,
                label=f"mean = {np.mean(v_sim):.3f}",
            )
            ax.set_xlabel("|v| (LJ velocity units)")
            ax.set_ylabel("Probability density")
            ax.set_title("Simulation velocity |v|")
            ax.legend()
        else:
            ax.text(
                0.5,
                0.5,
                "No vx/vy in trajectory",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="gray",
            )
            ax.set_title("Simulation velocity |v|")
    else:
        axes[0].set_visible(False)

    if df_tracks is not None and "track_id" in df_tracks.columns:
        v_trk = _track_velocity_magnitudes(df_tracks)
        ax = axes[1]
        if len(v_trk) > 0:
            ax.hist(v_trk, bins=60, density=True, color=_TRK_COLOR, alpha=0.75, edgecolor="none")
            ax.axvline(
                float(np.mean(v_trk)),
                color="sienna",
                ls="--",
                lw=1.5,
                label=f"mean = {np.mean(v_trk):.1f} px/frame",
            )
            ax.set_xlabel("|v| (pixels per frame)")
            ax.set_ylabel("Probability density")
            ax.set_title("Tracked particle velocity |v|")
            ax.legend()
        else:
            ax.text(
                0.5,
                0.5,
                "Too few tracks (<3 frames)",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="gray",
            )
            ax.set_title("Tracked particle velocity |v|")
    else:
        axes[1].set_visible(False)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close()
    print(f"Saved {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Compare simulation vs. tracking physics observables"
    )
    parser.add_argument("--lammps", default=None, help="Path to .lammpstrj file")
    parser.add_argument("--tracks", default=None, help="Path to tracks.csv")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg_full = _load_config(args.config)
    cfg = cfg_full.get("compare", {})

    lammps_path = args.lammps or _cfg_get(cfg, "lammps_file")
    tracks_path = args.tracks or _cfg_get(cfg, "tracks_file")

    if not lammps_path and not tracks_path:
        print("Error: provide --lammps, --tracks, or both.")
        sys.exit(1)

    pixel_scale = float(_cfg_get(cfg, "pixel_scale", default=0.108))
    lj_to_um = float(_cfg_get(cfg, "lj_to_um", default=2.0))
    max_lag = int(_cfg_get(cfg, "max_lag_frames", default=50))
    output_dir = Path(_cfg_get(cfg, "output_dir", default="verification_output/"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    sim_frames = None
    df_sim_long = None
    if lammps_path:
        print(f"Loading LAMMPS trajectory: {lammps_path}")
        sim_frames = load_lammpstrj(lammps_path)
        parts = []
        for i, df in enumerate(sim_frames):
            cols = ["id", "x", "y"] + [c for c in ("vx", "vy") if c in df.columns]
            df2 = df[cols].copy()
            df2["frame"] = i
            parts.append(df2)
        df_sim_long = pd.concat(parts, ignore_index=True)
        print(f"  {len(sim_frames)} timesteps, {len(sim_frames[0])} particles/timestep")

    df_tracks = None
    if tracks_path:
        print(f"Loading tracks: {tracks_path}")
        df_tracks = pd.read_csv(tracks_path)
        if "track_id" not in df_tracks.columns and "id" in df_tracks.columns:
            df_tracks = df_tracks.rename(columns={"id": "track_id"})
        n_tracks = df_tracks["track_id"].nunique() if "track_id" in df_tracks.columns else 0
        print(f"  {n_tracks} tracks, {df_tracks['frame'].nunique()} frames")

    # Generate plots
    print("\nComputing hexatic order...")
    _plot_hexatic(
        Path(lammps_path) if lammps_path else None,
        df_tracks,
        output_dir / "hexatic_order.png",
    )

    print("Computing MSD...")
    _plot_msd(df_sim_long, df_tracks, pixel_scale, lj_to_um, max_lag, output_dir / "msd.png")

    print("Computing velocity distributions...")
    _plot_velocity_dist(sim_frames, df_tracks, output_dir / "velocity_dist.png")

    print("\nDone. Outputs in:", output_dir)


if __name__ == "__main__":
    main()
