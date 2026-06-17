#!/usr/bin/env python3
"""Render synthetic 16-bit TIFF frames from a LAMMPS trajectory.

Each LAMMPS timestep becomes one TIFF with particles rendered as Gaussian
spots, plus Poisson shot noise and Gaussian readout noise to mimic real
epi-fluorescence microscopy.  A ground_truth.json is written alongside
the TIFFs with pixel-coordinate positions for each frame.  A
ground_truth_tracks.csv is also written for tracking metric computation.

Usage:
    uv run python render.py --lammps ../lammps-scripts/results/sim.lammpstrj
    uv run python render.py --lammps sim.lammpstrj --frames 20 --config config.yaml

Assumptions:
    LAMMPS atom IDs must be stable across all timesteps (NVT/NVE without
    fix/deposit/evaporate).  An assertion enforces this before writing the
    tracking CSV.

Render strategies (set via synthetic.render_strategy in config.yaml):
    procedural  — flat 2D Gaussian PSF (default; unchanged from original)
    deeptrack   — physics-accurate scalar-diffraction PSF via DeepTrack2
    randomized  — procedural with per-frame stochastic parameter sampling
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import tifffile
import yaml

# lammps_parser.py lives in lammps-scripts/ (pure Python, no venv needed)
sys.path.insert(0, str(Path(__file__).parent / ".." / "lammps-scripts"))
from lammps_parser import parse_lammps_dump


def _load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def _parse_box(box_bounds):
    """Extract (x_lo, x_hi, y_lo, y_hi) from LAMMPS box_bounds strings."""
    x_lo, x_hi = map(float, box_bounds[0].split())
    y_lo, y_hi = map(float, box_bounds[1].split())
    return x_lo, x_hi, y_lo, y_hi


def _parse_atoms(atom_header, atoms):
    """Return positions (N, 2) in LJ units and atom IDs (N,) as int arrays.

    Prefers unwrapped coords (xu, yu) over real coords (x, y).
    If the 'id' column is absent, returns sequential IDs starting from 1.
    """
    cols = atom_header.replace("ITEM: ATOMS", "").split()
    xc = "xu" if "xu" in cols else "x"
    yc = "yu" if "yu" in cols else "y"
    if xc not in cols or yc not in cols:
        return np.zeros((0, 2)), np.zeros(0, dtype=np.int64)

    xi, yi = cols.index(xc), cols.index(yc)
    id_col = cols.index("id") if "id" in cols else None

    pos, ids = [], []
    for idx, line in enumerate(atoms):
        vals = line.split()
        max_needed = max(xi, yi) if id_col is None else max(xi, yi, id_col)
        if len(vals) > max_needed:
            pos.append([float(vals[xi]), float(vals[yi])])
            ids.append(int(vals[id_col]) if id_col is not None else idx + 1)

    if not pos:
        return np.zeros((0, 2)), np.zeros(0, dtype=np.int64)
    return np.array(pos, dtype=np.float64), np.array(ids, dtype=np.int64)


def _parse_positions(atom_header, atoms):
    """Return (N, 2) float array of (x, y) in LJ units.

    Prefers unwrapped coords (xu, yu) over real coords (x, y).
    """
    positions, _ = _parse_atoms(atom_header, atoms)
    return positions


def render_frame(positions_lj, box, cfg, rng):
    """Render one synthetic microscopy frame.

    Args:
        positions_lj: (N, 2) array of particle positions in LJ units
        box: (x_lo, x_hi, y_lo, y_hi) simulation box bounds
        cfg: synthetic config dict
        rng: numpy random Generator

    Returns:
        uint16 numpy array of shape (H, W)
    """
    H = cfg["image_height"]
    W = cfg["image_width"]
    sigma = cfg["psf_sigma"]
    peak = cfg["peak_intensity"]
    x_lo, x_hi, y_lo, y_hi = box

    img = np.zeros((H, W), dtype=np.float64)
    r = int(3 * sigma) + 1

    for x, y in positions_lj:
        # Map LJ → pixel coordinates (auto-scales to any box size)
        cx = (x - x_lo) / (x_hi - x_lo) * W
        cy = (y - y_lo) / (y_hi - y_lo) * H

        # Stamp a Gaussian PSF onto a small ROI (avoids full-frame ops)
        x0, x1 = max(0, int(cx) - r), min(W, int(cx) + r + 1)
        y0, y1 = max(0, int(cy) - r), min(H, int(cy) + r + 1)
        if x0 >= x1 or y0 >= y1:
            continue

        xs = np.arange(x0, x1, dtype=np.float64)
        ys = np.arange(y0, y1, dtype=np.float64)
        gx = np.exp(-0.5 * ((xs - cx) / sigma) ** 2)
        gy = np.exp(-0.5 * ((ys - cy) / sigma) ** 2)
        img[y0:y1, x0:x1] += peak * np.outer(gy, gx)

    if cfg.get("shot_noise", True):
        img = rng.poisson(img).astype(np.float64)
    img += rng.normal(0.0, cfg.get("readout_noise", 15.0), img.shape)
    return np.clip(img, 0, 65535).astype(np.uint16)


def _lj_to_pixels(positions_lj, box, H, W):
    """Convert (N, 2) LJ positions to pixel coordinates, clipping to image boundary."""
    x_lo, x_hi, y_lo, y_hi = box
    px = np.clip((positions_lj[:, 0] - x_lo) / (x_hi - x_lo) * W, 0, W - 1)
    py = np.clip((positions_lj[:, 1] - y_lo) / (y_hi - y_lo) * H, 0, H - 1)
    return np.stack([px, py], axis=1)


def _dispatch_render(positions_lj, box, cfg, rng, strategy):
    """Dispatch to the appropriate render function based on strategy.

    Args:
        strategy: 'procedural' | 'deeptrack' | 'randomized'

    Returns:
        uint16 numpy array of shape (H, W)
    """
    if strategy == "deeptrack":
        # U2 will implement this; import guard gives a clear error until then
        try:
            from render_deeptrack import render_frame_deeptrack

            return render_frame_deeptrack(positions_lj, box, cfg, rng)
        except ImportError:
            raise ImportError(
                "DeepTrack2 rendering requires 'deeptrack==2.0.1'. "
                "Run 'uv add deeptrack==2.0.1' inside verification/. "
                "(render_strategy: deeptrack — implemented in U2)"
            )
    elif strategy == "randomized":
        # U3 will implement this
        try:
            from render_randomized import render_frame_randomized

            return render_frame_randomized(positions_lj, box, cfg, rng)
        except ImportError:
            raise ImportError(
                "Randomized rendering not yet implemented. "
                "(render_strategy: randomized — implemented in U3)"
            )
    else:
        # Default: procedural Gaussian PSF
        return render_frame(positions_lj, box, cfg, rng)


def main():
    parser = argparse.ArgumentParser(
        description="Render synthetic TIFF frames from LAMMPS trajectory"
    )
    parser.add_argument("--lammps", required=True, help="Path to .lammpstrj file")
    parser.add_argument(
        "--config", default="config.yaml", help="Config file (default: config.yaml)"
    )
    parser.add_argument("--frames", type=int, default=None, help="Limit to first N timesteps")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    args = parser.parse_args()

    cfg = _load_config(args.config).get("synthetic", {})
    strategy = cfg.get("render_strategy", "procedural")

    output_dir = Path(cfg.get("output_dir", "verification_output/synthetic_frames/"))
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    ground_truth = []
    # Collect per-frame data for tracks CSV: list of (atom_ids, px_positions)
    all_frame_ids = []
    track_rows = []

    print(f"Rendering from: {args.lammps}")
    print(f"Image size:     {cfg['image_width']}×{cfg['image_height']} px")
    print(f"PSF sigma:      {cfg.get('psf_sigma', cfg.get('psf', {}).get('sigma_px', 5.0))} px")
    print(f"Render strategy: {strategy}")
    print(f"Output:         {output_dir}")

    for i, block in enumerate(parse_lammps_dump(args.lammps)):
        if args.frames is not None and i >= args.frames:
            break

        box = _parse_box(block["box_bounds"])
        positions_lj, atom_ids = _parse_atoms(block["atom_header"], block["atoms"])

        img = _dispatch_render(positions_lj, box, cfg, rng, strategy)

        tiff_path = output_dir / f"frame_{i:05d}.tif"
        tifffile.imwrite(str(tiff_path), img)

        H, W = cfg["image_height"], cfg["image_width"]
        px_pos = (
            _lj_to_pixels(positions_lj, box, H, W) if len(positions_lj) > 0 else np.zeros((0, 2))
        )

        ground_truth.append(
            {
                "frame": i,
                "timestep": block["timestep"],
                "n_particles": len(positions_lj),
                "positions": px_pos.tolist(),
            }
        )

        all_frame_ids.append(atom_ids)
        for atom_id, (px, py) in zip(atom_ids, px_pos):
            track_rows.append(
                {"frame": i, "particle_id": int(atom_id), "x": float(px), "y": float(py)}
            )

        if i == 0 or (i + 1) % 10 == 0:
            print(
                f"  frame {i:4d} (timestep {block['timestep']:8d}): {len(positions_lj):4d} particles"
            )

    gt_path = output_dir.parent / "ground_truth.json"
    with open(gt_path, "w") as f:
        json.dump(ground_truth, f)

    # Validate atom ID stability (required for tracking ground truth)
    if all_frame_ids:
        id_sets = [set(ids.tolist()) for ids in all_frame_ids]
        if not all(s == id_sets[0] for s in id_sets):
            raise AssertionError(
                "Atom ID set changed between frames — assumes NVT/NVE without "
                "fix/deposit/evaporate. Check your LAMMPS trajectory."
            )

        tracks_path = output_dir.parent / "ground_truth_tracks.csv"
        with open(tracks_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["frame", "particle_id", "x", "y"])
            writer.writeheader()
            writer.writerows(track_rows)
        print(f"Tracking GT   → {tracks_path}")

    print(f"\nRendered {len(ground_truth)} frames → {output_dir}")
    print(f"Ground truth  → {gt_path}")


if __name__ == "__main__":
    main()
