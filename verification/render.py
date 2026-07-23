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
import re
import sys
from pathlib import Path

import matplotlib.image as mplimg
import numpy as np
import yaml

from frames_to_video import frames_to_video

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

    # Dark-ring parameters (R3): a difference-of-Gaussians term, expressed as
    # factors of psf_sigma, subtracted from the bright core. Missing/absent
    # `ring` config falls back to these documented defaults rather than
    # raising KeyError — see config.yaml's synthetic.ring block and
    # docs/plans/2026-07-22-004-feat-procedural-renderer-ring-and-noise-plan.md U2.
    ring_cfg = cfg.get("ring", {})
    ring_radius_factor = ring_cfg.get("radius_factor", 2.2)
    ring_width_factor = ring_cfg.get("width_factor", 0.5)
    ring_depth = ring_cfg.get("depth", 0.4)
    ring_width = ring_width_factor * sigma

    img = np.zeros((H, W), dtype=np.float64)
    # ROI radius (R4) must cover both the bright core and the ring's outer
    # Gaussian tail (radius_factor*sigma) plus a margin of a few
    # width_factor*sigma, or the ring would be clipped by the ROI bounds.
    core_extent = 3 * sigma
    ring_extent = ring_radius_factor * sigma + 3 * ring_width
    r = int(max(core_extent, ring_extent)) + 1

    for x, y in positions_lj:
        # Map LJ → pixel coordinates (auto-scales to any box size)
        cx = (x - x_lo) / (x_hi - x_lo) * W
        cy = (y - y_lo) / (y_hi - y_lo) * H

        # Stamp a core-plus-ring PSF onto a small ROI (avoids full-frame ops)
        x0, x1 = max(0, int(cx) - r), min(W, int(cx) + r + 1)
        y0, y1 = max(0, int(cy) - r), min(H, int(cy) + r + 1)
        if x0 >= x1 or y0 >= y1:
            continue

        xs = np.arange(x0, x1, dtype=np.float64)
        ys = np.arange(y0, y1, dtype=np.float64)
        # The ring term depends on Euclidean radius from the particle center,
        # not independent x/y offsets, so — unlike the core — it does NOT
        # factor into a separable outer product. A separable/outer-product
        # implementation of the ring would produce a diamond/star-shaped
        # non-isotropic artifact instead of a circular ring, so it must be
        # evaluated over an explicit 2D radius grid.
        X, Y = np.meshgrid(xs, ys)
        r_grid = np.hypot(X - cx, Y - cy)
        core = np.exp(-0.5 * (r_grid / sigma) ** 2)
        if ring_depth > 0 and ring_width > 0:
            ring = ring_depth * np.exp(
                -0.5 * ((r_grid - ring_radius_factor * sigma) / ring_width) ** 2
            )
        else:
            ring = 0.0
        img[y0:y1, x0:x1] += peak * (core - ring)

    # R5: the ring's negative dip pushes some pixels well below zero (e.g.
    # roughly -6,500 ADU at peak_intensity=40000 for the default ring
    # parameters at an isolated particle's trough) — numpy.random.Generator
    # .poisson raises ValueError on any negative input, so this clip must
    # run before the shot-noise branch below, not just at this function's
    # existing final clip. Mirrors render_deeptrack.py's clip-before-Poisson
    # precedent (there: `np.clip(frame * gain, 0.0, None)` before
    # `rng.poisson`). The dip landing at exactly 0 (rather than deeply
    # negative) after this clip is the intended dark-ring appearance.
    img = np.clip(img, 0, None)

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


_FWHM_TO_SIGMA = 2.355  # FWHM = 2*sqrt(2*ln2)*sigma ~= 2.355*sigma


def _parse_particle_diameter_lj(lammps_in_path):
    """Extract a particle diameter in LJ units from a LAMMPS .in script.

    Prefers a `set type <N> shape <sx> <sy> <sz>` line (diameter = 2*sx,
    assuming a spherical particle where sx == sy == sz — the common case
    for this repo's ellipsoid-as-sphere sims, e.g.
    lammps-scripts/central_pair_interaction.in:27).

    Falls back to the `variable sigma equal <value>` LJ parameter (e.g.
    central_pair_interaction.in:18) if no `shape` line is present — this
    repo's sims set the LJ pair-interaction sigma equal to the particle
    diameter, so the bare sigma value is used directly as the diameter.

    Raises:
        FileNotFoundError: if lammps_in_path does not exist.
        ValueError: if the file exists but neither a `shape` line nor a
            `variable sigma equal <value>` line with a parseable literal
            number can be found.
    """
    path = Path(lammps_in_path)
    if not path.is_file():
        raise FileNotFoundError(f"LAMMPS input script not found: {lammps_in_path}")

    text = path.read_text()
    lines = [line.split("#", 1)[0].strip() for line in text.splitlines()]

    shape_re = re.compile(r"^set\s+type\s+\d+\s+shape\s+(\S+)\s+(\S+)\s+(\S+)")
    for line in lines:
        m = shape_re.match(line)
        if m:
            try:
                sx = float(m.group(1))
            except ValueError as exc:
                raise ValueError(
                    f"Could not parse numeric shape value from '{line}' in " f"{lammps_in_path}"
                ) from exc
            return 2.0 * sx

    sigma_re = re.compile(r"^variable\s+sigma\s+equal\s+(\S+)")
    for line in lines:
        m = sigma_re.match(line)
        if m:
            try:
                return float(m.group(1))
            except ValueError as exc:
                raise ValueError(
                    f"Could not parse numeric sigma value from '{line}' in "
                    f"{lammps_in_path} (likely an unresolved LAMMPS variable "
                    "reference, e.g. '${sigma}' — a literal number is required)"
                ) from exc

    raise ValueError(
        f"Could not find a 'set type <N> shape ...' line or a "
        f"'variable sigma equal <value>' line in {lammps_in_path}; "
        "cannot derive particle diameter."
    )


def _derive_psf_sigma_from_lammps_in(lammps_in_path, box, image_width):
    """Derive a psf_sigma (px) from a LAMMPS .in script's particle diameter.

    Converts the LJ-unit diameter to pixels using the same per-axis
    LJ-to-pixel scale `_lj_to_pixels` derives from the box bounds
    (image_width / (x_hi - x_lo)), then converts pixel diameter to a
    Gaussian sigma via the FWHM relationship: sigma = FWHM / 2.355.
    """
    diameter_lj = _parse_particle_diameter_lj(lammps_in_path)
    x_lo, x_hi, _y_lo, _y_hi = box
    scale = image_width / (x_hi - x_lo)
    diameter_px = diameter_lj * scale
    return diameter_px / _FWHM_TO_SIGMA


def _dispatch_render(positions_lj, box, cfg, rng, strategy):
    """Dispatch to the appropriate render function based on strategy.

    Args:
        strategy: 'procedural' | 'deeptrack' | 'randomized'

    Returns:
        uint16 numpy array of shape (H, W)
    """
    if strategy == "deeptrack":
        try:
            from render_deeptrack import render_frame_deeptrack

            return render_frame_deeptrack(positions_lj, box, cfg, rng)
        except ImportError:
            raise ImportError(
                "DeepTrack2 rendering requires 'deeptrack==2.0.1'. "
                "Run 'uv add deeptrack==2.0.1' inside verification/. "
            )
    elif strategy == "randomized":
        try:
            from render_randomized import render_frame_randomized

            return render_frame_randomized(positions_lj, box, cfg, rng)
        except ImportError:
            raise ImportError(
                "Randomized rendering requires render_randomized.py. "
                "Ensure the file exists in the verification/ directory."
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
    parser.add_argument(
        "--lammps-in",
        default=None,
        help=(
            "Path to the LAMMPS .in script that produced --lammps's trajectory. "
            "When given, psf_sigma is derived from the script's particle diameter "
            "(overriding config.yaml's psf_sigma for this run) instead of using "
            "an arbitrary constant."
        ),
    )
    parser.add_argument("--frames", type=int, default=None, help="Limit to first N timesteps")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    parser.add_argument("--video", action="store_true", help="Also encode frames into preview.mp4")
    parser.add_argument(
        "--fps", type=float, default=10.0, help="Frame rate for --video output (default: 10)"
    )
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
    if not args.lammps_in:
        print(f"PSF sigma:      {cfg.get('psf_sigma', cfg.get('psf', {}).get('sigma_px', 5.0))} px")
    print(f"Render strategy: {strategy}")
    print(f"Output:         {output_dir}")

    for i, block in enumerate(parse_lammps_dump(args.lammps)):
        if args.frames is not None and i >= args.frames:
            break

        box = _parse_box(block["box_bounds"])

        if args.lammps_in and i == 0:
            cfg["psf_sigma"] = _derive_psf_sigma_from_lammps_in(
                args.lammps_in, box, cfg["image_width"]
            )
            print(
                f"PSF sigma:      {cfg['psf_sigma']:.3f} px "
                f"(derived from --lammps-in {args.lammps_in})"
            )

        positions_lj, atom_ids = _parse_atoms(block["atom_header"], block["atoms"])

        img = _dispatch_render(positions_lj, box, cfg, rng, strategy)

        img_f = img.astype(np.float32)
        lo, hi = img_f.min(), img_f.max()
        img8 = (
            ((img_f - lo) / (hi - lo) * 255).clip(0, 255).astype(np.uint8)
            if hi > lo
            else np.zeros_like(img, dtype=np.uint8)
        )
        png_path = output_dir / f"frame_{i:05d}.png"
        mplimg.imsave(str(png_path), img8, cmap="gray")

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

    if args.video and ground_truth:
        video_path = output_dir / "preview.mp4"
        try:
            frames_to_video(output_dir, video_path, fps=args.fps)
        except (ValueError, RuntimeError) as exc:
            print(f"Video generation failed: {exc}")
        else:
            print(f"Video         → {video_path}")


if __name__ == "__main__":
    main()
