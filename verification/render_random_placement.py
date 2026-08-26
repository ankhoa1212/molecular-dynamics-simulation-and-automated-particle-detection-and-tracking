#!/usr/bin/env python3
"""Generate synthetic microscopy frames with randomly-placed particles (DeepTrack-style baseline).

Particles are placed uniformly at random within the simulation box — no physics, no LAMMPS
trajectory, no minimum separation. All rendering is delegated to render.py's existing
_dispatch_render / render_frame functions so the renderer, PSF parameters, and noise model
are identical to the physics-grounded condition.

Output format exactly matches render.py:
  <output-dir>/frame_NNNNN.png      — 8-bit PNG previews
  <output-dir>/../ground_truth.json  — pixel positions per frame
  <output-dir>/../ground_truth_tracks.csv — per-particle tracks

Usage:
    uv run python render_random_placement.py --frames 30 --config config.yaml --seed 42
    uv run python render_random_placement.py --frames 30 \\
        --config configs/render_deeptrack_comparison.yaml \\
        --seed 42 \\
        --output-dir verification_output/deeptrack_comparison/random/synthetic_frames/
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.image as mplimg
import numpy as np

from render import (
    _dispatch_render,
    _lj_to_pixels,
    _load_config,
    _stretch_to_uint8,
)
from run_provenance import write_manifest


def generate_random_positions(W, H, N, rng):
    """Return (N, 2) float64 array of uniformly random pixel-space positions.

    x is drawn from Uniform(0, W) and y from Uniform(0, H) independently,
    matching a simple Poisson spatial process (DeepTrack-style random placement).

    Args:
        W: image width in pixels
        H: image height in pixels
        N: number of particles
        rng: numpy.random.Generator instance

    Returns:
        positions: (N, 2) float64 array, columns are [x_px, y_px]
    """
    x_px = rng.uniform(0.0, float(W), size=N)
    y_px = rng.uniform(0.0, float(H), size=N)
    return np.stack([x_px, y_px], axis=1)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Render synthetic microscopy frames with randomly-placed particles "
            "(DeepTrack-style baseline, no LAMMPS trajectory required)."
        )
    )
    parser.add_argument(
        "--frames",
        type=int,
        required=True,
        help="Number of frames to generate.",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config YAML (default: config.yaml).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--output-dir",
        default="verification_output/random_placement_frames/",
        help=(
            "Output directory for frame PNGs "
            "(default: verification_output/random_placement_frames/)."
        ),
    )
    args = parser.parse_args()

    cfg = _load_config(args.config).get("synthetic", {})

    # Read n_particles from config — required; raise a clear error if absent.
    rp_cfg = cfg.get("random_placement", {})
    if "n_particles" not in rp_cfg:
        raise KeyError(
            "Missing required config key 'synthetic.random_placement.n_particles'. "
            f"Add it to {args.config} under the 'synthetic:' section, e.g.:\n"
            "  synthetic:\n"
            "    random_placement:\n"
            "      n_particles: 1446"
        )
    n_particles = rp_cfg["n_particles"]

    strategy = cfg.get("render_strategy", "procedural")
    W = cfg["image_width"]
    H = cfg["image_height"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # _stretch_to_uint8 uses a fixed per-run peak_intensity/background_fraction
    # reference (same dict as passed to _dispatch_render) so the stretch is
    # consistent across all frames — same pattern as render.py's stretch_cfg.
    stretch_cfg = cfg

    rng = np.random.default_rng(args.seed)

    # Cross-frame state dict for brightfield/brightfield_fast — keeps each
    # particle's rendered appearance stable across frames (same pattern as
    # render.py's main()).
    state = {} if strategy in ("brightfield", "brightfield_fast") else None

    # "Unit LJ box = pixel space" trick: set box to (0, W, 0, H) so that
    # _dispatch_render's internal LJ→pixel rescaling is a no-op:
    #   pixel_x = (x_lj - x_lo) / (x_hi - x_lo) * W
    #           = (x_px - 0) / (W - 0) * W = x_px  ✓
    box = (0.0, float(W), 0.0, float(H))

    # Assign stable particle IDs once — same convention as render.py's LAMMPS
    # atom IDs (1-indexed). IDs are constant across all frames.
    atom_ids = np.arange(1, n_particles + 1, dtype=np.int64)

    ground_truth = []
    track_rows = []

    print("Random-placement render")
    print(f"Image size:      {W}×{H} px")
    print(f"Particles/frame: {n_particles}")
    print(f"Render strategy: {strategy}")
    print(f"Frames:          {args.frames}")
    print(f"Seed:            {args.seed}")
    print(f"Output:          {output_dir}")

    for i in range(args.frames):
        positions_px = generate_random_positions(W, H, n_particles, rng)

        img = _dispatch_render(
            positions_px,
            box,
            cfg,
            rng,
            strategy,
            state=state,
            atom_ids=atom_ids,
        )

        img8 = _stretch_to_uint8(img, stretch_cfg)
        png_path = output_dir / f"frame_{i:05d}.png"
        # vmin/vmax=0/255 are required — without them imsave re-normalises
        # img8 against this frame's own min/max, defeating _stretch_to_uint8's
        # fixed-reference scaling and reintroducing frame-to-frame drift.
        # See AGENTS.md's matplotlib footgun note.
        mplimg.imsave(str(png_path), img8, cmap="gray", vmin=0, vmax=255)

        clipped_positions_px = _lj_to_pixels(positions_px, box, H, W)

        ground_truth.append(
            {
                "frame": i,
                "n_particles": n_particles,
                "positions": clipped_positions_px.tolist(),
            }
        )

        for atom_id, (px, py) in zip(atom_ids, clipped_positions_px):
            track_rows.append(
                {
                    "frame": i,
                    "particle_id": int(atom_id),
                    "x": float(px),
                    "y": float(py),
                }
            )

        if i == 0 or (i + 1) % 10 == 0:
            print(f"  frame {i:4d}: {n_particles:4d} particles")

    # --- Output files ---
    gt_path = output_dir.parent / "ground_truth.json"
    with open(gt_path, "w") as f:
        json.dump(ground_truth, f)

    tracks_path = output_dir.parent / "ground_truth_tracks.csv"
    with open(tracks_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "particle_id", "x", "y"])
        writer.writeheader()
        writer.writerows(track_rows)

    # Alongside ground_truth.json, not inside output_dir -- keeps this
    # separate from benchmark.py's own benchmark_manifest_{model_type}.json
    # written later into the same experiment root. See run_provenance.py.
    write_manifest(
        output_dir.parent,
        "render_manifest.json",
        script="render_random_placement.py",
        cli_args=dict(vars(args)),
        resolved_params={
            "placement": "i.i.d. uniform",
            "n_particles": n_particles,
            "render_strategy": strategy,
            "seed": args.seed,
            "frames": args.frames,
            **cfg,
        },
        repo_root=Path(__file__).resolve().parent.parent,
    )

    print(f"\nRendered {len(ground_truth)} frames → {output_dir}")
    print(f"Ground truth  → {gt_path}")
    print(f"Tracking GT   → {tracks_path}")


if __name__ == "__main__":
    main()
