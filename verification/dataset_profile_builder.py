#!/usr/bin/env python3
"""Build a dataset-profile YAML from a LAMMPS trajectory and a known size_px.

A dataset profile (see dataset-profiles/README.md) stores two calibrated
per-dataset pixel values -- `size_px` and `spacing_px` -- that
detectors-common's and trackers-common's scale derivation read. `size_px`
is NOT computed here: it comes from an already-calibrated/configured PSF
value (e.g. `synthetic.psf_sigma` in verification/config.yaml), passed in
as --size-px. `spacing_px` IS computed here, from the trajectory's own
particle positions.

Method (must not drift from this without re-validating against the
documented ~10.9px reference -- see AGENTS.md and verification/config.yaml):
each particle's position is converted from LAMMPS LJ units into the same
pixel space render.py stamps particles into (via render.py's own
`_lj_to_pixels`, reusing its box/atom parsing rather than a new parser),
then each particle's nearest OTHER particle is found via
`scipy.spatial.cKDTree.query(..., k=2)` -- k=1 would just match every point
to itself at distance 0 -- and `spacing_px` is the MEDIAN of those
per-particle nearest-neighbor distances (matching AGENTS.md's own "median
nearest-neighbor spacing" language, not the mean). A
sqrt(box_area / particle_count) density-proxy formula was considered and
rejected: verified directly against this repo's reference trajectory
(lammps-scripts/single_continuous_force_test/continuous_force_1500_5.0.lammpstrj)
that it overestimates the true median nearest-neighbor spacing by ~24%.

Usage:
    uv run python dataset_profile_builder.py --lammps <path/to/trajectory.lammpstrj> \\
        --size-px 5.0 --output ../dataset-profiles/synthetic-default.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial import cKDTree

# render.py lives alongside this script (verification/); reuse its own
# LAMMPS-dump/box/atom parsing and LJ-to-pixel conversion rather than writing
# a second parser. render.py's own module-level code already puts
# lammps-scripts/ on sys.path for lammps_parser, so importing it also makes
# parse_lammps_dump reachable.
sys.path.insert(0, str(Path(__file__).parent))
from render import _lj_to_pixels, _parse_atoms, _parse_box, parse_lammps_dump  # noqa: E402


def compute_spacing_px(lammps_path, image_width=512, image_height=512, frame_index=0):
    """Median nearest-neighbor spacing (px) among a LAMMPS trajectory's own
    particles, in the same pixel space render.py stamps particles into.

    Reads `frame_index` (default: the trajectory's first frame -- particle
    spacing at t=0 before any dynamics-driven clustering/dispersal is the
    dataset's own designed density, matching how psf_sigma/spacing are
    otherwise treated as fixed per-dataset constants elsewhere in this
    repo). Raises ValueError if the trajectory has fewer than `frame_index +
    1` frames, or fewer than 2 particles in that frame (a nearest-neighbor
    distance is undefined for 0 or 1 particles).
    """
    block = None
    for i, candidate in enumerate(parse_lammps_dump(str(lammps_path))):
        if i == frame_index:
            block = candidate
            break
    if block is None:
        raise ValueError(
            f"LAMMPS trajectory {lammps_path} has fewer than {frame_index + 1} frame(s)"
        )

    box = _parse_box(block["box_bounds"])
    positions_lj, _atom_ids = _parse_atoms(block["atom_header"], block["atoms"])
    if len(positions_lj) < 2:
        raise ValueError(
            f"LAMMPS trajectory {lammps_path} frame {frame_index} has "
            f"{len(positions_lj)} particle(s); need at least 2 to compute a "
            "nearest-neighbor spacing"
        )

    positions_px = _lj_to_pixels(positions_lj, box, image_height, image_width)
    tree = cKDTree(positions_px)
    dists, _indices = tree.query(positions_px, k=2)
    nearest_neighbor_dists = dists[:, 1]
    return float(np.median(nearest_neighbor_dists))


def build_dataset_profile(
    lammps_path, size_px, image_width=512, image_height=512, frame_index=0, description=None
):
    """Build a dataset-profile dict: `size_px` taken as-is (see module
    docstring), `spacing_px` computed via `compute_spacing_px`.
    """
    if not isinstance(size_px, (int, float)) or isinstance(size_px, bool) or size_px <= 0:
        raise ValueError(f"size_px must be a positive number, got {size_px!r}")

    spacing_px = compute_spacing_px(
        lammps_path, image_width=image_width, image_height=image_height, frame_index=frame_index
    )
    profile = {"size_px": float(size_px), "spacing_px": round(spacing_px, 4)}
    if description:
        profile["description"] = description
    return profile


def write_dataset_profile(profile, output_path):
    """Write `profile` to `output_path` as plain YAML, creating parent
    directories if needed."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.safe_dump(profile, f, default_flow_style=False, sort_keys=False)


def main():
    parser = argparse.ArgumentParser(
        description="Build a dataset-profile YAML from a LAMMPS trajectory and a known size_px"
    )
    parser.add_argument("--lammps", required=True, help="Path to a .lammpstrj trajectory file")
    parser.add_argument(
        "--size-px",
        type=float,
        required=True,
        help="Known/calibrated particle size in pixels (e.g. synthetic.psf_sigma); "
        "not computed by this script",
    )
    parser.add_argument("--output", required=True, help="Output path for the profile YAML")
    parser.add_argument("--image-width", type=int, default=512)
    parser.add_argument("--image-height", type=int, default=512)
    parser.add_argument(
        "--frame-index", type=int, default=0, help="Which trajectory frame to measure (default: 0)"
    )
    parser.add_argument("--description", default=None, help="Optional free-text description")
    args = parser.parse_args()

    lammps_path = Path(args.lammps)
    if not lammps_path.exists():
        print(f"ERROR: --lammps file does not exist: {lammps_path}", file=sys.stderr)
        sys.exit(1)

    profile = build_dataset_profile(
        lammps_path,
        args.size_px,
        image_width=args.image_width,
        image_height=args.image_height,
        frame_index=args.frame_index,
        description=args.description,
    )
    write_dataset_profile(profile, args.output)
    print(f"Dataset profile written to: {args.output}")
    print(f"  size_px:    {profile['size_px']}")
    print(f"  spacing_px: {profile['spacing_px']:.4f}")


if __name__ == "__main__":
    main()
