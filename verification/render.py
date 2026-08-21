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
    brightfield — coherent whole-frame optical-field solve via DeepTrack2's
                  Brightfield optics; particles placed at real trajectory
                  positions, small-batch/reference-quality (see
                  render_brightfield.py)
    brightfield_fast — same coherent optics, reimplemented directly in
                  numpy/scipy (no deeptrack dependency) so cost is
                  independent of particle count; validated against
                  brightfield as a fast, production-density-capable
                  equivalent (see render_brightfield_fast.py)
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
from scipy.special import erf

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


def _gaussian_ring_profile(
    r_grid, sigma, ring_radius_factor=1.0, ring_width_factor=0.3, ring_depth=0.65
):
    """Core-minus-ring difference-of-Gaussians profile, peak-normalized
    (may dip below 0 near the ring edge; caller clips before Poisson noise).
    The caller multiplies by peak_intensity and clips to non-negative --
    this function does neither.

    Same math render_frame has always used inline: a bright Gaussian core
    with a dark ring subtracted at ring_radius_factor*sigma. r_grid must be
    a 2D array of Euclidean distances from the particle center (never x/y
    offsets independently) -- the ring term is not separable into an outer
    product, or it produces a non-isotropic diamond-shaped artifact instead
    of a circular ring.
    """
    ring_width = ring_width_factor * sigma
    core = np.exp(-0.5 * (r_grid / sigma) ** 2)
    if ring_depth > 0 and ring_width > 0:
        ring = ring_depth * np.exp(-0.5 * ((r_grid - ring_radius_factor * sigma) / ring_width) ** 2)
    else:
        ring = 0.0
    return core - ring


def _gaussian_ring_extent(sigma, ring_radius_factor=1.0, ring_width_factor=0.3, ring_depth=0.65):
    """Pixel ROI radius needed to contain the core and the ring's outer tail."""
    ring_width = ring_width_factor * sigma
    core_extent = 3 * sigma
    ring_extent = ring_radius_factor * sigma + 3 * ring_width
    return int(max(core_extent, ring_extent)) + 1


def _disk_rim_profile(
    r_grid,
    disk_radius_px,
    blur_sigma_px,
    rim_depth=0.0,
    rim_width_px=1.0,
    rim_offset_px=0.0,
):
    """Flat-top disk (smoothed step) with an optional dark rim near its edge,
    peak-normalized (may dip below 0 near the rim edge; caller clips before
    Poisson noise). The caller multiplies by peak_intensity and clips to
    non-negative -- this function does neither.

    Two disks that are merely touching have ~zero geometric overlap, unlike
    two Gaussian cores of comparable width -- summing two of these under
    plain additive compositing does not overshoot the way two overlapping
    Gaussian tails do. The rim gives touching particles a visible seam
    rather than a flat continuous plateau. See
    docs/superpowers/specs/2026-07-23-particle-render-profiles-design.md.
    """
    flat_top = 0.5 * (1 - erf((r_grid - disk_radius_px) / (np.sqrt(2) * blur_sigma_px)))
    if rim_depth > 0 and rim_width_px > 0:
        rim_radius = disk_radius_px - rim_offset_px
        rim = rim_depth * np.exp(-0.5 * ((r_grid - rim_radius) / rim_width_px) ** 2)
    else:
        rim = 0.0
    return flat_top - rim


def _disk_rim_extent(
    disk_radius_px, blur_sigma_px, rim_depth=0.0, rim_width_px=1.0, rim_offset_px=0.0
):
    """Pixel ROI radius needed to contain the disk and its blurred edge.

    rim_depth/rim_width_px/rim_offset_px are accepted (not just
    disk_radius_px/blur_sigma_px) so every profile type's extent function
    has the same call signature as its params dict -- the rim never needs a
    larger ROI than the disk-plus-blur margin alone, since rim_offset_px is
    subtracted from disk_radius_px, not added.
    """
    return int(disk_radius_px + 4 * blur_sigma_px) + 1


_PARTICLE_PROFILES = {
    "disk_rim": (_disk_rim_profile, _disk_rim_extent),
    "gaussian_ring": (_gaussian_ring_profile, _gaussian_ring_extent),
}


def _assign_particle_profiles(atom_ids, profiles_cfg, default_seed=42):
    """Weighted-random, seeded, persistent-for-the-run assignment of a named
    profile to each particle, keyed by atom_id.

    Never reads a LAMMPS atom-type column -- this function's inputs are
    atom_ids and profiles_cfg only, so it produces the same kind of
    proportion-respecting split whether the trajectory has one LAMMPS atom
    type or many.

    Args:
        atom_ids: (N,) array of atom IDs, typically from the first parsed
            frame. Safe to use only frame 0's IDs because render.py's
            main() already asserts atom IDs are stable across the whole
            trajectory before writing tracking output.
        profiles_cfg: synthetic.particle_render_profiles config dict, with a
            "profiles" list of {"name": str, "proportion": float, ...}
            dicts. "proportion" values are normalized by their sum -- they
            are not required to total 1.
        default_seed: used when profiles_cfg has no "seed" key.

    Returns:
        dict mapping int(atom_id) -> profile name (str).
    """
    rng = np.random.default_rng(profiles_cfg.get("seed", default_seed))
    profiles = profiles_cfg["profiles"]
    names = [p["name"] for p in profiles]
    proportions = np.array([p["proportion"] for p in profiles], dtype=np.float64)
    proportions = proportions / proportions.sum()
    choices = rng.choice(names, size=len(atom_ids), p=proportions)
    return {int(aid): name for aid, name in zip(atom_ids, choices)}


def render_frame(positions_lj, box, cfg, rng, atom_ids=None, profile_map=None):
    """Render one synthetic microscopy frame.

    Args:
        positions_lj: (N, 2) array of particle positions in LJ units
        box: (x_lo, x_hi, y_lo, y_hi) simulation box bounds
        cfg: synthetic config dict
        rng: numpy random Generator
        atom_ids: optional (N,) array of atom IDs, parallel to
            positions_lj. Required together with profile_map -- omitting
            atom_ids while passing profile_map raises TypeError from the
            zip() below, not a bespoke validation error.
        profile_map: optional dict of atom_id -> profile name, from
            _assign_particle_profiles. When None (the default), every
            particle renders with the single gaussian_ring shape from
            cfg["psf_sigma"]/cfg["ring"] -- unchanged from before this
            feature existed. When given, each particle's shape and ROI
            extent come from cfg["particle_render_profiles"]["profiles"]
            (looked up by name via profile_map[atom_id]) through
            _PARTICLE_PROFILES[profile["type"]], using that profile's own
            "params".

    Returns:
        uint16 numpy array of shape (H, W)
    """
    H = cfg["image_height"]
    W = cfg["image_width"]
    peak = cfg["peak_intensity"]
    x_lo, x_hi, y_lo, y_hi = box
    background_level = peak * cfg.get("background_fraction", 0.25)
    img = np.full((H, W), background_level, dtype=np.float64)
    # Per-pixel signed deviation from background_level, one layer shared by
    # every particle -- NOT accumulated with `+=`. Plain additive compositing
    # sums every overlapping particle's contribution, so two overlapping
    # bright cores pile up into a single brighter blob with no visible
    # boundary between them. Real opaque/reflective particles don't add
    # brightness where they overlap -- whichever one is locally most
    # prominent (furthest from background, in either direction) is what's
    # visible there, so each stamp competes on |deviation| and the
    # larger-magnitude one wins that pixel outright. See
    # docs/superpowers/specs/2026-08-07-no-blob-merging-design.md.
    deviation = np.zeros((H, W), dtype=np.float64)

    def _stamp(cx, cy, extent, intensity):
        x0, x1 = max(0, int(cx) - extent), min(W, int(cx) + extent + 1)
        y0, y1 = max(0, int(cy) - extent), min(H, int(cy) + extent + 1)
        if x0 >= x1 or y0 >= y1:
            return
        xs = np.arange(x0, x1, dtype=np.float64)
        ys = np.arange(y0, y1, dtype=np.float64)
        X, Y = np.meshgrid(xs, ys)
        r_grid = np.hypot(X - cx, Y - cy)
        contribution = intensity(r_grid)
        region = deviation[y0:y1, x0:x1]
        winner = np.abs(contribution) > np.abs(region)
        region[winner] = contribution[winner]

    if profile_map is None:
        sigma = cfg["psf_sigma"]
        ring_cfg = cfg.get("ring", {})
        ring_radius_factor = ring_cfg.get("radius_factor", 1.0)
        ring_width_factor = ring_cfg.get("width_factor", 0.3)
        ring_depth = ring_cfg.get("depth", 0.65)
        extent = _gaussian_ring_extent(sigma, ring_radius_factor, ring_width_factor, ring_depth)

        for x, y in positions_lj:
            cx = (x - x_lo) / (x_hi - x_lo) * W
            cy = (y - y_lo) / (y_hi - y_lo) * H
            _stamp(
                cx,
                cy,
                extent,
                lambda r_grid: peak
                * _gaussian_ring_profile(
                    r_grid, sigma, ring_radius_factor, ring_width_factor, ring_depth
                ),
            )
    else:
        profiles_by_name = {p["name"]: p for p in cfg["particle_render_profiles"]["profiles"]}
        for (x, y), atom_id in zip(positions_lj, atom_ids):
            profile = profiles_by_name[profile_map[int(atom_id)]]
            intensity_fn, extent_fn = _PARTICLE_PROFILES[profile["type"]]
            params = profile.get("params", {})
            cx = (x - x_lo) / (x_hi - x_lo) * W
            cy = (y - y_lo) / (y_hi - y_lo) * H
            _stamp(
                cx,
                cy,
                extent_fn(**params),
                lambda r_grid, fn=intensity_fn, p=params: peak * fn(r_grid, **p),
            )

    img += deviation

    # The ring/rim's negative dip can push some pixels below zero; rng.poisson
    # raises ValueError on negative input, so this clip must run before the
    # shot-noise branch below (not just at the function's final clip).
    img = np.clip(img, 0, None)

    if cfg.get("shot_noise", True):
        img = rng.poisson(img).astype(np.float64)
    img += rng.normal(0.0, cfg.get("readout_noise", 200.0), img.shape)
    return np.clip(img, 0, 65535).astype(np.uint16)


def _lj_to_pixels(positions_lj, box, H, W):
    """Convert (N, 2) LJ positions to pixel coordinates, clipping to image boundary."""
    x_lo, x_hi, y_lo, y_hi = box
    px = np.clip((positions_lj[:, 0] - x_lo) / (x_hi - x_lo) * W, 0, W - 1)
    py = np.clip((positions_lj[:, 1] - y_lo) / (y_hi - y_lo) * H, 0, H - 1)
    return np.stack([px, py], axis=1)


# Public (no leading underscore): imported by benchmark.py's lodestar box_size
# derivation, not just used internally here.
FWHM_TO_SIGMA = 2.355  # FWHM = 2*sqrt(2*ln2)*sigma ~= 2.355*sigma


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
            number can be found, or if the parsed diameter is not a
            positive, finite number -- a zero/negative/NaN diameter would
            otherwise propagate into a zero/negative psf_sigma and crash
            rng.poisson downstream with a much less legible error.
    """
    path = Path(lammps_in_path)
    if not path.is_file():
        raise FileNotFoundError(f"LAMMPS input script not found: {lammps_in_path}")

    text = path.read_text()
    lines = [line.split("#", 1)[0].strip() for line in text.splitlines()]

    def _require_positive(value, source_desc):
        if not (value > 0) or not np.isfinite(value):
            raise ValueError(
                f"{source_desc} in {lammps_in_path} yields a non-positive or "
                f"non-finite diameter ({value!r}) -- cannot derive a particle size from it."
            )
        return value

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
            return _require_positive(2.0 * sx, f"'set type ... shape' line ('{line}')")

    sigma_re = re.compile(r"^variable\s+sigma\s+equal\s+(\S+)")
    for line in lines:
        m = sigma_re.match(line)
        if m:
            try:
                sigma_value = float(m.group(1))
            except ValueError as exc:
                raise ValueError(
                    f"Could not parse numeric sigma value from '{line}' in "
                    f"{lammps_in_path} (likely an unresolved LAMMPS variable "
                    "reference, e.g. '${sigma}' — a literal number is required)"
                ) from exc
            return _require_positive(sigma_value, f"'variable sigma' line ('{line}')")

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
    return diameter_px / FWHM_TO_SIGMA


def _dispatch_render(
    positions_lj, box, cfg, rng, strategy, state=None, atom_ids=None, profile_map=None
):
    """Dispatch to the appropriate render function based on strategy.

    Args:
        strategy: 'procedural' | 'brightfield' | 'brightfield_fast'
        state: optional dict carrying cross-frame state. For
            'brightfield'/'brightfield_fast', a per-atom_id cache of
            sampled radius/refractive_index/z so each physical particle's
            properties stay constant across frames instead of flickering
            (see render_brightfield._sample_particle_properties). Ignored
            entirely by 'procedural' — its signature/call is unchanged.
        atom_ids: optional (N,) array of atom IDs, parallel to positions_lj.
            Passed through to the 'procedural' branch's render_frame (for
            particle_render_profiles lookup) and to 'brightfield'/
            'brightfield_fast' (for the per-particle state cache above).
        profile_map: optional dict of atom_id -> profile name from
            _assign_particle_profiles. Passed through only to the
            'procedural' branch.

    Returns:
        uint16 numpy array of shape (H, W)
    """
    if strategy == "brightfield":
        try:
            from render_brightfield import render_frame_brightfield

            return render_frame_brightfield(
                positions_lj, box, cfg, rng, atom_ids=atom_ids, state=state
            )
        except ImportError:
            raise ImportError(
                "Brightfield rendering requires 'deeptrack==2.0.1'. "
                "Run 'uv add deeptrack==2.0.1' inside verification/. "
            )
    elif strategy == "brightfield_fast":
        try:
            from render_brightfield_fast import render_frame_brightfield_fast

            return render_frame_brightfield_fast(
                positions_lj, box, cfg, rng, atom_ids=atom_ids, state=state
            )
        except ImportError:
            raise ImportError(
                "brightfield_fast rendering requires render_brightfield_fast.py. "
                "Ensure the file exists in the verification/ directory."
            )
    else:
        # Default: procedural
        return render_frame(positions_lj, box, cfg, rng, atom_ids=atom_ids, profile_map=profile_map)


def _stretch_to_uint8(img, cfg):
    """Convert a rendered uint16 frame to an 8-bit PNG-ready array.

    lo/hi are fixed per-run, not recomputed per frame from that frame's own
    observed min/max. A genuinely per-frame stretch makes a constant-sized
    particle look like it's pulsing/breathing in a video: shot noise and
    frame-to-frame overlap shift each frame's own min/max slightly, which
    shifts the effective display scale, which shifts the apparent radius
    where a Gaussian tail crosses the eye's visible threshold -- even though
    the underlying psf_sigma never changes. lo=0 matches render_frame's own
    floor (its output is always clipped to >= 0). hi is derived from
    peak_intensity/background_fraction when the strategy exposes them
    (procedural) -- the same fixed reference for every frame in the run,
    regardless of that frame's own noise or particle count/overlap --
    falling back to this frame's own max otherwise. See
    docs/superpowers/specs/2026-08-08-tight-ring-and-fixed-stretch-design.md.
    """
    img_f = img.astype(np.float32)
    lo = 0.0
    if "peak_intensity" in cfg:
        hi = cfg["peak_intensity"] * (1.0 + cfg.get("background_fraction", 0.25))
    else:
        hi = float(img_f.max())
    if hi <= lo:
        return np.zeros_like(img, dtype=np.uint8)
    return ((img_f - lo) / (hi - lo) * 255).clip(0, 255).astype(np.uint8)


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
            "an arbitrary constant. Only affects render_strategy: procedural -- "
            "brightfield/brightfield_fast derive their own particle size and "
            "never read this override; a warning is printed if combined with them."
        ),
    )
    parser.add_argument("--frames", type=int, default=None, help="Limit to first N timesteps")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    parser.add_argument("--video", action="store_true", help="Also encode frames into preview.mp4")
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Frame rate for --video output (default: 10)",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config).get("synthetic", {})
    strategy = cfg.get("render_strategy", "procedural")

    output_dir = Path(cfg.get("output_dir", "verification_output/synthetic_frames/"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # _stretch_to_uint8 needs one fixed peak_intensity/background_fraction
    # reference for the whole run (see its docstring on why per-frame
    # stretching causes apparent pulsing).
    stretch_cfg = cfg

    rng = np.random.default_rng(args.seed)
    # Cross-frame state — a small dict owned by this run, created once
    # alongside rng, and threaded through _dispatch_render the same way rng
    # already is. Deliberately not stuffed into cfg: carrying runtime state
    # through cfg would let a callee's private dict(cfg) copy silently drop
    # it. For 'brightfield'/'brightfield_fast' it holds the per-atom_id
    # particle-property cache that keeps each particle's rendered
    # appearance stable across frames instead of flickering (see
    # render_brightfield._sample_particle_properties). Other strategies
    # never see this — it stays None for them.
    state = {} if strategy in ("brightfield", "brightfield_fast") else None
    profile_map = None

    ground_truth = []
    # Collect per-frame data for tracks CSV: list of (atom_ids, px_positions)
    all_frame_ids = []
    track_rows = []

    print(f"Rendering from: {args.lammps}")
    print(f"Image size:     {cfg['image_width']}×{cfg['image_height']} px")
    if not args.lammps_in:
        print(f"PSF sigma:      {cfg.get('psf_sigma', cfg.get('psf', {}).get('sigma_px', 5.0))} px")
    elif strategy != "procedural":
        # brightfield/brightfield_fast derive particle appearance from their
        # own brightfield.na/wavelength/resolution config, never reading
        # cfg["psf_sigma"], so overriding it here would be a silent no-op.
        # Warn instead of letting --lammps-in's derived value (and the
        # "derived from --lammps-in" print below) misleadingly imply it's in
        # effect for this run's actual rendered output.
        print(
            f"WARNING:        --lammps-in has no effect on render_strategy: {strategy} -- "
            "only procedural reads the derived psf_sigma."
        )
    elif cfg.get("particle_render_profiles"):
        # Same reasoning as the strategy!=procedural branch above, but for
        # particle_render_profiles: each profile's own params (e.g.
        # disk_radius_px) already sets its size explicitly, so there's no
        # longer one unambiguous cfg["psf_sigma"] target to override.
        print(
            "WARNING:        --lammps-in has no effect when synthetic.particle_render_profiles "
            "is configured -- each profile's own params already set its size explicitly."
        )
    print(f"Render strategy: {strategy}")
    print(f"Output:         {output_dir}")

    for i, block in enumerate(parse_lammps_dump(args.lammps)):
        if args.frames is not None and i >= args.frames:
            break

        box = _parse_box(block["box_bounds"])

        if (
            args.lammps_in
            and i == 0
            and strategy == "procedural"
            and not cfg.get("particle_render_profiles")
        ):
            cfg["psf_sigma"] = _derive_psf_sigma_from_lammps_in(
                args.lammps_in, box, cfg["image_width"]
            )
            print(
                f"PSF sigma:      {cfg['psf_sigma']:.3f} px "
                f"(derived from --lammps-in {args.lammps_in})"
            )

        positions_lj, atom_ids = _parse_atoms(block["atom_header"], block["atoms"])

        if i == 0 and strategy == "procedural" and cfg.get("particle_render_profiles"):
            profile_map = _assign_particle_profiles(atom_ids, cfg["particle_render_profiles"])
            n_profiles = len(cfg["particle_render_profiles"]["profiles"])
            print(
                f"Particle profiles: {n_profiles} configured, {len(profile_map)} particles assigned "
                f"(seed={cfg['particle_render_profiles'].get('seed', 42)})"
            )

        img = _dispatch_render(
            positions_lj,
            box,
            cfg,
            rng,
            strategy,
            state=state,
            atom_ids=atom_ids,
            profile_map=profile_map,
        )

        img8 = _stretch_to_uint8(img, stretch_cfg)
        png_path = output_dir / f"frame_{i:05d}.png"
        # vmin/vmax=0/255 are required, not cosmetic: without them, imsave's
        # own colormap normalization re-stretches img8 a second time against
        # *this frame's own* observed min/max -- silently overriding
        # _stretch_to_uint8's fixed-reference computation and reintroducing
        # exactly the frame-to-frame drift that function exists to prevent.
        # Confirmed directly: two uint8 arrays sharing a pixel value of 128
        # but with different own-maxima (200 vs 255) round-tripped through
        # imsave(cmap="gray") with no vmin/vmax read back as 0 and 0 --
        # neither matching the literal value written -- until vmin=0/vmax=255
        # was added, after which both correctly read back as 128. See
        # docs/superpowers/specs/2026-08-08-tight-ring-and-fixed-stretch-
        # design.md.
        mplimg.imsave(str(png_path), img8, cmap="gray", vmin=0, vmax=255)

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
                {
                    "frame": i,
                    "particle_id": int(atom_id),
                    "x": float(px),
                    "y": float(py),
                }
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
