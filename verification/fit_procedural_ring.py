#!/usr/bin/env python3
"""Fit crop_source: procedural's ring parameters from crop_source: real's
harvested empirical template library.

Originally planned to fit against crop_source: physics's own PSF kernel
instead (avoiding any real-data dependency) -- abandoned after measuring
that this config's physics kernel has no meaningful secondary ring to fit
(the ring-model fit converges to a near-zero-amplitude ring far outside any
practical crop radius, consistent with the 2026-07-21 intensity-
normalization work's own finding that this kernel is "broad and smooth,"
not ring-structured). crop_source: real's harvested crops do show a
genuine measured ring, so this script fits against those instead. See
docs/plans/2026-07-22-001-fix-procedural-particle-realism-plan.md.

Usage:
    uv run python fit_procedural_ring.py [--config config.yaml] \
        [--cache-path crop_templates.npz] [--merge-config config.yaml]

Requires a template library already built via
render_crop_templates.build_template_library() (the same precondition
crop_source: real has at render time). Prints the fitted
ring_B/ring_A0/ring_s0/ring_A1/ring_r1/ring_s1 parameters as a YAML
fragment ready to paste into config.yaml's synthetic.procedural_shape, or
writes them directly with --merge-config.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

import calibrate_psf
from render_crop_templates import fit_ring_model, load_template_library, radial_profile_from_crops

RING_PARAM_NAMES = ("ring_B", "ring_A0", "ring_s0", "ring_A1", "ring_r1", "ring_s1")


def fit_procedural_ring_params(templates: np.ndarray, n_bins: int = 60) -> tuple:
    """Fit `_ring_model` against a harvested template library's radial
    profile. Returns the (B, A0, s0, A1, r1, s1) tuple.

    Raises ValueError if the profile's minimum sits at the outermost bin
    (the dark ring is likely truncated by the templates' own stored size,
    not actually captured). Raises RuntimeError if `curve_fit` fails to
    converge. Callers should surface these as a clear, actionable error
    rather than writing partial or invalid parameters.
    """
    radii, profile = radial_profile_from_crops(list(templates), n_bins=n_bins)

    if int(np.argmin(profile)) == len(profile) - 1:
        raise ValueError(
            f"Radial profile's minimum is at the outermost bin (r={radii[-1]:.1f}px) -- "
            "the dark ring may extend past the template library's own stored size "
            "(synthetic.crop_template.target_half). Retry with a larger target_half and "
            "rebuild the template library."
        )

    return fit_ring_model(radii, profile)


def main():
    parser = argparse.ArgumentParser(
        description="Fit procedural-shape ring parameters from crop_source: real's "
        "harvested template library"
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--cache-path", default=None)
    parser.add_argument("--n-bins", type=int, default=60)
    parser.add_argument("--merge-config", default=None, metavar="PATH")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: --config file does not exist: {config_path}", file=sys.stderr)
        sys.exit(1)

    full_cfg = yaml.safe_load(config_path.read_text()) or {}
    crop_template_cfg = full_cfg.get("synthetic", {}).get("crop_template", {})
    cache_path = args.cache_path or crop_template_cfg.get("cache_path")
    if not cache_path:
        print(
            "ERROR: no --cache-path given and synthetic.crop_template.cache_path is unset "
            f"in {config_path}.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        templates = load_template_library(cache_path)
    except FileNotFoundError:
        print(
            f"ERROR: no pre-built template library at '{cache_path}'. Build one via "
            "render_crop_templates.build_template_library() first (same precondition "
            "crop_source: real has at render time).",
            file=sys.stderr,
        )
        sys.exit(1)

    if len(templates) == 0:
        print(f"ERROR: template library at '{cache_path}' is empty.", file=sys.stderr)
        sys.exit(1)

    try:
        params = fit_procedural_ring_params(templates, n_bins=args.n_bins)
    except (ValueError, RuntimeError) as exc:
        print(f"ERROR: ring fit failed: {exc}", file=sys.stderr)
        sys.exit(1)

    # Round like calibrate_psf.py's own output (calibrate_psf.py:196-211):
    # amplitude-scale terms (B, A0, A1) to 4 decimals, pixel-scale terms
    # (s0, r1, s1) to 2 -- curve_fit's raw float64 precision isn't
    # meaningful past that, and config.yaml stays readable.
    rounded = [
        round(v, 4) if name in ("ring_B", "ring_A0", "ring_A1") else round(v, 2)
        for name, v in zip(RING_PARAM_NAMES, params)
    ]
    ring_params = dict(zip(RING_PARAM_NAMES, rounded))
    fragment = yaml.dump(
        {"procedural_shape": ring_params}, default_flow_style=False, sort_keys=False
    )
    yaml.safe_load(fragment)  # verify valid YAML before output

    if args.merge_config:
        calibrate_psf._merge_params_into_config(
            Path(args.merge_config), {"procedural_shape": ring_params}
        )
        print(f"Fitted ring parameters merged into: {args.merge_config}")
    else:
        print(fragment)


if __name__ == "__main__":
    main()
