#!/usr/bin/env python3
"""Fit crop_source: procedural's ring parameters from the physics crop_source's
own PSF kernel radial profile.

Usage:
    uv run python fit_procedural_ring.py [--config config.yaml] [--radius 80] \
        [--merge-config config.yaml]

Prints the fitted ring_B/ring_A0/ring_s0/ring_A1/ring_r1/ring_s1 parameters
as a YAML fragment ready to paste into config.yaml's synthetic.procedural_shape,
or writes them directly with --merge-config. See docs/plans/2026-07-22-001-
fix-procedural-particle-realism-plan.md.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

import calibrate_psf
from render_crop_templates import fit_ring_model, radial_profile_from_crops

RING_PARAM_NAMES = ("ring_B", "ring_A0", "ring_s0", "ring_A1", "ring_r1", "ring_s1")


def _build_wide_psf_kernel(psf_cfg: dict, radius: int) -> np.ndarray:
    """Build a physics PSF kernel cropped to `radius`, independent of
    render_deeptrack._build_psf_kernel's render-time r<=32px cap. That cap
    is a per-frame memory/performance bound and stays untouched (see the
    plan's Scope Boundaries) -- this function exists purely so the one-time
    ring fit below can see the whole ring (measured around r~50-58px at this
    module's default optics) rather than a render-time-truncated kernel.
    Deliberately duplicates render_deeptrack._build_psf_kernel's DeepTrack
    setup rather than sharing it, so the render-time path is never at risk
    of picking up this function's larger crop by accident.

    Raises ImportError if deeptrack isn't installed, ValueError if the
    resulting kernel is all-zero (optics parameters place the PSF entirely
    outside this crop).
    """
    try:
        import deeptrack  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "DeepTrack2 is required to fit procedural ring parameters. "
            "Run: uv add deeptrack==2.0.1  (inside verification/)"
        ) from exc

    na = float(psf_cfg.get("na", 1.4))
    wavelength = float(psf_cfg.get("wavelength", 520e-9))
    resolution = float(psf_cfg.get("resolution", 65e-9))

    size = 2 * radius + 1
    optics = deeptrack.Fluorescence(
        NA=na,
        wavelength=wavelength,
        resolution=resolution,
        output_region=(0, 0, size, size),
    )
    probe = deeptrack.PointParticle(position=(size // 2, size // 2), intensity=1.0, z=0)
    kernel = optics(probe).update().resolve()
    kernel = np.abs(np.array(kernel, dtype=np.float64).squeeze())

    if kernel.sum() == 0.0:
        raise ValueError(
            f"PSF kernel is all-zero at radius={radius} with na={na}, "
            f"wavelength={wavelength}, resolution={resolution} -- the optics "
            "parameters may place the whole PSF outside this crop."
        )

    peak = kernel.max()
    if peak > 0:
        kernel = kernel / peak
    return kernel.astype(np.float32)


def fit_procedural_ring_params(psf_cfg: dict, radius: int = 80, n_bins: int = 60) -> tuple:
    """Fit `_ring_model` against the physics PSF kernel's own radial
    profile. Returns the (B, A0, s0, A1, r1, s1) tuple.

    Raises ValueError if the kernel is all-zero, or if the profile's minimum
    sits at the outermost bin (the dark ring is likely truncated by
    `radius`, not actually captured -- retry with a larger radius). Raises
    RuntimeError if `curve_fit` fails to converge. Callers should surface
    these as a clear, actionable error rather than writing partial or
    invalid parameters.
    """
    kernel = _build_wide_psf_kernel(psf_cfg, radius)
    radii, profile = radial_profile_from_crops([kernel], n_bins=n_bins)

    if int(np.argmin(profile)) == len(profile) - 1:
        raise ValueError(
            f"Radial profile's minimum is at the outermost bin (r={radii[-1]:.1f}px) -- "
            f"the dark ring may extend past radius={radius}px. Retry with a larger --radius."
        )

    return fit_ring_model(radii, profile)


def main():
    parser = argparse.ArgumentParser(
        description="Fit procedural-shape ring parameters from the physics PSF kernel"
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--radius", type=int, default=80)
    parser.add_argument("--n-bins", type=int, default=60)
    parser.add_argument("--merge-config", default=None, metavar="PATH")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: --config file does not exist: {config_path}", file=sys.stderr)
        sys.exit(1)

    full_cfg = yaml.safe_load(config_path.read_text()) or {}
    psf_cfg = full_cfg.get("synthetic", {}).get("psf", {})

    try:
        params = fit_procedural_ring_params(psf_cfg, radius=args.radius, n_bins=args.n_bins)
    except (ImportError, ValueError, RuntimeError) as exc:
        print(f"ERROR: ring fit failed: {exc}", file=sys.stderr)
        sys.exit(1)

    ring_params = dict(zip(RING_PARAM_NAMES, params))
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
