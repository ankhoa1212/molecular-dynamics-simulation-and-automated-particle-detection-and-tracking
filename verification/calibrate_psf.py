#!/usr/bin/env python3
"""Calibrate PSF, background, intensity, and noise params from real microscopy .tif frames.

Usage:
    uv run python calibrate_psf.py --real-frames <dir> [--output-config <path>] [--dark-frames <dir>]

Prints a YAML fragment ready to paste into config.yaml under synthetic:.
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import scipy.ndimage
import scipy.optimize
import scipy.stats
import tifffile
import yaml

_MIN_GOOD_FITS = 20
_DEFAULT_SIGMA = 5.0
_CROP_HALF = 16  # 32×32 crop


def _load_tifs(directory: Path) -> list[np.ndarray]:
    paths = sorted(directory.glob("*.tif")) + sorted(directory.glob("*.tiff"))
    return [tifffile.imread(str(p)).astype(np.float32) for p in paths]


def _detect_spots(frame: np.ndarray, min_sep: float) -> np.ndarray:
    """Return (N, 2) array of local-maxima (row, col) with min_sep separation."""
    fs = max(3, int(min_sep) | 1)  # odd footprint size
    local_max = scipy.ndimage.maximum_filter(frame, footprint=np.ones((fs, fs), dtype=bool))
    threshold = np.percentile(frame, 90)
    rows, cols = np.where((frame == local_max) & (frame > threshold))
    return np.column_stack([rows, cols]) if len(rows) > 0 else np.zeros((0, 2), dtype=int)


def _gaussian_2d(xy, A, x0, y0, sx, sy, B):
    x, y = xy
    return A * np.exp(-((x - x0) ** 2 / (2 * sx**2) + (y - y0) ** 2 / (2 * sy**2))) + B


def _fit_gaussian(crop: np.ndarray):
    H, W = crop.shape
    xs, ys = np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys)
    # Use crop maximum as initial position guess (more robust than assuming center)
    peak_idx = np.argmax(crop)
    y0_init = float(peak_idx // W)
    x0_init = float(peak_idx % W)
    try:
        popt, _ = scipy.optimize.curve_fit(
            _gaussian_2d,
            (xx.ravel(), yy.ravel()),
            crop.ravel(),
            p0=[
                crop.max() - crop.min(),
                x0_init,
                y0_init,
                _DEFAULT_SIGMA,
                _DEFAULT_SIGMA,
                crop.min(),
            ],
            bounds=([0, 0, 0, 0.5, 0.5, -np.inf], [np.inf, W, H, H, W, np.inf]),
            maxfev=2000,
        )
        sx, sy = abs(popt[3]), abs(popt[4])
        if 0.5 < sx < _CROP_HALF and 0.5 < sy < _CROP_HALF:
            return float(sx), float(sy)
    except (RuntimeError, ValueError):
        pass
    return None


def _heterogeneity_scale(residual: np.ndarray) -> float:
    """Estimate background correlation length (px) from radial PSD decay."""
    H, W = residual.shape
    psd = np.abs(np.fft.rfft2(residual)) ** 2
    fy, fx = np.fft.fftfreq(H), np.fft.rfftfreq(W)
    fxx, fyy = np.meshgrid(fx, fy)
    r = np.sqrt(fxx**2 + fyy**2).ravel()
    p = psd.ravel()
    bins = np.linspace(0, 0.5, 31)
    centers, means = [], []
    for i in range(len(bins) - 1):
        m = (r >= bins[i]) & (r < bins[i + 1])
        if m.any():
            centers.append((bins[i] + bins[i + 1]) / 2)
            means.append(np.mean(p[m]))
    if len(centers) < 3:
        return 50.0
    try:
        coeffs = np.polyfit(centers, np.log(np.maximum(means, 1e-10)), 1)
        return float(np.clip(-max(H, W) / coeffs[0], 1.0, max(H, W) // 2))
    except (np.linalg.LinAlgError, ValueError):
        return 50.0


def calibrate_from_frames(
    frames: list[np.ndarray], dark_frames: list[np.ndarray] | None = None
) -> dict:
    """Fit PSF and imaging parameters from a list of microscopy frames.

    Returns dict with keys: psf, particle, background, noise, _meta.
    """
    if not frames:
        raise ValueError("frames list is empty")

    psf_sigma = _DEFAULT_SIGMA
    sigma_ests, peak_vals, all_bg, bg_residuals = [], [], [], []

    for frame in frames:
        spots = _detect_spots(frame, 3 * psf_sigma)
        if not len(spots):
            continue

        disk_r = int(2 * psf_sigma) + 1
        disk_y, disk_x = np.ogrid[-disk_r : disk_r + 1, -disk_r : disk_r + 1]
        disk = disk_x**2 + disk_y**2 <= disk_r**2
        H, W = frame.shape
        mask = np.zeros((H, W), dtype=bool)

        for row, col in spots.astype(int):
            # Fit crop
            r0, r1 = row - _CROP_HALF, row + _CROP_HALF
            c0, c1 = col - _CROP_HALF, col + _CROP_HALF
            if r0 >= 0 and r1 <= H and c0 >= 0 and c1 <= W:
                fit = _fit_gaussian(frame[r0:r1, c0:c1])
                if fit is not None:
                    sigma_ests.append((fit[0] + fit[1]) / 2)
                    peak_vals.append(float(frame[r0:r1, c0:c1].max()))
                    psf_sigma = float(np.mean(sigma_ests))

            # Mask particle for background
            mr0, mr1 = max(0, row - disk_r), min(H, row + disk_r + 1)
            mc0, mc1 = max(0, col - disk_r), min(W, col + disk_r + 1)
            dy = slice(max(0, disk_r - row), disk_r + min(H - row, disk_r + 1))
            dx = slice(max(0, disk_r - col), disk_r + min(W - col, disk_r + 1))
            mask[mr0:mr1, mc0:mc1] |= disk[dy, dx]

        bg = frame[~mask]
        if len(bg):
            all_bg.append(bg)
            bg_residuals.append(np.where(mask, float(np.mean(bg)), frame) - np.mean(bg))

    n_fits = len(sigma_ests)
    if n_fits < _MIN_GOOD_FITS:
        warnings.warn(
            f"Only {n_fits} good particle fits found (< {_MIN_GOOD_FITS} recommended). "
            "PSF sigma estimate may be noisy."
        )

    fitted_sigma = float(np.mean(sigma_ests)) if sigma_ests else _DEFAULT_SIGMA

    if peak_vals:
        shape, _, scale = scipy.stats.lognorm.fit(np.array(peak_vals), floc=0)
        peak_mean = float(np.exp(np.log(scale) + shape**2 / 2))
        intensity_sigma = float(shape)
    else:
        peak_mean, intensity_sigma = 40000.0, 0.3

    bg_amplitude = float(np.std(np.concatenate(all_bg))) if all_bg else 500.0
    scale_px = _heterogeneity_scale(bg_residuals[0]) if bg_residuals else 50.0

    gain_note = ""
    if dark_frames and len(dark_frames) >= 2:
        dark_stack = np.stack(dark_frames, axis=0)
        per_var = np.var(dark_stack, axis=0)
        read_noise = float(np.sqrt(np.mean(per_var)))
        gain_sigma = float(np.std(np.sqrt(per_var)) / max(np.mean(np.sqrt(per_var)), 1e-6))
    else:
        if not dark_frames:
            warnings.warn("No --dark-frames provided. Estimating read_noise from image stats.")
        ref = frames[0]
        low = ref[ref < np.percentile(ref, 10)]
        read_noise = float(np.std(low)) if len(low) else 15.0
        gain_sigma = 0.02
        gain_note = "  # WARNING: estimated from image stats; provide --dark-frames for accuracy"

    return {
        "psf": {
            "sigma_px": round(fitted_sigma, 2),
            "defocus": 0.0,
            "spherical_aberration": 0.0,
            "resolution": 65.0e-9,
        },
        "particle": {
            "peak_mean": round(peak_mean, 1),
            "intensity_sigma": round(intensity_sigma, 3),
        },
        "background": {
            "heterogeneity_scale": round(scale_px, 1),
            "amplitude": round(bg_amplitude, 1),
        },
        "noise": {
            "read_noise": round(read_noise, 2),
            "gain_sigma": round(gain_sigma, 4),
            "_gain_sigma_note": gain_note,
        },
        "_meta": {"n_fits": n_fits, "n_frames": len(frames)},
    }


def _format_yaml_fragment(params: dict) -> str:
    meta = params["_meta"]
    gain_note = params["noise"]["_gain_sigma_note"]
    return "\n".join(
        [
            f"# Calibrated parameters (paste into config.yaml under synthetic:)",
            f"# Fitted from {meta['n_fits']} particle crops in {meta['n_frames']} frames",
            "psf:",
            f"  sigma_px: {params['psf']['sigma_px']}      # mean fitted PSF sigma in pixels",
            "  defocus: 0.0           # manual — use a defocused test image to fit",
            "  spherical_aberration: 0.0",
            f"  resolution: {params['psf']['resolution']:.1e}    # assumed — verify from microscope spec",
            "particle:",
            f"  peak_mean: {params['particle']['peak_mean']}",
            f"  intensity_sigma: {params['particle']['intensity_sigma']}",
            "background:",
            f"  heterogeneity_scale: {params['background']['heterogeneity_scale']}",
            f"  amplitude: {params['background']['amplitude']}",
            "noise:",
            f"  read_noise: {params['noise']['read_noise']}",
            f"  gain_sigma: {params['noise']['gain_sigma']}{gain_note}",
        ]
    )


def _merge_params_into_config(config_path: Path, params: dict) -> None:
    """Merge calibrated params into an existing config.yaml file under synthetic:.

    Preserves all existing keys not in the four calibrated sub-dicts (psf, particle,
    background, noise). Strips _gain_sigma_note from noise and drops _meta entirely.

    Raises FileNotFoundError if config_path does not exist.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open() as f:
        config = yaml.safe_load(f) or {}

    if "synthetic" not in config:
        config["synthetic"] = {}

    synthetic = config["synthetic"]

    for section in ("psf", "particle", "background", "noise"):
        calibrated = dict(params[section])
        if section == "noise":
            calibrated.pop("_gain_sigma_note", None)
        if section not in synthetic:
            synthetic[section] = {}
        synthetic[section].update(calibrated)

    with config_path.open("w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def main():
    parser = argparse.ArgumentParser(description="Calibrate PSF and noise from real .tif frames")
    parser.add_argument("--real-frames", required=True)
    parser.add_argument("--output-config", default=None)
    parser.add_argument("--merge-config", default=None, metavar="PATH")
    parser.add_argument("--dark-frames", default=None)
    args = parser.parse_args()

    real_dir = Path(args.real_frames)
    if not real_dir.exists():
        print(f"ERROR: --real-frames directory does not exist: {real_dir}", file=sys.stderr)
        sys.exit(1)

    frames = _load_tifs(real_dir)
    if not frames:
        print(f"ERROR: No .tif files found in {real_dir}", file=sys.stderr)
        sys.exit(1)

    dark_frames = None
    if args.dark_frames:
        dark_dir = Path(args.dark_frames)
        dark_frames = _load_tifs(dark_dir) if dark_dir.exists() else None
        if not dark_frames:
            print(f"WARNING: No dark frames found in {args.dark_frames}", file=sys.stderr)

    params = calibrate_from_frames(frames, dark_frames=dark_frames)
    fragment = _format_yaml_fragment(params)
    yaml.safe_load(fragment)  # verify valid YAML before output

    if args.output_config:
        Path(args.output_config).write_text(fragment + "\n")
        print(f"Calibrated config written to: {args.output_config}")

    if args.merge_config:
        _merge_params_into_config(Path(args.merge_config), params)
        print(f"Calibrated parameters merged into: {args.merge_config}")

    if not args.output_config and not args.merge_config:
        print(fragment)


if __name__ == "__main__":
    main()
