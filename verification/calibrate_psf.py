#!/usr/bin/env python3
"""Calibrate PSF, background, intensity, and noise params from real microscopy .tif frames.

Usage:
    uv run python calibrate_psf.py --real-frames <dir> [--output-config <path>] [--dark-frames <dir>]

Prints a YAML fragment ready to paste into config.yaml under synthetic:.
"""

import argparse
import re
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
    """Fit a 2-D Gaussian to `crop`, returning (sx, sy, A) -- the fitted
    sigmas and the background-subtracted peak amplitude -- or None on
    failure. `A` is the particle's own fitted contribution, excluding the
    local background baseline `B` (also fitted, but not needed by callers
    today)."""
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
        A, sx, sy = popt[0], abs(popt[3]), abs(popt[4])
        if 0.5 < sx < _CROP_HALF and 0.5 < sy < _CROP_HALF:
            return float(sx), float(sy), float(A)
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
                    # fit[2] is the fitted amplitude A, background-subtracted --
                    # matches what render_deeptrack.py treats peak_mean as (the
                    # particle's own contribution, separate from background).
                    # The raw crop max used previously included the local
                    # background baseline, inflating peak_mean.
                    peak_vals.append(fit[2])
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


def _line_indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _is_key_line(line: str, indent: int, key: str) -> bool:
    """True if `line` is an active (non-comment) 'key:' line at exactly `indent`."""
    stripped = line.strip()
    if stripped == "" or stripped.startswith("#"):
        return False
    if _line_indent(line) != indent:
        return False
    rest = line[indent:]
    return rest == f"{key}:" or rest.startswith(f"{key}: ") or rest.startswith(f"{key}:\n")


def _find_key_line(lines: list[str], start: int, end: int, indent: int, key: str) -> int | None:
    """Find the first ACTIVE 'key:' line within lines[start:end] at `indent`.

    Deliberately skips commented-out lines (e.g. '# sigma_px: 4.2') so a
    commented-out placeholder is never mistaken for a live key.
    """
    for i in range(start, end):
        if _is_key_line(lines[i], indent, key):
            return i
    return None


def _block_end(lines: list[str], header_idx: int, header_indent: int) -> int:
    """Index one past the last line belonging to the block headed by lines[header_idx].

    Blank lines don't end the block; the first non-blank line at indent <=
    header_indent does.
    """
    i = header_idx + 1
    last_in_block = header_idx
    while i < len(lines):
        if lines[i].strip() == "":
            i += 1
            continue
        if _line_indent(lines[i]) > header_indent:
            last_in_block = i
            i += 1
            continue
        break
    return last_in_block + 1


def _render_value(value) -> str:
    return str(value)


def _replace_value_preserving_comment(line: str, key: str, indent: int, rendered_value: str) -> str:
    """Replace just the value portion of an existing 'key: value  # comment' line,
    keeping the trailing comment (if any) and the line's own indentation intact."""
    newline = "\n" if line.endswith("\n") else ""
    content = line[indent:].rstrip("\n")
    rest = content[len(f"{key}:") :]
    if "#" in rest:
        comment = rest[rest.index("#") :]
        return f"{' ' * indent}{key}: {rendered_value}  {comment}{newline}"
    return f"{' ' * indent}{key}: {rendered_value}{newline}"


def _normalize_flow_empty_mappings(lines: list[str]) -> list[str]:
    """Rewrite 'key: {}' lines to a bare 'key:' block header.

    yaml.dump renders an empty dict as flow-style '{}' even with
    default_flow_style=False (there's nothing to put in block style). Normalizing
    this away up front means the rest of this module only has to reason about one
    shape — a block header, possibly with zero children yet — instead of two.
    """
    pattern = re.compile(r"^(\s*)(\w[\w-]*):\s*\{\}\s*$")
    out = []
    for line in lines:
        m = pattern.match(line.rstrip("\n"))
        out.append(f"{m.group(1)}{m.group(2)}:\n" if m else line)
    return out


def _merge_params_into_config(config_path: Path, params: dict) -> None:
    """Merge calibrated params into an existing config.yaml file under synthetic:.

    Preserves all existing keys not in the four calibrated sub-dicts (psf, particle,
    background, noise), including comments and formatting — this patches only the
    specific lines being updated (or appends new ones) rather than re-dumping the
    whole file, since a full yaml.safe_load -> yaml.dump round-trip silently drops
    every comment in the file. Strips _gain_sigma_note from noise and drops _meta
    entirely (both are internal-only fields, never written).

    Raises FileNotFoundError if config_path does not exist.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    lines = _normalize_flow_empty_mappings(config_path.read_text().splitlines(keepends=True))

    synthetic_idx = _find_key_line(lines, 0, len(lines), indent=0, key="synthetic")
    if synthetic_idx is None:
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        lines.append("synthetic:\n")
        synthetic_idx = len(lines) - 1

    for section in ("psf", "particle", "background", "noise"):
        calibrated = dict(params[section])
        if section == "noise":
            calibrated.pop("_gain_sigma_note", None)

        synthetic_end = _block_end(lines, synthetic_idx, header_indent=0)
        section_idx = _find_key_line(lines, synthetic_idx + 1, synthetic_end, indent=2, key=section)
        if section_idx is None:
            lines.insert(synthetic_end, f"  {section}:\n")
            section_idx = synthetic_end

        section_end = _block_end(lines, section_idx, header_indent=2)
        for key, value in calibrated.items():
            rendered = _render_value(value)
            key_idx = _find_key_line(lines, section_idx + 1, section_end, indent=4, key=key)
            if key_idx is not None:
                lines[key_idx] = _replace_value_preserving_comment(
                    lines[key_idx], key, indent=4, rendered_value=rendered
                )
            else:
                lines.insert(section_end, f"    {key}: {rendered}\n")
                section_end += 1

    config_path.write_text("".join(lines))


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
