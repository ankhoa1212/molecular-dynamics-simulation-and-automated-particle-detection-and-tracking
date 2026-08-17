#!/usr/bin/env python3
"""Calibrate PSF, background, intensity, and noise params from real microscopy .tif frames.

Usage:
    uv run python calibrate_psf.py --real-frames <dir> [--output-config <path>] [--dark-frames <dir>]
    uv run python calibrate_psf.py --brightfield --lammps <path> [--real-frames <dir>] [--mie-frames <n>] ...

Prints a YAML fragment ready to paste into config.yaml under synthetic:.
--brightfield switches to calibrate_brightfield, a separate search over
synthetic.brightfield for render_strategy: brightfield (see that function's
docstring).

Real saturated bright-field data should tune --min-area/--max-area/--percentile
away from the small-particle-friendly defaults -- start from this repo's
config.yaml crop_template values (--min-area 100 --max-area 4000 --percentile 95.0)
and retune per dataset.
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


def _load_real_frames(directory: Path) -> list[np.ndarray]:
    """Load real reference frames from `directory`, auto-detecting format by
    extension: .tif/.tiff via tifffile (same as _load_tifs), .png via
    cv2.imread grayscale -- the LodeSTAR training crops
    (data-setup/models/lodestar_model_*/crops/*.png) calibrate_brightfield's
    z-range fit targets are stored as PNG, not TIFF. Both formats may
    coexist in the same directory; format is determined per-file by
    extension, not by directory-wide assumption.
    """
    import cv2  # noqa: PLC0415

    tif_paths = sorted(directory.glob("*.tif")) + sorted(directory.glob("*.tiff"))
    png_paths = sorted(directory.glob("*.png"))
    frames = [tifffile.imread(str(p)).astype(np.float32) for p in tif_paths]
    for p in png_paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"failed to load {p}: not a valid image file")
        frames.append(img.astype(np.float32))
    return frames


def _fit_to_canvas(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    """Pad or center-crop `frame` to exactly (height, width).

    LodeSTAR crops are smaller than and inconsistent in size with
    calibrate_brightfield's candidate canvas (~64x65 to ~173x176px
    measured, vs. the default 256x256 canvas), and
    compare_renders.compute_psd_similarity requires matching array shapes.
    Padding (not resizing) preserves the crop's true physical pixel scale
    -- resizing would distort exactly the size relationship the z-range fit
    is trying to calibrate. Center-crops instead on any axis where the
    source frame is larger than the target canvas.
    """
    h, w = frame.shape
    out = np.zeros((height, width), dtype=frame.dtype)

    if h <= height:
        pad_y = (height - h) // 2
        src_y0, src_y1 = 0, h
        dst_y0, dst_y1 = pad_y, pad_y + h
    else:
        crop_y = (h - height) // 2
        src_y0, src_y1 = crop_y, crop_y + height
        dst_y0, dst_y1 = 0, height

    if w <= width:
        pad_x = (width - w) // 2
        src_x0, src_x1 = 0, w
        dst_x0, dst_x1 = pad_x, pad_x + w
    else:
        crop_x = (w - width) // 2
        src_x0, src_x1 = crop_x, crop_x + width
        dst_x0, dst_x1 = 0, width

    out[dst_y0:dst_y1, dst_x0:dst_x1] = frame[src_y0:src_y1, src_x0:src_x1]
    return out


def _detect_particle_centers(frame, min_area, max_area, percentile):
    """Detect candidate particle centers via connected-component labeling on
    a percentile-thresholded mask, filtered by component pixel-area.

    A per-pixel local-maxima approach (frame == maximum_filter(frame))
    assumes isolated point-like peaks. Real bright-field particle frames can
    instead contain sensor-saturated intensity plateaus (measured on the 2 um
    dataset: up to ~4.5% of pixels at the sensor max) — every pixel in such a
    plateau ties for "local max" under that approach, producing tens of
    thousands of spurious candidates per frame instead of one per particle.
    Connected-component centroiding treats each contiguous bright region as
    one candidate regardless of internal saturation.

    `max_area=None` is treated as unbounded, so callers don't each need to
    repeat their own None-to-unbounded sentinel conversion.

    Returns:
        (N, 2) float array of (row, col) intensity-weighted centroids.
    """
    max_area = np.inf if max_area is None else max_area
    threshold = np.percentile(frame, percentile)
    mask = frame > threshold
    labels, n_labels = scipy.ndimage.label(mask)
    if n_labels == 0:
        return np.zeros((0, 2), dtype=np.float64)
    sizes = scipy.ndimage.sum(mask, labels, index=np.arange(1, n_labels + 1))
    keep = np.where((sizes >= min_area) & (sizes <= max_area))[0] + 1
    if len(keep) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    centroids = scipy.ndimage.center_of_mass(frame, labels, index=keep)
    return np.array(centroids).reshape(-1, 2)


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
        # curve_fit's lower bound on A is 0, not a meaningful noise floor -- on
        # some crops (candidate detection false positives, or a genuine
        # particle too large for this fixed-size window) it converges to a
        # numerically-near-zero amplitude that still satisfies the sigma
        # bounds below. That's a degenerate "flat" fit, not a real peak, and
        # left unfiltered it corrupts the population-level lognormal fit in
        # calibrate_from_frames (a handful of ~1e-10 amplitudes alongside
        # genuine hundreds-to-thousands-ADU peaks spans ~13 orders of
        # magnitude in log-space, blowing up the fitted shape parameter).
        # Scaled to the crop's own dynamic range rather than an absolute ADU
        # constant, since that range varies by dataset/exposure. The 1e-9
        # absolute floor backstops the relative check on an (near-)perfectly
        # flat crop, where crop_range itself is ~0 and "A < 0.01 * crop_range"
        # degenerates to "A < ~0", which a tiny positive A would pass.
        crop_range = float(crop.max() - crop.min())
        if A < 1e-9 or A < 0.01 * crop_range:
            return None
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
    frames: list[np.ndarray],
    dark_frames: list[np.ndarray] | None = None,
    min_area: float = 4.0,
    max_area: float | None = None,
    percentile: float = 90.0,
) -> dict:
    """Fit PSF and imaging parameters from a list of microscopy frames.

    min_area, max_area, percentile: tune _detect_particle_centers' connected-
    component candidate detection. Defaults suit small, isolated point-like
    data; real saturated bright-field data needs a higher percentile and an
    explicit max_area (see verification/config.yaml's crop_template section
    for values tuned to a real dataset).

    Returns dict with keys: psf, particle, background, noise, _meta.
    """
    if not frames:
        raise ValueError("frames list is empty")

    psf_sigma = _DEFAULT_SIGMA
    sigma_ests, peak_vals, all_bg, bg_residuals = [], [], [], []

    for frame in frames:
        spots = _detect_particle_centers(frame, min_area, max_area, percentile)
        if not len(spots):
            continue

        disk_r = int(2 * psf_sigma) + 1
        disk_y, disk_x = np.ogrid[-disk_r : disk_r + 1, -disk_r : disk_r + 1]
        disk = disk_x**2 + disk_y**2 <= disk_r**2
        H, W = frame.shape
        mask = np.zeros((H, W), dtype=bool)

        for row, col in np.round(spots).astype(int):
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


def calibrate_brightfield(
    real_frames: list[np.ndarray],
    mie_frames: list[np.ndarray],
    positions_lj: np.ndarray,
    box: tuple,
    image_height: int,
    image_width: int,
    rng,
    n_iterations: int = 15,
    param_bounds: dict | None = None,
) -> dict:
    """Bounded coarse search over render_strategy: brightfield's
    synthetic.brightfield parameters, scored against real_frames and/or
    mie_frames.

    calibrate_from_frames' isolated-spot Gaussian-fit method doesn't apply
    here -- this strategy's output is dense, touching, and ring-shaped, not
    isolated bright spots -- so this is new fitting logic, not a reuse of
    that method. It IS a coarse random search over a bounded parameter
    range, not a gradient-based or exhaustive optimizer, per this plan's
    scope: each iteration renders one candidate config via
    render_brightfield.render_frame_brightfield (one real Brightfield
    solve -- the expensive part; see that module's docstring for the
    per-frame cost data behind n_iterations' small default) and scores it
    with compare_renders.compute_psd_similarity's mid-band value against
    every supplied target frame, keeping the best-scoring candidate.

    Args:
        real_frames: real brightfield reference frames (may be empty if
            mie_frames alone are supplied).
        mie_frames: Mie ground-truth frames from
            render_brightfield.generate_mie_ground_truth (may be empty if
            real_frames alone are supplied). At least one of real_frames/
            mie_frames must be non-empty.
        positions_lj: (N, 2) float array of particle positions (LJ units)
            to render each candidate with -- typically the same trajectory
            frame generate_mie_ground_truth drew its subset from.
        box: (x_lo, x_hi, y_lo, y_hi) simulation box bounds.
        image_height, image_width: canvas size to render candidates at.
        rng: numpy.random.Generator instance.
        n_iterations: number of random candidates to evaluate.
        param_bounds: optional override of the search space; see the
            default bounds in this function's body for the expected keys.

    Returns:
        {"brightfield": {...}} shaped for _merge_params_into_config, plus
        an internal-only "_meta" key (best PSD score, iteration count) that
        _merge_params_into_config skips.

    Raises:
        ValueError: if both real_frames and mie_frames are empty, or if no
            candidate produced a valid PSD similarity score against the
            given targets.
    """
    if not real_frames and not mie_frames:
        raise ValueError(
            "calibrate_brightfield requires at least one of real_frames or "
            "mie_frames as a fitting target."
        )

    from compare_renders import compute_psd_similarity
    from render_brightfield import render_frame_brightfield

    bounds = param_bounds or {
        "na": (0.7, 1.4),
        "wavelength": (450e-9, 650e-9),
        "resolution": (65e-9, 150e-9),
        "refractive_index_medium": (1.33, 1.40),
        "radius": (0.3e-6, 0.8e-6),
        "refractive_index": (1.40, 1.60),
        "intensity_scale": (5000.0, 40000.0),
        # z_min_px/z_max_px are sampled independently within this range
        # (not mirrored like radius/refractive_index above, which
        # intentionally produce one monodisperse value per candidate) --
        # defocus needs a genuine spread for render_strategy: brightfield_fast's
        # z-bucketing to have anything to exercise once this feeds production
        # config. See docs/plans/2026-08-16-001-feat-brightfield-fast-
        # render-path-plan.md's U4 KTD.
        "z_range_px": (-15.0, 15.0),
    }
    # LodeSTAR crops (this function's real-world real_frames target) are
    # smaller than and inconsistent in size with the candidate canvas --
    # compute_psd_similarity requires matching shapes, so fit every real
    # frame to the candidate canvas once up front. mie_frames are already
    # rendered at (image_height, image_width) by generate_mie_ground_truth,
    # so they need no fitting.
    fitted_real_frames = [_fit_to_canvas(f, image_height, image_width) for f in real_frames]
    targets = list(mie_frames) + fitted_real_frames
    search_max_particles = min(max(len(positions_lj), 1), 50)

    best_score = -np.inf
    best_params = None

    for _ in range(n_iterations):
        radius = float(rng.uniform(*bounds["radius"]))
        refractive_index = float(rng.uniform(*bounds["refractive_index"]))
        # Independently sampled, then sorted -- a genuine [z_min, z_max]
        # spread, not the mirrored-constant pattern radius/refractive_index
        # use above (see the z_range_px bound's own comment).
        z_a = float(rng.uniform(*bounds["z_range_px"]))
        z_b = float(rng.uniform(*bounds["z_range_px"]))
        candidate = {
            "max_particles": search_max_particles,
            "intensity_scale": float(rng.uniform(*bounds["intensity_scale"])),
            "na": float(rng.uniform(*bounds["na"])),
            "wavelength": float(rng.uniform(*bounds["wavelength"])),
            "resolution": float(rng.uniform(*bounds["resolution"])),
            # Fixed, not searched: deeptrack.Brightfield's own default (10)
            # renders this dataset's particles at ~10x its real interparticle
            # spacing (invisible at small validation scales, catastrophic at
            # production density -- see render_brightfield.py's
            # _resolve_brightfield_intensity docstring). 1.0 matches every
            # other render strategy's particle scale; not reopened as a
            # search dimension since it's a fixed, decided value, not a
            # free physical parameter to fit against reference imagery.
            "magnification": 1.0,
            "refractive_index_medium": float(rng.uniform(*bounds["refractive_index_medium"])),
            "radius_min": radius,
            "radius_max": radius,
            "refractive_index_min": refractive_index,
            "refractive_index_max": refractive_index,
            "z_min_px": min(z_a, z_b),
            "z_max_px": max(z_a, z_b),
        }
        cfg = {
            "image_height": image_height,
            "image_width": image_width,
            "brightfield": candidate,
        }
        rendered = render_frame_brightfield(positions_lj, box, cfg, rng)

        scores = [
            mid
            for target in targets
            for (_, mid, _) in [compute_psd_similarity(rendered, target)]
            if not np.isnan(mid)
        ]
        if not scores:
            continue
        score = float(np.mean(scores))
        if score > best_score:
            best_score = score
            best_params = candidate

    if best_params is None:
        raise ValueError(
            "calibrate_brightfield: no candidate produced a valid PSD similarity "
            "score against the given targets."
        )

    return {
        "brightfield": best_params,
        "_meta": {"psd_mid_score": round(best_score, 4), "n_iterations": n_iterations},
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

    Preserves all existing keys not in `params`'s own top-level sections (e.g.
    psf, particle, background, noise -- or procedural_shape for
    fit_procedural_ring.py's ring parameters), including comments and
    formatting — this patches only the specific lines being updated (or
    appends new ones) rather than re-dumping the whole file, since a full
    yaml.safe_load -> yaml.dump round-trip silently drops every comment in
    the file. Strips _gain_sigma_note from noise and drops _meta entirely
    (both are internal-only fields, never written).

    Iterates over whatever sections `params` actually contains rather than a
    fixed tuple, so any caller-defined section (not just the four
    calibrate_psf.py itself always calibrates) can be merged without changes
    here.

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

    for section in params:
        if section.startswith("_"):
            continue  # internal-only, e.g. calibrate_from_frames' _meta -- never written
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


def _run_brightfield_calibration(args):
    """--brightfield mode: calibrate_brightfield instead of calibrate_from_frames.

    Real footage is optional here (calibrate_brightfield accepts real
    frames and/or Mie ground-truth frames), unlike the default Gaussian-fit
    mode where --real-frames is required.
    """
    sys.path.insert(0, str(Path(__file__).parent))
    sys.path.insert(0, str(Path(__file__).parent / ".." / "lammps-scripts"))
    from lammps_parser import parse_lammps_dump

    from render import _parse_atoms, _parse_box
    from render_brightfield import generate_mie_ground_truth

    real_frames = []
    if args.real_frames:
        real_dir = Path(args.real_frames)
        if not real_dir.exists():
            print(
                f"ERROR: --real-frames directory does not exist: {real_dir}",
                file=sys.stderr,
            )
            sys.exit(1)
        real_frames = _load_real_frames(real_dir)

    positions_lj, box = None, None
    for block in parse_lammps_dump(args.lammps):
        box = _parse_box(block["box_bounds"])
        positions_lj, _ = _parse_atoms(block["atom_header"], block["atoms"])
        break
    if positions_lj is None or len(positions_lj) == 0:
        print(f"ERROR: No timesteps/particles found in {args.lammps}", file=sys.stderr)
        sys.exit(1)

    rng = np.random.default_rng(args.seed)
    mie_frames = []
    if args.mie_frames > 0:
        n_particles = min(args.mie_frames_particles, len(positions_lj))
        cfg = {
            "image_height": args.image_height,
            "image_width": args.image_width,
            "brightfield": {
                "mie_max_particles": n_particles,
                "mie_max_frames": args.mie_frames,
                "magnification": 1.0,
            },
        }
        mie_frames = generate_mie_ground_truth(
            cfg,
            positions_lj,
            box,
            n_frames=args.mie_frames,
            n_particles=n_particles,
            rng=rng,
        )

    if not real_frames and not mie_frames:
        print(
            "ERROR: --brightfield needs at least one of --real-frames or --mie-frames > 0.",
            file=sys.stderr,
        )
        sys.exit(1)

    params = calibrate_brightfield(
        real_frames,
        mie_frames,
        positions_lj,
        box,
        image_height=args.image_height,
        image_width=args.image_width,
        rng=rng,
        n_iterations=args.n_iterations,
    )
    # _format_yaml_fragment is hardcoded to calibrate_from_frames' fixed
    # psf/particle/background/noise shape (unlike _merge_params_into_config,
    # which is genuinely section-agnostic) -- not reusable here. A plain
    # yaml.dump of the one flat "brightfield" section is sufficient for the
    # --output-config/stdout display path.
    fragment = "# Calibrated brightfield parameters (paste into config.yaml under synthetic:)\n"
    fragment += f"# Best PSD mid-band score: {params['_meta']['psd_mid_score']} "
    fragment += f"over {params['_meta']['n_iterations']} candidates\n"
    fragment += yaml.dump({"brightfield": params["brightfield"]}, sort_keys=False)
    yaml.safe_load(fragment)  # verify valid YAML before output

    if args.output_config:
        Path(args.output_config).write_text(fragment + "\n")
        print(f"Calibrated config written to: {args.output_config}")
    if args.merge_config:
        _merge_params_into_config(Path(args.merge_config), params)
        print(f"Calibrated parameters merged into: {args.merge_config}")
    if not args.output_config and not args.merge_config:
        print(fragment)


def main():
    parser = argparse.ArgumentParser(description="Calibrate PSF and noise from real .tif frames")
    parser.add_argument(
        "--real-frames",
        default=None,
        help="required unless --brightfield is given with --mie-frames > 0",
    )
    parser.add_argument("--output-config", default=None)
    parser.add_argument("--merge-config", default=None, metavar="PATH")
    parser.add_argument("--dark-frames", default=None)
    parser.add_argument(
        "--min-area",
        type=float,
        default=4.0,
        help="min connected-component pixel area to count as a detection candidate "
        "(default: 4.0, suited to small isolated point-like data)",
    )
    parser.add_argument(
        "--max-area",
        type=float,
        default=None,
        help="max connected-component pixel area, excludes merged/oversized blobs "
        "(default: unbounded)",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=90.0,
        help="candidate-detection brightness percentile (default: 90.0). Real "
        "saturated bright-field data should start from this repo's config.yaml "
        "crop_template values (--min-area 100 --max-area 4000 --percentile 95.0), "
        "retuned per dataset",
    )
    parser.add_argument(
        "--brightfield",
        action="store_true",
        help="calibrate render_strategy: brightfield's synthetic.brightfield section "
        "(calibrate_brightfield) instead of the default Gaussian-spot PSF fit "
        "(calibrate_from_frames).",
    )
    parser.add_argument(
        "--lammps",
        default=None,
        help="--brightfield only: trajectory to render candidates from",
    )
    parser.add_argument(
        "--mie-frames",
        type=int,
        default=0,
        help="--brightfield only: number of Mie ground-truth frames to generate as an "
        "additional fitting target (default: 0, real frames only)",
    )
    parser.add_argument(
        "--mie-frames-particles",
        type=int,
        default=10,
        help="--brightfield only: particles per generated Mie ground-truth frame (default: 10)",
    )
    parser.add_argument(
        "--n-iterations",
        type=int,
        default=15,
        help="--brightfield only: number of candidate configs to search (default: 15)",
    )
    parser.add_argument("--image-height", type=int, default=256, help="--brightfield only")
    parser.add_argument("--image-width", type=int, default=256, help="--brightfield only")
    parser.add_argument("--seed", type=int, default=42, help="--brightfield only")
    args = parser.parse_args()

    if args.brightfield:
        if not args.lammps:
            print("ERROR: --brightfield requires --lammps.", file=sys.stderr)
            sys.exit(1)
        _run_brightfield_calibration(args)
        return

    if not args.real_frames:
        print(
            "ERROR: --real-frames is required (unless --brightfield is given).",
            file=sys.stderr,
        )
        sys.exit(1)

    real_dir = Path(args.real_frames)
    if not real_dir.exists():
        print(
            f"ERROR: --real-frames directory does not exist: {real_dir}",
            file=sys.stderr,
        )
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

    params = calibrate_from_frames(
        frames,
        dark_frames=dark_frames,
        min_area=args.min_area,
        max_area=args.max_area,
        percentile=args.percentile,
    )
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
