"""Empirical crop-template rendering for the verification pipeline
(render_strategy: deeptrack, crop_source: real -- the module's name is
historical from when it also implemented a deeptrack-backed PSF kernel path;
that path (crop_source: physics) and the fitted-template path
(crop_source: procedural) were both removed once render_strategy:
brightfield_fast superseded them on realism at production scale).

Composites each particle's own template, drawn from a cached empirical
crop-template library, onto the canvas. No deeptrack dependency -- this
module is pure numpy/scipy.
"""

import numpy as np

try:
    import scipy.ndimage
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "scipy is required for render_deeptrack. " "Run: uv add scipy  (inside verification/)"
    ) from exc


def _lj_to_pixels(positions_lj, box, H, W):
    """Convert (N, 2) LJ positions to pixel (px, py) coordinates clipped to image."""
    x_lo, x_hi, y_lo, y_hi = box
    px = np.clip((positions_lj[:, 0] - x_lo) / (x_hi - x_lo) * W, 0, W - 1)
    py = np.clip((positions_lj[:, 1] - y_lo) / (y_hi - y_lo) * H, 0, H - 1)
    return np.stack([px, py], axis=1)


def _composite_crop_templates(pixel_positions, intensities, cfg, rng, H, W):
    """Composite each particle's template onto the canvas at its sub-pixel
    position (crop_source: real).

    Each particle draws its own template from the cached empirical library,
    picked uniformly at random, so appearance still varies particle-to-
    particle within a frame. A particle near the canvas edge has its
    template patch clipped to canvas bounds (matching _lj_to_pixels'
    clip-to-bounds contract) rather than indexed out of range.
    """
    from render_crop_templates import load_template_library

    canvas = np.zeros((H, W), dtype=np.float32)

    crop_template_cfg = cfg.get("crop_template", {})
    cache_path = crop_template_cfg.get("cache_path")
    if not cache_path:
        raise ValueError(
            "crop_source: 'real' requires synthetic.crop_template.cache_path — a template "
            "library built via render_crop_templates.build_template_library()."
        )
    try:
        templates = load_template_library(cache_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"crop_source: 'real' requires a pre-built template library at '{cache_path}'. "
            "Build one via render_crop_templates.build_template_library() first."
        ) from exc
    if len(templates) == 0:
        raise ValueError(f"Template library at '{cache_path}' is empty.")

    for (px, py), intensity in zip(pixel_positions, intensities):
        template = templates[int(rng.integers(0, len(templates)))]

        th, tw = template.shape
        half_h, half_w = th // 2, tw // 2

        iy, ix = int(round(float(py))), int(round(float(px)))
        frac_y, frac_x = float(py) - iy, float(px) - ix
        shifted = scipy.ndimage.shift(
            template, shift=(frac_y, frac_x), order=3, mode="constant", cval=0.0
        )
        patch = shifted.astype(np.float32) * float(intensity)

        r0, r1 = iy - half_h, iy - half_h + th
        c0, c1 = ix - half_w, ix - half_w + tw

        pr0, pr1 = max(0, r0), min(H, r1)
        pc0, pc1 = max(0, c0), min(W, c1)
        if pr0 >= pr1 or pc0 >= pc1:
            continue  # template patch falls entirely outside the canvas
        tr0, tr1 = pr0 - r0, pr1 - r0
        tc0, tc1 = pc0 - c0, pc1 - c0

        canvas[pr0:pr1, pc0:pc1] += patch[tr0:tr1, tc0:tc1]

    return canvas


def render_frame_deeptrack(positions_lj, box, cfg, rng):
    """Render one synthetic microscopy frame using an empirical crop-template
    library (crop_source: real -- the only supported value; physics and
    procedural were removed once render_strategy: brightfield_fast
    superseded both on realism at production scale).

    Strategy:
        1. Sample per-particle intensities from a log-normal distribution.
        2. Composite each particle's own template, drawn from the cached
           empirical library, onto the canvas (_composite_crop_templates).
        3. Add spatially varying background (smooth 2-D random field).
        4. Apply sCMOS noise model: per-pixel gain variation + Poisson photon
           noise + Gaussian read noise.
        5. Clip to [0, 65535] and cast to uint16.

    Args:
        positions_lj: (N, 2) float array of particle positions in LJ units.
        box: (x_lo, x_hi, y_lo, y_hi) simulation box bounds.
        cfg: synthetic config dict (sub-dicts: psf, background, particle,
            noise, crop_template). crop_source must be "real".
        rng: numpy.random.Generator instance.

    Returns:
        uint16 numpy array of shape (H, W).

    Raises:
        ValueError: if crop_source is unset or not "real".
    """
    H = cfg["image_height"]
    W = cfg["image_width"]
    crop_source = cfg.get("crop_source")

    if crop_source != "real":
        raise ValueError(
            f"Unknown crop_source: {crop_source!r}. Only 'real' is supported -- "
            "crop_source: physics and crop_source: procedural were removed once "
            "render_strategy: brightfield_fast superseded both on realism at production "
            "scale. Set synthetic.crop_source: real and configure crop_template.cache_path."
        )

    # --- LJ -> pixel coordinates ------------------------------------------
    if len(positions_lj) == 0:
        pixel_positions = np.zeros((0, 2))
    else:
        pixel_positions = _lj_to_pixels(positions_lj, box, H, W)

    # --- Per-particle intensities (log-normal) ----------------------------
    particle_cfg = cfg.get("particle", {})
    peak_mean = float(particle_cfg.get("peak_mean", 40000))
    intensity_sigma = float(particle_cfg.get("intensity_sigma", 0.3))
    n_particles = len(pixel_positions)

    if n_particles > 0:
        intensities = rng.lognormal(
            mean=np.log(peak_mean),
            sigma=intensity_sigma,
            size=n_particles,
        )
    else:
        intensities = np.array([], dtype=np.float32)

    frame = _composite_crop_templates(pixel_positions, intensities, cfg, rng, H, W)

    # --- Spatially varying background -------------------------------------
    bg_cfg = cfg.get("background", {})
    bg_scale = float(bg_cfg.get("heterogeneity_scale", 50))
    bg_amplitude = float(bg_cfg.get("amplitude", 500))

    if bg_amplitude > 0:
        bg_noise = rng.random((H, W)).astype(np.float32)
        bg_field = scipy.ndimage.gaussian_filter(bg_noise, sigma=bg_scale) * bg_amplitude
        frame = frame + bg_field

    # --- sCMOS noise model ------------------------------------------------
    noise_cfg = cfg.get("noise", {})
    gain_sigma = float(noise_cfg.get("gain_sigma", 0.02))
    read_noise = float(noise_cfg.get("read_noise", 15.0))

    # Per-pixel gain map (mean=1, std=gain_sigma)
    gain = rng.normal(1.0, gain_sigma, (H, W)).astype(np.float32)
    gain = np.clip(gain, 0.0, None)  # gain must be non-negative

    # Poisson photon noise on gain-scaled photons
    photons = np.clip(frame * gain, 0.0, None)
    shot = rng.poisson(photons).astype(np.float32)

    # Gaussian read noise
    if read_noise > 0:
        shot = shot + rng.normal(0.0, read_noise, (H, W)).astype(np.float32)

    return np.clip(shot, 0, 65535).astype(np.uint16)
