"""DeepTrack2-backed PSF rendering for the verification pipeline.

Implements the render-kernel-then-stamp approach: deeptrack is called once
per frame to produce a PSF kernel, which is then stamped onto an empty
canvas via scipy convolution.  This avoids the O(N) per-frame deeptrack
overhead measured at ~964× slower for the naive PointParticle-per-particle
path.

Requires deeptrack==2.0.1:
    uv add deeptrack==2.0.1    (inside verification/)
"""

import warnings

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


def _build_psf_kernel(cfg, H, W):
    """Build a normalised PSF kernel via a single deeptrack call.

    Uses deeptrack.Fluorescence (widefield) with NA, wavelength, and pixel
    size from the config.  A single on-axis PointParticle is rendered at the
    image centre so the kernel is spatially invariant (valid for the
    render-kernel-then-stamp approach).

    Args:
        cfg: synthetic config dict (must have a 'psf' sub-dict)
        H, W: image height/width in pixels

    Returns:
        2-D float32 array, normalised so kernel.max() == 1.
    """
    try:
        import deeptrack  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "DeepTrack2 rendering requires 'deeptrack==2.0.1'. "
            "Run: uv add deeptrack==2.0.1  (inside verification/)\n"
            "See https://github.com/DeepTrackAI/DeepTrack2 for details."
        ) from exc

    psf_cfg = cfg.get("psf", {})
    na = float(psf_cfg.get("na", 1.4))
    wavelength = float(psf_cfg.get("wavelength", 520e-9))
    resolution = float(psf_cfg.get("resolution", 65e-9))

    optics = deeptrack.Fluorescence(
        NA=na,
        wavelength=wavelength,
        resolution=resolution,
        output_region=(0, 0, H, W),
    )

    probe = deeptrack.PointParticle(
        position=(H // 2, W // 2),
        intensity=1.0,
        z=0,
    )
    pipeline = optics(probe)
    kernel = pipeline.update().resolve()

    # deeptrack may return complex arrays; take the absolute value
    kernel = np.abs(np.array(kernel, dtype=np.float64).squeeze())

    # Guard against all-zero kernel (e.g., PSF entirely outside output_region)
    if kernel.sum() == 0.0:
        # Fall back to a small unit-impulse so stamping still works
        kernel = np.zeros_like(kernel)
        kernel[H // 2, W // 2] = 1.0

    # Crop to a small patch around the PSF centre so that
    # scipy.ndimage.convolve doesn't allocate O(image_size^4) memory. The
    # crop radius is a memory/performance bound, not a guarantee of
    # capturing the PSF's full energy -- measured at this module's default
    # NA/wavelength/resolution, only ~41% of the kernel's total energy
    # lies within r=10px of a 65x65 (r=32) crop, so the PSF here is
    # considerably broader than a naive "tightly concentrated" assumption.
    # Peak-normalizing (rather than sum-normalizing) after the crop makes
    # rendered brightness insensitive to how much of that tail the crop
    # radius happens to capture.
    cy, cx = H // 2, W // 2
    r = min(32, cy, cx)
    kernel = kernel[cy - r : cy + r + 1, cx - r : cx + r + 1]
    peak = kernel.max()
    if peak > 0:
        kernel = kernel / peak

    return kernel.astype(np.float32)


def _composite_crop_templates(pixel_positions, intensities, cfg, rng, H, W):
    """Composite each particle's template onto the canvas at its sub-pixel
    position (crop_source: real / procedural).

    Unlike the physics path's single kernel convolved globally, each
    particle draws from its own template source. 'real' picks uniformly at
    random from the cached empirical library, so appearance still varies
    particle-to-particle within a frame. 'procedural' is deterministic given
    `procedural_shape` config (no per-particle randomization) -- the same
    template is reused for every particle in the frame. A particle near the
    canvas edge has its template patch clipped to canvas bounds (matching
    _lj_to_pixels' clip-to-bounds contract) rather than indexed out of range.
    """
    from render_crop_templates import RING_PARAM_NAMES, generate_procedural_shape
    from render_crop_templates import load_template_library

    crop_source = cfg["crop_source"]
    canvas = np.zeros((H, W), dtype=np.float32)

    if crop_source == "real":
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
    else:  # procedural
        proc_cfg = cfg.get("procedural_shape", {})
        proc_size = int(proc_cfg.get("size", 41))
        proc_sigma = float(proc_cfg.get("sigma", 5.0))
        present = [key for key in RING_PARAM_NAMES if key in proc_cfg]
        if present and len(present) < len(RING_PARAM_NAMES):
            missing = [key for key in RING_PARAM_NAMES if key not in proc_cfg]
            warnings.warn(
                f"synthetic.procedural_shape has {len(present)}/{len(RING_PARAM_NAMES)} "
                f"ring_* keys set (missing {missing}) -- falling back to the plain circular "
                "Gaussian. Set all six (via fit_procedural_ring.py --merge-config) or none.",
                stacklevel=2,
            )
        ring_params = (
            tuple(proc_cfg[key] for key in RING_PARAM_NAMES)
            if len(present) == len(RING_PARAM_NAMES)
            else None
        )
        procedural_template = generate_procedural_shape(proc_size, proc_sigma, ring_params)

    for (px, py), intensity in zip(pixel_positions, intensities):
        if crop_source == "real":
            template = templates[int(rng.integers(0, len(templates)))]
        else:
            template = procedural_template

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
    """Render one synthetic microscopy frame using a DeepTrack2 PSF kernel,
    an empirical crop-template library, or a procedural shape generator.

    Strategy (crop_source: physics, default — unchanged from before crop_source existed):
        1. Build a PSF kernel via a single deeptrack call (render-kernel-then-stamp).
        2. Sample per-particle intensities from a log-normal distribution.
        3. Stamp particles onto an empty canvas, then convolve with the kernel.
        4. Add spatially varying background (smooth 2-D random field).
        5. Apply sCMOS noise model: per-pixel gain variation + Poisson photon
           noise + Gaussian read noise.
        6. Clip to [0, 65535] and cast to uint16.

    Strategy (crop_source: real / procedural): steps 1-3 above are replaced by
    per-particle template compositing (_composite_crop_templates) — each
    particle draws its own template rather than sharing one convolved
    kernel. Steps 4-6 (background + noise) are unchanged and shared across
    all three crop_source values.

    Args:
        positions_lj: (N, 2) float array of particle positions in LJ units.
        box: (x_lo, x_hi, y_lo, y_hi) simulation box bounds.
        cfg: synthetic config dict (sub-dicts: psf, background, particle, noise,
            and — for crop_source real/procedural — crop_template/procedural_shape).
        rng: numpy.random.Generator instance.

    Returns:
        uint16 numpy array of shape (H, W).
    """
    H = cfg["image_height"]
    W = cfg["image_width"]
    crop_source = cfg.get("crop_source", "physics")

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

    if crop_source == "physics":
        # --- PSF kernel ------------------------------------------------------
        kernel = _build_psf_kernel(cfg, H, W)

        # --- Stamp particles ---------------------------------------------------
        canvas = np.zeros((H, W), dtype=np.float32)
        for (px, py), intensity in zip(pixel_positions, intensities):
            ix = int(np.clip(px, 0, W - 1))
            iy = int(np.clip(py, 0, H - 1))
            canvas[iy, ix] += float(intensity)

        # Convolve the delta-function stamps with the PSF kernel
        frame = scipy.ndimage.convolve(canvas, kernel, mode="reflect").astype(np.float32)
    elif crop_source in ("real", "procedural"):
        frame = _composite_crop_templates(pixel_positions, intensities, cfg, rng, H, W)
    else:
        raise ValueError(
            f"Unknown crop_source: {crop_source!r}. Expected 'physics', 'real', or 'procedural'."
        )

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
