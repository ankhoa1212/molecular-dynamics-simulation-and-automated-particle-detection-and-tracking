"""DeepTrack2-backed PSF rendering for the verification pipeline.

Implements the render-kernel-then-stamp approach: deeptrack is called once
per frame to produce a PSF kernel, which is then stamped onto an empty
canvas via scipy convolution.  This avoids the O(N) per-frame deeptrack
overhead measured at ~964× slower for the naive PointParticle-per-particle
path.

Requires deeptrack==2.0.1:
    uv add deeptrack==2.0.1    (inside verification/)
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
        2-D float32 array, normalised so kernel.sum() == 1.
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
    total = kernel.sum()
    if total == 0.0:
        # Fall back to a small unit-impulse so stamping still works
        kernel = np.zeros_like(kernel)
        kernel[H // 2, W // 2] = 1.0
        total = 1.0

    kernel = (kernel / total).astype(np.float32)
    return kernel


def render_frame_deeptrack(positions_lj, box, cfg, rng):
    """Render one synthetic microscopy frame using a DeepTrack2 PSF kernel.

    Strategy:
        1. Build a PSF kernel via a single deeptrack call (render-kernel-then-stamp).
        2. Sample per-particle intensities from a log-normal distribution.
        3. Stamp particles onto an empty canvas, then convolve with the kernel.
        4. Add spatially varying background (smooth 2-D random field).
        5. Apply sCMOS noise model: per-pixel gain variation + Poisson photon
           noise + Gaussian read noise.
        6. Clip to [0, 65535] and cast to uint16.

    Args:
        positions_lj: (N, 2) float array of particle positions in LJ units.
        box: (x_lo, x_hi, y_lo, y_hi) simulation box bounds.
        cfg: synthetic config dict (sub-dicts: psf, background, particle, noise).
        rng: numpy.random.Generator instance.

    Returns:
        uint16 numpy array of shape (H, W).
    """
    H = cfg["image_height"]
    W = cfg["image_width"]

    # --- PSF kernel --------------------------------------------------------
    kernel = _build_psf_kernel(cfg, H, W)

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

    # --- Stamp particles ---------------------------------------------------
    canvas = np.zeros((H, W), dtype=np.float32)
    for (px, py), intensity in zip(pixel_positions, intensities):
        ix = int(np.clip(px, 0, W - 1))
        iy = int(np.clip(py, 0, H - 1))
        canvas[iy, ix] += float(intensity)

    # Convolve the delta-function stamps with the PSF kernel
    frame = scipy.ndimage.convolve(canvas, kernel, mode="reflect").astype(np.float32)

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
