"""Background composite rendering strategy for render.py.

Composites a real experimental background frame (extracted as a temporal
median from a reference video) with synthetically stamped particle signal.
This approach preserves real camera noise texture, illumination gradients,
and sensor artefacts in the background while adding controlled particle
signal on top.

Usage:
    1. Call ``extract_temporal_median`` once (outside the frame loop) to
       build the background from a reference video.
    2. Store the result in ``cfg["_background_frame"]`` before the frame loop.
    3. Call ``render_frame_background_composite`` per frame.

Memory note: ``extract_temporal_median`` with n_frames=50 reads at most
50 pages from a TIFF stack.  For a 2200x3200 uint16 stack this is
50 × 2200 × 3200 × 2 bytes ≈ 704 MB peak; the 50-frame default is the
safety ceiling.

Requires scipy (``uv add scipy`` inside verification/).
"""

import numpy as np

try:
    import scipy.ndimage
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "scipy is required for render_background_composite. "
        "Run: uv add scipy  (inside verification/)"
    ) from exc


def extract_temporal_median(
    video_path,
    n_frames: int = 50,
    rng=None,
) -> np.ndarray:
    """Extract a temporal median background from a multi-page TIFF video.

    Sub-samples up to ``n_frames`` pages uniformly at random (without
    replacement), computes the per-pixel median, and returns it as a float32
    array.  The median suppresses moving particles while retaining the static
    background illumination and camera offset.

    Args:
        video_path: Path to the TIFF stack (str or pathlib.Path).
        n_frames: Maximum number of pages to sample.  If the file has fewer
            pages than this, all pages are used.  Default: 50.
        rng: numpy.random.Generator for reproducible sub-sampling.  Defaults
            to ``np.random.default_rng(0)`` if None.

    Returns:
        float32 numpy array of shape (H, W).

    Raises:
        FileNotFoundError / tifffile errors: propagated as-is if the file
            cannot be opened.
    """
    import tifffile  # noqa: PLC0415 — imported here so tests can stub it

    if rng is None:
        rng = np.random.default_rng(0)

    with tifffile.TiffFile(video_path) as tf:
        total_pages = len(tf.pages)

        if n_frames >= total_pages:
            indices = np.arange(total_pages)
        else:
            indices = rng.choice(total_pages, size=n_frames, replace=False)

        pages = [tf.pages[i].asarray() for i in sorted(indices)]

    return np.median(np.stack(pages, axis=0), axis=0).astype(np.float32)


def render_frame_background_composite(
    positions_lj,
    box,
    cfg,
    rng,
) -> np.ndarray:
    """Render one synthetic frame composited onto a real experimental background.

    Strategy:
        1. Load the pre-extracted background from ``cfg["_background_frame"]``.
        2. Build a Gaussian PSF kernel from the configured sigma.
        3. Convert LJ positions to pixel coordinates.
        4. Sample per-particle intensities from a log-normal distribution.
        5. Stamp particles onto a zero canvas, convolve with the PSF.
        6. Apply Poisson noise to the particle signal only.
        7. Add Gaussian read noise.
        8. Composite: background + particle signal.
        9. Clip to [0, 65535] and cast to uint16.

    Args:
        positions_lj: (N, 2) float array of particle positions in LJ units.
        box: (x_lo, x_hi, y_lo, y_hi) simulation box bounds.
        cfg: synthetic config dict.  Must contain ``_background_frame`` (set
            by render.py main before the frame loop), ``image_height``, and
            ``image_width``.  Optional sub-dicts: ``psf``, ``particle``,
            ``noise``.
        rng: numpy.random.Generator instance.

    Returns:
        uint16 numpy array of shape (H, W).

    Raises:
        KeyError: if ``cfg["_background_frame"]`` is absent.
    """
    # --- Background (must be pre-loaded by caller) -------------------------
    if "_background_frame" not in cfg:
        raise KeyError(
            "cfg['_background_frame'] not set — background must be pre-loaded "
            "in render.py main() before the frame loop"
        )
    background = cfg["_background_frame"]

    H = cfg["image_height"]
    W = cfg["image_width"]

    # --- PSF kernel (Gaussian) --------------------------------------------
    psf_cfg = cfg.get("psf", {})
    sigma = psf_cfg.get("sigma_px") or cfg.get("psf_sigma", 5.0)
    sigma = float(sigma)

    ksize = int(np.ceil(3 * sigma)) + 1
    impulse = np.zeros((2 * ksize + 1, 2 * ksize + 1), dtype=np.float32)
    impulse[ksize, ksize] = 1.0
    kernel = scipy.ndimage.gaussian_filter(impulse, sigma=sigma)
    kernel = kernel / kernel.sum()

    # --- LJ -> pixel coordinates ------------------------------------------
    x_lo, x_hi, y_lo, y_hi = box
    if len(positions_lj) == 0:
        pixel_positions = np.zeros((0, 2))
    else:
        px = np.clip((positions_lj[:, 0] - x_lo) / (x_hi - x_lo) * W, 0, W - 1)
        py = np.clip((positions_lj[:, 1] - y_lo) / (y_hi - y_lo) * H, 0, H - 1)
        pixel_positions = np.stack([px, py], axis=1)

    # --- Per-particle intensities (log-normal) ----------------------------
    particle_cfg = cfg.get("particle", {})
    peak_mean = float(particle_cfg.get("peak_mean", cfg.get("peak_intensity", 40000)))
    intensity_sigma = float(particle_cfg.get("intensity_sigma", 0.3))

    if len(pixel_positions) > 0:
        intensities = rng.lognormal(
            mean=np.log(peak_mean),
            sigma=intensity_sigma,
            size=len(pixel_positions),
        )
    else:
        intensities = np.array([], dtype=np.float32)

    # --- Stamp particles onto canvas --------------------------------------
    canvas = np.zeros((H, W), dtype=np.float32)
    for (px, py), intensity in zip(pixel_positions, intensities):
        ix = int(np.clip(px, 0, W - 1))
        iy = int(np.clip(py, 0, H - 1))
        canvas[iy, ix] += float(intensity)

    frame = scipy.ndimage.convolve(canvas, kernel, mode="reflect").astype(np.float32)

    # --- Poisson noise on particle signal only ----------------------------
    frame = rng.poisson(np.clip(frame, 0, None)).astype(np.float32)

    # --- Gaussian read noise ---------------------------------------------
    noise_cfg = cfg.get("noise", {})
    read_noise = float(noise_cfg.get("read_noise", cfg.get("readout_noise", 15.0)))
    frame = frame + rng.normal(0.0, read_noise, (H, W)).astype(np.float32)

    # --- Composite onto background ----------------------------------------
    result = background.astype(np.float32) + frame

    return np.clip(result, 0, 65535).astype(np.uint16)
