"""Domain randomization rendering strategy for render.py.

Samples PSF sigma, peak intensity, and readout noise independently per
frame from uniform distributions specified in the config.  Uses the
existing procedural ``render_frame`` internally — no deeptrack dependency
required.

This mode is intended for fast augmentation robustness evaluation: by
sampling a distribution of appearances for each frame, a detector trained
(or benchmarked) on randomized renders must generalize over a range of
imaging conditions rather than a single fixed parameter set.

Configuration (under ``synthetic.randomization`` in config.yaml)::

    randomization:
      psf_sigma_range: [3.0, 7.0]        # Uniform(min, max) for PSF sigma (px)
      peak_range: [20000, 60000]          # Uniform(min, max) for peak intensity (ADU)
      readout_noise_range: [10.0, 25.0]  # Uniform(min, max) for readout noise std (ADU)

Raises:
    ValueError: if any range has min > max (validated before rendering).
"""

import sys
from pathlib import Path

# render.py lives in the same directory; ensure it is importable.
sys.path.insert(0, str(Path(__file__).parent))
from render import render_frame


def render_frame_randomized(positions_lj, box, cfg, rng):
    """Render one synthetic frame with per-frame randomized imaging parameters.

    Reads randomization ranges from ``cfg.get('randomization', {})``.
    Samples PSF sigma, peak intensity, and readout noise uniformly within
    those ranges, then delegates to the procedural ``render_frame`` with
    an overridden cfg dict.

    Args:
        positions_lj: (N, 2) array of particle positions in LJ units.
        box: (x_lo, x_hi, y_lo, y_hi) simulation box bounds.
        cfg: synthetic config dict (may contain a ``randomization`` sub-dict).
        rng: numpy random Generator — caller is responsible for seeding.

    Returns:
        uint16 numpy array of shape (H, W).

    Raises:
        ValueError: if any range has min > max.
    """
    r = cfg.get("randomization", {})

    sigma_min, sigma_max = r.get("psf_sigma_range", [3.0, 7.0])
    peak_min, peak_max = r.get("peak_range", [20000, 60000])
    noise_min, noise_max = r.get("readout_noise_range", [10.0, 25.0])

    # Validate ranges before any rendering
    if sigma_min > sigma_max:
        raise ValueError(
            f"psf_sigma_range min ({sigma_min}) > max ({sigma_max}): "
            "lower bound must be <= upper bound."
        )
    if peak_min > peak_max:
        raise ValueError(
            f"peak_range min ({peak_min}) > max ({peak_max}): "
            "lower bound must be <= upper bound."
        )
    if noise_min > noise_max:
        raise ValueError(
            f"readout_noise_range min ({noise_min}) > max ({noise_max}): "
            "lower bound must be <= upper bound."
        )

    # Sample per-frame parameters
    sigma = rng.uniform(sigma_min, sigma_max)
    peak = rng.uniform(peak_min, peak_max)
    noise = rng.uniform(noise_min, noise_max)

    # Build a per-frame cfg override: copy base cfg, update sampled params
    frame_cfg = dict(cfg)
    frame_cfg["psf_sigma"] = float(sigma)
    frame_cfg["peak_intensity"] = float(peak)
    frame_cfg["readout_noise"] = float(noise)

    return render_frame(positions_lj, box, frame_cfg, rng)
