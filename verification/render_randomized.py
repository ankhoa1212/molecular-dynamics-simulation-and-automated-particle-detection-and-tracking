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
      smoothing:
        step_std_fraction: 0.15           # random-walk step std, as a fraction of
                                           # each parameter's own (max - min) range;
                                           # only used when a real `state` dict is
                                           # passed (see render_frame_randomized).

Raises:
    ValueError: if any range has min > max (validated before rendering).
"""

import sys
from pathlib import Path

# render.py lives in the same directory; ensure it is importable.
sys.path.insert(0, str(Path(__file__).parent))
from render import render_frame

# Cap on reflection iterations in _reflect_into_range: guards against a
# pathological loop for degenerate config (e.g. an inverted/zero-width range
# slipping past validation) rather than looping indefinitely. A final clamp
# after the loop is the actual safety net (see plan's Key Decisions /
# U4 Technical design "clamp as a final safety net after reflecting once, or
# reflect iteratively" — this does both).
_MAX_REFLECTIONS = 64


def _reflect_into_range(value, lo, hi):
    """Reflect ``value`` back into ``[lo, hi]`` instead of hard-clipping.

    A hard clip (``max(lo, min(hi, value))``) dwells at the boundary: once a
    random-walk value is near an edge, every step that overshoots truncates
    to the same boundary value, producing multi-frame stretches pinned at a
    range extreme. Reflecting mirrors the overshoot back into range instead
    (``value = lo + (lo - value)`` on underflow, mirrored on overflow),
    which preserves the step's magnitude and keeps the stationary
    distribution from concentrating at the edges.

    A single reflection can still overshoot the *opposite* bound for a very
    large step, so this reflects iteratively (bounded by
    ``_MAX_REFLECTIONS``) and clamps as a final safety net.
    """
    for _ in range(_MAX_REFLECTIONS):
        if value < lo:
            value = lo + (lo - value)
        elif value > hi:
            value = hi - (value - hi)
        else:
            break
    return min(max(value, lo), hi)


def _sample_smoothed(rng, state, key, lo, hi, step_std_fraction):
    """Sample one parameter with bounded frame-to-frame continuity.

    On first use (``key`` absent from ``state``), bootstraps with an
    independent uniform sample over ``[lo, hi]`` — identical in
    distribution to the stateless path. On subsequent calls, takes a
    Gaussian step from the previous value and reflects it back into range.
    Writes the new value into ``state`` before returning it.
    """
    prev = state.get(key)
    if prev is None:
        value = rng.uniform(lo, hi)
    else:
        step_std = step_std_fraction * (hi - lo)
        step = rng.normal(0.0, step_std)
        value = _reflect_into_range(prev + step, lo, hi)
    state[key] = value
    return value


def render_frame_randomized(positions_lj, box, cfg, rng, state=None):
    """Render one synthetic frame with per-frame randomized imaging parameters.

    Reads randomization ranges from ``cfg.get('randomization', {})``.
    Samples PSF sigma, peak intensity, and readout noise, then delegates
    to the procedural ``render_frame`` with an overridden cfg dict.

    Args:
        positions_lj: (N, 2) array of particle positions in LJ units.
        box: (x_lo, x_hi, y_lo, y_hi) simulation box bounds.
        cfg: synthetic config dict (may contain a ``randomization`` sub-dict).
        rng: numpy random Generator — caller is responsible for seeding.
        state: optional dict carrying the previous frame's sampled values,
            keyed by ``"psf_sigma"``/``"peak_intensity"``/``"readout_noise"``.
            When ``None`` (the default), sampling is unchanged from before
            this parameter existed: a fresh independent
            ``rng.uniform()`` per parameter every call — this is
            load-bearing for backward compatibility (R9), not incidental.
            When a real dict is passed, each parameter takes a bounded,
            reflecting random-walk step from its previous value instead
            (R8), and the dict is mutated in place with the new values so
            the caller can reuse it across frames.

    Returns:
        uint16 numpy array of shape (H, W).

    Raises:
        ValueError: if any range has min > max. Raised before any
            sampling, regardless of whether ``state`` is passed.
    """
    r = cfg.get("randomization", {})

    sigma_min, sigma_max = r.get("psf_sigma_range", [3.0, 7.0])
    peak_min, peak_max = r.get("peak_range", [20000, 60000])
    noise_min, noise_max = r.get("readout_noise_range", [150.0, 300.0])
    step_std_fraction = r.get("smoothing", {}).get("step_std_fraction", 0.15)

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
    if step_std_fraction < 0:
        raise ValueError(
            f"smoothing.step_std_fraction ({step_std_fraction}) must be >= 0 -- "
            "a negative value would pass a negative scale to rng.normal, which raises."
        )

    if state is None:
        # Unchanged stateless behavior (R9): fresh independent sample every
        # call, in the same order as before this parameter existed — must
        # stay byte-identical for a fixed seed (test_fixed_seed_is_reproducible).
        sigma = rng.uniform(sigma_min, sigma_max)
        peak = rng.uniform(peak_min, peak_max)
        noise = rng.uniform(noise_min, noise_max)
    else:
        sigma = _sample_smoothed(rng, state, "psf_sigma", sigma_min, sigma_max, step_std_fraction)
        peak = _sample_smoothed(rng, state, "peak_intensity", peak_min, peak_max, step_std_fraction)
        noise = _sample_smoothed(
            rng, state, "readout_noise", noise_min, noise_max, step_std_fraction
        )

    # Build a per-frame cfg override: copy base cfg, update sampled params
    frame_cfg = dict(cfg)
    frame_cfg["psf_sigma"] = float(sigma)
    frame_cfg["peak_intensity"] = float(peak)
    frame_cfg["readout_noise"] = float(noise)
    # render_strategy: randomized stays black regardless of config.yaml's
    # background_fraction (a procedural-only knob) -- see
    # docs/superpowers/specs/2026-08-07-gray-background-default-design.md's Scope.
    frame_cfg["background_fraction"] = 0.0

    return render_frame(positions_lj, box, frame_cfg, rng)
