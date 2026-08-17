"""Fast-vs-slow equivalence tests for render_brightfield_fast.py (R7, R8 in
docs/plans/2026-08-16-001-feat-brightfield-fast-render-path-plan.md).

Compares render_frame_brightfield_fast against the real, deeptrack-backed
render_frame_brightfield at particle counts the slow path can still complete
in test time (N=1-45), via compare_renders.compute_ssim_similarity. Requires
a real deeptrack install (pytest.importorskip below) -- there is no mocked
path here, since the whole point is validating against genuine physics.

SSIM threshold: this module pins **SSIM >= 0.7**, matching the plan's
original placeholder (arrived at independently, not just reused). The
threshold has moved twice since: down to 0.35 once real multi-particle
scenes were measured after fixing an FFT-padding bug and a magnification
bug (see render_brightfield_fast.py's module docstring and
render_brightfield.py's _resolve_brightfield_intensity docstring), then
back up to a real 0.7 once two more things were fixed:

1. render_brightfield.py/render_brightfield_fast.py both now apply
   _apply_partial_coherence_blur to the resolved intensity, suppressing
   the multi-ring Airy diffraction pattern real reference images
   (data-setup/models/lodestar_model_15/, lodestar_model_10/ crops) don't
   show -- this incidentally also removed a real source of fast-vs-slow
   disagreement (the exact ring structure was one of the harder things to
   match pixel-for-pixel), substantially improving true equivalence.
2. `_ssim_for` below now renders through `_NoShotNoiseRNG`, neutralizing
   Poisson shot noise so the comparison measures structure, not two
   independent per-path noise realizations -- the blur's reduced signal
   contrast made uncontrolled shot noise dominate the un-neutralized score
   entirely (N=1 in-focus measured SSIM 0.25 noisy vs 0.99 neutralized).

With both applied, real measured SSIM is 0.99+ for single particles and
0.79-0.87 for realistic (non-touching, production-scale) multi-particle
scenes -- a real, honest improvement over the earlier 0.32-0.46 floor, not
just a re-derivation of the same numbers under a different metric. 0.7 is
set below the lowest observed measurement (0.79, N=20 in-focus) rather than
at the average, so the gate stays meaningful against genuine regressions.

Uniform-random positions are deliberately NOT used for the N=20 scenarios:
coherent interference is chaotic near-degenerate configurations, and random
uniform sampling can produce near-touching particle pairs by chance that
real LAMMPS trajectory data never would (particles stay separated by their
interaction potential's excluded volume). _non_overlapping_positions below
enforces a minimum separation, matching that real-data property, while the
dedicated dense-cluster scenario (N=45) deliberately tests the
touching/overlapping case R2 claims to handle.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

pytest.importorskip("deeptrack")

from compare_renders import compute_ssim_similarity  # noqa: E402
from render_brightfield import render_frame_brightfield  # noqa: E402
from render_brightfield_fast import render_frame_brightfield_fast  # noqa: E402

SSIM_THRESHOLD = 0.7

_BOX = (0.0, 100.0, 0.0, 100.0)


def _cfg(image_size=256, z_min_px=0.0, z_max_px=0.0, n_z_slices=10):
    return {
        "image_width": image_size,
        "image_height": image_size,
        "brightfield": {
            "na": 1.0,
            "wavelength": 550e-9,
            "resolution": 100e-9,
            "magnification": 1.0,  # matches config.yaml's production value -- see
            # render_brightfield.py's _resolve_brightfield_intensity docstring.
            # The deeptrack library default (10) renders particles at 50px radius,
            # ~10x this dataset's real interparticle spacing, so a comparison at
            # that default scale wouldn't exercise a realistic, well-separated
            # multi-particle scene at all.
            "refractive_index_medium": 1.33,
            "radius_min": 0.5e-6,
            "radius_max": 0.5e-6,
            "refractive_index_min": 1.45,
            "refractive_index_max": 1.45,
            "z_min_px": z_min_px,
            "z_max_px": z_max_px,
            "intensity_scale": 20000.0,
            "max_particles": 100,
            "mie_max_particles": 5,
            "mie_max_frames": 2,
        },
        "brightfield_fast": {"max_particles": 5000, "n_z_slices": n_z_slices},
        "background": {"amplitude": 0},
        "noise": {"gain_sigma": 0.0, "read_noise": 0.0},
    }


def _non_overlapping_positions(n, lo, hi, min_sep, seed):
    """Rejection-sample n positions in [lo, hi]^2 with pairwise separation
    >= min_sep -- see module docstring for why this (not uniform-random) is
    used for the "well-separated, non-integer positions" scenarios."""
    rng = np.random.default_rng(seed)
    pts = []
    attempts = 0
    while len(pts) < n and attempts < 20000:
        attempts += 1
        p = rng.uniform(lo, hi, size=2)
        if all(np.hypot(*(p - q)) >= min_sep for q in pts):
            pts.append(p)
    assert len(pts) == n, f"rejection sampling only found {len(pts)}/{n} positions"
    return np.array(pts)


class _NoShotNoiseRNG:
    """Proxy around a real Generator that neutralizes Poisson shot noise
    (returns the mean/lambda unchanged) so two renders of the identical
    scene are directly, deterministically comparable -- everything else
    (particle-property sampling, gain sampling) passes through unchanged.

    Needed because gain_sigma/read_noise=0 in _cfg still leaves the shared
    noise tail's `rng.poisson(photons)` call active (Poisson shot noise is
    unconditional), and the two render paths don't consume rng identically
    before reaching it, so their independent noise realizations are
    uncorrelated. That was a minor SSIM-suppressing factor before
    _apply_partial_coherence_blur; post-blur, the reduced signal contrast
    makes uncontrolled shot noise dominate the score entirely (confirmed
    directly: N=1 in-focus measured SSIM 0.25 noisy vs 0.99 with this
    neutralized) -- exactly the noise floor this equivalence gate isn't
    meant to be testing.
    """

    def __init__(self, rng):
        self._rng = rng

    def poisson(self, lam):
        return np.asarray(lam, dtype=np.float64)

    def __getattr__(self, name):
        return getattr(self._rng, name)


def _ssim_for(positions, cfg, seed):
    rng_slow = _NoShotNoiseRNG(np.random.default_rng(seed))
    slow = render_frame_brightfield(positions, _BOX, cfg, rng_slow).astype(np.float64)
    rng_fast = _NoShotNoiseRNG(np.random.default_rng(seed))
    fast = render_frame_brightfield_fast(positions, _BOX, cfg, rng_fast).astype(np.float64)
    return compute_ssim_similarity(slow, fast)


class TestSingleParticleEquivalence:
    def test_in_focus(self):
        positions = np.array([[50.37, 50.82]])  # non-integer, per R7's subpixel requirement
        ssim = _ssim_for(positions, _cfg(), seed=10)
        assert ssim >= SSIM_THRESHOLD

    def test_defocused(self):
        positions = np.array([[50.37, 50.82]])
        ssim = _ssim_for(positions, _cfg(z_min_px=5.0, z_max_px=5.0), seed=11)
        assert ssim >= SSIM_THRESHOLD


class TestMultiParticleEquivalence:
    def test_in_focus_n20_non_overlapping(self):
        positions = _non_overlapping_positions(20, 15, 85, min_sep=6.0, seed=20)
        ssim = _ssim_for(positions, _cfg(), seed=20)
        assert ssim >= SSIM_THRESHOLD

    def test_defocused_n20_spanning_full_z_range(self):
        positions = _non_overlapping_positions(20, 15, 85, min_sep=6.0, seed=20)
        cfg = _cfg(z_min_px=-5.0, z_max_px=5.0, n_z_slices=10)
        ssim = _ssim_for(positions, cfg, seed=21)
        assert ssim >= SSIM_THRESHOLD

    def test_dense_touching_cluster_n45(self):
        """Directly exercises R2's overlap-handling claim: 45 particles
        packed into a small region (heavy footprint overlap given this
        dataset's ~5px particle radius), near the slow path's own
        practical N ceiling."""
        rng = np.random.default_rng(30)
        positions = rng.uniform(20, 44, size=(45, 2))
        ssim = _ssim_for(positions, _cfg(image_size=64), seed=30)
        assert ssim >= SSIM_THRESHOLD


class TestZBucketGranularityDiagnostic:
    """Diagnostic (not a pass/fail correctness gate): confirms that
    increasing n_z_slices does not blow up the defocused SSIM, i.e. any
    gap between fast and slow in the defocused case is a K/voxel-
    granularity mismatch (documented KTD in render_brightfield_fast.py),
    not a coarse-K implementation bug. See module docstring for how this
    was used to help interpret the pinned threshold above."""

    def test_ssim_does_not_collapse_as_k_increases(self):
        positions = _non_overlapping_positions(20, 15, 85, min_sep=6.0, seed=20)
        ssim_k10 = _ssim_for(positions, _cfg(z_min_px=-5.0, z_max_px=5.0, n_z_slices=10), seed=21)
        ssim_k20 = _ssim_for(positions, _cfg(z_min_px=-5.0, z_max_px=5.0, n_z_slices=20), seed=21)
        assert ssim_k20 >= ssim_k10 - 0.1


class TestProductionDensitySanity:
    """R8: one full-density (~1446-particle) frame renders without NaN/Inf
    and completes in a time the implementer judges practical. There is no
    slow-path baseline at this density to assert equivalence against --
    this is a one-time manual/CI sanity gate, not an automated
    correctness check (see the plan's own note on this scenario)."""

    def test_full_density_frame_renders_sanely(self):
        box = (0.0, 300.0, 0.0, 300.0)
        cfg = {
            "image_width": 512,
            "image_height": 512,
            "brightfield": {
                "na": 1.0,
                "wavelength": 550e-9,
                "resolution": 100e-9,
                "refractive_index_medium": 1.33,
                "radius_min": 0.5e-6,
                "radius_max": 0.5e-6,
                "refractive_index_min": 1.45,
                "refractive_index_max": 1.45,
                "z_min_px": 0.0,
                "z_max_px": 0.0,
                "intensity_scale": 20000.0,
            },
            "brightfield_fast": {"max_particles": 5000, "n_z_slices": 10},
            "background": {"heterogeneity_scale": 50, "amplitude": 500},
            "noise": {"gain_sigma": 0.02, "read_noise": 15.0},
        }
        rng = np.random.default_rng(0)
        positions = rng.uniform(10, 290, size=(1446, 2))
        frame = render_frame_brightfield_fast(positions, box, cfg, rng)
        assert frame.shape == (512, 512)
        assert frame.dtype == np.uint16
        assert np.all(np.isfinite(frame.astype(np.float64)))
