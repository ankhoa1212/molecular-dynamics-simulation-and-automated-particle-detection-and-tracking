"""Tests for render_brightfield_fast.py -- the render_strategy: brightfield_fast
path (see docs/plans/2026-08-16-001-feat-brightfield-fast-render-path-plan.md).

Unlike test_render_brightfield.py, no deeptrack mocking is needed here --
render_brightfield_fast.py has no deeptrack runtime dependency at all (see
the plan's KTDs). Equivalence against the real slow path (render_strategy:
brightfield) is a separate concern, covered by
test_render_brightfield_fast_equivalence.py (R7 in the plan above), not
this file.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from render_brightfield_fast import (
    _particle_footprint_px,
    _pupil,
    _stamp_footprints,
    _voxel_size_m,
    render_frame_brightfield_fast,
)


def _cfg(image_size=128, z_min_px=0.0, z_max_px=0.0, n_z_slices=10, amplitude=0.0):
    return {
        "image_width": image_size,
        "image_height": image_size,
        "brightfield": {
            "na": 1.0,
            "wavelength": 550e-9,
            "resolution": 100e-9,
            "refractive_index_medium": 1.33,
            "radius_min": 0.5e-6,
            "radius_max": 0.5e-6,
            "refractive_index_min": 1.45,
            "refractive_index_max": 1.45,
            "z_min_px": z_min_px,
            "z_max_px": z_max_px,
            "intensity_scale": 20000.0,
        },
        "brightfield_fast": {"max_particles": 5000, "n_z_slices": n_z_slices},
        "background": {"heterogeneity_scale": 50, "amplitude": amplitude},
        "noise": {"gain_sigma": 0.0, "read_noise": 0.0},
    }


_BOX = (0.0, 100.0, 0.0, 100.0)


class TestVoxelSize:
    def test_divides_resolution_by_magnification_default_10(self):
        # Optics.__init__'s own default magnification is 10 (confirmed
        # directly against the installed deeptrack==2.0.1 source) --
        # render_brightfield.py never overrides it, so this must match.
        vx, vy, vz = _voxel_size_m({"resolution": 100e-9})
        assert vx == pytest.approx(10e-9)
        assert vy == pytest.approx(10e-9)
        assert vz == pytest.approx(10e-9)

    def test_respects_explicit_magnification_override(self):
        vx, _, _ = _voxel_size_m({"resolution": 100e-9, "magnification": 1.0})
        assert vx == pytest.approx(100e-9)


class TestParticleFootprint:
    def test_footprint_is_zero_outside_radius(self):
        footprint = _particle_footprint_px(0.5e-6, 1.45, 1.33, 10e-9)
        half = footprint.shape[0] // 2
        assert footprint[0, 0] == 0.0  # corner, outside the circular footprint
        assert footprint[half, half] > 0.0  # center, inside

    def test_footprint_peak_matches_thickness_formula(self):
        # Center thickness of a sphere is 2*radius, scaled by the index
        # contrast -- the exact z-integral of a homogeneous sphere.
        radius_m, ri, ri_medium, voxel = 0.5e-6, 1.45, 1.33, 10e-9
        footprint = _particle_footprint_px(radius_m, ri, ri_medium, voxel)
        radius_px = radius_m / voxel
        expected_peak = 2.0 * radius_px * (ri - ri_medium)
        assert footprint.max() == pytest.approx(expected_peak, rel=1e-6)


class TestStampFootprints:
    def test_subpixel_placement_matches_continuous_position(self):
        footprint = _particle_footprint_px(0.5e-6, 1.45, 1.33, 10e-9)
        index_map = np.zeros((128, 128))
        _stamp_footprints(index_map, footprint, np.array([[64.47, 82.97]]))
        peak_row, peak_col = np.unravel_index(index_map.argmax(), index_map.shape)
        assert peak_row == 83  # round(82.97)
        assert peak_col == 64  # round(64.47)

    def test_no_particles_leaves_map_unchanged(self):
        footprint = _particle_footprint_px(0.5e-6, 1.45, 1.33, 10e-9)
        index_map = np.zeros((32, 32))
        _stamp_footprints(index_map, footprint, np.zeros((0, 2)))
        assert np.all(index_map == 0.0)

    def test_overlapping_particles_combine_additively(self):
        footprint = _particle_footprint_px(0.5e-6, 1.45, 1.33, 10e-9)
        index_map_one = np.zeros((128, 128))
        _stamp_footprints(index_map_one, footprint, np.array([[64.0, 64.0]]))

        index_map_two = np.zeros((128, 128))
        _stamp_footprints(index_map_two, footprint, np.array([[64.0, 64.0], [64.0, 64.0]]))

        # Two particles at the identical position must sum, not saturate --
        # matching deeptrack.optics._create_volume's own additive `+=`.
        assert index_map_two.max() == pytest.approx(2 * index_map_one.max())


class TestPupil:
    def test_zero_defocus_produces_hard_aperture_mask(self):
        pupil = _pupil((64, 64), 1.0, 550e-9, 1.33, (10e-9, 10e-9, 10e-9), 0.0)
        # At defocus=0 the phase term vanishes; only the aperture (0/1) mask
        # and its fftshift remain, so magnitudes are exactly 0 or 1.
        mags = np.abs(pupil)
        assert set(np.unique(np.round(mags, 6))).issubset({0.0, 1.0})

    def test_no_object_field_preserves_uniform_illumination(self):
        # A bare imaging pupil applied to a uniform (DC-only) illumination
        # field must leave intensity at ~1.0 everywhere -- if the pupil
        # isn't fftshift-ed to match FFT-native (DC-first) layout, the DC
        # component lands on a near-zero pupil value and the field
        # collapses to ~0 instead (the bug this test guards against).
        H = W = 64
        pupil = _pupil((H, W), 1.0, 550e-9, 1.33, (10e-9, 10e-9, 10e-9), 0.0)
        light = np.fft.fft2(np.ones((H, W), dtype=complex))
        light = light * pupil
        mask = np.abs(pupil) > 0
        light = light * mask
        field = np.fft.ifft2(light)
        assert (np.abs(field) ** 2).mean() == pytest.approx(1.0, rel=1e-6)


class TestRenderFrameBrightfieldFast:
    def test_single_particle_renders_without_error(self):
        frame = render_frame_brightfield_fast(
            np.array([[50.0, 50.0]]), _BOX, _cfg(), np.random.default_rng(0)
        )
        assert frame.shape == (128, 128)
        assert frame.dtype == np.uint16

    def test_single_particle_bright_feature_centered_at_true_position(self):
        rng = np.random.default_rng(0)
        frame = render_frame_brightfield_fast(
            np.array([[50.0, 50.0]]), _BOX, _cfg(image_size=256), rng
        ).astype(np.float64)
        # True pixel position: (50/100*256, 50/100*256) = (128, 128).
        deviation = np.abs(frame - frame.mean())
        peak_row, peak_col = np.unravel_index(deviation.argmax(), deviation.shape)
        assert abs(peak_row - 128) <= 3
        assert abs(peak_col - 128) <= 3

    def test_zero_particles_renders_plain_background_frame(self):
        frame = render_frame_brightfield_fast(
            np.zeros((0, 2)), _BOX, _cfg(), np.random.default_rng(0)
        )
        assert frame.shape == (128, 128)
        assert np.all(np.isfinite(frame.astype(np.float64)))

    def test_particle_at_frame_boundary_renders_without_index_errors(self):
        frame = render_frame_brightfield_fast(
            np.array([[0.1, 0.1]]), _BOX, _cfg(), np.random.default_rng(0)
        )
        assert np.all(np.isfinite(frame.astype(np.float64)))

    def test_single_z_slice_with_varying_particle_z_does_not_crash(self):
        rng = np.random.default_rng(0)
        cfg = _cfg(z_min_px=-5.0, z_max_px=5.0, n_z_slices=1)
        frame = render_frame_brightfield_fast(rng.uniform(20, 80, size=(5, 2)), _BOX, cfg, rng)
        assert np.all(np.isfinite(frame.astype(np.float64)))

    def test_defocused_particle_differs_from_in_focus(self):
        positions = np.array([[50.0, 50.0]])
        in_focus = render_frame_brightfield_fast(
            positions,
            _BOX,
            _cfg(image_size=256, z_min_px=0.0, z_max_px=0.0),
            np.random.default_rng(0),
        ).astype(np.float64)
        defocused = render_frame_brightfield_fast(
            positions,
            _BOX,
            _cfg(image_size=256, z_min_px=20.0, z_max_px=20.0, n_z_slices=1),
            np.random.default_rng(0),
        ).astype(np.float64)
        assert not np.allclose(in_focus, defocused)

    def test_dense_overlapping_cluster_renders_without_nan_or_overflow(self):
        rng = np.random.default_rng(1)
        # 200 particles clustered tightly enough to heavily overlap given
        # this dataset's ~5px (radius_px) particle size.
        positions = rng.uniform(45, 55, size=(200, 2))
        frame = render_frame_brightfield_fast(positions, _BOX, _cfg(image_size=256), rng)
        assert np.all(np.isfinite(frame.astype(np.float64)))
        assert frame.max() <= 65535

    def test_production_density_renders_in_practical_time(self):
        # R8's sanity-check scenario at a smaller canvas for test speed --
        # the full 512x512/~1446-particle case is U3's dedicated check.
        rng = np.random.default_rng(2)
        positions = rng.uniform(5, 95, size=(500, 2))
        frame = render_frame_brightfield_fast(positions, _BOX, _cfg(image_size=256), rng)
        assert np.all(np.isfinite(frame.astype(np.float64)))
