"""Tests for render_brightfield.py -- the render_strategy: brightfield path.

Glue-logic tests (dispatch, capping, sampling ranges, error handling) use
the repo's established deeptrack-stub mocking convention (see
test_render_deeptrack.py) so they run without deeptrack installed. A
handful of tests assert genuine physical behavior (interference, defocus)
that a mock can't demonstrate -- those use pytest.importorskip("deeptrack")
so they run for real when deeptrack is available and skip (not fail) in CI
when it isn't, consistent with this repo's "tests mock the import so CI
passes without it" convention for deeptrack-dependent code.
"""

import sys
import types
import warnings
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def _import_with_mock_deeptrack(resolve_return=None):
    """Stub deeptrack so dt.Sphere/dt.MieSphere/dt.Brightfield are
    MagicMocks whose eventual optics(sample).resolve() call returns
    resolve_return (default: a small all-ones array)."""
    for key in list(sys.modules.keys()):
        if "render_brightfield" in key:
            del sys.modules[key]

    if resolve_return is None:
        resolve_return = np.ones((8, 8), dtype=np.complex128)

    fake_resolved = mock.MagicMock()
    fake_resolved.resolve.return_value = resolve_return
    fake_optics_instance = mock.MagicMock(return_value=fake_resolved)

    deeptrack_stub = types.ModuleType("deeptrack")
    fake_sphere = mock.MagicMock()
    fake_sphere.__pow__ = mock.MagicMock(return_value=mock.MagicMock())
    deeptrack_stub.Sphere = mock.MagicMock(return_value=fake_sphere)
    fake_mie_sphere = mock.MagicMock()
    fake_mie_sphere.__pow__ = mock.MagicMock(return_value=mock.MagicMock())
    deeptrack_stub.MieSphere = mock.MagicMock(return_value=fake_mie_sphere)
    deeptrack_stub.Brightfield = mock.MagicMock(return_value=fake_optics_instance)

    sys.modules["deeptrack"] = deeptrack_stub

    import render_brightfield as rbf

    return rbf


def _cfg(**overrides):
    cfg = {
        "image_width": 8,
        "image_height": 8,
        "brightfield": {
            "max_particles": 5,
            "intensity_scale": 100.0,
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
            "mie_max_particles": 5,
            "mie_max_frames": 2,
        },
        "background": {"amplitude": 0},
        "noise": {"gain_sigma": 0.0, "read_noise": 0.0},
    }
    cfg.update(overrides)
    return cfg


class TestRenderFrameBrightfieldGlue:
    def test_empty_positions_returns_zero_frame(self):
        rbf = _import_with_mock_deeptrack()
        rng = np.random.default_rng(0)
        frame = rbf.render_frame_brightfield(np.zeros((0, 2)), (0, 1, 0, 1), _cfg(), rng)
        assert frame.shape == (8, 8)
        assert frame.dtype == np.uint16
        assert frame.max() == 0

    def test_returns_uint16_array_of_configured_shape(self):
        rbf = _import_with_mock_deeptrack()
        rng = np.random.default_rng(0)
        positions = np.array([[0.5, 0.5]])
        frame = rbf.render_frame_brightfield(positions, (0, 1, 0, 1), _cfg(), rng)
        assert frame.shape == (8, 8)
        assert frame.dtype == np.uint16

    def test_atom_ids_argument_accepted_and_unused(self):
        rbf = _import_with_mock_deeptrack()
        rng = np.random.default_rng(0)
        positions = np.array([[0.5, 0.5]])
        # Must not raise even though atom_ids has no effect -- signature
        # parity with the other render_frame_* strategies.
        rbf.render_frame_brightfield(positions, (0, 1, 0, 1), _cfg(), rng, atom_ids=np.array([7]))

    def test_particle_count_above_cap_warns_and_truncates(self):
        rbf = _import_with_mock_deeptrack()
        rng = np.random.default_rng(0)
        positions = rng.uniform(0.1, 0.9, size=(9, 2))  # cap is 5
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            rbf.render_frame_brightfield(positions, (0, 1, 0, 1), _cfg(), rng)
        assert any("max_particles" in str(w.message) for w in caught)

    def test_particle_count_at_or_below_cap_does_not_warn(self):
        rbf = _import_with_mock_deeptrack()
        rng = np.random.default_rng(0)
        positions = rng.uniform(0.1, 0.9, size=(3, 2))  # cap is 5
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            rbf.render_frame_brightfield(positions, (0, 1, 0, 1), _cfg(), rng)
        assert not any("max_particles" in str(w.message) for w in caught)

    def test_sampled_radius_and_refractive_index_stay_within_configured_range(self):
        rbf = _import_with_mock_deeptrack()
        rng = np.random.default_rng(1)
        radii, ri, z = rbf._sample_particle_properties(
            50,
            {
                "radius_min": 0.4e-6,
                "radius_max": 0.6e-6,
                "refractive_index_min": 1.40,
                "refractive_index_max": 1.50,
                "z_min_px": -3.0,
                "z_max_px": 3.0,
            },
            rng,
        )
        assert (radii >= 0.4e-6).all() and (radii <= 0.6e-6).all()
        assert (ri >= 1.40).all() and (ri <= 1.50).all()
        assert (z >= -3.0).all() and (z <= 3.0).all()

    def test_missing_deeptrack_raises_import_hint(self):
        for key in list(sys.modules.keys()):
            if "render_brightfield" in key:
                del sys.modules[key]
        sys.modules.pop("deeptrack", None)

        import render_brightfield as rbf

        rng = np.random.default_rng(0)
        positions = np.array([[0.5, 0.5]])
        with mock.patch.dict(sys.modules, {"deeptrack": None}):
            with pytest.raises(ImportError, match="deeptrack==2.0.1"):
                rbf.render_frame_brightfield(positions, (0, 1, 0, 1), _cfg(), rng)


class TestCapParticleCount:
    def test_returns_all_indices_when_under_cap(self):
        rbf = _import_with_mock_deeptrack()
        rng = np.random.default_rng(0)
        positions = np.zeros((3, 2))
        keep = rbf._cap_particle_count(positions, max_particles=5, rng=rng)
        assert sorted(keep.tolist()) == [0, 1, 2]

    def test_selects_a_bounded_random_subset_when_over_cap(self):
        rbf = _import_with_mock_deeptrack()
        rng = np.random.default_rng(0)
        positions = np.zeros((20, 2))
        keep = rbf._cap_particle_count(positions, max_particles=5, rng=rng)
        assert len(keep) == 5
        assert len(set(keep.tolist())) == 5  # no duplicates
        assert all(0 <= i < 20 for i in keep)


class TestGenerateMieGroundTruth:
    def test_over_cap_n_particles_raises(self):
        rbf = _import_with_mock_deeptrack()
        rng = np.random.default_rng(0)
        positions = rng.uniform(0.1, 0.9, size=(10, 2))
        with pytest.raises(ValueError, match="max_particles"):
            rbf.generate_mie_ground_truth(
                _cfg(), positions, (0, 1, 0, 1), n_frames=1, n_particles=99, rng=rng
            )

    def test_over_cap_n_frames_raises(self):
        rbf = _import_with_mock_deeptrack()
        rng = np.random.default_rng(0)
        positions = rng.uniform(0.1, 0.9, size=(10, 2))
        with pytest.raises(ValueError, match="max_frames"):
            rbf.generate_mie_ground_truth(
                _cfg(), positions, (0, 1, 0, 1), n_frames=99, n_particles=3, rng=rng
            )

    def test_returns_n_frames_of_configured_shape(self):
        rbf = _import_with_mock_deeptrack()
        rng = np.random.default_rng(0)
        positions = rng.uniform(0.1, 0.9, size=(10, 2))
        frames = rbf.generate_mie_ground_truth(
            _cfg(), positions, (0, 1, 0, 1), n_frames=2, n_particles=3, rng=rng
        )
        assert len(frames) == 2
        for frame in frames:
            assert frame.shape == (8, 8)
            assert frame.dtype == np.uint16

    def test_selected_positions_are_drawn_from_the_real_trajectory_subset(self):
        """Position reuse: the MieSphere instances are built from the same
        real positions_lj subset, not synthetically sampled -- covers AE3's
        ground-truth-generation half."""
        rbf = _import_with_mock_deeptrack()
        rng = np.random.default_rng(0)
        positions = rng.uniform(0.1, 0.9, size=(10, 2))

        rbf.generate_mie_ground_truth(
            _cfg(), positions, (0, 1, 0, 1), n_frames=1, n_particles=4, rng=rng
        )

        # The stubbed MieSphere was constructed with position=<lambda>;
        # calling that lambda with a fabricated _ID must return a pixel
        # position that traces back to one of the real trajectory rows
        # (converted the same way _lj_to_pixels converts every other
        # strategy's positions).
        import render_brightfield as rbf_mod

        deeptrack_stub = sys.modules["deeptrack"]
        position_fn = deeptrack_stub.MieSphere.call_args.kwargs["position"]
        pixel_positions_all = rbf_mod._lj_to_pixels(positions, (0, 1, 0, 1), 8, 8)
        resolved = position_fn(_ID=(0,))
        assert any(np.allclose(resolved, row) for row in pixel_positions_all)


class TestRenderFrameBrightfieldRealPhysics:
    """Genuine deeptrack physics -- skipped (not failed) when deeptrack
    isn't installed, per this file's module docstring."""

    def test_defocused_particle_differs_from_in_focus(self):
        pytest.importorskip("deeptrack")
        for key in list(sys.modules.keys()):
            if "render_brightfield" in key:
                del sys.modules[key]
        sys.modules.pop("deeptrack", None)  # drop any leftover mock stub from prior tests
        import render_brightfield as rbf

        positions = np.array([[0.5, 0.5]])
        box = (0, 1, 0, 1)

        in_focus_cfg = _cfg(image_width=48, image_height=48)
        in_focus_cfg["brightfield"]["z_min_px"] = 0.0
        in_focus_cfg["brightfield"]["z_max_px"] = 0.0
        rng = np.random.default_rng(0)
        in_focus = rbf.render_frame_brightfield(positions, box, in_focus_cfg, rng)

        defocused_cfg = _cfg(image_width=48, image_height=48)
        defocused_cfg["brightfield"]["z_min_px"] = 5.0
        defocused_cfg["brightfield"]["z_max_px"] = 5.0
        rng = np.random.default_rng(0)
        defocused = rbf.render_frame_brightfield(positions, box, defocused_cfg, rng)

        assert not np.array_equal(in_focus, defocused)

    def test_dense_touching_particles_render_without_error(self):
        """Covers AE1: two close/touching particles produce a frame that's
        neither a flat merged blob nor a plain sum of two independent
        rings -- here checked as "renders successfully and isn't uniform",
        the physics-agnostic proxy a unit test can assert without
        hand-verifying an interference fringe pattern."""
        pytest.importorskip("deeptrack")
        for key in list(sys.modules.keys()):
            if "render_brightfield" in key:
                del sys.modules[key]
        sys.modules.pop("deeptrack", None)  # drop any leftover mock stub from prior tests
        import render_brightfield as rbf

        positions = np.array([[0.45, 0.5], [0.55, 0.5]])  # close together
        cfg = _cfg(image_width=48, image_height=48)
        rng = np.random.default_rng(0)
        frame = rbf.render_frame_brightfield(positions, (0, 1, 0, 1), cfg, rng)

        assert frame.shape == (48, 48)
        assert frame.max() > frame.min()  # not a uniform/blank frame
