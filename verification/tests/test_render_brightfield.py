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


class TestApplyPartialCoherenceBlur:
    def test_default_sigma_blurs_a_sharp_feature(self):
        rbf = _import_with_mock_deeptrack()
        frame = np.zeros((32, 32))
        frame[16, 16] = 1000.0
        blurred = rbf._apply_partial_coherence_blur(frame, {})
        assert blurred[16, 16] < 1000.0  # peak spread out
        assert blurred[16, 17] > 0.0  # energy spread to neighbors

    def test_zero_sigma_disables_blur(self):
        rbf = _import_with_mock_deeptrack()
        frame = np.zeros((32, 32))
        frame[16, 16] = 1000.0
        blurred = rbf._apply_partial_coherence_blur(frame, {"coherence_blur_sigma_px": 0})
        assert np.array_equal(blurred, frame)

    def test_configured_sigma_overrides_default(self):
        rbf = _import_with_mock_deeptrack()
        frame = np.zeros((32, 32))
        frame[16, 16] = 1000.0
        light_blur = rbf._apply_partial_coherence_blur(frame, {"coherence_blur_sigma_px": 0.5})
        heavy_blur = rbf._apply_partial_coherence_blur(frame, {"coherence_blur_sigma_px": 5.0})
        assert heavy_blur[16, 16] < light_blur[16, 16]  # more blur, lower peak


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

    def test_atom_ids_without_state_does_not_raise(self):
        rbf = _import_with_mock_deeptrack()
        rng = np.random.default_rng(0)
        positions = np.array([[0.5, 0.5]])
        # atom_ids alone (no state) must not raise -- falls back to fresh
        # random sampling, same as neither being passed at all.
        rbf.render_frame_brightfield(positions, (0, 1, 0, 1), _cfg(), rng, atom_ids=np.array([7]))

    def test_atom_ids_and_state_keep_particle_properties_stable_across_frames(self):
        """End-to-end version of the _sample_particle_properties persistence
        test, through the full render_frame_brightfield entry point (the
        frame-to-frame flicker fix)."""
        rbf = _import_with_mock_deeptrack()
        rng = np.random.default_rng(0)
        positions = np.array([[0.5, 0.5], [0.7, 0.7]])
        atom_ids = np.array([1, 2])
        state = {}

        rbf.render_frame_brightfield(
            positions, (0, 1, 0, 1), _cfg(), rng, atom_ids=atom_ids, state=state
        )
        cached_after_frame_1 = dict(state["particle_properties"])

        rbf.render_frame_brightfield(
            positions, (0, 1, 0, 1), _cfg(), rng, atom_ids=atom_ids, state=state
        )
        cached_after_frame_2 = dict(state["particle_properties"])

        assert cached_after_frame_1 == cached_after_frame_2

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

    def test_atom_id_keyed_properties_stay_constant_across_calls(self):
        """Frame-to-frame flicker fix: with atom_ids/state given, the same
        physical particle's radius/refractive_index/z must be identical
        across separate calls (separate frames), not resampled fresh each
        time -- a real particle's depth/size doesn't randomly jump every
        frame."""
        rbf = _import_with_mock_deeptrack()
        rng = np.random.default_rng(0)
        bf_cfg = {
            "radius_min": 0.4e-6,
            "radius_max": 0.6e-6,
            "refractive_index_min": 1.40,
            "refractive_index_max": 1.50,
            "z_min_px": -13.0,
            "z_max_px": 13.0,
        }
        state = {}
        atom_ids = np.array([10, 20, 30])

        radii1, ri1, z1 = rbf._sample_particle_properties(
            3, bf_cfg, rng, atom_ids=atom_ids, state=state
        )
        radii2, ri2, z2 = rbf._sample_particle_properties(
            3, bf_cfg, rng, atom_ids=atom_ids, state=state
        )

        assert np.array_equal(radii1, radii2)
        assert np.array_equal(ri1, ri2)
        assert np.array_equal(z1, z2)

    def test_new_atom_id_seen_later_gets_sampled_and_cached(self):
        rbf = _import_with_mock_deeptrack()
        rng = np.random.default_rng(0)
        bf_cfg = {"z_min_px": -13.0, "z_max_px": 13.0}
        state = {}

        rbf._sample_particle_properties(2, bf_cfg, rng, atom_ids=np.array([1, 2]), state=state)
        assert set(state["particle_properties"].keys()) == {1, 2}

        _, _, z2 = rbf._sample_particle_properties(
            3, bf_cfg, rng, atom_ids=np.array([1, 2, 3]), state=state
        )
        assert set(state["particle_properties"].keys()) == {1, 2, 3}
        assert z2[2] == state["particle_properties"][3][2]

    def test_without_atom_ids_or_state_falls_back_to_fresh_random_sampling(self):
        """Preserves the original behavior for callers with no cross-frame
        continuity to preserve (e.g. calibrate_psf.py's search loop)."""
        rbf = _import_with_mock_deeptrack()
        rng1 = np.random.default_rng(0)
        rng2 = np.random.default_rng(0)
        bf_cfg = {"z_min_px": -13.0, "z_max_px": 13.0}

        _, _, z_a = rbf._sample_particle_properties(5, bf_cfg, rng1)
        _, _, z_b = rbf._sample_particle_properties(5, bf_cfg, rng1)
        _, _, z_c = rbf._sample_particle_properties(5, bf_cfg, rng2)

        assert not np.array_equal(z_a, z_b)  # two independent fresh draws differ
        assert np.array_equal(z_a, z_c)  # same rng seed/sequence position reproduces

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

    def test_single_particle_secondary_ring_suppressed_relative_to_primary_feature(self):
        """Task: suppress the unrealistic outer diffraction ring. A single
        in-focus particle's radially-averaged intensity profile has a
        primary feature (core + first ring) near the particle center and,
        without _apply_partial_coherence_blur, a real secondary bright-ring
        bump well beyond it that real reference images
        (data-setup/models/lodestar_model_15/, lodestar_model_10/ crops)
        don't show. Asserts the outer secondary-ring deviation from
        background stays small relative to the primary feature's own
        deviation -- confirmed empirically to separate blurred (~0.11) from
        unblurred (~0.29) at this dataset's particle scale."""
        pytest.importorskip("deeptrack")
        for key in list(sys.modules.keys()):
            if "render_brightfield" in key:
                del sys.modules[key]
        sys.modules.pop("deeptrack", None)
        import render_brightfield as rbf

        cfg = {
            "image_width": 64,
            "image_height": 64,
            "brightfield": {
                "na": 1.0,
                "wavelength": 550e-9,
                "resolution": 100e-9,
                "magnification": 1.0,
                "refractive_index_medium": 1.33,
                "radius_min": 0.5e-6,
                "radius_max": 0.5e-6,
                "refractive_index_min": 1.45,
                "refractive_index_max": 1.45,
                "z_min_px": 0.0,
                "z_max_px": 0.0,
                "intensity_scale": 20000.0,
                "max_particles": 5,
                "mie_max_particles": 5,
                "mie_max_frames": 2,
            },
            "background": {"amplitude": 0},
            "noise": {"gain_sigma": 0.0, "read_noise": 0.0},
        }
        rng = np.random.default_rng(0)
        frame = rbf.render_frame_brightfield(
            np.array([[0.5, 0.5]]), (0.0, 1.0, 0.0, 1.0), cfg, rng
        ).astype(np.float64)

        cy, cx = 32, 32
        angles = np.linspace(0, 2 * np.pi, 32, endpoint=False)
        profile = np.array(
            [
                frame[
                    np.clip(np.round(cy + r * np.sin(angles)).astype(int), 0, 63),
                    np.clip(np.round(cx + r * np.cos(angles)).astype(int), 0, 63),
                ].mean()
                for r in range(20)
            ]
        )
        background = profile[-3:].mean()
        primary_deviation = np.abs(profile[:8] - background).max()
        secondary_deviation = np.abs(profile[10:18] - background).max()

        assert secondary_deviation < 0.2 * primary_deviation
