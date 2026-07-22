"""Tests for fit_procedural_ring.py.

docs/plans/2026-07-22-001-fix-procedural-particle-realism-plan.md U2 test
scenarios:
- happy path: fitting against a kernel built from config's optics settings
  returns a six-parameter tuple with a plausible ring radius
- happy path: --merge-config writes the fitted params into procedural_shape,
  preserving existing config.yaml keys/comments
- edge case: the wider kernel's crop radius must be large enough to contain
  the dark ring before fitting (radial profile's minimum isn't at the last
  bin)
- error path: a non-convergent fit exits with a clear message, no partial
  config write
- integration: parameters written are consumable by generate_procedural_shape
  without further transformation
"""

import sys
import types
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

import fit_procedural_ring as fpr
import render_crop_templates as rct

_TRUE_RING_PARAMS = (0.5, 1.0, 8.0, 0.4, 25.0, 5.0)  # B, A0, s0, A1, r1, s1
# B - A1 > 0 -- a real (non-negative) intensity kernel never dips below
# zero, and fit_procedural_ring._build_wide_psf_kernel takes np.abs() of
# deeptrack's raw (possibly-complex) output the same way _build_psf_kernel
# does; abs() would otherwise flip a negative-going ring dip into a
# spurious bump and corrupt the fixture, not just the fit. Values mirror
# TestFitRingModel.test_recovers_known_parameters_within_tolerance's known-
# convergent shape (render_crop_templates.py) rather than inventing new
# ones -- fit_ring_model's extrema-derived initial guess is sensitive to
# the ratio between the core width and the profile's overall radial range.


def _import_with_mock_deeptrack(fake_kernel):
    """Same stub pattern as tests/test_render_deeptrack.py's
    _import_with_mock_deeptrack -- fit_procedural_ring's deeptrack call
    chain (Fluorescence(...)(probe).update().resolve()) is identical."""
    for key in list(sys.modules.keys()):
        if key == "fit_procedural_ring":
            del sys.modules[key]

    fake_pipeline = mock.MagicMock()
    fake_pipeline.update.return_value = fake_pipeline
    fake_pipeline.resolve.return_value = fake_kernel

    fake_optics_instance = mock.MagicMock(return_value=fake_pipeline)
    deeptrack_stub = types.ModuleType("deeptrack")
    deeptrack_stub.Fluorescence = mock.MagicMock(return_value=fake_optics_instance)
    deeptrack_stub.PointParticle = mock.MagicMock()

    sys.modules["deeptrack"] = deeptrack_stub

    import fit_procedural_ring as mod

    return mod


def _fake_ring_kernel(radius, ring_params=_TRUE_RING_PARAMS):
    size = 2 * radius + 1
    return rct.generate_ring_template(size, ring_params)


class TestFitProceduralRingParams:
    def test_recovers_plausible_ring_parameters(self):
        radius = 80
        kernel = _fake_ring_kernel(radius)
        mod = _import_with_mock_deeptrack(kernel)

        fitted = mod.fit_procedural_ring_params({"na": 1.4}, radius=radius)

        assert len(fitted) == 6
        fitted_r1 = fitted[4]
        # Loose bound, not a numerical-accuracy check on fit_ring_model
        # itself (covered by TestFitRingModel elsewhere) -- this test is
        # about the wiring (wide kernel -> radial profile -> fit), so it
        # only needs "found a ring near the true radius", not exact recovery.
        assert 10 < fitted_r1 < 40

    def test_ring_truncated_by_radius_raises_value_error(self):
        # Ring radius (60) is close to the crop radius (35) -- the profile's
        # minimum lands at (or past) the outermost bin instead of a genuine
        # interior trough.
        radius = 35
        kernel = _fake_ring_kernel(radius, ring_params=(0.5, 1.0, 8.0, 0.4, 60.0, 5.0))
        mod = _import_with_mock_deeptrack(kernel)

        with pytest.raises(ValueError, match="outermost bin"):
            mod.fit_procedural_ring_params({"na": 1.4}, radius=radius)

    def test_all_zero_kernel_raises_value_error(self):
        radius = 40
        kernel = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.float64)
        mod = _import_with_mock_deeptrack(kernel)

        with pytest.raises(ValueError, match="all-zero"):
            mod.fit_procedural_ring_params({"na": 1.4}, radius=radius)


class TestFittedParamsConsumableByGenerateProceduralShape:
    def test_fitted_tuple_feeds_generate_procedural_shape_without_transformation(self):
        radius = 80
        kernel = _fake_ring_kernel(radius)
        mod = _import_with_mock_deeptrack(kernel)

        fitted = mod.fit_procedural_ring_params({"na": 1.4}, radius=radius)
        shape = rct.generate_procedural_shape(size=41, sigma=5.0, ring_params=fitted)

        assert abs(float(shape.max()) - 1.0) < 1e-4


class TestMergeConfigIntegration:
    def test_merge_config_writes_ring_params_preserving_existing_keys(self, tmp_path):
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(
            yaml.dump(
                {
                    "synthetic": {
                        "render_strategy": "deeptrack",
                        "procedural_shape": {"size": 41, "sigma": 5.0},
                    }
                },
                default_flow_style=False,
                sort_keys=False,
            )
        )
        radius = 80
        kernel = _fake_ring_kernel(radius)
        mod = _import_with_mock_deeptrack(kernel)

        fitted = mod.fit_procedural_ring_params({"na": 1.4}, radius=radius)
        ring_params = dict(zip(mod.RING_PARAM_NAMES, fitted))

        import calibrate_psf

        calibrate_psf._merge_params_into_config(cfg_path, {"procedural_shape": ring_params})

        result = yaml.safe_load(cfg_path.read_text())
        assert result["synthetic"]["render_strategy"] == "deeptrack"
        assert result["synthetic"]["procedural_shape"]["size"] == 41
        assert 10 < result["synthetic"]["procedural_shape"]["ring_r1"] < 40

    def test_cli_error_path_does_not_write_config_on_fit_failure(self, tmp_path, capsys):
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump({"synthetic": {"psf": {"na": 1.4}}}))
        original_text = cfg_path.read_text()

        radius = 35
        kernel = _fake_ring_kernel(radius, ring_params=(0.5, 1.0, 8.0, 0.4, 60.0, 5.0))
        mod = _import_with_mock_deeptrack(kernel)

        with mock.patch.object(
            sys,
            "argv",
            ["fit_procedural_ring.py", "--config", str(cfg_path), "--radius", str(radius)],
        ):
            with pytest.raises(SystemExit) as exc_info:
                mod.main()

        assert exc_info.value.code == 1
        assert "ERROR" in capsys.readouterr().err
        assert cfg_path.read_text() == original_text
