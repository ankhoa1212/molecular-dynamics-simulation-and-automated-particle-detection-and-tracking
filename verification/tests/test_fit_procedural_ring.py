"""Tests for fit_procedural_ring.py.

docs/plans/2026-07-22-001-fix-procedural-particle-realism-plan.md U2 test
scenarios (revised mid-execution: fits against crop_source: real's harvested
template library instead of crop_source: physics's PSF kernel, after
measuring that this config's physics kernel has no meaningful ring to fit):
- happy path: fitting against a template library's radial profile returns
  a six-parameter tuple with a plausible ring radius
- happy path: --merge-config writes the fitted params into procedural_shape,
  preserving existing config.yaml keys/comments
- error path: no pre-built template library at cache_path exits with a
  clear message pointing at build_template_library
- edge case: the templates' own stored size must be large enough to contain
  the dark ring before fitting (radial profile's minimum isn't at the last
  bin)
- error path: a non-convergent fit exits with a clear message, no partial
  config write
- integration: parameters written are consumable by generate_procedural_shape
  without further transformation
"""

import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

import fit_procedural_ring as fpr
import render_crop_templates as rct

_TRUE_RING_PARAMS = (0.5, 1.0, 8.0, 0.4, 25.0, 5.0)  # B, A0, s0, A1, r1, s1
# B - A1 > 0 -- a real (non-negative) intensity template never dips below
# zero. Values mirror TestFitRingModel.test_recovers_known_parameters_
# within_tolerance's known-convergent shape (render_crop_templates.py)
# rather than inventing new ones -- fit_ring_model's extrema-derived
# initial guess is sensitive to the ratio between the core width and the
# profile's overall radial range.


def _fake_template_library(n=3, half=80, ring_params=_TRUE_RING_PARAMS):
    size = 2 * half + 1
    template = rct.generate_ring_template(size, ring_params)
    template = template / template.max()
    return np.stack([template.astype(np.float32)] * n, axis=0)


class TestFitProceduralRingParams:
    def test_recovers_plausible_ring_parameters(self):
        templates = _fake_template_library()

        fitted = fpr.fit_procedural_ring_params(templates)

        assert len(fitted) == 6
        fitted_r1 = fitted[4]
        # Loose bound, not a numerical-accuracy check on fit_ring_model
        # itself (covered by TestFitRingModel elsewhere) -- this test is
        # about the wiring (templates -> radial profile -> fit), so it only
        # needs "found a ring near the true radius", not exact recovery.
        assert 10 < fitted_r1 < 40

    def test_ring_truncated_by_template_size_raises_value_error(self):
        # Ring radius (60) is close to the template half-width (35) -- the
        # profile's minimum lands at (or past) the outermost bin instead of
        # a genuine interior trough.
        templates = _fake_template_library(half=35, ring_params=(0.5, 1.0, 8.0, 0.4, 60.0, 5.0))

        with pytest.raises(ValueError, match="outermost bin"):
            fpr.fit_procedural_ring_params(templates)

    def test_fit_beyond_half_width_raises_value_error(self):
        # Ring radius (42) sits between the template half-width (35) and the
        # profile's corner-diagonal reach (~49.5) -- visible enough that the
        # profile's minimum isn't at the outermost bin, but still past the
        # well-sampled half-width, extrapolating into the sparse corner-only
        # region rather than measuring a real feature.
        templates = _fake_template_library(half=35, ring_params=(0.5, 1.0, 8.0, 0.4, 42.0, 5.0))

        with pytest.raises(ValueError, match="half-width"):
            fpr.fit_procedural_ring_params(templates)


class TestFittedParamsConsumableByGenerateProceduralShape:
    def test_fitted_tuple_feeds_generate_procedural_shape_without_transformation(self):
        templates = _fake_template_library()

        fitted = fpr.fit_procedural_ring_params(templates)
        shape = rct.generate_procedural_shape(size=41, sigma=5.0, ring_params=fitted)

        assert abs(float(shape.max()) - 1.0) < 1e-4


class TestMainCliFlow:
    def _write_config(self, cfg_path, cache_path):
        cfg_path.write_text(
            yaml.dump(
                {
                    "synthetic": {
                        "render_strategy": "deeptrack",
                        "crop_template": {"cache_path": str(cache_path)},
                        "procedural_shape": {"size": 41, "sigma": 5.0},
                    }
                },
                default_flow_style=False,
                sort_keys=False,
            )
        )

    def test_missing_template_library_exits_with_clear_error(self, tmp_path, capsys):
        cfg_path = tmp_path / "config.yaml"
        self._write_config(cfg_path, tmp_path / "absent.npz")

        with mock.patch.object(sys, "argv", ["fit_procedural_ring.py", "--config", str(cfg_path)]):
            with pytest.raises(SystemExit) as exc_info:
                fpr.main()

        assert exc_info.value.code == 1
        err = capsys.readouterr().err
        assert "ERROR" in err
        assert "build_template_library" in err

    def test_merge_config_writes_ring_params_preserving_existing_keys(self, tmp_path):
        cache_path = tmp_path / "templates.npz"
        np.savez(cache_path, templates=_fake_template_library())
        cfg_path = tmp_path / "config.yaml"
        self._write_config(cfg_path, cache_path)

        with mock.patch.object(
            sys,
            "argv",
            [
                "fit_procedural_ring.py",
                "--config",
                str(cfg_path),
                "--merge-config",
                str(cfg_path),
            ],
        ):
            fpr.main()

        result = yaml.safe_load(cfg_path.read_text())
        assert result["synthetic"]["render_strategy"] == "deeptrack"
        assert result["synthetic"]["procedural_shape"]["size"] == 41
        assert 10 < result["synthetic"]["procedural_shape"]["ring_r1"] < 40

    def test_cli_error_path_does_not_write_config_on_fit_failure(self, tmp_path, capsys):
        cache_path = tmp_path / "templates.npz"
        np.savez(
            cache_path,
            templates=_fake_template_library(half=35, ring_params=(0.5, 1.0, 8.0, 0.4, 60.0, 5.0)),
        )
        cfg_path = tmp_path / "config.yaml"
        self._write_config(cfg_path, cache_path)
        original_text = cfg_path.read_text()

        with mock.patch.object(sys, "argv", ["fit_procedural_ring.py", "--config", str(cfg_path)]):
            with pytest.raises(SystemExit) as exc_info:
                fpr.main()

        assert exc_info.value.code == 1
        assert "ERROR" in capsys.readouterr().err
        assert cfg_path.read_text() == original_text
