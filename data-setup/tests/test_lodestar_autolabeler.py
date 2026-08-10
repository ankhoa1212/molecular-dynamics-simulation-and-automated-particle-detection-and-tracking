"""Tests for lodestar_autolabeler.py's --dataset-profile wiring (U6: R5, R10).

Covers the precedence chain box_size/nms_distance resolve through:
    CLI-explicit -> --config JSON value -> --dataset-profile-derived
    -> today's existing hardcoded default (40 / 0.0, unchanged from
    pre-plan behavior -- regression coverage).

Exercises the real production functions (`parse_args`, `_apply_config_file`,
`_resolve_scale_params`) in the same order `main()` calls them, rather than
re-implementing the merge/resolve logic in the test -- so these tests would
actually catch a wiring regression in main() itself.
"""

import json
import sys

import pytest
import yaml

from detectors_common.scale_derivation import FWHM_TO_SIGMA
from lodestar_autolabeler import parse_args, _apply_config_file, _resolve_scale_params


def _parse(monkeypatch, argv):
    """Parse CLI args as if invoked with `argv` (excluding the program name)."""
    monkeypatch.setattr(sys, "argv", ["lodestar_autolabeler.py"] + argv)
    return parse_args()


def _write_profile(tmp_path, size_px=8.0, spacing_px=12.0):
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(yaml.safe_dump({"size_px": size_px, "spacing_px": spacing_px}))
    return str(profile_path)


def _write_json_config(tmp_path, **kwargs):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(kwargs))
    return str(config_path)


class TestDetectorsCommonEditableInstall:
    """uv sync must install detectors-common as an editable dependency into
    data-setup/.venv; this module already imports from it at collection
    time (see the top of this file and lodestar_autolabeler.py's own
    imports), so a broken editable install would fail before any test
    here runs -- this test just makes that failure mode explicit."""

    def test_import_detectors_common(self):
        import detectors_common
        from detectors_common.scale_derivation import resolve_box_size
        from detectors_common.dataset_profile import load_dataset_profile

        assert detectors_common is not None
        assert callable(resolve_box_size)
        assert callable(load_dataset_profile)


class TestArgparseDefaults:
    """--box-size/--nms-distance must default to None at the argparse level
    (not 40/0.0) -- see the Execution note in the U6 plan: the 40/0.0
    defaults now live in _resolve_scale_params, not in argparse itself."""

    def test_box_size_and_nms_distance_default_to_none(self, monkeypatch):
        args = _parse(monkeypatch, ["--model", "m.pt", "--input", "in/"])

        assert args.box_size is None
        assert args.nms_distance is None

    def test_dataset_profile_defaults_to_none(self, monkeypatch):
        args = _parse(monkeypatch, ["--model", "m.pt", "--input", "in/"])

        assert args.dataset_profile is None


class TestResolveScaleParamsWithProfile:
    """--dataset-profile supplied, no explicit --box-size/--nms-distance:
    both derive via detectors_common.scale_derivation (AE4, R10)."""

    def test_both_derive_from_profile(self, monkeypatch, tmp_path):
        profile_path = _write_profile(tmp_path, size_px=8.0, spacing_px=12.0)
        args = _parse(
            monkeypatch,
            ["--model", "m.pt", "--input", "in/", "--dataset-profile", profile_path],
        )

        _resolve_scale_params(args)

        assert args.box_size == pytest.approx(8.0 * FWHM_TO_SIGMA)
        assert args.nms_distance == pytest.approx(min(8.0 * 1.0, 12.0 * 0.5))


class TestExplicitOverrideWinsOverProfile:
    """--dataset-profile supplied alongside an explicit --box-size: the
    explicit value wins (AE4's override case)."""

    def test_explicit_box_size_wins(self, monkeypatch, tmp_path):
        profile_path = _write_profile(tmp_path, size_px=8.0, spacing_px=12.0)
        args = _parse(
            monkeypatch,
            [
                "--model",
                "m.pt",
                "--input",
                "in/",
                "--dataset-profile",
                profile_path,
                "--box-size",
                "99",
            ],
        )

        _resolve_scale_params(args)

        assert args.box_size == 99
        # nms_distance was not explicitly passed, so it still derives.
        assert args.nms_distance == pytest.approx(min(8.0 * 1.0, 12.0 * 0.5))

    def test_explicit_nms_distance_wins(self, monkeypatch, tmp_path):
        profile_path = _write_profile(tmp_path, size_px=8.0, spacing_px=12.0)
        args = _parse(
            monkeypatch,
            [
                "--model",
                "m.pt",
                "--input",
                "in/",
                "--dataset-profile",
                profile_path,
                "--nms-distance",
                "7",
            ],
        )

        _resolve_scale_params(args)

        assert args.nms_distance == 7
        assert args.box_size == pytest.approx(8.0 * FWHM_TO_SIGMA)


class TestNoDatasetProfileRegression:
    """No --dataset-profile: today's hardcoded defaults (40, 0.0) are
    unchanged (regression) -- the most important scenario given the
    argparse-default-relocation from 40/0.0 to None."""

    def test_hardcoded_defaults_unchanged(self, monkeypatch):
        args = _parse(monkeypatch, ["--model", "m.pt", "--input", "in/"])

        _resolve_scale_params(args)

        assert args.box_size == 40
        assert args.nms_distance == 0.0

    def test_explicit_cli_values_still_work_without_a_profile(self, monkeypatch):
        args = _parse(
            monkeypatch,
            ["--model", "m.pt", "--input", "in/", "--box-size", "50", "--nms-distance", "12"],
        )

        _resolve_scale_params(args)

        assert args.box_size == 50
        assert args.nms_distance == 12


class TestConfigFileAndProfilePrecedence:
    """--dataset-profile supplied alongside the existing --config <path.json>
    JSON merge: CLI-explicit > JSON > profile-derived > hardcoded must hold
    -- a new interaction the plan flags as needing its own test given how
    fragile the pre-existing JSON-merge heuristic already is."""

    def test_json_value_wins_over_profile_derivation(self, monkeypatch, tmp_path):
        profile_path = _write_profile(tmp_path, size_px=8.0, spacing_px=12.0)
        config_path = _write_json_config(tmp_path, box_size=77, nms_distance=3.5)
        args = _parse(
            monkeypatch,
            [
                "--model",
                "m.pt",
                "--input",
                "in/",
                "--config",
                config_path,
                "--dataset-profile",
                profile_path,
            ],
        )

        _apply_config_file(args, args.config)
        _resolve_scale_params(args)

        assert args.box_size == 77
        assert args.nms_distance == 3.5

    def test_cli_explicit_wins_over_json_and_profile(self, monkeypatch, tmp_path):
        profile_path = _write_profile(tmp_path, size_px=8.0, spacing_px=12.0)
        config_path = _write_json_config(tmp_path, box_size=77, nms_distance=3.5)
        args = _parse(
            monkeypatch,
            [
                "--model",
                "m.pt",
                "--input",
                "in/",
                "--config",
                config_path,
                "--dataset-profile",
                profile_path,
                "--box-size",
                "13",
            ],
        )

        _apply_config_file(args, args.config)
        _resolve_scale_params(args)

        assert args.box_size == 13
        # nms_distance has no CLI override, so JSON still wins over the profile.
        assert args.nms_distance == 3.5

    def test_profile_derives_when_json_omits_the_key(self, monkeypatch, tmp_path):
        profile_path = _write_profile(tmp_path, size_px=8.0, spacing_px=12.0)
        # JSON config sets unrelated keys only -- box_size/nms_distance absent.
        config_path = _write_json_config(tmp_path, alpha=0.9, cutoff=0.1)
        args = _parse(
            monkeypatch,
            [
                "--model",
                "m.pt",
                "--input",
                "in/",
                "--config",
                config_path,
                "--dataset-profile",
                profile_path,
            ],
        )

        _apply_config_file(args, args.config)
        _resolve_scale_params(args)

        assert args.box_size == pytest.approx(8.0 * FWHM_TO_SIGMA)
        assert args.nms_distance == pytest.approx(min(8.0 * 1.0, 12.0 * 0.5))

    def test_hardcoded_default_when_neither_json_nor_profile_set_it(self, monkeypatch, tmp_path):
        config_path = _write_json_config(tmp_path, alpha=0.9)
        args = _parse(
            monkeypatch,
            ["--model", "m.pt", "--input", "in/", "--config", config_path],
        )

        _apply_config_file(args, args.config)
        _resolve_scale_params(args)

        assert args.box_size == 40
        assert args.nms_distance == 0.0
