"""Tests for detectors_common.dataset_profile — U2: shared dataset scale
profile format loader."""

import importlib.util
from pathlib import Path

import pytest

from detectors_common.dataset_profile import load_dataset_profile


def _write_profile(tmp_path, content, name="profile.yaml"):
    path = tmp_path / name
    path.write_text(content)
    return path


def _load_trackers_common_loader():
    """Load trackers_common.dataset_profile directly from its source file,
    without adding a cross-package install dependency between the two
    otherwise-independent sibling packages -- this test's only job is
    proving the two deliberately-duplicated loader implementations agree,
    not introducing a new dependency to do it."""
    module_path = (
        Path(__file__).resolve().parent.parent.parent
        / "trackers-common"
        / "trackers_common"
        / "dataset_profile.py"
    )
    spec = importlib.util.spec_from_file_location("trackers_common_dataset_profile", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_dataset_profile


class TestLoadDatasetProfile:
    def test_valid_profile_with_both_required_keys_loads_successfully(self, tmp_path):
        path = _write_profile(tmp_path, "size_px: 5.0\nspacing_px: 10.9\n")

        result = load_dataset_profile(path)

        assert result["size_px"] == 5.0
        assert result["spacing_px"] == 10.9

    def test_missing_size_px_raises_clear_error(self, tmp_path):
        path = _write_profile(tmp_path, "spacing_px: 10.9\n")

        with pytest.raises(ValueError, match="size_px"):
            load_dataset_profile(path)

    def test_missing_spacing_px_raises_clear_error(self, tmp_path):
        path = _write_profile(tmp_path, "size_px: 5.0\n")

        with pytest.raises(ValueError, match="spacing_px"):
            load_dataset_profile(path)

    def test_non_positive_size_px_raises_clear_error(self, tmp_path):
        path = _write_profile(tmp_path, "size_px: 0\nspacing_px: 10.9\n")

        with pytest.raises(ValueError, match="size_px"):
            load_dataset_profile(path)

    def test_negative_spacing_px_raises_clear_error(self, tmp_path):
        path = _write_profile(tmp_path, "size_px: 5.0\nspacing_px: -3.0\n")

        with pytest.raises(ValueError, match="spacing_px"):
            load_dataset_profile(path)

    def test_non_numeric_size_px_raises_clear_error(self, tmp_path):
        path = _write_profile(tmp_path, "size_px: not-a-number\nspacing_px: 10.9\n")

        with pytest.raises(ValueError, match="size_px"):
            load_dataset_profile(path)

    def test_unknown_extra_keys_are_ignored_without_error(self, tmp_path):
        path = _write_profile(
            tmp_path,
            "size_px: 5.0\nspacing_px: 10.9\ndescription: some dataset\nfuture_key: 42\n",
        )

        result = load_dataset_profile(path)

        assert result["size_px"] == 5.0
        assert result["spacing_px"] == 10.9
        assert result["description"] == "some dataset"
        assert result["future_key"] == 42

    def test_missing_file_raises_file_not_found_error(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_dataset_profile(tmp_path / "does-not-exist.yaml")

    def test_non_mapping_yaml_raises_clear_error(self, tmp_path):
        path = _write_profile(tmp_path, "- 1\n- 2\n")

        with pytest.raises(ValueError, match="mapping"):
            load_dataset_profile(path)


class TestCrossPackageAgreement:
    """Regression guard: detectors_common's and trackers_common's independently
    duplicated dataset_profile loaders must parse the same file identically."""

    def test_detectors_common_and_trackers_common_loaders_agree(self, tmp_path):
        path = _write_profile(
            tmp_path,
            "size_px: 5.0\nspacing_px: 10.8658\ndescription: shared profile\nextra: 1\n",
        )
        trackers_load_dataset_profile = _load_trackers_common_loader()

        detectors_result = load_dataset_profile(path)
        trackers_result = trackers_load_dataset_profile(path)

        assert detectors_result == trackers_result
