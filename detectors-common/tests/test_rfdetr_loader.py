"""Tests for detectors_common.rfdetr_loader — U2: get_rfdetr_model, RFDETR_VARIANTS,
_normalize_device, and the venv-site-packages-injection failure mode."""

import sys
from unittest import mock

import pytest

from detectors_common import rfdetr_loader


def _make_fake_venv(base_dir, pyver="python3.11"):
    """Build a fake venv directory with a versioned site-packages dir, mimicking
    a real uv-managed venv layout closely enough for get_rfdetr_model to glob."""
    venv_dir = base_dir / "fake.venv"
    site_pkgs = venv_dir / "lib" / pyver / "site-packages"
    site_pkgs.mkdir(parents=True)
    return venv_dir


class TestNormalizeDevice:
    def test_none_stays_none(self):
        assert rfdetr_loader._normalize_device(None) is None

    def test_bare_integer_string_becomes_cuda_device(self):
        assert rfdetr_loader._normalize_device("0") == "cuda:0"

    def test_already_prefixed_string_is_unchanged(self):
        assert rfdetr_loader._normalize_device("cpu") == "cpu"


class TestGetRfdetrModel:
    def test_missing_site_packages_raises_venv_not_synced_error(self, tmp_path):
        empty_venv_dir = tmp_path / "never-synced.venv"
        empty_venv_dir.mkdir()

        with pytest.raises(rfdetr_loader.VenvNotSyncedError):
            rfdetr_loader.get_rfdetr_model("large", "ckpt.pth", "0", empty_venv_dir)

    def test_valid_venv_injects_site_packages_and_threads_num_classes(self, tmp_path):
        venv_dir = _make_fake_venv(tmp_path)
        fake_model = mock.Mock()
        fake_model.optimize_for_inference = mock.Mock()
        fake_rfdetr = mock.MagicMock()
        fake_rfdetr.RFDETRLarge.return_value = fake_model

        with mock.patch.dict(sys.modules, {"rfdetr": fake_rfdetr}):
            result = rfdetr_loader.get_rfdetr_model(
                "large", "ckpt.pth", "0", venv_dir, num_classes=2
            )

        assert str(venv_dir / "lib" / "python3.11" / "site-packages") in sys.path
        fake_rfdetr.RFDETRLarge.assert_called_once_with(
            pretrain_weights="ckpt.pth", device="cuda:0", num_classes=2
        )
        assert result is fake_model
        fake_model.optimize_for_inference.assert_called_once()

    def test_unknown_variant_exits_cleanly_not_attribute_error(self, tmp_path):
        venv_dir = _make_fake_venv(tmp_path)
        fake_rfdetr = mock.MagicMock()

        with mock.patch.dict(sys.modules, {"rfdetr": fake_rfdetr}):
            with pytest.raises(SystemExit):
                rfdetr_loader.get_rfdetr_model("not-a-real-variant", "ckpt.pth", "0", venv_dir)

    def test_variant_absent_from_installed_rfdetr_package_exits_cleanly(self, tmp_path):
        """cls_name resolves via RFDETR_VARIANTS, but the installed rfdetr
        package doesn't have that attribute (version skew) — must not raise
        a raw AttributeError."""
        venv_dir = _make_fake_venv(tmp_path)
        fake_rfdetr = mock.MagicMock(spec=[])  # no RFDETRLarge attribute at all

        with mock.patch.dict(sys.modules, {"rfdetr": fake_rfdetr}):
            with pytest.raises(SystemExit):
                rfdetr_loader.get_rfdetr_model("large", "ckpt.pth", "0", venv_dir)
