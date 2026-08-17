"""Tests for detectors_common.yolo_loader -- native-vs-inject duality
(mirroring test_lodestar_loader.py) and detect_yolo's sv.Detections wrapping."""

import sys
from unittest import mock

import numpy as np
import pytest

# Force supervision's own real import to happen now, before any test mocks
# sys.modules["torch"] or sys.modules["ultralytics"] -- see
# test_lodestar_loader.py's identical preload for the full rationale.
import supervision as _sv_preload  # noqa: F401

from detectors_common import yolo_loader
from detectors_common.rfdetr_loader import VenvNotSyncedError


def _make_fake_venv(base_dir, pyver="python3.11"):
    venv_dir = base_dir / "fake.venv"
    site_pkgs = venv_dir / "lib" / pyver / "site-packages"
    site_pkgs.mkdir(parents=True)
    return venv_dir


class TestGetYoloModelNativeMode:
    """inject_venv_site_packages=None -- particle-tracking's real case:
    ultralytics already importable, no sys.path manipulation."""

    def test_loads_checkpoint_via_ultralytics_yolo(self, tmp_path):
        checkpoint = tmp_path / "best.pt"
        checkpoint.write_bytes(b"")

        built_model = mock.MagicMock()
        fake_yolo_cls = mock.MagicMock(return_value=built_model)
        fake_ultralytics = mock.MagicMock(YOLO=fake_yolo_cls)

        with mock.patch.dict(sys.modules, {"ultralytics": fake_ultralytics}):
            result = yolo_loader.get_yolo_model(checkpoint)

        fake_yolo_cls.assert_called_once_with(str(checkpoint))
        assert result is built_model

    def test_missing_ultralytics_prints_error_and_exits(self, tmp_path):
        checkpoint = tmp_path / "best.pt"
        checkpoint.write_bytes(b"")

        with mock.patch.dict(sys.modules, {"ultralytics": None}):
            with pytest.raises(SystemExit):
                yolo_loader.get_yolo_model(checkpoint)


class TestGetYoloModelInjectMode:
    """inject_venv_site_packages=<path> -- verification's real case."""

    def test_missing_site_packages_raises_venv_not_synced_error(self, tmp_path):
        checkpoint = tmp_path / "best.pt"
        checkpoint.write_bytes(b"")
        never_synced = tmp_path / "never-synced.venv"
        never_synced.mkdir()

        with pytest.raises(VenvNotSyncedError):
            yolo_loader.get_yolo_model(checkpoint, inject_venv_site_packages=never_synced)

    def test_valid_venv_injects_site_packages_and_evicts_stale_modules(self, tmp_path):
        """A fresh injection must both add the venv's site-packages to sys.path
        and evict any stale ultralytics/torch already loaded from elsewhere --
        proven by pre-loading fakes and confirming they're evicted (their
        absence forces the subsequent real `import ultralytics` to fail,
        since this test's own venv has no real ultralytics installed -- that
        failure is the proof the eviction ran, not a test bug)."""
        checkpoint = tmp_path / "best.pt"
        checkpoint.write_bytes(b"")
        venv_dir = _make_fake_venv(tmp_path)

        stale_ultralytics = mock.MagicMock()
        stale_torch = mock.MagicMock()

        with mock.patch.dict(sys.modules, {"ultralytics": stale_ultralytics, "torch": stale_torch}):
            with pytest.raises(SystemExit):
                yolo_loader.get_yolo_model(checkpoint, inject_venv_site_packages=venv_dir)
            assert "ultralytics" not in sys.modules
            assert "torch" not in sys.modules

        assert str(venv_dir / "lib" / "python3.11" / "site-packages") in sys.path

    def test_second_call_after_injection_does_not_reevict(self, tmp_path, monkeypatch):
        checkpoint = tmp_path / "best.pt"
        checkpoint.write_bytes(b"")
        venv_dir = _make_fake_venv(tmp_path)
        site_pkgs_str = str(venv_dir / "lib" / "python3.11" / "site-packages")
        monkeypatch.syspath_prepend(site_pkgs_str)

        built_model = mock.MagicMock()
        fake_yolo_cls = mock.MagicMock(return_value=built_model)
        fake_ultralytics = mock.MagicMock(YOLO=fake_yolo_cls)

        with mock.patch.dict(sys.modules, {"ultralytics": fake_ultralytics}):
            yolo_loader.get_yolo_model(checkpoint, inject_venv_site_packages=venv_dir)

        fake_yolo_cls.assert_called_once()


class TestDetectYolo:
    def test_wraps_ultralytics_results_as_sv_detections(self):
        model = mock.MagicMock()
        fake_result = mock.MagicMock()
        model.predict.return_value = [fake_result]
        frame = np.zeros((64, 64, 3), dtype=np.uint8)

        expected = mock.MagicMock()
        with mock.patch("supervision.Detections.from_ultralytics", return_value=expected) as m:
            result = yolo_loader.detect_yolo(model, frame, threshold=0.3, device="cpu")

        model.predict.assert_called_once_with(
            frame, conf=0.3, device="cpu", verbose=False, max_det=5000
        )
        m.assert_called_once_with(fake_result)
        assert result is expected

    def test_max_det_override_passed_through(self):
        """max_det defaults to 5000 (well above this project's measured particle
        density) but is overridable -- proven by asserting a non-default value
        actually reaches ultralytics' predict() call."""
        model = mock.MagicMock()
        fake_result = mock.MagicMock()
        model.predict.return_value = [fake_result]
        frame = np.zeros((64, 64, 3), dtype=np.uint8)

        with mock.patch("supervision.Detections.from_ultralytics"):
            yolo_loader.detect_yolo(model, frame, threshold=0.3, device="cpu", max_det=100)

        model.predict.assert_called_once_with(
            frame, conf=0.3, device="cpu", verbose=False, max_det=100
        )
