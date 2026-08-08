"""Tests for detectors_common.lodestar_loader — U3: get_lodestar_model's
native-vs-inject duality, and detect_lodestar's sigma scaling / NMS."""

import sys
from unittest import mock

import numpy as np
import pytest

# Force supervision's own real import (and its internal torch-availability
# check) to happen now, before any test mocks sys.modules["torch"]. If a
# mocked torch is present the first time supervision is imported, its
# torch-integration code path initializes against the mock and later tests
# that import supervision for the first time under a *different* mocked
# torch instance can hit stale/inconsistent internal state.
import supervision as _sv_preload  # noqa: F401

from detectors_common import lodestar_loader
from detectors_common.rfdetr_loader import VenvNotSyncedError


def _fake_deeplay_module(built_model):
    fake_dl = mock.MagicMock()
    fake_dl.LodeSTAR.return_value.build.return_value = built_model
    return fake_dl


def _make_fake_venv(base_dir, pyver="python3.11"):
    venv_dir = base_dir / "fake.venv"
    site_pkgs = venv_dir / "lib" / pyver / "site-packages"
    site_pkgs.mkdir(parents=True)
    return venv_dir


class TestGetLodestarModelNativeMode:
    """inject_venv_site_packages=None — particle-tracking's real case: deeplay/
    torch already importable, no sys.path manipulation."""

    def test_reads_companion_json_for_n_transforms_and_num_outputs(self, tmp_path):
        checkpoint = tmp_path / "model.pt"
        checkpoint.write_bytes(b"")
        (tmp_path / "model.json").write_text('{"n_transforms": 4, "num_outputs": 5}')

        built_model = mock.MagicMock()
        fake_dl = _fake_deeplay_module(built_model)
        fake_torch = mock.MagicMock()

        with mock.patch.dict(sys.modules, {"deeplay": fake_dl, "torch": fake_torch}):
            result = lodestar_loader.get_lodestar_model(str(checkpoint), device="cpu")

        fake_dl.LodeSTAR.assert_called_once_with(n_transforms=4, num_outputs=5)
        assert result is built_model
        built_model.eval.assert_called_once()

    def test_missing_companion_json_falls_back_to_defaults(self, tmp_path):
        checkpoint = tmp_path / "model.pt"
        checkpoint.write_bytes(b"")  # no sibling .json

        built_model = mock.MagicMock()
        fake_dl = _fake_deeplay_module(built_model)
        fake_torch = mock.MagicMock()

        with mock.patch.dict(sys.modules, {"deeplay": fake_dl, "torch": fake_torch}):
            lodestar_loader.get_lodestar_model(str(checkpoint), device="cpu")

        fake_dl.LodeSTAR.assert_called_once_with(n_transforms=8, num_outputs=3)

    def test_fp16_calls_half_on_model(self, tmp_path):
        checkpoint = tmp_path / "model.pt"
        checkpoint.write_bytes(b"")

        built_model = mock.MagicMock()
        fake_dl = _fake_deeplay_module(built_model)
        fake_torch = mock.MagicMock()

        with mock.patch.dict(sys.modules, {"deeplay": fake_dl, "torch": fake_torch}):
            lodestar_loader.get_lodestar_model(str(checkpoint), device="cpu", fp16=True)

        built_model.half.assert_called_once()

    def test_missing_deeplay_prints_error_and_exits(self, tmp_path):
        checkpoint = tmp_path / "model.pt"
        checkpoint.write_bytes(b"")

        with mock.patch.dict(sys.modules, {"deeplay": None}):
            with pytest.raises(SystemExit):
                lodestar_loader.get_lodestar_model(str(checkpoint), device="cpu")


class TestGetLodestarModelInjectMode:
    """inject_venv_site_packages=<path> — verification's real case."""

    def test_missing_site_packages_raises_venv_not_synced_error(self, tmp_path):
        checkpoint = tmp_path / "model.pt"
        checkpoint.write_bytes(b"")
        never_synced = tmp_path / "never-synced.venv"
        never_synced.mkdir()

        with pytest.raises(VenvNotSyncedError):
            lodestar_loader.get_lodestar_model(
                str(checkpoint), device="cpu", inject_venv_site_packages=never_synced
            )

    def test_valid_venv_injects_site_packages_and_evicts_stale_modules(self, tmp_path):
        """A fresh injection must both add the venv's site-packages to sys.path
        and evict any stale torch/deeplay already loaded from elsewhere — proven
        here by pre-loading fakes and confirming they're evicted (their absence
        forces the subsequent real `import deeplay` to fail, since this test's
        own venv has no real deeplay installed — that failure is the proof the
        eviction ran, not a test bug)."""
        checkpoint = tmp_path / "model.pt"
        checkpoint.write_bytes(b"")
        venv_dir = _make_fake_venv(tmp_path)

        stale_dl = _fake_deeplay_module(mock.MagicMock())
        stale_torch = mock.MagicMock()

        with mock.patch.dict(sys.modules, {"deeplay": stale_dl, "torch": stale_torch}):
            with pytest.raises(SystemExit):
                lodestar_loader.get_lodestar_model(
                    str(checkpoint), device="cpu", inject_venv_site_packages=venv_dir
                )
            assert "deeplay" not in sys.modules
            assert "torch" not in sys.modules

        assert str(venv_dir / "lib" / "python3.11" / "site-packages") in sys.path

    def test_second_call_after_injection_does_not_reevict(self, tmp_path, monkeypatch):
        checkpoint = tmp_path / "model.pt"
        checkpoint.write_bytes(b"")
        venv_dir = _make_fake_venv(tmp_path)
        site_pkgs_str = str(venv_dir / "lib" / "python3.11" / "site-packages")
        # Simulate the injection having already happened on a prior call.
        monkeypatch.syspath_prepend(site_pkgs_str)

        built_model = mock.MagicMock()
        fake_dl = _fake_deeplay_module(built_model)
        fake_torch = mock.MagicMock()

        with mock.patch.dict(sys.modules, {"deeplay": fake_dl, "torch": fake_torch}):
            # site_pkgs_str is already on sys.path, so the eviction guard must
            # not fire — the already-injected fake deeplay/torch survive untouched.
            lodestar_loader.get_lodestar_model(
                str(checkpoint), device="cpu", inject_venv_site_packages=venv_dir
            )

        fake_dl.LodeSTAR.assert_called_once()


class TestDetectLodestar:
    @staticmethod
    def _fake_torch_module():
        fake_torch = mock.MagicMock()
        fake_model = mock.MagicMock()
        fake_model.parameters.return_value = iter([mock.Mock(dtype="float32")])
        return fake_torch, fake_model

    def test_box_size_governs_radius_regardless_of_small_sigma(self):
        fake_torch, fake_model = self._fake_torch_module()
        # sigma=0.01 -- small, previously would have been misread as a normalized
        # frame-fraction and scaled by the frame size. Must have no effect now.
        fake_model.detect.return_value = np.array([[100.0, 200.0, 0.01]])
        frame = np.zeros((512, 512), dtype=np.float32)

        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            result = lodestar_loader.detect_lodestar(
                fake_model, frame, threshold=0.1, device="cpu", box_size=30
            )

        assert len(result) == 1
        cx = (result.xyxy[0][0] + result.xyxy[0][2]) / 2
        cy = (result.xyxy[0][1] + result.xyxy[0][3]) / 2
        assert cx == pytest.approx(200.0, abs=0.5)
        assert cy == pytest.approx(100.0, abs=0.5)
        radius = (result.xyxy[0][2] - result.xyxy[0][0]) / 2
        assert radius == pytest.approx(30 / 2, rel=0.01)

    def test_box_size_governs_radius_regardless_of_large_sigma(self):
        fake_torch, fake_model = self._fake_torch_module()
        # sigma=12.0 -- previously would have been read as an already-pixel-scale
        # radius and used directly. Must have no effect now.
        fake_model.detect.return_value = np.array([[100.0, 200.0, 12.0]])
        frame = np.zeros((512, 512), dtype=np.float32)

        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            result = lodestar_loader.detect_lodestar(
                fake_model, frame, threshold=0.1, device="cpu", box_size=30
            )

        radius = (result.xyxy[0][2] - result.xyxy[0][0]) / 2
        assert radius == pytest.approx(30 / 2, rel=0.01)

    def test_default_box_size_used_when_not_specified(self):
        fake_torch, fake_model = self._fake_torch_module()
        fake_model.detect.return_value = np.array([[100.0, 200.0, 0.01]])
        frame = np.zeros((512, 512), dtype=np.float32)

        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            result = lodestar_loader.detect_lodestar(fake_model, frame, threshold=0.1, device="cpu")

        radius = (result.xyxy[0][2] - result.xyxy[0][0]) / 2
        assert radius == pytest.approx(40 / 2, rel=0.01)  # detect_lodestar's own default

    def test_empty_detections_returns_empty(self):
        fake_torch, fake_model = self._fake_torch_module()
        fake_model.detect.return_value = np.zeros((0, 3))
        frame = np.zeros((64, 64), dtype=np.float32)

        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            result = lodestar_loader.detect_lodestar(fake_model, frame, threshold=0.1, device="cpu")

        assert len(result) == 0

    def test_none_detections_returns_empty(self):
        fake_torch, fake_model = self._fake_torch_module()
        fake_model.detect.return_value = None
        frame = np.zeros((64, 64), dtype=np.float32)

        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            result = lodestar_loader.detect_lodestar(fake_model, frame, threshold=0.1, device="cpu")

        assert len(result) == 0

    def test_list_wrapped_detections_unwraps_first_element(self):
        fake_torch, fake_model = self._fake_torch_module()
        fake_model.detect.return_value = [np.array([[10.0, 20.0, 0.02]])]
        frame = np.zeros((100, 100), dtype=np.float32)

        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            result = lodestar_loader.detect_lodestar(fake_model, frame, threshold=0.1, device="cpu")

        assert len(result) == 1

    def test_nms_distance_suppresses_nearby_duplicates(self):
        fake_torch, fake_model = self._fake_torch_module()
        fake_model.detect.return_value = np.array(
            [
                [100.0, 100.0, 0.01],
                [100.0, 103.0, 0.01],
                [400.0, 400.0, 0.01],
            ]
        )
        frame = np.zeros((512, 512), dtype=np.float32)

        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            result = lodestar_loader.detect_lodestar(
                fake_model, frame, threshold=0.1, device="cpu", nms_distance=10
            )

        assert len(result) == 2  # the near-duplicate pair collapses to one
