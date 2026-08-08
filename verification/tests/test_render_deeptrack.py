"""Direct unit tests for render_deeptrack._build_psf_kernel -- specifically the
peak-normalization contract (docs/plans/2026-07-21-001-fix-peak-normalized-
particle-brightness-plan.md, U3). No test file previously exercised this
function directly; crop_source integration tests in test_render.py cover it
only indirectly through render_frame_deeptrack.
"""

import sys
import types
from pathlib import Path
from unittest import mock

import numpy as np
import pytest


def _import_with_mock_deeptrack(fake_kernel):
    """Stub deeptrack so _build_psf_kernel's pipeline.resolve() returns
    fake_kernel, mirroring the pattern in tests/test_render.py."""
    for key in list(sys.modules.keys()):
        if "render_deeptrack" in key:
            del sys.modules[key]

    fake_pipeline = mock.MagicMock()
    fake_pipeline.update.return_value = fake_pipeline
    fake_pipeline.resolve.return_value = fake_kernel

    fake_optics_instance = mock.MagicMock(return_value=fake_pipeline)
    deeptrack_stub = types.ModuleType("deeptrack")
    deeptrack_stub.Fluorescence = mock.MagicMock(return_value=fake_optics_instance)
    deeptrack_stub.PointParticle = mock.MagicMock()

    sys.modules["deeptrack"] = deeptrack_stub
    sys.path.insert(0, str(Path(__file__).parent.parent))

    import render_deeptrack as rdt

    return rdt


def _cfg():
    return {"psf": {"na": 1.4, "wavelength": 520e-9, "resolution": 65e-9}}


class TestBuildPsfKernelPeakNormalization:
    def test_kernel_peaks_at_one_for_a_tight_kernel(self):
        H, W = 128, 128
        cy, cx = H // 2, W // 2
        raw = np.zeros((H, W), dtype=np.float64)
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                raw[cy + dy, cx + dx] = np.exp(-0.5 * (dy**2 + dx**2))
        rdt = _import_with_mock_deeptrack(raw)

        kernel = rdt._build_psf_kernel(_cfg(), H, W)

        assert abs(float(kernel.max()) - 1.0) < 1e-6

    def test_kernel_peaks_at_one_for_a_broad_kernel_spanning_the_full_crop(self):
        """A PSF whose energy is broadly spread across the whole crop window
        -- the failure mode measured this session for physics's actual
        DeepTrack2 output at config.yaml defaults -- must still peak-
        normalize to exactly 1, unlike the old sum-normalized contract where
        breadth diluted the peak far below 1."""
        H, W = 128, 128
        cy, cx = H // 2, W // 2
        yy, xx = np.mgrid[0:H, 0:W]
        r2 = (yy - cy) ** 2 + (xx - cx) ** 2
        raw = np.exp(-r2 / (2 * 40.0**2))  # sigma=40px, broad relative to a 32px crop half-width
        rdt = _import_with_mock_deeptrack(raw)

        kernel = rdt._build_psf_kernel(_cfg(), H, W)

        assert abs(float(kernel.max()) - 1.0) < 1e-6

    def test_peak_normalized_kernel_is_insensitive_to_psf_breadth(self):
        """Two kernels with identical peak amplitude but very different
        breadth must both peak-normalize to the same max value -- proving
        rendered brightness no longer depends on how spread out the PSF is
        (R3/R4)."""
        H, W = 128, 128
        cy, cx = H // 2, W // 2
        yy, xx = np.mgrid[0:H, 0:W]
        r2 = (yy - cy) ** 2 + (xx - cx) ** 2

        tight = np.exp(-r2 / (2 * 2.0**2))
        broad = np.exp(-r2 / (2 * 30.0**2))

        rdt_tight = _import_with_mock_deeptrack(tight)
        kernel_tight = rdt_tight._build_psf_kernel(_cfg(), H, W)
        rdt_broad = _import_with_mock_deeptrack(broad)
        kernel_broad = rdt_broad._build_psf_kernel(_cfg(), H, W)

        assert abs(float(kernel_tight.max()) - float(kernel_broad.max())) < 1e-6

    def test_all_zero_kernel_fallback_still_peaks_at_one(self):
        """If deeptrack's PSF lands entirely outside output_region, the
        unit-impulse fallback must still satisfy the peak-normalization
        contract."""
        H, W = 64, 64
        raw = np.zeros((H, W), dtype=np.float64)
        rdt = _import_with_mock_deeptrack(raw)

        kernel = rdt._build_psf_kernel(_cfg(), H, W)

        assert abs(float(kernel.max()) - 1.0) < 1e-6
        cy, cx = H // 2, W // 2
        r = min(32, cy, cx)
        assert kernel[r, r] == pytest.approx(1.0)

    def test_kernel_dtype_is_float32(self):
        H, W = 64, 64
        cy, cx = H // 2, W // 2
        raw = np.zeros((H, W), dtype=np.float64)
        raw[cy, cx] = 5.0
        rdt = _import_with_mock_deeptrack(raw)

        kernel = rdt._build_psf_kernel(_cfg(), H, W)

        assert kernel.dtype == np.float32
