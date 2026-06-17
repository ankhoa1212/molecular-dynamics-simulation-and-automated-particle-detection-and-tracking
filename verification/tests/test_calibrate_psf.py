"""Tests for calibrate_psf.py — U4: PSF calibration from real frames.

Test scenarios from the plan:
- test_recovered_sigma_within_10_percent_of_ground_truth
- test_fewer_than_20_particles_prints_warning_and_continues
- test_absent_real_frames_dir_exits_with_error
- test_empty_real_frames_dir_exits_with_error
- test_output_config_is_valid_yaml
"""

import sys
import warnings
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import tifffile
import yaml

# Ensure verification/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent))
# Also stub lammps_parser so render.py is importable from calibrate_psf tests
sys.modules.setdefault("lammps_parser", mock.MagicMock())

import calibrate_psf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synthetic_frame(
    height: int = 128,
    width: int = 128,
    n_particles: int = 30,
    psf_sigma: float = 5.0,
    peak: float = 40000.0,
    seed: int = 0,
) -> np.ndarray:
    """Generate a synthetic frame with known PSF sigma using render.render_frame.

    Particles are placed on a grid with 8σ spacing to avoid PSF tail overlap
    in the 32×32 calibration crops, which would otherwise bias sigma estimates.
    """
    # Import render here to use the procedural renderer directly
    if "render" in sys.modules:
        del sys.modules["render"]
    import render

    rng = np.random.default_rng(seed)
    # Use 8σ spacing (40px for sigma=5) so neighboring particle tails don't
    # contaminate the 32×32 crop used for fitting.
    spacing = max(int(psf_sigma * 8), 40)
    positions = []
    x, y = spacing, spacing
    while y < height - spacing:
        while x < width - spacing:
            positions.append([x / width * 10, y / height * 10])  # LJ coords in [0,10]
            x += spacing
        x = spacing
        y += spacing

    positions = (
        np.array(positions[:n_particles]) if len(positions) >= n_particles else np.array(positions)
    )

    box = (0.0, 10.0, 0.0, 10.0)
    cfg = {
        "image_height": height,
        "image_width": width,
        "psf_sigma": psf_sigma,
        "peak_intensity": peak,
        "shot_noise": False,
        "readout_noise": 0.0,
    }
    frame = render.render_frame(positions, box, cfg, rng)
    return frame.astype(np.float32)


def _write_tif(path: Path, frame: np.ndarray):
    tifffile.imwrite(str(path), frame.astype(np.float32))


# ---------------------------------------------------------------------------
# test_recovered_sigma_within_10_percent_of_ground_truth
# ---------------------------------------------------------------------------


class TestRecoveredSigma:
    def test_recovered_sigma_within_10_percent_of_ground_truth(self, tmp_path):
        """Calibration on a synthetic frame with known psf_sigma=5.0 should recover
        sigma within 10% of ground truth.

        Particles are placed at 8σ separation to ensure no PSF tail contamination
        in the 32×32 calibration crops.
        """
        true_sigma = 5.0
        # Use 512×512 so enough particles fit at 8σ spacing (≥20 for calibration)
        frame = _make_synthetic_frame(height=512, width=512, n_particles=50, psf_sigma=true_sigma)

        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        # Single frame is sufficient for sigma test
        _write_tif(frames_dir / "frame_000.tif", frame)

        frames = calibrate_psf._load_tifs(frames_dir)
        params = calibrate_psf.calibrate_from_frames(frames)

        recovered = params["psf"]["sigma_px"]
        assert (
            abs(recovered - true_sigma) / true_sigma <= 0.10
        ), f"Recovered sigma {recovered:.3f} is not within 10% of ground truth {true_sigma}"


# ---------------------------------------------------------------------------
# test_fewer_than_20_particles_prints_warning_and_continues
# ---------------------------------------------------------------------------


class TestFewParticlesWarning:
    def test_fewer_than_20_particles_prints_warning_and_continues(self, tmp_path):
        """Frame with only ~5 isolated spots: warning emitted, non-None result returned."""
        # Provide exactly 5 particles, widely spaced so all are detected
        frame = _make_synthetic_frame(
            height=256, width=256, n_particles=5, psf_sigma=5.0, peak=50000.0
        )

        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        _write_tif(frames_dir / "frame_000.tif", frame)

        frames = calibrate_psf._load_tifs(frames_dir)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            params = calibrate_psf.calibrate_from_frames(frames)

        # Should warn about few fits
        warning_msgs = [str(w.message) for w in caught]
        assert any(
            "good particle fits" in m or "noisy" in m or "estimated" in m for m in warning_msgs
        ), f"Expected a warning about few particles, got: {warning_msgs}"
        # Should still return a result dict, not crash
        assert params is not None
        assert "psf" in params
        assert params["psf"]["sigma_px"] > 0


# ---------------------------------------------------------------------------
# test_absent_real_frames_dir_exits_with_error
# ---------------------------------------------------------------------------


class TestAbsentDirectory:
    def test_absent_real_frames_dir_exits_with_error(self, tmp_path):
        """Passing a non-existent --real-frames path should cause SystemExit."""
        absent = str(tmp_path / "does_not_exist")
        with mock.patch.object(sys, "argv", ["calibrate_psf.py", "--real-frames", absent]):
            with pytest.raises(SystemExit) as exc_info:
                calibrate_psf.main()
        assert exc_info.value.code != 0


# ---------------------------------------------------------------------------
# test_empty_real_frames_dir_exits_with_error
# ---------------------------------------------------------------------------


class TestEmptyDirectory:
    def test_empty_real_frames_dir_exits_with_error(self, tmp_path):
        """Empty directory (no .tif files) should cause SystemExit with error."""
        empty_dir = tmp_path / "empty_frames"
        empty_dir.mkdir()

        with mock.patch.object(sys, "argv", ["calibrate_psf.py", "--real-frames", str(empty_dir)]):
            with pytest.raises(SystemExit) as exc_info:
                calibrate_psf.main()
        assert exc_info.value.code != 0


# ---------------------------------------------------------------------------
# test_output_config_is_valid_yaml
# ---------------------------------------------------------------------------


class TestOutputConfigYaml:
    def test_output_config_is_valid_yaml(self, tmp_path):
        """Calibration output should parse as valid YAML with expected top-level keys."""
        frame = _make_synthetic_frame(height=128, width=128, n_particles=30, psf_sigma=5.0)

        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        _write_tif(frames_dir / "frame_000.tif", frame)

        out_config = tmp_path / "calibrated.yaml"
        with mock.patch.object(
            sys,
            "argv",
            [
                "calibrate_psf.py",
                "--real-frames",
                str(frames_dir),
                "--output-config",
                str(out_config),
            ],
        ):
            calibrate_psf.main()

        assert out_config.exists()
        content = out_config.read_text()
        parsed = yaml.safe_load(content)

        assert parsed is not None, "Output is not valid YAML"
        assert "psf" in parsed, f"Missing 'psf' key; got: {list(parsed.keys())}"
        assert "particle" in parsed, f"Missing 'particle' key; got: {list(parsed.keys())}"
        assert "background" in parsed, f"Missing 'background' key; got: {list(parsed.keys())}"
        assert "noise" in parsed, f"Missing 'noise' key; got: {list(parsed.keys())}"

    def test_stdout_output_is_valid_yaml(self, tmp_path, capsys):
        """When no --output-config given, stdout should be parseable YAML with expected keys."""
        frame = _make_synthetic_frame(height=128, width=128, n_particles=30, psf_sigma=5.0)

        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        _write_tif(frames_dir / "frame_000.tif", frame)

        with mock.patch.object(sys, "argv", ["calibrate_psf.py", "--real-frames", str(frames_dir)]):
            calibrate_psf.main()

        captured = capsys.readouterr().out
        parsed = yaml.safe_load(captured)
        assert parsed is not None
        for key in ("psf", "particle", "background", "noise"):
            assert key in parsed, f"Missing '{key}' in YAML output"
