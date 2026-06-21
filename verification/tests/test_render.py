"""Tests for render.py — U1: ground-truth track export and render_strategy dispatch.
Also contains U2 tests for render_deeptrack.py (DeepTrack2 PSF + enhanced noise).
"""

import csv
import io
import json
import sys
import textwrap
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

# Make render.py importable without requiring lammps_parser on sys.path by
# injecting a minimal stub before the import.
_LAMMPS_STUB = mock.MagicMock()


def _make_block(timestep, atom_ids, xs, ys):
    """Build a minimal LAMMPS dump block dict for testing."""
    lines = [f"{aid} 1 {x} {y} 0.0" for aid, x, y in zip(atom_ids, xs, ys)]
    return {
        "timestep": timestep,
        "num_atoms": len(atom_ids),
        "box_bounds": ["0.0 10.0\n", "0.0 10.0\n", "0.0 1.0\n"],
        "atom_header": "ITEM: ATOMS id type x y z",
        "atoms": lines,
    }


@pytest.fixture(autouse=True)
def _patch_lammps_parser(monkeypatch):
    """Stub out lammps_parser so render.py imports without the real module."""
    monkeypatch.setitem(sys.modules, "lammps_parser", _LAMMPS_STUB)


@pytest.fixture()
def render_module(tmp_path, monkeypatch):
    """Import render.py with lammps-scripts on sys.path patched away."""
    # Prevent sys.path.insert inside render.py from finding a real lammps_parser
    monkeypatch.chdir(tmp_path)
    # Force re-import so the autouse stub is in place
    if "render" in sys.modules:
        del sys.modules["render"]
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import render as r

    return r


# ---------------------------------------------------------------------------
# _parse_atoms
# ---------------------------------------------------------------------------


class TestParseAtoms:
    def test_returns_positions_and_ids(self, render_module):
        header = "ITEM: ATOMS id type x y z"
        atoms = ["1 1 1.0 2.0 0.0", "2 1 3.0 4.0 0.0", "3 1 5.0 6.0 0.0"]
        positions, ids = render_module._parse_atoms(header, atoms)

        assert positions.shape == (3, 2)
        np.testing.assert_array_equal(ids, [1, 2, 3])
        np.testing.assert_allclose(positions[:, 0], [1.0, 3.0, 5.0])
        np.testing.assert_allclose(positions[:, 1], [2.0, 4.0, 6.0])

    def test_prefers_unwrapped_coords(self, render_module):
        header = "ITEM: ATOMS id type x y xu yu z"
        atoms = ["1 1 0.5 0.5 1.5 2.5 0.0"]
        positions, _ = render_module._parse_atoms(header, atoms)
        np.testing.assert_allclose(positions[0], [1.5, 2.5])

    def test_sequential_ids_when_no_id_column(self, render_module):
        header = "ITEM: ATOMS type x y z"
        atoms = ["1 1.0 2.0 0.0", "1 3.0 4.0 0.0"]
        _, ids = render_module._parse_atoms(header, atoms)
        np.testing.assert_array_equal(ids, [1, 2])

    def test_empty_atoms_returns_empty(self, render_module):
        header = "ITEM: ATOMS id type x y z"
        positions, ids = render_module._parse_atoms(header, [])
        assert positions.shape == (0, 2)
        assert ids.shape == (0,)


# ---------------------------------------------------------------------------
# _parse_positions backward-compatibility
# ---------------------------------------------------------------------------


def test_parse_positions_still_works(render_module):
    header = "ITEM: ATOMS id type x y z"
    atoms = ["1 1 1.0 2.0 0.0", "2 1 3.0 4.0 0.0"]
    pos = render_module._parse_positions(header, atoms)
    assert pos.shape == (2, 2)


# ---------------------------------------------------------------------------
# _lj_to_pixels
# ---------------------------------------------------------------------------


class TestLjToPixels:
    def test_center_maps_to_half_image(self, render_module):
        box = (0.0, 10.0, 0.0, 10.0)
        positions = np.array([[5.0, 5.0]])
        px = render_module._lj_to_pixels(positions, box, H=512, W=512)
        np.testing.assert_allclose(px[0], [256.0, 256.0])

    def test_clips_to_boundary(self, render_module):
        box = (0.0, 10.0, 0.0, 10.0)
        positions = np.array([[-1.0, -1.0], [11.0, 11.0]])
        px = render_module._lj_to_pixels(positions, box, H=512, W=512)
        assert px[0, 0] == 0
        assert px[0, 1] == 0
        assert px[1, 0] == 511
        assert px[1, 1] == 511


# ---------------------------------------------------------------------------
# ground_truth_tracks.csv written by main()
# ---------------------------------------------------------------------------


def _minimal_cfg(tmp_path):
    import yaml

    cfg = {
        "synthetic": {
            "render_strategy": "procedural",
            "image_width": 64,
            "image_height": 64,
            "psf_sigma": 2.0,
            "peak_intensity": 1000,
            "shot_noise": False,
            "readout_noise": 0.0,
            "output_dir": str(tmp_path / "frames"),
        }
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.dump(cfg))
    return str(cfg_path)


def _run_main_with_blocks(render_module, tmp_path, blocks):
    """Patch parse_lammps_dump to yield `blocks`, run main()."""
    cfg_path = _minimal_cfg(tmp_path)
    _LAMMPS_STUB.parse_lammps_dump.return_value = iter(blocks)

    with mock.patch.object(
        sys, "argv", ["render.py", "--lammps", "fake.lammpstrj", "--config", cfg_path]
    ):
        render_module.main()


class TestGroundTruthTracksCSV:
    def test_csv_row_count_equals_frames_times_atoms(self, render_module, tmp_path):
        # 3 frames × 5 atoms = 15 rows
        blocks = [
            _make_block(t * 100, [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
            for t in range(3)
        ]
        _run_main_with_blocks(render_module, tmp_path, blocks)

        tracks_path = tmp_path / "ground_truth_tracks.csv"
        assert tracks_path.exists()
        with open(tracks_path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 15

    def test_csv_columns_are_correct(self, render_module, tmp_path):
        blocks = [_make_block(0, [1, 2], [1.0, 2.0], [3.0, 4.0])]
        _run_main_with_blocks(render_module, tmp_path, blocks)
        tracks_path = tmp_path / "ground_truth_tracks.csv"
        with open(tracks_path) as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == ["frame", "particle_id", "x", "y"]

    def test_same_particle_id_in_all_frames(self, render_module, tmp_path):
        blocks = [_make_block(t * 100, [10, 20, 30], [1, 2, 3], [1, 2, 3]) for t in range(3)]
        _run_main_with_blocks(render_module, tmp_path, blocks)
        tracks_path = tmp_path / "ground_truth_tracks.csv"
        with open(tracks_path) as f:
            rows = list(csv.DictReader(f))
        id_by_frame = {}
        for row in rows:
            id_by_frame.setdefault(int(row["frame"]), set()).add(int(row["particle_id"]))
        # All frames should have same particle IDs
        frame_sets = list(id_by_frame.values())
        assert all(s == frame_sets[0] for s in frame_sets)
        assert frame_sets[0] == {10, 20, 30}

    def test_ground_truth_json_still_written(self, render_module, tmp_path):
        blocks = [_make_block(0, [1], [5.0], [5.0])]
        _run_main_with_blocks(render_module, tmp_path, blocks)
        gt_path = tmp_path / "ground_truth.json"
        assert gt_path.exists()
        with open(gt_path) as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["frame"] == 0
        assert data[0]["n_particles"] == 1

    def test_out_of_box_particle_clipped_not_omitted(self, render_module, tmp_path):
        # Particle at x=15 (beyond box max of 10) should be clipped to image edge
        blocks = [_make_block(0, [1, 2], [15.0, 5.0], [5.0, 5.0])]
        _run_main_with_blocks(render_module, tmp_path, blocks)
        tracks_path = tmp_path / "ground_truth_tracks.csv"
        with open(tracks_path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2  # both particles present, first clipped

    def test_unstable_atom_ids_raises_assertion_error(self, render_module, tmp_path):
        # Frame 0: atoms [1,2,3], frame 1: atoms [1,2,4] — set differs
        blocks = [
            _make_block(0, [1, 2, 3], [1, 2, 3], [1, 2, 3]),
            _make_block(100, [1, 2, 4], [1, 2, 3], [1, 2, 3]),
        ]
        with pytest.raises(AssertionError, match="Atom ID set changed"):
            _run_main_with_blocks(render_module, tmp_path, blocks)


# ---------------------------------------------------------------------------
# render_strategy dispatch
# ---------------------------------------------------------------------------


class TestRenderStrategyDispatch:
    def test_procedural_strategy_produces_uint16(self, render_module):
        positions = np.array([[5.0, 5.0]])
        box = (0.0, 10.0, 0.0, 10.0)
        cfg = {
            "image_height": 64,
            "image_width": 64,
            "psf_sigma": 2.0,
            "peak_intensity": 1000,
            "shot_noise": False,
            "readout_noise": 0.0,
        }
        rng = np.random.default_rng(42)
        frame = render_module._dispatch_render(positions, box, cfg, rng, "procedural")
        assert frame.dtype == np.uint16
        assert frame.shape == (64, 64)

    def test_unknown_strategy_falls_back_to_procedural(self, render_module):
        """Unknown strategies route to procedural (the else branch)."""
        positions = np.array([[5.0, 5.0]])
        box = (0.0, 10.0, 0.0, 10.0)
        cfg = {
            "image_height": 32,
            "image_width": 32,
            "psf_sigma": 2.0,
            "peak_intensity": 500,
            "shot_noise": False,
            "readout_noise": 0.0,
        }
        rng = np.random.default_rng(0)
        frame = render_module._dispatch_render(positions, box, cfg, rng, "unknown_strategy")
        assert frame.dtype == np.uint16


# ---------------------------------------------------------------------------
# U2: render_deeptrack.py — DeepTrack2 PSF + enhanced noise model
# ---------------------------------------------------------------------------


def _make_fake_kernel(H, W):
    """Return a small normalised float32 kernel that looks like a PSF."""
    kernel = np.zeros((H, W), dtype=np.float32)
    cy, cx = H // 2, W // 2
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            kernel[cy + dy, cx + dx] = np.exp(-0.5 * (dy**2 + dx**2))
    kernel /= kernel.sum()
    return kernel


def _deeptrack_cfg(H=32, W=32):
    """Minimal config dict for render_frame_deeptrack."""
    return {
        "image_height": H,
        "image_width": W,
        "psf": {"na": 1.4, "wavelength": 520e-9, "resolution": 65e-9},
        "background": {"heterogeneity_scale": 5, "amplitude": 200},
        "particle": {
            "intensity_distribution": "lognormal",
            "peak_mean": 5000,
            "intensity_sigma": 0.3,
        },
        "noise": {"gain_sigma": 0.02, "read_noise": 10.0},
    }


@pytest.fixture()
def render_deeptrack_module(tmp_path, monkeypatch):
    """Import render_deeptrack with deeptrack stubbed out."""
    # Ensure render_deeptrack is re-imported fresh each test
    for mod in list(sys.modules.keys()):
        if "render_deeptrack" in mod:
            del sys.modules[mod]
    sys.path.insert(0, str(Path(__file__).parent.parent))
    return None  # modules imported inside tests for finer control


class TestDeeptrackStrategy:
    """U2: render_frame_deeptrack — using mocked deeptrack so tests run without the real package."""

    def _import_with_mock_deeptrack(self, fake_kernel):
        """Import render_deeptrack with deeptrack mocked to return fake_kernel."""
        import types

        # Remove any cached imports
        for key in list(sys.modules.keys()):
            if "render_deeptrack" in key:
                del sys.modules[key]

        # Build a minimal deeptrack stub
        fake_pipeline = mock.MagicMock()
        fake_pipeline.update.return_value = fake_pipeline
        fake_pipeline.resolve.return_value = fake_kernel

        fake_optics_instance = mock.MagicMock(return_value=fake_pipeline)
        fake_point_particle = mock.MagicMock()

        deeptrack_stub = types.ModuleType("deeptrack")
        deeptrack_stub.Fluorescence = mock.MagicMock(return_value=fake_optics_instance)
        deeptrack_stub.PointParticle = mock.MagicMock(return_value=fake_point_particle)

        sys.modules["deeptrack"] = deeptrack_stub
        sys.path.insert(0, str(Path(__file__).parent.parent))

        import render_deeptrack as rdt

        return rdt

    def test_deeptrack_strategy_produces_uint16_frame(self):
        H, W = 32, 32
        fake_kernel = _make_fake_kernel(H, W)
        rdt = self._import_with_mock_deeptrack(fake_kernel)

        positions = np.array([[5.0, 5.0], [3.0, 7.0]])
        box = (0.0, 10.0, 0.0, 10.0)
        cfg = _deeptrack_cfg(H, W)
        rng = np.random.default_rng(42)

        frame = rdt.render_frame_deeptrack(positions, box, cfg, rng)

        assert frame.dtype == np.uint16
        assert frame.shape == (H, W)

    def test_background_field_has_spatial_variation(self):
        """When background amplitude > 0, the frame should have measurable std."""
        H, W = 32, 32
        fake_kernel = _make_fake_kernel(H, W)
        rdt = self._import_with_mock_deeptrack(fake_kernel)

        # Use amplitude=1000 so background dominates, no particles
        positions = np.zeros((0, 2))
        box = (0.0, 10.0, 0.0, 10.0)
        cfg = _deeptrack_cfg(H, W)
        cfg["background"]["amplitude"] = 1000
        cfg["noise"]["gain_sigma"] = 0.0  # disable gain variation for isolation
        cfg["noise"]["read_noise"] = 0.0  # disable read noise for isolation
        # Use only 1 particle to avoid random intensity dominating
        rng = np.random.default_rng(7)

        frame = rdt.render_frame_deeptrack(positions, box, cfg, rng)

        # Background should produce non-zero std (frame is not flat)
        assert frame.std() > 0

    def test_lognormal_intensity_varies(self):
        """Over 20 renders, the sampled peak intensities should have std > 0."""
        H, W = 16, 16
        fake_kernel = _make_fake_kernel(H, W)
        rdt = self._import_with_mock_deeptrack(fake_kernel)

        # Single particle at centre; measure peak value of output
        box = (0.0, 10.0, 0.0, 10.0)
        cfg = _deeptrack_cfg(H, W)
        cfg["background"]["amplitude"] = 0  # disable background
        cfg["noise"]["gain_sigma"] = 0.0
        cfg["noise"]["read_noise"] = 0.0

        peak_vals = []
        rng = np.random.default_rng(99)
        for _ in range(20):
            positions = np.array([[5.0, 5.0]])
            frame = rdt.render_frame_deeptrack(positions, box, cfg, rng)
            peak_vals.append(float(frame.max()))

        assert np.std(peak_vals) > 0, "Log-normal intensities should vary across renders"

    def test_missing_deeptrack_package_raises_import_error(self):
        """When deeptrack is absent, render_frame_deeptrack raises ImportError with install hint."""
        # Remove any cached imports
        for key in list(sys.modules.keys()):
            if "render_deeptrack" in key:
                del sys.modules[key]

        # Remove deeptrack from sys.modules and block its import
        sys.modules.pop("deeptrack", None)

        sys.path.insert(0, str(Path(__file__).parent.parent))

        with mock.patch.dict(sys.modules, {"deeptrack": None}):
            import render_deeptrack as rdt  # noqa: F811

            for key in list(sys.modules.keys()):
                if "render_deeptrack" in key:
                    del sys.modules[key]

        # Re-import with deeptrack blocked
        with mock.patch.dict(sys.modules, {"deeptrack": None}):
            # Force a fresh import
            import importlib

            spec = importlib.util.spec_from_file_location(
                "render_deeptrack_fresh",
                str(Path(__file__).parent.parent / "render_deeptrack.py"),
            )
            fresh_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(fresh_mod)

            positions = np.array([[5.0, 5.0]])
            box = (0.0, 10.0, 0.0, 10.0)
            cfg = _deeptrack_cfg()
            rng = np.random.default_rng(1)

            with pytest.raises(ImportError, match="deeptrack==2.0.1"):
                fresh_mod.render_frame_deeptrack(positions, box, cfg, rng)

    def test_procedural_strategy_unchanged_by_u2(self):
        """Procedural strategy still produces (H, W) uint16 after U2 changes."""
        # This uses the existing render_module fixture pattern inline
        for mod in list(sys.modules.keys()):
            if mod == "render":
                del sys.modules[mod]

        sys.modules.setdefault("lammps_parser", mock.MagicMock())
        sys.path.insert(0, str(Path(__file__).parent.parent))
        import render as r

        positions = np.array([[5.0, 5.0]])
        box = (0.0, 10.0, 0.0, 10.0)
        cfg = {
            "image_height": 32,
            "image_width": 32,
            "psf_sigma": 2.0,
            "peak_intensity": 1000,
            "shot_noise": False,
            "readout_noise": 0.0,
        }
        rng = np.random.default_rng(42)
        frame = r._dispatch_render(positions, box, cfg, rng, "procedural")
        assert frame.dtype == np.uint16
        assert frame.shape == (32, 32)


# ---------------------------------------------------------------------------
# U3: TestRandomizedStrategy
# ---------------------------------------------------------------------------


@pytest.fixture()
def rr_module():
    """Import render_randomized with render.py importable from verification/."""
    verification_dir = str(Path(__file__).parent.parent)
    if verification_dir not in sys.path:
        sys.path.insert(0, verification_dir)

    for mod_name in list(sys.modules):
        if mod_name in ("render_randomized", "render"):
            del sys.modules[mod_name]

    import render_randomized as rr

    return rr


def _base_cfg():
    """Minimal procedural cfg suitable for passing to render_frame_randomized."""
    return {
        "image_height": 32,
        "image_width": 32,
        "psf_sigma": 5.0,
        "peak_intensity": 40000,
        "shot_noise": False,
        "readout_noise": 15.0,
        "randomization": {
            "psf_sigma_range": [3.0, 7.0],
            "peak_range": [20000, 60000],
            "readout_noise_range": [10.0, 25.0],
        },
    }


class TestRandomizedStrategy:
    def test_randomized_produces_uint16_frame(self, rr_module):
        cfg = _base_cfg()
        positions = np.array([[5.0, 5.0], [8.0, 3.0]])
        box = (0.0, 10.0, 0.0, 10.0)
        rng = np.random.default_rng(0)
        frame = rr_module.render_frame_randomized(positions, box, cfg, rng)
        assert frame.dtype == np.uint16
        assert frame.shape == (32, 32)

    def test_over_20_frames_at_least_3_distinct_sigma_values(self, rr_module):
        cfg = _base_cfg()
        positions = np.array([[5.0, 5.0]])
        box = (0.0, 10.0, 0.0, 10.0)
        rng = np.random.default_rng(42)
        sampled_sigmas = []

        original_render_frame = rr_module.render_frame

        def capturing_render_frame(positions_lj, box_arg, frame_cfg, rng_inner):
            sampled_sigmas.append(frame_cfg["psf_sigma"])
            return original_render_frame(positions_lj, box_arg, frame_cfg, rng_inner)

        with mock.patch.object(rr_module, "render_frame", side_effect=capturing_render_frame):
            for _ in range(20):
                rr_module.render_frame_randomized(positions, box, cfg, rng)

        distinct_values = set(round(s, 6) for s in sampled_sigmas)
        assert len(distinct_values) >= 3

    def test_fixed_seed_is_reproducible(self, rr_module):
        cfg = _base_cfg()
        positions = np.array([[2.0, 3.0], [7.0, 6.0]])
        box = (0.0, 10.0, 0.0, 10.0)
        frame1 = rr_module.render_frame_randomized(positions, box, cfg, np.random.default_rng(99))
        frame2 = rr_module.render_frame_randomized(positions, box, cfg, np.random.default_rng(99))
        np.testing.assert_array_equal(frame1, frame2)

    def test_invalid_range_min_greater_than_max_raises_value_error(self, rr_module):
        cfg = _base_cfg()
        cfg["randomization"]["psf_sigma_range"] = [7.0, 3.0]
        with pytest.raises(ValueError, match="psf_sigma_range"):
            rr_module.render_frame_randomized(
                np.array([[5.0, 5.0]]), (0.0, 10.0, 0.0, 10.0), cfg, np.random.default_rng(0)
            )

    def test_invalid_peak_range_raises_value_error(self, rr_module):
        cfg = _base_cfg()
        cfg["randomization"]["peak_range"] = [60000, 20000]
        with pytest.raises(ValueError, match="peak_range"):
            rr_module.render_frame_randomized(
                np.array([[5.0, 5.0]]), (0.0, 10.0, 0.0, 10.0), cfg, np.random.default_rng(0)
            )

    def test_invalid_noise_range_raises_value_error(self, rr_module):
        cfg = _base_cfg()
        cfg["randomization"]["readout_noise_range"] = [25.0, 10.0]
        with pytest.raises(ValueError, match="readout_noise_range"):
            rr_module.render_frame_randomized(
                np.array([[5.0, 5.0]]), (0.0, 10.0, 0.0, 10.0), cfg, np.random.default_rng(0)
            )

    def test_randomized_uses_procedural_render_internally(self, rr_module):
        cfg = _base_cfg()
        positions = np.array([[5.0, 5.0]])
        box = (0.0, 10.0, 0.0, 10.0)
        rng = np.random.default_rng(7)
        call_args = {}
        original = rr_module.render_frame

        def spy(positions_lj, box_arg, frame_cfg, rng_arg):
            call_args["psf_sigma"] = frame_cfg["psf_sigma"]
            call_args["peak_intensity"] = frame_cfg["peak_intensity"]
            call_args["readout_noise"] = frame_cfg["readout_noise"]
            return original(positions_lj, box_arg, frame_cfg, rng_arg)

        with mock.patch.object(rr_module, "render_frame", side_effect=spy):
            rr_module.render_frame_randomized(positions, box, cfg, rng)

        assert 3.0 <= call_args["psf_sigma"] <= 7.0
        assert 20000 <= call_args["peak_intensity"] <= 60000
        assert 10.0 <= call_args["readout_noise"] <= 25.0

    def test_default_ranges_used_when_randomization_key_absent(self, rr_module):
        cfg = {
            "image_height": 32,
            "image_width": 32,
            "psf_sigma": 5.0,
            "peak_intensity": 40000,
            "shot_noise": False,
            "readout_noise": 15.0,
        }
        frame = rr_module.render_frame_randomized(
            np.array([[5.0, 5.0]]), (0.0, 10.0, 0.0, 10.0), cfg, np.random.default_rng(1)
        )
        assert frame.dtype == np.uint16
        assert frame.shape == (32, 32)


# ---------------------------------------------------------------------------
# U2 (background composite): render_background_composite.py
# ---------------------------------------------------------------------------


def _make_tifffile_stub(n_pages: int, page_shape=(16, 16)):
    """Build a tifffile stub whose TiffFile context manager returns n_pages of synthetic data.

    Each page returns a distinct uint16 array filled with the page index value
    (page 0 → all zeros, page 1 → all ones, …).
    """
    import types

    stub = types.ModuleType("tifffile")

    class _FakePage:
        def __init__(self, idx):
            self._idx = idx
            self._shape = page_shape

        def asarray(self):
            return np.full(self._shape, self._idx, dtype=np.uint16)

    class _FakeTiffFile:
        def __init__(self, path):
            self.pages = [_FakePage(i) for i in range(n_pages)]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    stub.TiffFile = _FakeTiffFile
    return stub


def _bc_module(tifffile_stub=None):
    """Import render_background_composite with optional tifffile stub injected."""
    # Remove any cached import
    for key in list(sys.modules.keys()):
        if "render_background_composite" in key:
            del sys.modules[key]

    if tifffile_stub is not None:
        sys.modules["tifffile"] = tifffile_stub

    sys.path.insert(0, str(Path(__file__).parent.parent))
    import render_background_composite as rbc

    return rbc


def _bg_cfg(H=16, W=16):
    """Minimal config dict for render_frame_background_composite tests."""
    return {
        "image_height": H,
        "image_width": W,
        "psf": {"sigma_px": 2.0},
        "particle": {"peak_mean": 1000, "intensity_sigma": 0.1},
        "noise": {"read_noise": 5.0},
        "_background_frame": np.zeros((H, W), dtype=np.float32),
    }


class TestBackgroundCompositeStrategy:
    """Tests for render_background_composite.py — background extraction and composite render."""

    # ------------------------------------------------------------------
    # extract_temporal_median
    # ------------------------------------------------------------------

    def test_extract_temporal_median_shape_and_dtype(self):
        """10-page stub: returns float32 (16, 16)."""
        stub = _make_tifffile_stub(n_pages=10, page_shape=(16, 16))
        rbc = _bc_module(stub)
        result = rbc.extract_temporal_median("fake.tif", n_frames=5)
        assert result.dtype == np.float32
        assert result.shape == (16, 16)

    def test_extract_temporal_median_plausible_values(self):
        """Median of pages 0..9 should be between 0 and 9."""
        stub = _make_tifffile_stub(n_pages=10, page_shape=(16, 16))
        rbc = _bc_module(stub)
        result = rbc.extract_temporal_median("fake.tif", n_frames=10)
        # page values are 0..9; median of a subset of these must be in [0, 9]
        assert float(result.min()) >= 0.0
        assert float(result.max()) <= 9.0

    def test_extract_temporal_median_fewer_pages_than_n_frames(self):
        """3 available pages, n_frames=50: uses all 3, no IndexError."""
        stub = _make_tifffile_stub(n_pages=3, page_shape=(16, 16))
        rbc = _bc_module(stub)
        result = rbc.extract_temporal_median("fake.tif", n_frames=50)
        assert result.shape == (16, 16)
        assert result.dtype == np.float32

    # ------------------------------------------------------------------
    # render_frame_background_composite — basic output
    # ------------------------------------------------------------------

    def test_render_returns_uint16_correct_shape(self):
        """Output is uint16 with shape (H, W) from cfg."""
        rbc = _bc_module()
        cfg = _bg_cfg(H=16, W=16)
        positions = np.array([[5.0, 5.0], [3.0, 7.0]])
        box = (0.0, 10.0, 0.0, 10.0)
        rng = np.random.default_rng(42)
        frame = rbc.render_frame_background_composite(positions, box, cfg, rng)
        assert frame.dtype == np.uint16
        assert frame.shape == (16, 16)

    def test_render_background_only_non_negative(self):
        """Zero particles: output is non-negative everywhere."""
        rbc = _bc_module()
        cfg = _bg_cfg(H=16, W=16)
        positions = np.zeros((0, 2))
        box = (0.0, 10.0, 0.0, 10.0)
        rng = np.random.default_rng(1)
        frame = rbc.render_frame_background_composite(positions, box, cfg, rng)
        assert frame.dtype == np.uint16
        assert frame.shape == (16, 16)
        # uint16 is always >= 0 by type, but confirm the clip is correct
        assert int(frame.min()) >= 0

    # ------------------------------------------------------------------
    # Particle signal is stamped
    # ------------------------------------------------------------------

    def test_particle_at_known_position_elevates_region(self):
        """Particle near image centre should raise values vs. background-only."""
        rbc = _bc_module()
        box = (0.0, 10.0, 0.0, 10.0)
        H, W = 16, 16

        # Background-only baseline (no noise for clean comparison)
        cfg_bg = _bg_cfg(H, W)
        cfg_bg["noise"]["read_noise"] = 0.0
        cfg_bg["particle"]["peak_mean"] = 50000
        cfg_bg["particle"]["intensity_sigma"] = 0.0  # constant intensity

        rng_bg = np.random.default_rng(0)
        frame_bg = rbc.render_frame_background_composite(np.zeros((0, 2)), box, cfg_bg, rng_bg)

        # With a bright particle at the image centre
        cfg_p = _bg_cfg(H, W)
        cfg_p["noise"]["read_noise"] = 0.0
        cfg_p["particle"]["peak_mean"] = 50000
        cfg_p["particle"]["intensity_sigma"] = 0.0

        rng_p = np.random.default_rng(0)
        centre = np.array([[5.0, 5.0]])  # maps to pixel ~(8, 8)
        frame_p = rbc.render_frame_background_composite(centre, box, cfg_p, rng_p)

        # Total signal should be higher when a bright particle is present
        assert int(frame_p.sum()) > int(frame_bg.sum())

    # ------------------------------------------------------------------
    # Noise randomness
    # ------------------------------------------------------------------

    def test_different_seeds_produce_different_output(self):
        """Two calls with different rng seeds should differ (Poisson + read noise)."""
        rbc = _bc_module()
        cfg = _bg_cfg()
        positions = np.array([[5.0, 5.0]])
        box = (0.0, 10.0, 0.0, 10.0)
        frame1 = rbc.render_frame_background_composite(
            positions, box, cfg, np.random.default_rng(0)
        )
        frame2 = rbc.render_frame_background_composite(
            positions, box, cfg, np.random.default_rng(999)
        )
        assert not np.array_equal(frame1, frame2)

    # ------------------------------------------------------------------
    # Error conditions
    # ------------------------------------------------------------------

    def test_missing_background_frame_raises_key_error(self):
        """KeyError raised with informative message when _background_frame absent."""
        rbc = _bc_module()
        cfg = _bg_cfg()
        del cfg["_background_frame"]
        with pytest.raises(KeyError, match="_background_frame"):
            rbc.render_frame_background_composite(
                np.zeros((0, 2)), (0.0, 10.0, 0.0, 10.0), cfg, np.random.default_rng(0)
            )

    # ------------------------------------------------------------------
    # PSF sigma fallback
    # ------------------------------------------------------------------

    def test_sigma_fallback_to_psf_sigma_key(self):
        """When psf.sigma_px absent, psf_sigma top-level key is used without error."""
        rbc = _bc_module()
        cfg = _bg_cfg()
        # Remove the nested psf.sigma_px and use flat psf_sigma key instead
        cfg["psf"] = {}
        cfg["psf_sigma"] = 3.0
        frame = rbc.render_frame_background_composite(
            np.array([[5.0, 5.0]]), (0.0, 10.0, 0.0, 10.0), cfg, np.random.default_rng(7)
        )
        assert frame.dtype == np.uint16

    # ------------------------------------------------------------------
    # Clip to [0, 65535]
    # ------------------------------------------------------------------

    def test_output_clipped_to_uint16_range(self):
        """Very bright background + bright particles must not overflow uint16."""
        rbc = _bc_module()
        cfg = _bg_cfg(H=16, W=16)
        # Saturating background
        cfg["_background_frame"] = np.full((16, 16), 60000, dtype=np.float32)
        cfg["particle"]["peak_mean"] = 50000
        cfg["noise"]["read_noise"] = 0.0

        positions = np.array([[5.0, 5.0]])
        box = (0.0, 10.0, 0.0, 10.0)
        frame = rbc.render_frame_background_composite(
            positions, box, cfg, np.random.default_rng(42)
        )
        assert frame.dtype == np.uint16
        assert int(frame.max()) <= 65535


# ---------------------------------------------------------------------------
# U3: _dispatch_render routes to background_composite; main() pre-loads once
# ---------------------------------------------------------------------------


class TestDispatchBackgroundComposite:
    """U3 wiring: _dispatch_render delegates to render_frame_background_composite."""

    def _fresh_render(self):
        for key in list(sys.modules):
            if key == "render":
                del sys.modules[key]
        sys.modules.setdefault("lammps_parser", mock.MagicMock())
        sys.path.insert(0, str(Path(__file__).parent.parent))
        import render as r

        return r

    def test_dispatch_calls_render_frame_background_composite(self):
        """_dispatch_render('background_composite') invokes render_frame_background_composite."""
        r = self._fresh_render()
        positions = np.array([[5.0, 5.0]])
        box = (0.0, 10.0, 0.0, 10.0)
        sentinel = np.zeros((16, 16), dtype=np.uint16)
        cfg = {
            "image_height": 16,
            "image_width": 16,
            "_background_frame": np.zeros((16, 16), dtype=np.float32),
            "psf": {"sigma_px": 2.0},
            "particle": {"peak_mean": 1000, "intensity_sigma": 0.1},
            "noise": {"read_noise": 5.0},
        }
        rng = np.random.default_rng(0)

        fake_rbc = mock.MagicMock()
        fake_rbc.render_frame_background_composite.return_value = sentinel
        # Remove any previously cached real module so the mock is picked up.
        sys.modules.pop("render_background_composite", None)
        with mock.patch.dict(sys.modules, {"render_background_composite": fake_rbc}):
            result = r._dispatch_render(positions, box, cfg, rng, "background_composite")

        fake_rbc.render_frame_background_composite.assert_called_once()
        assert result is sentinel

    def test_dispatch_background_composite_missing_module_raises_import_error(self):
        """ImportError with helpful message when render_background_composite.py is absent."""
        r = self._fresh_render()
        cfg = {
            "image_height": 16,
            "image_width": 16,
            "_background_frame": np.zeros((16, 16), dtype=np.float32),
        }
        with mock.patch.dict(sys.modules, {"render_background_composite": None}):
            with pytest.raises(ImportError, match="render_background_composite"):
                r._dispatch_render(
                    np.zeros((0, 2)),
                    (0.0, 10.0, 0.0, 10.0),
                    cfg,
                    np.random.default_rng(0),
                    "background_composite",
                )

    def test_main_preloads_background_exactly_once(self, tmp_path):
        """render.py main() calls extract_temporal_median once before the frame loop."""
        import yaml

        for key in list(sys.modules):
            if key == "render":
                del sys.modules[key]
        sys.modules.setdefault("lammps_parser", mock.MagicMock())
        sys.path.insert(0, str(Path(__file__).parent.parent))
        import render as r

        # Config with background_composite strategy and a dummy video_path
        cfg_dict = {
            "synthetic": {
                "render_strategy": "background_composite",
                "image_width": 16,
                "image_height": 16,
                "psf_sigma": 2.0,
                "peak_intensity": 1000,
                "shot_noise": False,
                "readout_noise": 0.0,
                "output_dir": str(tmp_path / "frames"),
                "psf": {"sigma_px": 2.0},
                "particle": {"peak_mean": 1000, "intensity_sigma": 0.1},
                "noise": {"read_noise": 5.0},
                "background_composite": {
                    "video_path": "fake_video.tif",
                    "n_frames_for_median": 5,
                },
            }
        }
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump(cfg_dict))

        fake_bg = np.zeros((16, 16), dtype=np.float32)

        # One LAMMPS block so the frame loop runs exactly once
        block = {
            "timestep": 0,
            "num_atoms": 1,
            "box_bounds": ["0.0 10.0\n", "0.0 10.0\n", "0.0 1.0\n"],
            "atom_header": "ITEM: ATOMS id type x y z",
            "atoms": ["1 1 5.0 5.0 0.0"],
        }
        sys.modules["lammps_parser"].parse_lammps_dump.return_value = iter([block])

        extract_calls = []

        def fake_extract(video_path, n_frames, rng):
            extract_calls.append(video_path)
            return fake_bg

        fake_rbc = mock.MagicMock()
        fake_rbc.extract_temporal_median.side_effect = fake_extract
        fake_rbc.render_frame_background_composite.return_value = np.zeros(
            (16, 16), dtype=np.uint16
        )

        # Earlier tests may leave a tifffile stub in sys.modules; patch imwrite on the
        # render module object so it uses a no-op regardless of what tifffile resolves to.
        sys.modules.pop("render_background_composite", None)
        with mock.patch.dict(sys.modules, {"render_background_composite": fake_rbc}):
            with mock.patch.object(r, "tifffile") as mock_tifffile:
                mock_tifffile.imwrite = mock.MagicMock()
                with mock.patch.object(
                    sys,
                    "argv",
                    ["render.py", "--lammps", "fake.lammpstrj", "--config", str(cfg_path)],
                ):
                    r.main()

        # extract_temporal_median called exactly once (pre-load, not per-frame)
        assert len(extract_calls) == 1
        assert extract_calls[0] == "fake_video.tif"
