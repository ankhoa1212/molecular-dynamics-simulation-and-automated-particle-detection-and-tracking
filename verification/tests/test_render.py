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
# U4: crop_source (physics | real | procedural)
# ---------------------------------------------------------------------------


def _import_deeptrack_with_mock(fake_kernel):
    """Same stub pattern as TestDeeptrackStrategy._import_with_mock_deeptrack,
    factored out for reuse by crop_source tests that don't subclass it."""
    import types

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


def _fake_template_library(n=3, size=9):
    """A small library of distinguishable normalized templates (each with a
    single bright pixel at a different location, sum == 1)."""
    templates = []
    for i in range(n):
        t = np.zeros((size, size), dtype=np.float32)
        t[size // 2, (size // 2 + i) % size] = 1.0
        templates.append(t)
    return np.stack(templates, axis=0)


class TestCropSourcePhysicsRegression:
    def test_default_and_explicit_physics_are_byte_identical(self):
        """crop_source omitted vs. crop_source: 'physics' must render
        byte-identical output for the same seed — the default fallback must
        route through the exact same code path as an explicit 'physics'."""
        H, W = 32, 32
        fake_kernel = _make_fake_kernel(H, W)
        rdt = _import_deeptrack_with_mock(fake_kernel)

        positions = np.array([[5.0, 5.0], [3.0, 7.0]])
        box = (0.0, 10.0, 0.0, 10.0)

        cfg_default = _deeptrack_cfg(H, W)
        cfg_explicit = _deeptrack_cfg(H, W)
        cfg_explicit["crop_source"] = "physics"

        frame_default = rdt.render_frame_deeptrack(
            positions, box, cfg_default, np.random.default_rng(42)
        )
        frame_explicit = rdt.render_frame_deeptrack(
            positions, box, cfg_explicit, np.random.default_rng(42)
        )

        assert np.array_equal(frame_default, frame_explicit)

    def test_unknown_crop_source_raises_value_error(self):
        H, W = 32, 32
        fake_kernel = _make_fake_kernel(H, W)
        rdt = _import_deeptrack_with_mock(fake_kernel)

        cfg = _deeptrack_cfg(H, W)
        cfg["crop_source"] = "not_a_real_source"

        with pytest.raises(ValueError, match="not_a_real_source"):
            rdt.render_frame_deeptrack(
                np.array([[5.0, 5.0]]), (0.0, 10.0, 0.0, 10.0), cfg, np.random.default_rng(0)
            )


class TestCropSourceReal:
    def test_elevated_intensity_at_particle_positions_with_varying_templates(self, monkeypatch):
        H, W = 64, 64
        fake_kernel = _make_fake_kernel(H, W)
        rdt = _import_deeptrack_with_mock(fake_kernel)

        import render_crop_templates as rct

        templates = _fake_template_library(n=4, size=9)
        monkeypatch.setattr(rct, "load_template_library", lambda path: templates)

        cfg = _deeptrack_cfg(H, W)
        cfg["crop_source"] = "real"
        cfg["crop_template"] = {"cache_path": "unused-because-mocked.npz"}
        cfg["background"]["amplitude"] = 0
        cfg["noise"]["gain_sigma"] = 0.0
        cfg["noise"]["read_noise"] = 0.0

        positions = np.array([[2.0, 2.0], [8.0, 8.0]])
        box = (0.0, 10.0, 0.0, 10.0)
        frame = rdt.render_frame_deeptrack(positions, box, cfg, np.random.default_rng(3))

        assert frame.dtype == np.uint16
        assert frame.max() > 0  # some particle intensity landed on the canvas

    def test_missing_cache_path_key_raises_informative_error(self):
        H, W = 32, 32
        fake_kernel = _make_fake_kernel(H, W)
        rdt = _import_deeptrack_with_mock(fake_kernel)

        cfg = _deeptrack_cfg(H, W)
        cfg["crop_source"] = "real"
        cfg["crop_template"] = {}  # no cache_path

        with pytest.raises(ValueError, match="crop_template.cache_path"):
            rdt.render_frame_deeptrack(
                np.array([[5.0, 5.0]]), (0.0, 10.0, 0.0, 10.0), cfg, np.random.default_rng(0)
            )

    def test_nonexistent_cache_file_raises_informative_error_not_bare_exception(self):
        H, W = 32, 32
        fake_kernel = _make_fake_kernel(H, W)
        rdt = _import_deeptrack_with_mock(fake_kernel)

        cfg = _deeptrack_cfg(H, W)
        cfg["crop_source"] = "real"
        cfg["crop_template"] = {"cache_path": "/nonexistent/path/templates.npz"}

        with pytest.raises(FileNotFoundError, match="pre-built template library"):
            rdt.render_frame_deeptrack(
                np.array([[5.0, 5.0]]), (0.0, 10.0, 0.0, 10.0), cfg, np.random.default_rng(0)
            )

    def test_ground_truth_position_matches_lj_to_pixels_rounded_location(self, monkeypatch):
        H, W = 64, 64
        fake_kernel = _make_fake_kernel(H, W)
        rdt = _import_deeptrack_with_mock(fake_kernel)

        import render_crop_templates as rct

        # Single-peak template so the brightest canvas pixel is unambiguous.
        template = np.zeros((9, 9), dtype=np.float32)
        template[4, 4] = 1.0
        monkeypatch.setattr(rct, "load_template_library", lambda path: np.stack([template]))

        cfg = _deeptrack_cfg(H, W)
        cfg["crop_source"] = "real"
        cfg["crop_template"] = {"cache_path": "unused.npz"}
        cfg["background"]["amplitude"] = 0
        cfg["noise"]["gain_sigma"] = 0.0
        cfg["noise"]["read_noise"] = 0.0

        box = (0.0, 10.0, 0.0, 10.0)
        positions = np.array([[6.3, 4.7]])
        expected_px = rdt._lj_to_pixels(positions, box, H, W)[0]
        expected_row, expected_col = int(round(expected_px[1])), int(round(expected_px[0]))

        frame = rdt.render_frame_deeptrack(positions, box, cfg, np.random.default_rng(5))

        brightest = np.unravel_index(np.argmax(frame), frame.shape)
        assert abs(brightest[0] - expected_row) <= 1
        assert abs(brightest[1] - expected_col) <= 1

    def test_particle_near_edge_is_clipped_not_out_of_range(self, monkeypatch):
        H, W = 32, 32
        fake_kernel = _make_fake_kernel(H, W)
        rdt = _import_deeptrack_with_mock(fake_kernel)

        import render_crop_templates as rct

        templates = _fake_template_library(n=2, size=21)  # half-width 10, larger than edge margin
        monkeypatch.setattr(rct, "load_template_library", lambda path: templates)

        cfg = _deeptrack_cfg(H, W)
        cfg["crop_source"] = "real"
        cfg["crop_template"] = {"cache_path": "unused.npz"}

        # Particle right at the corner — template patch extends past all four edges.
        positions = np.array([[0.0, 0.0]])
        box = (0.0, 10.0, 0.0, 10.0)

        frame = rdt.render_frame_deeptrack(positions, box, cfg, np.random.default_rng(0))

        assert frame.shape == (H, W)  # no exception, canvas size unaffected


class TestCropSourceProcedural:
    def test_produces_frame_with_no_cache_dependency(self):
        H, W = 48, 48
        fake_kernel = _make_fake_kernel(H, W)
        rdt = _import_deeptrack_with_mock(fake_kernel)

        cfg = _deeptrack_cfg(H, W)
        cfg["crop_source"] = "procedural"
        cfg["procedural_shape"] = {"size": 15, "sigma": 3.0}

        positions = np.array([[3.0, 3.0], [7.0, 7.0]])
        box = (0.0, 10.0, 0.0, 10.0)
        frame = rdt.render_frame_deeptrack(positions, box, cfg, np.random.default_rng(9))

        assert frame.dtype == np.uint16
        assert frame.shape == (H, W)

    def test_ring_params_produce_frame_with_no_cache_dependency(self):
        H, W = 48, 48
        fake_kernel = _make_fake_kernel(H, W)
        rdt = _import_deeptrack_with_mock(fake_kernel)

        cfg = _deeptrack_cfg(H, W)
        cfg["crop_source"] = "procedural"
        cfg["procedural_shape"] = {
            "size": 15,
            "sigma": 3.0,
            "ring_B": 0.0,
            "ring_A0": 1.0,
            "ring_s0": 2.0,
            "ring_A1": 0.8,
            "ring_r1": 8.0,
            "ring_s1": 2.0,
        }

        positions = np.array([[3.0, 3.0], [7.0, 7.0]])
        box = (0.0, 10.0, 0.0, 10.0)
        frame = rdt.render_frame_deeptrack(positions, box, cfg, np.random.default_rng(9))

        assert frame.dtype == np.uint16
        assert frame.shape == (H, W)

    def test_multiple_particles_share_identical_template(self, monkeypatch):
        H, W = 48, 48
        fake_kernel = _make_fake_kernel(H, W)
        rdt = _import_deeptrack_with_mock(fake_kernel)

        import render_crop_templates as rct

        captured = []
        real_generate = rct.generate_procedural_shape

        def _spy(*args, **kwargs):
            shape = real_generate(*args, **kwargs)
            captured.append(shape)
            return shape

        monkeypatch.setattr(rct, "generate_procedural_shape", _spy)

        cfg = _deeptrack_cfg(H, W)
        cfg["crop_source"] = "procedural"
        cfg["procedural_shape"] = {"size": 15, "sigma": 3.0}
        positions = np.array([[2.0, 2.0], [5.0, 5.0], [8.0, 8.0]])
        box = (0.0, 10.0, 0.0, 10.0)
        rdt.render_frame_deeptrack(positions, box, cfg, np.random.default_rng(9))

        # Deterministic now (no per-particle ellipticity/rotation), so the
        # shape is built once per render, not once per particle.
        assert len(captured) == 1


class TestCropSourceBackgroundNoiseInvariance:
    def test_background_and_noise_statistics_match_across_crop_sources(self, monkeypatch):
        """Given the same clean canvas (no particles, so crop_source's own
        compositing/convolution differences are moot), background + noise
        output statistics must be identical regardless of crop_source."""
        H, W = 32, 32
        fake_kernel = _make_fake_kernel(H, W)
        rdt = _import_deeptrack_with_mock(fake_kernel)

        import render_crop_templates as rct

        monkeypatch.setattr(rct, "load_template_library", lambda path: _fake_template_library())

        box = (0.0, 10.0, 0.0, 10.0)
        positions = np.zeros((0, 2))

        frames = {}
        for source in ("physics", "real", "procedural"):
            cfg = _deeptrack_cfg(H, W)
            cfg["crop_source"] = source
            cfg["crop_template"] = {"cache_path": "unused.npz"}
            cfg["procedural_shape"] = {"size": 9, "sigma": 2.0}
            cfg["background"]["amplitude"] = 500
            frames[source] = rdt.render_frame_deeptrack(
                positions, box, cfg, np.random.default_rng(11)
            )

        assert np.array_equal(frames["physics"], frames["real"])
        assert np.array_equal(frames["physics"], frames["procedural"])


class TestPeakBrightnessMatchesSampledIntensity:
    """R4 regression (docs/plans/2026-07-21-001-fix-peak-normalized-particle-
    brightness-plan.md): a single isolated particle's peak pixel (before
    background/noise) matches its own sampled intensity within tolerance,
    consistently across physics/real/procedural. This is the direct
    regression test for the dilution bug found this session -- it fails
    against the pre-fix sum-normalized contract (where a wide `real`
    template or a broad `physics` kernel dilutes the peak far below the
    sampled intensity) and passes once each strategy peak-normalizes.
    """

    def _isolated_particle_cfg(self, H, W, peak_mean):
        cfg = _deeptrack_cfg(H, W)
        cfg["particle"]["peak_mean"] = peak_mean
        cfg["particle"]["intensity_sigma"] = 0.0  # deterministic: sampled intensity == peak_mean
        cfg["background"]["amplitude"] = 0
        cfg["noise"]["gain_sigma"] = 0.0
        cfg["noise"]["read_noise"] = 0.0
        return cfg

    def _assert_particle_pixel_matches_intensity(self, frame, row, col, intensity):
        # Check the value at the particle's own pixel, not frame.max() --
        # scipy.ndimage.convolve(mode="reflect") can produce a brighter
        # boundary-reflection artifact elsewhere in the frame when the
        # kernel/template radius approaches the image size, which is a
        # pre-existing boundary-handling property unrelated to this fix.
        # Only remaining noise source is Poisson shot noise (std = sqrt(intensity));
        # a 5% relative tolerance is comfortably wider than that at these peak_mean scales.
        assert abs(float(frame[row, col]) - intensity) < 0.05 * intensity

    def test_physics_peak_matches_sampled_intensity(self):
        H, W = 128, 128
        peak_mean = 50000
        # A broad fake kernel spanning most of the crop window -- the same
        # shape category measured for physics's real DeepTrack2 output at
        # config.yaml defaults this session.
        cy, cx = H // 2, W // 2
        yy, xx = np.mgrid[0:H, 0:W]
        r2 = (yy - cy) ** 2 + (xx - cx) ** 2
        broad_kernel = np.exp(-r2 / (2 * 30.0**2))
        rdt = _import_deeptrack_with_mock(broad_kernel)

        cfg = self._isolated_particle_cfg(H, W, peak_mean)
        positions = np.array([[5.0, 5.0]])
        box = (0.0, 10.0, 0.0, 10.0)
        row, col = rdt._lj_to_pixels(positions, box, H, W)[0][::-1].astype(int)
        frame = rdt.render_frame_deeptrack(positions, box, cfg, np.random.default_rng(0))

        self._assert_particle_pixel_matches_intensity(frame, row, col, peak_mean)

    def test_real_peak_matches_sampled_intensity_regardless_of_template_size(self, monkeypatch):
        H, W = 128, 128
        peak_mean = 50000
        fake_kernel = _make_fake_kernel(H, W)
        rdt = _import_deeptrack_with_mock(fake_kernel)

        import render_crop_templates as rct

        positions = np.array([[5.0, 5.0]])
        box = (0.0, 10.0, 0.0, 10.0)
        row, col = rdt._lj_to_pixels(positions, box, H, W)[0][::-1].astype(int)

        for size in (9, 65):  # narrow vs. the wide template_half that caused this session's bug
            # Single-peak template so the checked pixel receives exactly the
            # sampled intensity (not diluted by which offset pixel is 1.0).
            template = np.zeros((size, size), dtype=np.float32)
            template[size // 2, size // 2] = 1.0
            monkeypatch.setattr(
                rct, "load_template_library", lambda path, t=template: np.stack([t])
            )

            cfg = self._isolated_particle_cfg(H, W, peak_mean)
            cfg["crop_source"] = "real"
            cfg["crop_template"] = {"cache_path": "unused.npz"}
            frame = rdt.render_frame_deeptrack(positions, box, cfg, np.random.default_rng(0))

            self._assert_particle_pixel_matches_intensity(frame, row, col, peak_mean)

    def test_procedural_peak_matches_sampled_intensity(self):
        H, W = 128, 128
        peak_mean = 50000
        fake_kernel = _make_fake_kernel(H, W)
        rdt = _import_deeptrack_with_mock(fake_kernel)

        cfg = self._isolated_particle_cfg(H, W, peak_mean)
        cfg["crop_source"] = "procedural"
        cfg["procedural_shape"] = {"size": 15, "sigma": 3.0}
        positions = np.array([[5.0, 5.0]])
        box = (0.0, 10.0, 0.0, 10.0)
        row, col = rdt._lj_to_pixels(positions, box, H, W)[0][::-1].astype(int)
        frame = rdt.render_frame_deeptrack(positions, box, cfg, np.random.default_rng(0))

        self._assert_particle_pixel_matches_intensity(frame, row, col, peak_mean)


class TestCropSourceDispatchIntegration:
    """Mirrors TestRenderStrategyDispatch + TestDeeptrackStrategy's stub setup —
    exercises config -> _dispatch_render -> render_frame_deeptrack -> compositing
    -> noise together, not just individual functions in isolation."""

    def test_dispatch_real_and_procedural_each_return_valid_frame(self, render_module, monkeypatch):
        H, W = 40, 40
        fake_kernel = _make_fake_kernel(H, W)
        rdt = _import_deeptrack_with_mock(fake_kernel)
        monkeypatch.setitem(sys.modules, "render_deeptrack", rdt)

        import render_crop_templates as rct

        templates = _fake_template_library(n=3, size=9)
        monkeypatch.setattr(rct, "load_template_library", lambda path: templates)

        positions = np.array([[4.0, 4.0]])
        box = (0.0, 10.0, 0.0, 10.0)

        cfg_real = _deeptrack_cfg(H, W)
        cfg_real["crop_source"] = "real"
        cfg_real["crop_template"] = {"cache_path": "unused.npz"}
        frame_real = render_module._dispatch_render(
            positions, box, cfg_real, np.random.default_rng(1), "deeptrack"
        )

        cfg_proc = _deeptrack_cfg(H, W)
        cfg_proc["crop_source"] = "procedural"
        cfg_proc["procedural_shape"] = {"size": 11, "sigma": 2.5}
        frame_proc = render_module._dispatch_render(
            positions, box, cfg_proc, np.random.default_rng(2), "deeptrack"
        )

        for frame in (frame_real, frame_proc):
            assert frame.dtype == np.uint16
            assert frame.shape == (H, W)
