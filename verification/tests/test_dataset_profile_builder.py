"""Tests for dataset_profile_builder.py — U2: builds a dataset-profile YAML
from a LAMMPS trajectory and a known size_px.

Test scenarios from the plan:
- compute_spacing_px computes the correct median nearest-neighbor distance
  on a small, controlled trajectory (grid of known positions).
- build_dataset_profile rejects a non-positive size_px.
- build_dataset_profile includes/omits the optional description key.
- write_dataset_profile round-trips a valid, loadable YAML file.
- Frame-index-out-of-range and fewer-than-2-particle error cases.
- dataset_profile_builder.py run against the repo's reference trajectory
  (lammps-scripts/single_continuous_force_test/continuous_force_1500_5.0.lammpstrj)
  produces a spacing_px matching the documented ~10.9px reference value
  (within +/-5%) -- regression guard against the builder's formula drifting
  from established ground truth (covers R4, AE6). Skipped when that file
  isn't present locally: *.lammpstrj is gitignored (large binary trajectory
  data), so it isn't guaranteed to exist in every checkout/worktree.
"""

import sys
from pathlib import Path

import pytest
import yaml

# Ensure verification/ is importable. Unlike other verification/tests/*.py
# (e.g. test_calibrate_psf.py), this suite deliberately does NOT stub out
# lammps_parser -- dataset_profile_builder.py's whole job is real LAMMPS-
# trajectory parsing, so these tests need the genuine parse_lammps_dump
# (render.py's own sys.path insert makes the real lammps-scripts/
# lammps_parser.py importable).
sys.path.insert(0, str(Path(__file__).parent.parent))

import dataset_profile_builder as builder  # noqa: E402

_REFERENCE_TRAJECTORY = (
    Path(__file__).parent.parent.parent
    / "lammps-scripts"
    / "single_continuous_force_test"
    / "continuous_force_1500_5.0.lammpstrj"
)


def _write_lammpstrj(path, box_bounds, atom_lines, atom_header="ITEM: ATOMS id type x y"):
    """Write a minimal, real-format LAMMPS dump file with a single frame."""
    lines = [
        "ITEM: TIMESTEP",
        "0",
        "ITEM: NUMBER OF ATOMS",
        str(len(atom_lines)),
        "ITEM: BOX BOUNDS pp pp pp",
        *box_bounds,
        atom_header,
        *atom_lines,
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


class TestComputeSpacingPx:
    def test_median_nearest_neighbor_distance_on_known_grid(self, tmp_path):
        """Four particles: three at mutual distance 1.0, one far outlier at
        distance ~6.4 from its own nearest neighbor. Box and image size are
        both 10x10 (1:1 LJ-to-pixel mapping) so expected pixel distances are
        exact. Median of [1.0, 1.0, 1.0, 6.403] == 1.0."""
        path = _write_lammpstrj(
            tmp_path / "grid.lammpstrj",
            box_bounds=["0.0 10.0", "0.0 10.0", "0.0 1.0"],
            atom_lines=["1 1 0.0 0.0", "2 1 1.0 0.0", "3 1 0.0 1.0", "4 1 5.0 5.0"],
        )

        spacing_px = builder.compute_spacing_px(path, image_width=10, image_height=10)

        assert spacing_px == pytest.approx(1.0, abs=1e-6)

    def test_frame_index_out_of_range_raises_value_error(self, tmp_path):
        path = _write_lammpstrj(
            tmp_path / "one_frame.lammpstrj",
            box_bounds=["0.0 10.0", "0.0 10.0", "0.0 1.0"],
            atom_lines=["1 1 0.0 0.0", "2 1 1.0 0.0"],
        )

        with pytest.raises(ValueError, match="frame"):
            builder.compute_spacing_px(path, image_width=10, image_height=10, frame_index=5)

    def test_fewer_than_two_particles_raises_value_error(self, tmp_path):
        path = _write_lammpstrj(
            tmp_path / "single_particle.lammpstrj",
            box_bounds=["0.0 10.0", "0.0 10.0", "0.0 1.0"],
            atom_lines=["1 1 5.0 5.0"],
        )

        with pytest.raises(ValueError, match="particle"):
            builder.compute_spacing_px(path, image_width=10, image_height=10)


class TestBuildDatasetProfile:
    def _grid_trajectory(self, tmp_path):
        return _write_lammpstrj(
            tmp_path / "grid.lammpstrj",
            box_bounds=["0.0 10.0", "0.0 10.0", "0.0 1.0"],
            atom_lines=["1 1 0.0 0.0", "2 1 1.0 0.0", "3 1 0.0 1.0", "4 1 5.0 5.0"],
        )

    def test_non_positive_size_px_raises_value_error(self, tmp_path):
        path = self._grid_trajectory(tmp_path)

        with pytest.raises(ValueError, match="size_px"):
            builder.build_dataset_profile(path, size_px=0, image_width=10, image_height=10)

    def test_profile_omits_description_when_not_given(self, tmp_path):
        path = self._grid_trajectory(tmp_path)

        profile = builder.build_dataset_profile(path, size_px=5.0, image_width=10, image_height=10)

        assert profile["size_px"] == 5.0
        assert profile["spacing_px"] == pytest.approx(1.0, abs=1e-6)
        assert "description" not in profile

    def test_profile_includes_description_when_given(self, tmp_path):
        path = self._grid_trajectory(tmp_path)

        profile = builder.build_dataset_profile(
            path, size_px=5.0, image_width=10, image_height=10, description="test profile"
        )

        assert profile["description"] == "test profile"


class TestWriteDatasetProfile:
    def test_written_profile_round_trips_as_valid_yaml(self, tmp_path):
        output_path = tmp_path / "nested" / "profile.yaml"
        profile = {"size_px": 5.0, "spacing_px": 10.8658, "description": "a profile"}

        builder.write_dataset_profile(profile, output_path)

        assert output_path.exists()
        loaded = yaml.safe_load(output_path.read_text())
        assert loaded == profile


@pytest.mark.skipif(
    not _REFERENCE_TRAJECTORY.exists(),
    reason=(
        "*.lammpstrj is gitignored; continuous_force_1500_5.0.lammpstrj is not "
        "guaranteed to exist in every checkout"
    ),
)
class TestReferenceTrajectoryRegression:
    def test_spacing_px_matches_documented_reference_value(self):
        """AGENTS.md and verification/config.yaml document this trajectory's
        median nearest-neighbor spacing as ~10.9px at 512x512 -- this is the
        established ground truth the builder's cKDTree-based formula must
        keep matching."""
        profile = builder.build_dataset_profile(
            _REFERENCE_TRAJECTORY, size_px=5.0, image_width=512, image_height=512
        )

        assert profile["spacing_px"] == pytest.approx(10.9, rel=0.05)
