"""Tests for compare.py's unit conversion of tracked positions."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import compare  # noqa: E402

VERIFICATION_DIR = Path(__file__).parent.parent
LAMMPS_FRAME = (
    "ITEM: TIMESTEP\n{step}\nITEM: NUMBER OF ATOMS\n2\n"
    "ITEM: BOX BOUNDS ss ss pp\n0.0 256.0\n0.0 256.0\n-0.5 0.5\n"
    "ITEM: ATOMS id type x y\n1 1 50.0 50.0\n2 1 80.0 60.0\n"
)


def _run_main(monkeypatch, tmp_path, config_text, with_lammps=False):
    """Runs compare.main() on a tiny tracks.csv; returns the pixel scale given to the MSD plot."""
    tracks = pd.DataFrame(
        [
            {"frame": f, "track_id": t, "x": 10.0 * t + f, "y": 5.0}
            for t in range(3)
            for f in range(5)
        ]
    )
    tracks_path = tmp_path / "tracks.csv"
    tracks.to_csv(tracks_path, index=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_text)
    argv = ["compare.py", "--tracks", str(tracks_path), "--config", str(config_path)]
    if with_lammps:
        lammps_path = tmp_path / "sim.lammpstrj"
        lammps_path.write_text(LAMMPS_FRAME.format(step=0) + LAMMPS_FRAME.format(step=100))
        argv += ["--lammps", str(lammps_path)]

    seen = {}
    monkeypatch.setattr(compare, "_plot_hexatic", lambda *args, **kwargs: None)
    monkeypatch.setattr(compare, "_plot_velocity_dist", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        compare,
        "_plot_msd",
        lambda df_sim, df_tracks, pixel_scale, *rest: seen.update(pixel_scale=pixel_scale),
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", argv)
    compare.main()
    return seen["pixel_scale"]


class TestTrackPixelScale:
    def test_defaults_to_the_real_microscope_scale(self, monkeypatch, tmp_path):
        config = f"compare:\n  output_dir: {tmp_path / 'out'}\n"
        assert _run_main(monkeypatch, tmp_path, config) == pytest.approx(0.045)

    def test_a_configured_scale_wins(self, monkeypatch, tmp_path):
        config = f"compare:\n  pixel_scale: 0.78\n  output_dir: {tmp_path / 'out'}\n"
        assert _run_main(monkeypatch, tmp_path, config) == pytest.approx(0.78)

    def test_render_uses_the_synthetic_render_s_own_scale(self, monkeypatch, tmp_path):
        config = (
            "synthetic:\n  image_width: 512\n"
            f"compare:\n  pixel_scale: render\n  lj_to_um: 2.0\n  output_dir: {tmp_path / 'out'}\n"
        )
        # A 256-LJ-wide box on 512 px is 2 px per LJ unit, and one LJ unit is 2 um
        scale = _run_main(monkeypatch, tmp_path, config, with_lammps=True)
        assert scale == pytest.approx(1.0)

    def test_render_without_a_trajectory_is_rejected(self, monkeypatch, tmp_path):
        config = f"compare:\n  pixel_scale: render\n  output_dir: {tmp_path / 'out'}\n"
        with pytest.raises(SystemExit, match="needs --lammps"):
            _run_main(monkeypatch, tmp_path, config)

    def test_the_shipped_config_converts_real_tracks_at_the_microscope_scale(
        self, monkeypatch, tmp_path
    ):
        import yaml

        shipped = yaml.safe_load((VERIFICATION_DIR / "config.yaml").read_text())
        shipped["compare"]["output_dir"] = str(tmp_path / "out")
        scale = _run_main(monkeypatch, tmp_path, yaml.safe_dump(shipped), with_lammps=True)
        assert scale == pytest.approx(0.045)
