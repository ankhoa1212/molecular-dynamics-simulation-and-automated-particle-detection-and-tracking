"""Tests for compare.py's unit conversion of tracked positions."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import compare  # noqa: E402

VERIFICATION_DIR = Path(__file__).parent.parent


def _run_main(monkeypatch, tmp_path, config_path):
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

    seen = {}
    monkeypatch.setattr(compare, "_plot_hexatic", lambda *args, **kwargs: None)
    monkeypatch.setattr(compare, "_plot_velocity_dist", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        compare,
        "_plot_msd",
        lambda df_sim, df_tracks, pixel_scale, *rest: seen.update(pixel_scale=pixel_scale),
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["compare.py", "--tracks", str(tracks_path), "--config", str(config_path)],
    )
    compare.main()
    return seen["pixel_scale"]


class TestTrackPixelScale:
    def test_defaults_to_the_real_microscope_scale(self, monkeypatch, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(f"compare:\n  output_dir: {tmp_path / 'out'}\n")
        assert _run_main(monkeypatch, tmp_path, config_path) == pytest.approx(0.045)

    def test_the_shipped_config_uses_the_real_microscope_scale(self, monkeypatch, tmp_path):
        import yaml

        shipped = yaml.safe_load((VERIFICATION_DIR / "config.yaml").read_text())
        shipped["compare"]["output_dir"] = str(tmp_path / "out")
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump(shipped))
        assert _run_main(monkeypatch, tmp_path, config_path) == pytest.approx(0.045)

    def test_a_configured_scale_wins(self, monkeypatch, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(f"compare:\n  pixel_scale: 0.78\n  output_dir: {tmp_path / 'out'}\n")
        assert _run_main(monkeypatch, tmp_path, config_path) == pytest.approx(0.78)
