"""Smoke tests: each plot entry point runs end to end and writes its PNG."""

import sys

import numpy as np
import pytest

from temp_graph import plot_log_temperature, plot_temperatures
from velocity_graph import plot_velocity_over_time

NAME = "continuous_force_100_4.0"
# How continuous_force.in's release step appears in a LAMMPS log (final echo)
RELEASE_LINE = "variable        pull_release_step equal floor(0.5*400)\n"


def _write_dump(path, n_frames=5, n_atoms=40):
    """A small continuous_force-style dump: atoms around the box center, steps 0-400."""
    rng = np.random.default_rng(0)
    lines = []
    for frame in range(n_frames):
        xy = 100 + rng.uniform(-8, 8, size=(n_atoms, 2))
        v = rng.normal(0, 1, size=(n_atoms, 2))
        lines += [
            "ITEM: TIMESTEP",
            str(frame * 100),
            "ITEM: NUMBER OF ATOMS",
            str(n_atoms),
            "ITEM: BOX BOUNDS ss ss pp",
            "0 200",
            "0 200",
            "-0.5 0.5",
            "ITEM: ATOMS id type x y vx vy fx fy v_dist",
        ]
        lines += [
            f"{i + 1} 1 {x:.4f} {y:.4f} {vx:.4f} {vy:.4f} 0 0 0"
            for i, ((x, y), (vx, vy)) in enumerate(zip(xy, v))
        ]
    path.write_text("\n".join(lines) + "\n")
    return path


def _run(tmp_path):
    """A trajectory with its run log in logs/, as run.py writes them."""
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / f"{NAME}.log").write_text(RELEASE_LINE)
    return _write_dump(tmp_path / f"{NAME}.lammpstrj")


def test_temperature_plot_from_a_trajectory(tmp_path):
    out = tmp_path / "out"
    plot_temperatures(str(_run(tmp_path)), str(out), no_show=True)
    assert (out / f"{NAME}_temp_graph_lammpstrj.png").stat().st_size > 0


def test_temperature_plot_from_a_log(tmp_path):
    log = tmp_path / f"{NAME}.log"
    log.write_text(
        RELEASE_LINE
        + "   Step          Temp     \n"
        + "".join(f"{step:8d}   {1.0 + 0.001 * step:.4f}\n" for step in range(0, 401, 100))
        + "Loop time of 1.0 on 1 procs\n"
    )
    out = tmp_path / "out"
    plot_log_temperature(str(log), str(out), no_show=True)
    assert (out / f"{NAME}_temp_graph_log.png").stat().st_size > 0


def test_velocity_plot(tmp_path):
    out = tmp_path / "out"
    plot_velocity_over_time(str(_run(tmp_path)), str(out), no_show=True)
    assert (out / f"{NAME}_velocity_graph.png").stat().st_size > 0


def test_velocity_plot_of_an_empty_trajectory(tmp_path):
    empty = tmp_path / f"{NAME}.lammpstrj"
    empty.write_text("")
    out = tmp_path / "out"
    plot_velocity_over_time(str(empty), str(out), no_show=True)
    assert (out / f"{NAME}_velocity_graph.png").stat().st_size > 0


def test_temperature_plot_of_an_empty_trajectory_writes_nothing(tmp_path, capsys):
    empty = tmp_path / f"{NAME}.lammpstrj"
    empty.write_text("")
    out = tmp_path / "out"
    plot_temperatures(str(empty), str(out), no_show=True)
    assert "No valid temperature data found." in capsys.readouterr().out
    assert not out.exists()


def test_hexatic_plot_from_a_trajectory(tmp_path, monkeypatch):
    pytest.importorskip("freud")
    import hexatic_order_graph  # needs freud, so only after the skip check

    out = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        ["hexatic_order_graph.py", str(_run(tmp_path)), "--output_dir", str(out), "--no-show"],
    )
    hexatic_order_graph.main()
    assert (out / f"{NAME}_hexatic_order.png").stat().st_size > 0


def test_hexatic_plot_from_tracks_csv(tmp_path, monkeypatch):
    pytest.importorskip("freud")
    import hexatic_order_graph  # needs freud, so only after the skip check

    rng = np.random.default_rng(1)
    rows = ["frame,particle,x,y"]
    for frame in range(3):
        rows += [
            f"{frame},{i},{x:.2f},{y:.2f}"
            for i, (x, y) in enumerate(rng.uniform(0, 256, size=(30, 2)))
        ]
    tracks = tmp_path / "tracks.csv"
    tracks.write_text("\n".join(rows) + "\n")
    out = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hexatic_order_graph.py",
            "--tracks-csv",
            str(tracks),
            "--image-width",
            "256",
            "--image-height",
            "256",
            "--output_dir",
            str(out),
            "--no-show",
        ],
    )
    hexatic_order_graph.main()
    assert (out / "tracks_hexatic_order.png").stat().st_size > 0
