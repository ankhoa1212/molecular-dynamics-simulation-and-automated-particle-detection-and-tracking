"""Tests for graph_style.py's release lookup, titles and start/release/end markers."""

import matplotlib.pyplot as plt
import pytest

from graph_style import find_pull_release_step, mark_start_release_end, run_title

# How continuous_force.in's release step appears in a LAMMPS log: echoed once
# per substitution pass, and only the last line has every value filled in
RELEASE_ECHO = (
    "variable        pull_release_step equal floor(${{pull_release_fraction}}*${{steps}})\n"
    "variable        pull_release_step equal floor({fraction}*${{steps}})\n"
    "variable        pull_release_step equal floor({fraction}*{steps})\n"
)


def _run_with_log(tmp_path, fraction, steps, name="continuous_force_700_1.5"):
    """A trajectory path whose logs/<name>.log records the given release."""
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / f"{name}.log").write_text(
        "LAMMPS (4 Feb 2025)\n" + RELEASE_ECHO.format(fraction=fraction, steps=steps)
    )
    return tmp_path / f"{name}.lammpstrj"


def test_release_step_from_the_log_beside_a_trajectory(tmp_path):
    assert find_pull_release_step(_run_with_log(tmp_path, 0.666667, 15000)) == 10000


def test_release_step_from_a_log_file_itself(tmp_path):
    log = tmp_path / "run.log"
    log.write_text(RELEASE_ECHO.format(fraction=0.666667, steps=25000))
    assert find_pull_release_step(log) == 16666


def test_no_release_without_a_log_or_the_line(tmp_path):
    assert find_pull_release_step(tmp_path / "continuous_force_700_1.5.lammpstrj") is None
    log = tmp_path / "old_input.log"
    log.write_text("variable        cutoff_radius equal 10.0\n")
    assert find_pull_release_step(log) is None


@pytest.mark.parametrize(
    "name, title",
    [
        ("continuous_force_700_1.5.lammpstrj", "Velocity, N=700, ε=1.5"),
        ("continuous_force_200_5.0_seed11111.lammpstrj", "Velocity, N=200, ε=5.0"),
        ("continuous_force_700_1.5.log", "Velocity, N=700, ε=1.5"),
        ("tracks.csv", "Velocity"),
    ],
)
def test_run_title(name, title):
    assert run_title("Velocity", name) == title


def _plot(steps):
    fig, ax = plt.subplots()
    values = [1.0 + 0.0001 * step for step in steps]
    ax.plot(steps, values)
    return fig, ax, values


def _release_lines(ax):
    return [line for line in ax.lines if line.get_linestyle() == "--"]


def test_release_inside_the_run_is_marked(tmp_path):
    steps = list(range(0, 15001, 100))
    fig, ax, values = _plot(steps)

    mark_start_release_end(ax, _run_with_log(tmp_path, 0.666667, 15000), steps, values)

    assert [line.get_xdata()[0] for line in _release_lines(ax)] == [10000]
    labels = [text.get_text() for text in ax.texts]
    assert any(label.startswith("Pull released (step 10000)") for label in labels)
    assert any(label.startswith("Start:") for label in labels)
    assert any(label.startswith("End:") for label in labels)
    plt.close(fig)


def test_release_at_the_last_step_is_not_marked(tmp_path, capsys):
    steps = list(range(0, 15001, 100))
    fig, ax, values = _plot(steps)

    mark_start_release_end(ax, _run_with_log(tmp_path, 1, 15000), steps, values)

    assert _release_lines(ax) == []
    assert "outside the plotted steps" in capsys.readouterr().out
    plt.close(fig)


def test_late_release_label_sits_left_of_the_line(tmp_path):
    steps = list(range(0, 3001, 100))
    fig, ax, values = _plot(steps)

    mark_start_release_end(ax, _run_with_log(tmp_path, 0.95, 3000), steps, values)

    release_label = next(t for t in ax.texts if t.get_text().startswith("Pull released"))
    assert release_label.get_horizontalalignment() == "right"
    plt.close(fig)


def test_empty_run_draws_nothing(tmp_path):
    fig, ax = plt.subplots()

    mark_start_release_end(ax, _run_with_log(tmp_path, 0.666667, 15000), [], [])

    assert len(ax.texts) == 0 and len(ax.lines) == 0
    plt.close(fig)
