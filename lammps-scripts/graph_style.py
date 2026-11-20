"""
Shared look for the per-run graphs graph.py makes (hexatic order, velocity and
temperature), so the three plots of one run read as a set: same size, colors,
labels, and the same start / pull release / end markers.
"""

import math
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt

# run.py names runs <script>_<N>_<epsilon>[_seed<seed>].<ext>
RUN_NAME_PATTERN = re.compile(r"_(\d+)_(\d+(?:\.\d+)?)(?:_seed\d+)?\.(?:lammpstrj|log)$")

FIGSIZE = (10, 6)
DPI = 300

# Okabe-Ito colors, which stay distinguishable under the common forms of color
# vision deficiency
MAIN_COLOR = "#0072B2"  # blue: the measured quantity
SECONDARY_COLOR = "#E69F00"  # orange: a second curve to compare against
REFERENCE_COLOR = "#7F7F7F"  # gray: the pull release

LABEL_BOX = {"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none"}

# continuous_force.in echoes this line into the LAMMPS log with its values filled in
PULL_RELEASE_PATTERN = re.compile(
    r"^variable\s+pull_release_step\s+equal\s+floor\(([0-9.eE+-]+)\*([0-9]+)\)$"
)


def find_pull_release_step(path):
    """
    Returns the step at which continuous_force.in released the pull, read from the
    run's LAMMPS log: the file itself for a .log, otherwise logs/<name>.log beside
    the trajectory (where run.py writes it). Returns None when there is no log or
    the input has no pull release.
    """
    path = Path(path)
    log_path = path if path.suffix == ".log" else path.parent / "logs" / f"{path.stem}.log"
    if not log_path.exists():
        return None

    release_step = None
    with open(log_path, "r", encoding="utf-8") as log_file:
        for line in log_file:
            match = PULL_RELEASE_PATTERN.match(line.strip())
            if match:
                release_step = math.floor(float(match.group(1)) * int(match.group(2)))
    return release_step


def run_title(quantity, path):
    """Title naming the quantity and, when the filename gives them, N and epsilon."""
    match = RUN_NAME_PATTERN.search(os.path.basename(path))
    if match is None:
        return quantity
    return f"{quantity}, N={match.group(1)}, ε={match.group(2)}"


def new_figure():
    """Returns a figure and axes of the shared size."""
    return plt.subplots(figsize=FIGSIZE)


def _label_point(ax, step, value, text, align):
    """Marks a point on the main curve and labels it with its value."""
    ax.scatter([step], [value], color=MAIN_COLOR, s=30, zorder=3)
    # Label above the point, or below it when the point is near the top
    y_low, y_high = ax.get_ylim()
    below = (value - y_low) > 0.8 * (y_high - y_low)
    ax.annotate(
        f"{text}: {value:.2f}",
        xy=(step, value),
        xytext=(8 if align == "left" else -8, -14 if below else 12),
        textcoords="offset points",
        ha=align,
        va="top" if below else "bottom",
        fontsize=9,
        bbox=LABEL_BOX,
    )


def mark_start_release_end(ax, path, steps, values):
    """
    Labels the main curve's value at the start and end of the run and, when the
    run's log records a pull release inside the run, draws the release as a
    dashed line and labels the value there (first frame at or after it).
    """
    _label_point(ax, steps[0], values[0], "Start", "left")
    _label_point(ax, steps[-1], values[-1], "End", "right")

    release_step = find_pull_release_step(path)
    if release_step is None or not steps[0] < release_step < steps[-1]:
        print("No pull release found in the run's log; not marking it.")
        return

    ax.axvline(release_step, color=REFERENCE_COLOR, linestyle="--", linewidth=1.2)
    release_index = next(i for i, step in enumerate(steps) if step >= release_step)
    ax.scatter([steps[release_index]], [values[release_index]], color=MAIN_COLOR, s=30, zorder=3)
    # Label at mid-height beside the line, clear of the start and end labels
    ax.annotate(
        f"Pull released (step {release_step})\n{values[release_index]:.2f}",
        xy=(release_step, 0.5),
        xycoords=("data", "axes fraction"),
        xytext=(6, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=9,
        color=REFERENCE_COLOR,
        bbox=LABEL_BOX,
    )


def finish_and_save(fig, ax, title, ylabel, output_path, no_show, xlabel="Timestep"):
    """Applies the shared title, labels, grid and legend, then saves at the shared dpi."""
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.5)
    # A legend only when there is more than one thing to tell apart
    if len(ax.get_legend_handles_labels()[0]) > 1:
        ax.legend(loc="upper right")
    fig.tight_layout()

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(output_path, dpi=DPI)
    print(f"Graph saved to {output_path}")
    if not no_show:
        plt.show()
    plt.close(fig)
