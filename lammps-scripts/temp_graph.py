"""
This module processes LAMMPS log and dump files to plot temperature evolution over time.
It supports both standard thermodynamic output from logs and drift-corrected temperature
calculations from trajectory dump files.
"""

import argparse
import os
import sys

import numpy as np

from graph_style import (
    MAIN_COLOR,
    SECONDARY_COLOR,
    finish_and_save,
    mark_start_release_end,
    new_figure,
    run_title,
)
from lammps_parser import parse_lammps_dump


def generate_temp_graph_filename(filename, ending, output_dir=None):
    """
    Generates a consistent output filename for the temperature graph.
    """
    base_name = os.path.basename(filename)
    if "." in base_name:
        base_name = base_name[: base_name.rfind(".")]

    out_name = f"{base_name}_temp_graph_{ending}.png"

    if output_dir:
        out_name = os.path.join(output_dir, out_name)

    return out_name


def plot_log_temperature(filename, output_dir=None, no_show=False):
    """
    Plots temperature from a LAMMPS log file.
    """
    steps = []
    temps = []

    print(f"Reading log file: {filename}...")

    reading_data = False

    try:
        with open(filename, "r", encoding="utf-8") as log_file:
            for line in log_file:
                # Detect the start of the data table in log.lammps
                if "Step" in line and "Temp" in line:
                    reading_data = True
                    continue

                # Stop reading if we hit the end loop info
                if "Loop time" in line:
                    reading_data = False
                    continue

                if reading_data:
                    parts = line.split()
                    if len(parts) > 1 and parts[0].isdigit():
                        try:
                            # Step is col 0, Temp is col 1, per thermo_style custom step temp ...
                            step = int(parts[0])
                            temp = float(parts[1])
                            steps.append(step)
                            temps.append(temp)
                        except ValueError:
                            continue
    except FileNotFoundError:
        print(f"Error: Could not find file {filename}")
        sys.exit(1)

    if not steps:
        print("No temperature data found. Did you point to the correct .log file?")
        sys.exit(1)

    fig, ax = new_figure()
    ax.plot(steps, temps, color=MAIN_COLOR, linewidth=2.0, label="LAMMPS Temp")
    mark_start_release_end(ax, filename, steps, temps)
    finish_and_save(
        fig,
        ax,
        run_title("Temperature (LAMMPS log)", filename),
        "Temperature (LJ units)",
        generate_temp_graph_filename(filename, "log", output_dir),
        no_show,
    )


def _process_atom_lines(atom_lines):
    """
    Parses atom lines and returns position and velocity arrays.
    """
    data_list = []
    for line in atom_lines:
        parts = line.split()
        # x, y, vx, vy are indices 2, 3, 4, 5
        data_list.append([float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])])

    data_arr = np.array(data_list)
    return data_arr[:, 0], data_arr[:, 1], data_arr[:, 2], data_arr[:, 3]


# Width of the rings (in sigma) over which the inward collapse flow is averaged.
# 1-4 sigma give the same temperature within ~0.1; much wider rings start
# counting the flow's growth with radius as heat.
RING_WIDTH = 2.0


def _compute_radial_projection(x, y, vx, vy):
    """Computes distance from the pull center, radial velocity and tangential squared velocity."""
    center_x = 100.0
    center_y = 100.0
    delta_x = x - center_x
    delta_y = y - center_y
    dist = np.sqrt(delta_x**2 + delta_y**2)

    with np.errstate(divide="ignore", invalid="ignore"):
        unit_radial_x = np.divide(delta_x, dist, out=np.zeros_like(delta_x), where=dist != 0)
        unit_radial_y = np.divide(delta_y, dist, out=np.zeros_like(delta_y), where=dist != 0)

    v_rad = vx * unit_radial_x + vy * unit_radial_y
    vel_sq = vx**2 + vy**2
    vel_tan_sq = vel_sq - v_rad**2
    return dist, v_rad, vel_sq, vel_tan_sq


def _compute_frame_temperature(x, y, vx, vy, num_atoms):
    """
    Computes total and thermal (collapse flow removed) temperatures for a single frame.
    """
    dist, v_rad, vel_sq, vel_tan_sq = _compute_radial_projection(x, y, vx, vy)

    # Raw KE = 0.5 * m * (vx^2 + vy^2); T = Sum(v^2) / (2 * N) [2 degrees of freedom]
    total_temp = np.sum(vel_sq) / (2.0 * num_atoms)

    # The pull is a linear spring, so the inward flow grows with distance from
    # the center. Remove it ring by ring: subtract each ring's mean radial
    # velocity from the atoms in that ring.
    _, ring_index = np.unique((dist // RING_WIDTH).astype(int), return_inverse=True)
    ring_mean_v_rad = np.bincount(ring_index, weights=v_rad) / np.bincount(ring_index)
    v_rad_fluctuation = v_rad - ring_mean_v_rad[ring_index]

    # Each ring's mean removes one degree of freedom
    dof = max(2 * num_atoms - len(ring_mean_v_rad), 1)
    corrected_temp = np.sum(vel_tan_sq + v_rad_fluctuation**2) / dof

    return total_temp, corrected_temp


def _calculate_temps_for_frame(frame):
    """calculates temperature for a single frame"""
    x, y, vx, vy = _process_atom_lines(frame["atoms"])
    return _compute_frame_temperature(x, y, vx, vy, frame["num_atoms"])


def plot_temperatures(filename, output_dir=None, no_show=False):
    """
    Plots total and thermal temperature from a LAMMPS dump file, marking the pull
    release and labelling the thermal temperature at the start, release and end.
    """
    timesteps = []
    total_temps = []
    corrected_temps = []  # Drift-Corrected (True Thermal)

    print(f"Reading file: {filename}...")

    try:
        for frame in parse_lammps_dump(filename):
            if frame["num_atoms"] == 0:
                continue

            total_temp, corrected_temp = _calculate_temps_for_frame(frame)

            timesteps.append(frame["timestep"])
            total_temps.append(total_temp)
            corrected_temps.append(corrected_temp)

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return

    if not timesteps:
        print("No valid temperature data found.")
        return

    fig, ax = new_figure()
    ax.plot(
        timesteps,
        corrected_temps,
        color=MAIN_COLOR,
        linewidth=2.0,
        label="Temperature (pull motion removed)",
    )
    # Drawn after (over) the thermal curve, and thinner, so where the two agree
    # (no pull motion left) the orange line shows on top of the blue one
    ax.plot(
        timesteps,
        total_temps,
        color=SECONDARY_COLOR,
        linewidth=1.2,
        label="Temperature (including pull motion)",
    )
    mark_start_release_end(ax, filename, timesteps, corrected_temps)
    finish_and_save(
        fig,
        ax,
        run_title("Temperature", filename),
        "Temperature (LJ units)",
        generate_temp_graph_filename(filename, "lammpstrj", output_dir),
        no_show,
    )


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description="Plot drift-corrected temperature from LAMMPS trajectory"
    )
    PARSER.add_argument(
        "--filename", "-f", required=True, help="Path to a .log or .lammpstrj file (required)"
    )
    PARSER.add_argument("--output_dir", default=None, help="Output directory")
    PARSER.add_argument("--no-show", action="store_true", help="Do not display the graph")
    ARGS = PARSER.parse_args()
    if ARGS.filename.endswith(".log"):
        plot_log_temperature(ARGS.filename, ARGS.output_dir, ARGS.no_show)
    elif ARGS.filename.endswith(".lammpstrj"):
        plot_temperatures(ARGS.filename, ARGS.output_dir, ARGS.no_show)
    else:
        print("Error: File should be either .log or .lammpstrj")
        sys.exit(1)
