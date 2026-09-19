"""
This module reads a LAMMPS trajectory file and plots the average velocity magnitude
of atoms over time. It calculates the mean and standard deviation of velocities
for each timestep and generates a plot.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from lammps_parser import parse_lammps_dump


def _process_velocity_data(filename):
    """
    Reads the LAMMPS trajectory file and computes velocity statistics per frame.
    Returns lists of timesteps, means, and standard deviations.
    """
    timesteps = []
    avg_velocities = []
    std_velocities = []

    for frame in parse_lammps_dump(filename):
        timesteps.append(frame["timestep"])
        atom_lines = frame["atoms"]

        atom_header = frame["atom_header"]
        # Format is usually "ITEM: ATOMS id type x y vx vy"
        header_parts = atom_header.split()[2:]
        try:
            vx_idx = header_parts.index("vx")
            vy_idx = header_parts.index("vy")
        except ValueError:
            # vx/vy columns missing from this frame -- record a zero rather than skip
            avg_velocities.append(0)
            std_velocities.append(0)
            continue

        raw_data = []
        for line in atom_lines:
            parts = line.split()
            if len(parts) > max(vx_idx, vy_idx):
                vx = float(parts[vx_idx])
                vy = float(parts[vy_idx])
                raw_data.append(np.sqrt(vx**2 + vy**2))

        if raw_data:
            avg_velocities.append(np.mean(raw_data))
            std_velocities.append(np.std(raw_data))
        else:
            avg_velocities.append(0)
            std_velocities.append(0)

    return timesteps, avg_velocities, std_velocities


def plot_velocity_over_time(filename, output_dir, no_show=False):
    """
    Orchestrates the reading of data and plotting of the velocity graph.
    """
    timesteps, avg_velocities, std_velocities = _process_velocity_data(filename)

    plt.figure(figsize=(10, 6))
    plt.errorbar(
        timesteps,
        avg_velocities,
        yerr=std_velocities,
        marker="o",
        linestyle="-",
        color="b",
        capsize=5,
    )
    plt.xlabel("Timestep")
    plt.ylabel("Mean Velocity Magnitude")
    plt.title("Average Velocity Magnitude Over Time")
    plt.grid(True, linestyle="--", alpha=0.7)

    base_name = os.path.basename(filename)
    if "." in base_name:
        base_name = base_name[: base_name.rfind(".")]

    output_filename = f"{base_name}_velocity_graph.png"

    if output_dir:
        output = os.path.join(output_dir, output_filename)
    else:
        output = output_filename

    plt.savefig(output)
    print(f"Graph saved to {output}")
    if not no_show:
        plt.show()


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description="Plot velocity over time from LAMMPS trajectory file"
    )
    PARSER.add_argument(
        "--filename",
        "-f",
        default=os.path.join(os.getcwd(), "test_same", "test.in_100_5.0.lammpstrj"),
        help="Path to the LAMMPS trajectory file",
    )
    PARSER.add_argument("--output_dir", default=None, help="output directory to save graph file to")
    PARSER.add_argument("--no-show", action="store_true", help="Do not display the graph")
    ARGS = PARSER.parse_args()

    plot_velocity_over_time(ARGS.filename, ARGS.output_dir, ARGS.no_show)
