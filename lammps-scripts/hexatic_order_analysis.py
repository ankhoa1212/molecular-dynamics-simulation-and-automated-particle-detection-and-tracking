"""
This module contains functions to parse LAMMPS trajectory files and calculate the
hexatic order parameter using the freud library.
"""

import argparse

import freud
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lammps_parser import parse_lammps_dump


def _process_box(box_lines):
    """Parses box bounds from list of strings."""
    x_range = [float(x) for x in box_lines[0].split()]
    y_range = [float(y) for y in box_lines[1].split()]

    box_lx = x_range[1] - x_range[0]
    box_ly = y_range[1] - y_range[0]
    return freud.Box(Lx=box_lx, Ly=box_ly, is2D=True)


def _process_atoms(atom_lines):
    """
    Parses atom data from list of strings.
    Returns positions array (Nx3).
    """
    data_list = [line.split() for line in atom_lines]
    data = np.array(data_list, dtype=float)
    positions = data[:, 2:4]  # Grab x and y columns (index 2 and 3)
    # Add a zero z-column for freud compatibility
    return np.column_stack((positions, np.zeros(len(atom_lines))))


def parse_and_calc_hexatic(filename, verbose=1):
    """
    Parses a LAMMPS trajectory file and calculates the hexatic order parameter.

    Args:
        filename (str): Path to the LAMMPS trajectory file.
        verbose (int): Verbosity level (default: 1).

    Returns:
        tuple: A tuple containing two lists: steps and mean psi6 values.
    """
    steps = []
    psi6_means = []

    for frame in parse_lammps_dump(filename):
        step = frame["timestep"]
        steps.append(step)

        current_box = _process_box(frame["box_bounds"])
        positions = _process_atoms(frame["atoms"])

        hexatic_order_calculator = freud.order.Hexatic(k=6)
        hexatic_order_calculator.compute(
            system=(current_box, positions), neighbors={"num_neighbors": 6}
        )

        mag_psi6 = np.abs(hexatic_order_calculator.particle_order)
        mean_psi6 = np.mean(mag_psi6)
        psi6_means.append(mean_psi6)

        if verbose:
            print(f"Step {step}: Avg |psi6| = {mean_psi6:.4f}")

    return steps, psi6_means


def calc_hexatic_from_tracks(df, frame_width, frame_height, verbose=1):
    """Calculate hexatic order per frame from a particle tracking DataFrame.

    Args:
        df: DataFrame with columns frame, x, y (from tracks.csv)
        frame_width: Image width in pixels — used as freud box Lx
        frame_height: Image height in pixels — used as freud box Ly
        verbose: print per-frame values when 1

    Returns:
        tuple: (frames, psi6_means) — parallel lists, same signature as parse_and_calc_hexatic
    """
    box = freud.Box(Lx=frame_width, Ly=frame_height, is2D=True)
    frames_out, psi6_means = [], []

    for frame_idx, group in df.groupby("frame"):
        positions = group[["x", "y"]].to_numpy()
        if len(positions) < 6:
            continue
        positions_3d = np.column_stack((positions, np.zeros(len(positions))))

        hexatic = freud.order.Hexatic(k=6)
        hexatic.compute(system=(box, positions_3d), neighbors={"num_neighbors": 6})
        mean_psi6 = float(np.mean(np.abs(hexatic.particle_order)))

        frames_out.append(int(frame_idx))
        psi6_means.append(mean_psi6)
        if verbose:
            print(f"Frame {frame_idx}: Avg |psi6| = {mean_psi6:.4f}")

    return frames_out, psi6_means


def main():
    parser = argparse.ArgumentParser(description="Compute hexatic order from a LAMMPS dump file.")
    parser.add_argument(
        "filename", nargs="?", default="dump.lammps", help="Path to LAMMPS dump file"
    )
    args = parser.parse_args()

    timesteps, values = parse_and_calc_hexatic(args.filename)

    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, values, "o-", color="#2c3e50")
    plt.xlabel("Timestep")
    plt.ylabel(r"Average Hexatic Order Parameter $\langle|\psi_6|\rangle$")
    plt.title("Hexatic Order over Time")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
