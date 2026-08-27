"""
This module processes LAMMPS trajectory files to generate a phase diagram based on
hexatic order parameters.
"""

import argparse
import csv
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from hexatic_order_analysis import parse_and_calc_hexatic


def extract_epsilon_and_molecules(filename):
    """
    Extracts the epsilon and molecules values from the filename.
    Expected format:
      - *.in_{molecules}_{epsilon}.lammpstrj (Old format)
      - *_{molecules}_{epsilon}.lammpstrj (New format)
    """
    # Regex for new format (flexible script name, followed by _NUM_FLOAT.ext)
    match_new = re.search(r"_(\d+)_([+-]?\d*\.\d+|\d+)\.(lammpstrj|log)$", filename)
    if match_new:
        molecules = int(match_new.group(1))
        epsilon = float(match_new.group(2))
        return molecules, epsilon

    # Regex for old format (*.in_NUM_FLOAT.ext)
    match_old = re.search(r"\.in_(\d+)_([+-]?\d*\.\d+|\d+)\.(lammpstrj|log)", filename)
    if match_old:
        molecules = int(match_old.group(1))
        epsilon = float(match_old.group(2))
        return molecules, epsilon

    return None, None


def test_single_file(filename):
    """
    Test and display hexatic analysis results for a single lammpstrj file.
    """
    n_mols, eps = extract_epsilon_and_molecules(filename)

    if eps is None or n_mols is None:
        print(f"Could not extract parameters from filename: {filename}")
        return

    print(f"Analyzing: {filename}")
    print(f"  Molecules: {n_mols}")
    print(f"  Epsilon: {eps}")

    _, values = parse_and_calc_hexatic(filename)

    final_frame_psi6 = np.mean(values[-1])
    all_frames_mean = np.mean(values)

    print(f"  Final frame avg |psi6|: {final_frame_psi6:.4f}")
    print(f"  All frames avg |psi6|: {all_frames_mean:.4f}")
    print(f"  Total frames: {len(values)}")


def collect_phase_data(filenames, verbose):
    """
    Collects phase data (epsilon, molecules, psi6) from files.
    Returns three lists: epsilons, num_molecules, avg_psi6_list.
    """
    epsilons = []
    num_molecules = []
    avg_psi6_list = []

    for fname in filenames:
        n_mols, eps = extract_epsilon_and_molecules(fname)
        if eps is None or n_mols is None:
            if verbose:
                print(f"Skipping invalid filename: {fname}")
            continue

        # Use your existing function to get timesteps and psi6 values
        _, values = parse_and_calc_hexatic(fname, verbose)

        if not values:
            if verbose:
                print(f"No data found in: {fname}")
            continue

        # We take the mean of the last frame to represent the "stable" state
        final_frame_psi6 = np.mean(values[-1])

        epsilons.append(eps)
        num_molecules.append(n_mols)
        avg_psi6_list.append(final_frame_psi6)

    return epsilons, num_molecules, avg_psi6_list


def generate_stability_plot(data_dir, pattern, verbose):
    """Generate a phase diagram plot based on hexatic order analysis."""
    file_pattern = os.path.join(data_dir, pattern)
    filenames = glob.glob(file_pattern)

    if verbose:
        print(f"Found {len(filenames)} files.")

    epsilons, num_molecules, _avg_psi6 = collect_phase_data(filenames, verbose)

    if not epsilons:
        print("No valid data found to plot.")
        return

    # Persist the grid values so prose claims and caption ranges are verifiable
    csv_path = f"{os.path.basename(data_dir)}_hexatic_order_data.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n_molecules", "epsilon", "psi6_final_frame"])
        for n, eps, psi6 in sorted(zip(num_molecules, epsilons, _avg_psi6)):
            writer.writerow([n, eps, round(psi6, 6)])
    if verbose:
        print(f"Saved grid data to {csv_path}")

    # Get max values
    max_epsilon = max(epsilons) * 1.05  # add 5% buffer to graph
    max_molecules = max(num_molecules) * 1.05  # add 5% buffer to graph

    # Plotting
    # WACV reviewer guidance on color vision deficiency: color must not be
    # the only discriminative feature. RdYlGn is a textbook worst case for
    # red-green colorblindness (~8% of male readers); cividis is a
    # perceptually-uniform sequential map designed to remain legible under
    # the common forms of CVD. Marker size, scaled with the value itself,
    # is the second, non-color feature -- higher order shows as a visibly
    # larger point, independent of the color channel entirely.
    plt.figure(figsize=(10, 7))
    marker_sizes = [40 + 220 * v for v in _avg_psi6]
    scatter = plt.scatter(
        epsilons,
        num_molecules,
        c=_avg_psi6,
        cmap="cividis",
        s=marker_sizes,
        edgecolor="black",
        alpha=0.85,
        vmin=0,
        vmax=1,
    )

    cbar = plt.colorbar(scatter)
    cbar.set_label(r"Average Hexatic Order Parameter $\langle |\psi_6| \rangle$", fontsize=12)

    plt.xlabel(r"Epsilon ($\epsilon$)", fontsize=12)
    plt.title(r"Phase Behavior: $N$ vs $\epsilon$ colored by Hexatic Order", fontsize=14)
    plt.ylabel("Number of Molecules ($N$)", fontsize=12)
    plt.xlim(0, max_epsilon)
    plt.ylim(0, max_molecules)
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.savefig(f"{os.path.basename(data_dir)}_hexatic_order_phase_diagram.png")
    plt.show()


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description="Generate hexatic order phase diagram from LAMMPS trajectory files."
    )
    # REQUIRED unless --test is given -- no default, pass your own data folder.
    PARSER.add_argument(
        "data_dir",
        nargs="?",
        default=None,
        help="Path to folder containing .lammpstrj files (required unless --test is given)",
    )
    PARSER.add_argument(
        "--pattern",
        default="*.in_*_*.lammpstrj",
        help="Filename pattern inside the folder (default: %(default)s)",
    )
    PARSER.add_argument("--test", help="Test a single lammpstrj file instead of generating plot")
    PARSER.add_argument("--verbose", default=0, help="Set to 1 to print out results")

    ARGS = PARSER.parse_args()

    if ARGS.test:
        test_single_file(ARGS.test)
    else:
        if ARGS.data_dir is None:
            PARSER.error("data_dir is required unless --test is given")
        if ARGS.verbose:
            print(ARGS.data_dir)
        generate_stability_plot(ARGS.data_dir, ARGS.pattern, verbose=ARGS.verbose)
