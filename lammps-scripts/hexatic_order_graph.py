"""
This script calculates and plots the Hexatic Order Parameter from a LAMMPS trajectory
file. It uses the hexatic_order_analysis module to perform the calculations.
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from hexatic_order_analysis import calc_hexatic_from_tracks, parse_and_calc_hexatic
from phase_diagram import extract_epsilon_and_molecules


def main():
    """Main function to parse arguments and plot hexatic order."""
    parser = argparse.ArgumentParser(
        description="Calculate Hexatic Order Parameter from LAMMPS trajectory or tracks.csv."
    )
    parser.add_argument(
        "filename",
        nargs="?",
        help="Specific LAMMPS trajectory file to process. "
        "If not provided, processes all in results/.",
    )
    parser.add_argument("--output_dir", default=None, help="Output directory")
    parser.add_argument("--no-show", action="store_true", help="Do not display the graph")
    parser.add_argument(
        "--tracks-csv",
        help="Path to tracks.csv from particle tracking (alternative to LAMMPS file)",
    )
    parser.add_argument(
        "--image-width", type=int, help="Frame width in pixels (required with --tracks-csv)"
    )
    parser.add_argument(
        "--image-height", type=int, help="Frame height in pixels (required with --tracks-csv)"
    )
    args = parser.parse_args()

    if args.tracks_csv:
        if not args.image_width or not args.image_height:
            parser.error("--image-width and --image-height are required with --tracks-csv")

        df = pd.read_csv(args.tracks_csv)
        frames, hexatic_order = calc_hexatic_from_tracks(df, args.image_width, args.image_height)

        plt.figure(figsize=(10, 6))
        plt.plot(frames, hexatic_order, alpha=0.7)
        plt.xlabel("Frame")
        plt.ylabel(r"Global Hexatic Order $|\Psi_6|$")
        plt.title("Hexatic Order Parameter — Particle Tracking")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output_filename = f"{Path(args.tracks_csv).stem}_hexatic_order.png"
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            output_path = os.path.join(args.output_dir, output_filename)
        else:
            output_path = output_filename

        plt.savefig(output_path, dpi=300)
        print(f"Graph saved to {output_path}")
        if not args.no_show:
            plt.show()
        return

    if args.filename:
        filepath = args.filename

        plt.figure(figsize=(10, 6))
        filename = os.path.basename(filepath)

        num_molecules, epsilon = extract_epsilon_and_molecules(filename)

        frames, hexatic_order = parse_and_calc_hexatic(filepath)

        label_str = f"N={num_molecules}, ε={epsilon}"
        plt.plot(frames, hexatic_order, label=label_str, alpha=0.7)

        plt.xlabel("Frame")
        plt.ylabel(r"Global Hexatic Order $|\Psi_6|$")
        plt.title(f"Hexatic Order Parameter: N={num_molecules}, ε={epsilon}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if "." in filename:
            base = filename[: filename.rfind(".")]
        else:
            base = filename

        output_filename = f"{base}_hexatic_order.png"

        if args.output_dir:
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
            output_path = os.path.join(args.output_dir, output_filename)
        else:
            output_path = output_filename

        plt.savefig(output_path, dpi=300)
        print(f"Graph saved to {output_path}")
        if not args.no_show:
            plt.show()


if __name__ == "__main__":
    main()
