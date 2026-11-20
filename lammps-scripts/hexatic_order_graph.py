"""
This script calculates and plots the Hexatic Order Parameter from a LAMMPS trajectory
file. It uses the hexatic_order_analysis module to perform the calculations.
"""

import argparse
import os
from pathlib import Path

import pandas as pd
from graph_style import MAIN_COLOR, finish_and_save, mark_start_release_end, new_figure, run_title
from hexatic_order_analysis import calc_hexatic_from_tracks, parse_and_calc_hexatic

HEXATIC_YLABEL = r"Global hexatic order $|\Psi_6|$"


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

        fig, ax = new_figure()
        ax.plot(frames, hexatic_order, color=MAIN_COLOR, linewidth=2.0)
        output_filename = f"{Path(args.tracks_csv).stem}_hexatic_order.png"
        output_path = (
            os.path.join(args.output_dir, output_filename) if args.output_dir else output_filename
        )
        # Video frames from particle tracking: no LAMMPS log, so no pull release to mark
        finish_and_save(
            fig,
            ax,
            "Hexatic order, particle tracking",
            HEXATIC_YLABEL,
            output_path,
            args.no_show,
            xlabel="Frame",
        )
        return

    if args.filename:
        filepath = args.filename
        filename = os.path.basename(filepath)

        steps, hexatic_order = parse_and_calc_hexatic(filepath)

        fig, ax = new_figure()
        ax.plot(steps, hexatic_order, color=MAIN_COLOR, linewidth=2.0)
        mark_start_release_end(ax, filepath, steps, hexatic_order)

        base = filename[: filename.rfind(".")] if "." in filename else filename
        output_filename = f"{base}_hexatic_order.png"
        output_path = (
            os.path.join(args.output_dir, output_filename) if args.output_dir else output_filename
        )
        finish_and_save(
            fig, ax, run_title("Hexatic order", filepath), HEXATIC_YLABEL, output_path, args.no_show
        )


if __name__ == "__main__":
    main()
