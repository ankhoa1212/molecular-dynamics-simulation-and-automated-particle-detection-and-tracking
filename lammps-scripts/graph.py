"""
This module serves as a wrapper to run multiple analysis scripts
(hexatic_order_graph.py, velocity_graph.py, temp_graph.py) on LAMMPS
trajectory files. It can process individual files or directories of files.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_script(script_name, file_path, output_dir=None, no_show=False):
    """
    Executes a single analysis script on the target file.

    Args:
        script_name (str): The name of the Python script to run.
        file_path (str or Path): Path to the input file (e.g. .lammpstrj).
        output_dir (str or Path, optional): Directory to save output.
        no_show (bool, optional): If True, suppresses plot display.
    """
    script_dir = Path(__file__).resolve().parent
    script_path = script_dir / script_name

    cmd = ["python3", str(script_path)]

    if script_name == "hexatic_order_graph.py":
        cmd.append(str(file_path))
    else:
        cmd.extend(["--filename", str(file_path)])

    if output_dir:
        cmd.extend(["--output_dir", str(output_dir)])

    if no_show:
        cmd.append("--no-show")

    print(f"Running {script_name} on {file_path}...")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")


def process_file(file_path, output_dir=None, no_show=False):
    """
    Runs all analysis scripts on a single LAMMPS trajectory file.

    Args:
        file_path (str or Path): Path to the input file.
        output_dir (str or Path, optional): Directory to save output.
        no_show (bool, optional): If True, suppresses plot display.
    """
    scripts = ["hexatic_order_graph.py", "velocity_graph.py", "temp_graph.py"]
    for script in scripts:
        run_script(script, file_path, output_dir, no_show)


def main():
    """
    Main function to parse arguments and run the analysis scripts.
    It supports processing a single file or a directory.
    """
    parser = argparse.ArgumentParser(description="Run analysis scripts on LAMMPS trajectory files.")
    parser.add_argument(
        "input_path", help="Path to a .lammpstrj file or a directory containing them."
    )
    parser.add_argument(
        "--output_dir", "-o", default=None, help="Optional output directory for graphs."
    )
    parser.add_argument(
        "--no-show", action="store_true", help="Do not display the graphs interactively."
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_dir = args.output_dir
    no_show = args.no_show

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if input_path.is_file():
        if input_path.suffix == ".lammpstrj":
            process_file(input_path, output_dir, no_show)
        else:
            print(f"Error: {input_path} is not a .lammpstrj file.")
            sys.exit(1)

    elif input_path.is_dir():
        files = sorted(list(input_path.glob("*.lammpstrj")))
        if not files:
            print(f"No .lammpstrj files found in {input_path}")
            sys.exit(0)

        print(f"Found {len(files)} files processing...")
        for file_path in files:
            process_file(file_path, output_dir, no_show)

    else:
        print(f"Error: {input_path} does not exist.")
        sys.exit(1)


if __name__ == "__main__":
    main()
