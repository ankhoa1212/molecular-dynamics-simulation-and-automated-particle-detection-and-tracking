"""
Script to run LAMMPS simulations in parallel across multiple input parameters.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor


def run_simulation(cmd, filename, output_dir):
    """Runs a single simulation command and moves the output file."""
    try:
        # Run the command
        # Using shell=True to match bash behavior, though shell=False with list is usually safer.
        # Given the command string construction, shell=True is easier here.
        subprocess.run(cmd, shell=True, check=True)

        # Move the trajectory file
        traj_file = f"{filename}.lammpstrj"
        if os.path.exists(traj_file):
            shutil.move(traj_file, os.path.join(output_dir, traj_file))
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {cmd}\n{e}")


def _parse_json_config(config_file):
    """Parses a JSON configuration file."""
    config_args = []
    with open(config_file, "r", encoding="utf-8") as cfg_file:
        data = json.load(cfg_file)
        if isinstance(data, list):
            config_args = [str(x) for x in data]
        elif isinstance(data, dict):
            # Handle positional arguments in order defined in parser
            positional_keys = [
                "input_file",
                "output_dir",
                "molecules",
                "molecules_end",
                "molecules_step",
                "var_epsilon",
                "var_epsilon_end",
                "var_epsilon_step",
            ]
            for key in positional_keys:
                if key in data:
                    config_args.append(str(data[key]))

            # Handle optional arguments (flags)
            for key, value in data.items():
                if key in positional_keys:
                    continue
                # Determine flag prefix
                prefix = "-" if len(key) == 1 else "--"
                # If key doesn't start with -, add prefix
                arg_name = key if key.startswith("-") else f"{prefix}{key}"
                config_args.append(arg_name)
                config_args.append(str(value))
    return config_args


def _parse_text_config(config_file):
    """Parses a text configuration file."""
    config_args = []
    with open(config_file, "r", encoding="utf-8") as cfg_file:
        # Read content, ignore comments, split by whitespace
        for line in cfg_file:
            line = line.split("#", 1)[0].strip()
            if line:
                config_args.extend(line.split())
    return config_args


def parse_config(argv_list):
    """
    Parses a config file if provided in the command line arguments.
    Returns the modified argument list.
    """
    args_list = list(argv_list)
    while "--config" in args_list:
        try:
            config_index = args_list.index("--config")
            if config_index + 1 < len(args_list):
                config_file = args_list[config_index + 1]
                config_args = []

                if config_file.endswith(".json"):
                    config_args = _parse_json_config(config_file)
                else:
                    config_args = _parse_text_config(config_file)

                # Replace --config and its value with the file arguments
                args_list = args_list[:config_index] + config_args + args_list[config_index + 2 :]
            else:
                print("Error: --config requires a file path")
                sys.exit(1)
        except (IOError, json.JSONDecodeError) as e:
            print(f"Error processing config file: {e}")
            sys.exit(1)
    return args_list


def setup_arg_parser():
    """Sets up the argument parser."""
    parser = argparse.ArgumentParser(description="Run LAMMPS simulation.")

    parser.add_argument("--config", help="Path to file containing arguments", required=False)
    parser.add_argument("input_file", help="LAMMPS input file")
    parser.add_argument("output_dir", nargs="?", help="Output directory")
    parser.add_argument("molecules", nargs="?", help="Start molecules")
    parser.add_argument("molecules_end", nargs="?", help="End molecules")
    parser.add_argument("molecules_step", nargs="?", help="Step molecules")
    parser.add_argument("var_epsilon", nargs="?", help="Start epsilon")
    parser.add_argument("var_epsilon_end", nargs="?", help="End epsilon")
    parser.add_argument("var_epsilon_step", nargs="?", help="Step epsilon")

    # New options
    parser.add_argument(
        "--var_tstart",
        "--t_start",
        dest="var_tstart",
        default="1.0",
        help="Variable tstart (default: 1.0)",
    )
    parser.add_argument(
        "--var_tstop",
        "--t_stop",
        dest="var_tstop",
        default="1.0",
        help="Variable tstop (default: 1.0)",
    )

    parser.add_argument(
        "--vel_force_scale",
        dest="vel_force_scale",
        default=None,
        help="Scaling factor for initial velocity or continuous "
        "force (default: 9 for velocity_initialization.in, 1.0 for continuous_force.in)",
    )
    parser.add_argument("--steps", default="10000", help="Simulation steps (default: 10000)")
    return parser


def check_epsilon_loop(current_e, end_e, step_e):
    """
    Checks if the epsilon loop should continue.
    Float comparison with tolerance.
    """
    if step_e > 0:
        return current_e <= end_e + 1e-9
    return current_e >= end_e - 1e-9


def resolve_arguments(args):
    """
    Resolves arguments to their final values for the simulation configuration.
    Returns a dictionary or object with the resolved settings.
    """
    # Default values
    config = {
        "output_dir": "results",
        "m_start": 100,
        "m_end": None,
        "m_step": 100,
        "e_start": 5.0,
        "e_end": None,
        "e_step": 5.0,
        "vel_force_scale": args.vel_force_scale,
        "input_script": args.input_file,
    }

    if args.output_dir is not None:
        config["output_dir"] = args.output_dir

    if args.molecules is not None:
        config["m_start"] = int(args.molecules)
        config["m_end"] = (
            int(args.molecules_end) if args.molecules_end is not None else config["m_start"]
        )
        step = args.molecules_step if args.molecules_step is not None else "1"
        config["m_step"] = int(step)

    if args.var_epsilon is not None:
        config["e_start"] = float(args.var_epsilon)
        config["e_end"] = (
            float(args.var_epsilon_end) if args.var_epsilon_end is not None else config["e_start"]
        )
        step = args.var_epsilon_step if args.var_epsilon_step is not None else "1"
        config["e_step"] = float(step)
        if config["e_step"] == 0:
            config["e_step"] = 1.0

    # Needs to handle the None case for loop ends if not set
    if config["m_end"] is None:
        config["m_end"] = config["m_start"]
    if config["e_end"] is None:
        config["e_end"] = config["e_start"]

    # Set default scale if not provided
    if config["vel_force_scale"] is None:
        if "velocity_initialization" in config["input_script"]:
            config["vel_force_scale"] = "9"
        else:
            config["vel_force_scale"] = "1.0"

    return config


def generate_commands(config, args, lammps_executable):
    """Generates the list of LAMMPS commands to run."""
    commands = []

    # Strip .in extension
    base_script_name = os.path.basename(config["input_script"])
    if base_script_name.endswith(".in"):
        base_script_name = base_script_name[:-3]

    log_dir = os.path.join(config["output_dir"], "logs")
    os.makedirs(config["output_dir"], exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Clear commands.txt
    with open("commands.txt", "w", encoding="utf-8") as cmd_file:
        pass

    current_molecules = config["m_start"]

    while True:  # Molecules loop
        # Check condition
        if config["m_step"] > 0:
            if current_molecules > config["m_end"]:
                break
        else:
            if current_molecules < config["m_end"]:
                break

        current_epsilon = config["e_start"]

        while check_epsilon_loop(current_epsilon, config["e_end"], config["e_step"]):
            epsilon_str = f"{current_epsilon:.1f}"
            filename = f"{base_script_name}_{current_molecules}_{epsilon_str}"
            log_file = os.path.join(log_dir, f"{filename}.log")

            current_tstart = epsilon_str if args.var_tstart == "epsilon" else args.var_tstart
            current_tstop = epsilon_str if args.var_tstop == "epsilon" else args.var_tstop

            cmd = (
                f'"{lammps_executable}" -in "{config["input_script"]}" '
                f'-log "{log_file}" '
                f'-var filename "{filename}" -var molecules "{current_molecules}" '
                f'-var var_epsilon "{epsilon_str}" '
                f'-var var_tstart "{current_tstart}" '
                f'-var var_tstop "{current_tstop}" '
                f'-var steps "{args.steps}" '
                f'-var vel_force_scale "{config["vel_force_scale"]}"'
            )
            commands.append((cmd, filename))

            with open("commands.txt", "a", encoding="utf-8") as cmd_file:
                cmd_file.write(cmd + "\n")

            current_epsilon += config["e_step"]

        current_molecules += config["m_step"]

    return commands, base_script_name, log_dir


def get_num_cpus():
    """Determines the number of available CPUs."""
    try:
        # Try to get the number of CPUs available to the process
        return len(os.sched_getaffinity(0))
    except AttributeError:
        # Fallback for systems where sched_getaffinity is not available
        return os.cpu_count() or 1


def run_parallel_tasks(commands, output_dir, max_workers):
    """Runs simulation commands in parallel."""
    print(f"Running simulations in parallel with up to {max_workers} jobs...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(run_simulation, cmd, filename, output_dir) for cmd, filename in commands
        ]
        for future in futures:
            future.result()  # Wait for all to complete
    print("All parallel jobs finished.")


def cleanup_files(output_dir, base_script_name):
    """Moves any remaining .lammpstrj files belonging to this simulation to the output directory."""
    for file in os.listdir("."):
        if file.endswith(".lammpstrj") and file.startswith(base_script_name):
            dst = os.path.join(output_dir, file)
            try:
                shutil.move(file, dst)
                print(f"Moved trajectory file '{file}' to '{output_dir}/'")
            except FileNotFoundError:
                pass  # another parallel process already moved it


def print_visualization_commands(config, base_name):
    """Prints commands to help the user visualize results."""
    example_epsilon = f"{config['e_start']:.1f}"
    out_dir = config["output_dir"]
    m_start = config["m_start"]

    file_path = f"{out_dir}/{base_name}_{m_start}_{example_epsilon}.lammpstrj"

    print("=" * 100)
    print(f"Visualize: 'ovito {file_path}'")
    print(f"Temp Graph: 'python temp_graph.py --filename {file_path}'")
    print(f"Vel Graph: 'python velocity_graph.py --filename {file_path}'")
    print(f"Hexatic Graph: 'python hexatic_order_graph.py {file_path}'")
    print(f"Phase Diagram: 'python phase_diagram.py {out_dir}'")
    print("=" * 100)


def main():
    """Main function to parse arguments and run LAMMPS simulations in parallel."""
    # Set LAMMPS executable
    lammps_executable = "lmp"

    num_cpus = get_num_cpus()
    os.environ["OMP_NUM_THREADS"] = str(num_cpus)
    print(f"Setting OMP_NUM_THREADS to {num_cpus}")

    # Pre-process sys.argv to handle --config file
    sys.argv = parse_config(sys.argv)

    # Parse arguments
    parser = setup_arg_parser()
    args = parser.parse_args()

    # Convert to types & resolve defaults
    try:
        config = resolve_arguments(args)
    except ValueError as e:
        print(f"Error parsing arguments: {e}")
        sys.exit(1)

    print("=" * 100)
    print("Starting LAMMPS simulation...")
    print(f"  Executable: {lammps_executable}")
    print(f"  Input file: {config['input_script']}")
    print(f"  Output dir: {config['output_dir']}")
    print("=" * 100)

    commands, base_script_name, log_dir = generate_commands(config, args, lammps_executable)
    print(f"  Log dir:    {log_dir}")
    print("=" * 100)

    # Run commands in parallel
    run_parallel_tasks(commands, config["output_dir"], num_cpus)

    # Cleanup
    cleanup_files(config["output_dir"], base_script_name)

    print("=" * 100)
    print("LAMMPS simulation finished.")
    print(f"Check the log directory '{log_dir}' for details.")

    print_visualization_commands(config, base_script_name)


if __name__ == "__main__":
    main()
