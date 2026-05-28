### Usage Guide

- [Running Simulations with `run.py`](#running-simulations-with-runpy)
- [Plotting results with `velocity_graph.py`](#plotting-results-with-velocity_graphpy)
- [Plotting results with `temp_graph.py`](#plotting-results-with-temp_graphpy)
- [Plotting results with `phase_diagram.py`](#plotting-results-with-phase_diagrampy)
- [Older scripts (may not be relevant)](#older-scripts-may-not-be-as-relevant)
    - [Running Simulations with `run.sh`](#running-simulations-with-runsh)
    - [Plotting Results with `graph.py`](#plotting-results-with-graphpy)

### Running Simulations with `run.py`

The `run.py` script automates running LAMMPS simulations with configurable parameters. To use it:

```bash
python3 run.py [options]
```

To list possible options:

```bash
python3 run.py -h
```

Usage examples:

```bash
python3 run.py --config config/continuous_force_test.json
```

Or with positional arguments:

```bash
python3 run.py test.in results 1000 1000 100 5.0 10.0 1.0
```

This will execute a LAMMPS simulation with the specified input file, number of molecules, and epsilon value, storing results in the output directory.

---

### Plotting results with `velocity_graph.py`

The `velocity_graph.py` script generates velocity distribution plots from simulation data. To use it:

```bash
python3 velocity_graph.py --filename <path_to_lammpstrj_file>
```

- `--filename / -f`: Path to a `.lammpstrj` trajectory file

For example:

```bash
python3 velocity_graph.py --filename results/simulation.lammpstrj
```

This will analyze the velocity data and produce velocity distribution plots.

---

### Plotting results with `temp_graph.py`

The `temp_graph.py` script generates temperature plots from simulation data. To use it:

```bash
python3 temp_graph.py --filename <path_to_file>
```

- `--filename / -f`: Path to a `.lammpstrj` or log file

For example:

```bash
python3 temp_graph.py --filename results/simulation.lammpstrj
```

This will analyze the temperature data and produce a temperature vs. time plot.

---

### Plotting results with `phase_diagram.py`

The `phase_diagram.py` script generates phase diagrams from simulation data. To use it:

```bash
python3 phase_diagram.py <output_dir>
```

- `<output_dir>`: Directory containing simulation output files

For example:

```bash
python3 phase_diagram.py results
```

This will analyze the simulation results and produce a phase diagram plot.

---


### Older scripts (may not be as relevant)

### Running Simulations with `run.sh`
The `run.sh` script automates running multiple LAMMPS simulations by sweeping over a range of molecule counts and epsilon values. The usage is:

```bash
./run.sh <input_file> <output_dir> <min_molecules> <max_molecules> <molecules_step> <min_epsilon> <max_epsilon> <epsilon_step>
```

- `<input_file>`: LAMMPS input script (e.g., `central_pair_interaction.in`)
- `<output_dir>`: Directory to store simulation outputs (e.g., `tests`)
- `<min_molecules>`: Minimum number of molecules to simulate
- `<max_molecules>`: Maximum number of molecules to simulate
- `<molecules_step>`: Step size for molecule count
- `<min_epsilon>`: Minimum epsilon value (interaction strength)
- `<max_epsilon>`: Maximum epsilon value
- `<epsilon_step>`: Step size for epsilon

The script will run simulations for each combination of molecule count and epsilon value in the specified ranges.
To run a LAMMPS simulation, use the following command format:

```bash
./run.sh <input_file> <output_dir> <min_molecules> <max_molecules> <molecules_step> <min_epsilon> <max_epsilon> <epsilon_step>
```

For example:

```bash
./run.sh central_pair_interaction.in tests 1000 2000 1000 5.0 10.0 5.0
```

---

### Plotting Results with `graph.py`

Install requirements

```bash
pip install -r requirements.txt
```

To generate plots from simulation output data, use:

```bash
python3 graph.py <filename>
```

For example:

```bash
python3 graph.py test.lammpstrj
```

- `<filename>`: Trajectory result file to plot data from

This will produce a plot of the specified property over time using the data in the given output directory.
