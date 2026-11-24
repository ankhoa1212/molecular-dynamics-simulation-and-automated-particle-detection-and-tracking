"""Make lammps-scripts' top-level modules importable and plot without a display."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
