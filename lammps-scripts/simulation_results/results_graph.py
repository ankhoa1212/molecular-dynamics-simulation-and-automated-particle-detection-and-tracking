"""Script to generate a scatter plot of Molecule Count vs. Epsilon
Requires Pandas and Matplotlib libraries.
To install the required libraries, run:
pip install pandas matplotlib
"""

import os

import matplotlib.colors
import matplotlib.pyplot as plt
import pandas as pd

# Expects columns named exactly: 'molecule count', 'epsilon', and 'result'
df = pd.read_csv(os.path.join(os.getcwd(), "simulation_results.csv"))

# result: 0 = not crystallized, 1 = partially, 2 = fully
COLORS = ["red", "yellow", "green"]
CMAP = matplotlib.colors.ListedColormap(COLORS)
BOUNDS = [0, 1, 2, 3]
NORM = matplotlib.colors.BoundaryNorm(BOUNDS, CMAP.N)

plt.figure(figsize=(8, 6))
SCATTER = plt.scatter(
    df["epsilon"],
    df["molecule count"],
    c=df["result"],
    cmap=CMAP,
    norm=NORM,
    s=50,
    alpha=0.8,
)

plt.xlabel(r"$\epsilon$ (Epsilon)", fontsize=14)
plt.ylabel("Molecule Count", fontsize=14)
plt.title("Molecule Count vs. $\\epsilon$ (Epsilon) (Result in Color)", fontsize=12)

CBAR = plt.colorbar(SCATTER, ticks=[0.5, 1.5, 2.5])
CBAR.set_label("Result (0=Red, 1=Yellow, 2=Green)")
CBAR.set_ticklabels(
    ["0 (Not crystallized)", "1 (Partially crystallized)", "2 (Fully crystallized)"]
)

plt.savefig("molecule_count_vs_epsilon.png", dpi=300, bbox_inches="tight")

plt.grid(True)
plt.show()
