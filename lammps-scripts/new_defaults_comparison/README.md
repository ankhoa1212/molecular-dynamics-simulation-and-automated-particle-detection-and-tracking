# continuous_force.in: old vs new defaults

Figures comparing trajectories from `continuous_force.in` before and after the
grid-collapse fix (old: binned `temp/profile` thermostat bias, pull 1, pull always on,
no-pull radius 10; new: plain Langevin, pull 40, pull off at 2/3 of the run, no-pull radius 2).

Every existing trajectory set was re-run with the new defaults, keeping each run's
original N, epsilon, steps, temperatures and seed (152 runs: the 140-cell phase-diagram
grid, the density ablation, the production N=1500 run, the 5-seed variance set and the
single test runs). The trajectories themselves are gitignored; locally they are in
`lammps-scripts/results/new_defaults/`, next to the old sets in `lammps-scripts/results/`.

| File | What it shows |
|---|---|
| `snapshots_*.png` | Particle positions over the run, old (top row) vs new (bottom row) |
| `phase_diagram_old_vs_new.png` | `phase_diagram.py` final-frame psi6 over the N x epsilon grid |
| `renders_old_vs_new.png` | `verification/render.py` frames from the production N=1500 run, same render settings |
| `graph_{old,new}_production_1500_5.0_*.png` | `graph.py` hexatic order, temperature and velocity for the production run |
| `summary.txt` | Final-frame metrics per set (psi6, fraction in largest cluster, cluster count, neighbor spacing, central void, squareness, final temperature) |

Final-frame metrics are computed on the last dump frame: clusters use a 1.5 sigma
neighbor cutoff; squareness is |sum r^4 e^(4i theta)| / sum r^4 of the largest cluster
(0 = round); central void is the largest empty circle within 4 sigma of the cluster center.
