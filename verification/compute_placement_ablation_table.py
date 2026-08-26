#!/usr/bin/env python3
"""Regenerate the placement-ablation table (WACV paper Table 7,
tab:placement_ablation in wacv2027-paper/sec/results.tex): physics-grounded
LAMMPS placement vs. i.i.d. uniform placement (both RF-DETR), plus the
classical Trackpy baseline.

Reuses compare_deeptrack_results.summarize() for the physics/random rows, so
this table and the physics/random/delta comparison table stay derived from
the same per-frame mean-precision/recall/f1 computation.

The Trackpy row is NOT independently recomputed here: no per-frame CSV
currently in verification_output/ reproduces tab:synth_results' published
55.8%/44.5%/49.5% values via a plain mean (checked accuracy_metrics_trackpy.csv
at the top level, procedural_benchmark/, benchmark_1500_250/, reruns/, and
density_ablation/N1446/ -- none match). It's passed through as a documented
constant reused from that table until its own source data is located or
regenerated.

Usage:
    cd verification
    uv run python compute_placement_ablation_table.py
    uv run python compute_placement_ablation_table.py --latex
"""

import argparse
from pathlib import Path

from compare_deeptrack_results import summarize

_DEEPTRACK_COMPARISON = Path("verification_output/deeptrack_comparison")
DEFAULT_PHYSICS = _DEEPTRACK_COMPARISON / "physics" / "accuracy_metrics_rf-detr.csv"
DEFAULT_RANDOM = _DEEPTRACK_COMPARISON / "random" / "accuracy_metrics_rf-detr.csv"

# Trackpy classical baseline, physics-grounded placement, N~1446
# (brightfield_fast rendering) -- reused verbatim from tab:synth_results
# (Table 6) in wacv2027-paper/sec/results.tex. See module docstring: not
# independently recomputed by this script.
TRACKPY_BASELINE = {"precision": 0.558, "recall": 0.445, "f1": 0.495}

ROWS_ORDER = ["physics", "random", "trackpy"]
ROW_LABELS = {
    "physics": "Physics-grounded (RF-DETR)",
    "random": "i.i.d. uniform (RF-DETR)",
    "trackpy": "Classical (Trackpy)",
}


def build_rows(physics_path: Path, random_path: Path) -> dict:
    """Return {condition: {precision, recall, f1}} for all three table rows."""
    return {
        "physics": summarize(physics_path),
        "random": summarize(random_path),
        "trackpy": dict(TRACKPY_BASELINE),
    }


def _print_table(rows: dict) -> None:
    header = f"{'Placement / Method':<28}{'Precision':>11}{'Recall':>9}{'F1':>9}"
    print(header)
    print("-" * len(header))
    for key in ROWS_ORDER:
        row = rows[key]
        print(
            f"{ROW_LABELS[key]:<28}"
            f"{row['precision'] * 100:>10.1f}%"
            f"{row['recall'] * 100:>8.1f}%"
            f"{row['f1'] * 100:>8.1f}%"
        )


def _print_latex(rows: dict) -> None:
    best_f1_key = max(ROWS_ORDER, key=lambda k: rows[k]["f1"])
    print(r"\begin{tabular}{lccc}")
    print(r"\toprule")
    print(
        r"\textbf{Placement / Method} & \textbf{Precision}$\uparrow$ & "
        r"\textbf{Recall}$\uparrow$ & \textbf{F1}$\uparrow$ \\"
    )
    print(r"\midrule")
    for key in ROWS_ORDER:
        row = rows[key]
        f1_cell = f"{row['f1'] * 100:.1f}\\%"
        if key == best_f1_key:
            f1_cell = f"\\textbf{{{f1_cell}}}"
        print(
            f"{ROW_LABELS[key]} & {row['precision'] * 100:.1f}\\% & "
            f"{row['recall'] * 100:.1f}\\% & {f1_cell} \\\\"
        )
    print(r"\bottomrule")
    print(r"\end{tabular}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--physics",
        type=Path,
        default=DEFAULT_PHYSICS,
        help="physics-grounded accuracy_metrics_rf-detr.csv",
    )
    parser.add_argument(
        "--random",
        type=Path,
        default=DEFAULT_RANDOM,
        help="i.i.d.-placement accuracy_metrics_rf-detr.csv",
    )
    parser.add_argument(
        "--latex", action="store_true", help="print as a LaTeX tabular block instead of plain text"
    )
    args = parser.parse_args()

    rows = build_rows(args.physics, args.random)
    if args.latex:
        _print_latex(rows)
    else:
        _print_table(rows)


if __name__ == "__main__":
    main()
