#!/usr/bin/env python3
"""Plot detection accuracy vs. particle count across the density/overlap
ablation sweep produced by run_density_ablation.sh: one point per
(model, N) in verification_output/density_ablation/N<N>/accuracy_metrics_*.csv,
aggregated the same way plot_benchmark.py aggregates a single run's per-frame
CSV into one summary number. Reuses plot_benchmark.py's aggregation formula
and color/style conventions directly rather than re-deriving them, so this
figure stays visually and numerically consistent with benchmark_comparison.png.

CLI: uv run python plot_density_ablation.py [--ablation-dir verification_output/density_ablation/]
     [--models rf-detr yolo12m]
"""
import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_benchmark import (
    _AXIS_COLOR,
    _GRID_COLOR,
    _INK,
    _METRICS,
    _MUTED,
    _aggregate_from_csv,
    _color_for,
)

_N_DIR_RE = re.compile(r"^N(\d+)$")


def _discover_densities(ablation_dir):
    densities = []
    for p in ablation_dir.iterdir():
        if not p.is_dir():
            continue
        m = _N_DIR_RE.match(p.name)
        if m:
            densities.append((int(m.group(1)), p))
    densities.sort(key=lambda t: t[0])
    return densities


def main():
    parser = argparse.ArgumentParser(
        description="Plot precision/recall/F1/localization error vs. particle count"
    )
    parser.add_argument("--ablation-dir", default="verification_output/density_ablation/")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model types to include (default: autodetect from accuracy_metrics_*.csv present)",
    )
    args = parser.parse_args()

    ablation_dir = Path(args.ablation_dir)
    densities = _discover_densities(ablation_dir)
    if not densities:
        print(f"Error: no N<density> subdirectories found in {ablation_dir}")
        raise SystemExit(1)

    if args.models:
        model_types = list(args.models)
    else:
        seen = set()
        for _, n_dir in densities:
            for p in n_dir.glob("accuracy_metrics_*.csv"):
                seen.add(p.stem.replace("accuracy_metrics_", ""))
        model_types = sorted(seen)

    if not model_types:
        print(f"Error: no accuracy_metrics_*.csv files found under {ablation_dir}")
        raise SystemExit(1)

    # per_model[model] = {"n": [...], "series": {metric: [...]}}
    per_model = {m: {"n": [], "series": {key: [] for key, _ in _METRICS}} for m in model_types}

    for n_value, n_dir in densities:
        for model_type in model_types:
            csv_path = n_dir / f"accuracy_metrics_{model_type}.csv"
            if not csv_path.exists():
                print(f"Warning: {csv_path} not found -- skipping N={n_value} for '{model_type}'.")
                continue
            aggregate = _aggregate_from_csv(csv_path)
            per_model[model_type]["n"].append(n_value)
            for key, _ in _METRICS:
                per_model[model_type]["series"][key].append(aggregate[key])

    # --- Per-model discovery confirmation ---
    for model_type in model_types:
        if len(per_model[model_type]["n"]) == len(densities):
            print(f"Found all N-points for '{model_type}'")

    # --- Summary table (stdout) ---
    header = f"{'model':<10} {'N':>6} {'precision':>10} {'recall':>8} {'f1':>8} {'mean_err_px':>12}"
    print(header)
    print("-" * len(header))
    for model_type, data in per_model.items():
        for i, n_value in enumerate(data["n"]):
            s = data["series"]
            print(
                f"{model_type:<10} {n_value:>6} {s['precision'][i]:>10.4f} {s['recall'][i]:>8.4f} "
                f"{s['f1'][i]:>8.4f} {s['mean_pos_error_px'][i]:>12.2f}"
            )

    # --- Plot: one panel per metric, one line per model, vs. particle count ---
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()

    for ax, (key, label) in zip(axes, _METRICS):
        for model_type, data in per_model.items():
            if not data["n"]:
                continue
            ax.plot(
                data["n"],
                data["series"][key],
                label=model_type,
                color=_color_for(model_type, per_model.keys()),
                linewidth=1.75,
                marker="o",
                markersize=4,
                solid_capstyle="round",
            )
        ax.set_title(label, fontsize=10, color=_INK)
        ax.set_xlabel("particles per frame (N)", fontsize=9, color=_MUTED)
        ax.grid(True, color=_GRID_COLOR, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(_AXIS_COLOR)
        ax.tick_params(colors=_MUTED, labelsize=8)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(per_model), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    png_path = ablation_dir / "density_ablation.png"
    fig.savefig(str(png_path), dpi=100)
    plt.close(fig)
    print(f"\nPlot -> {png_path}")


if __name__ == "__main__":
    main()
