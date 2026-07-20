#!/usr/bin/env python3
"""Plot per-frame precision/recall/F1/position-error across benchmark.py's
per-model-type outputs, for comparing detector performance side by side.

CLI: uv run python plot_benchmark.py [--output-dir verification_output/]
     [--models rf-detr lodestar]
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Fixed categorical order (never cycled/reassigned) — see verification/README.md
# and the repo's dataviz conventions. New model types append the next slot.
_MODEL_COLORS = {
    "rf-detr": "#2a78d6",  # slot 1: blue
    "lodestar": "#008300",  # slot 2: green
    "trackpy": "#e87ba4",  # slot 3: pink
    "yolo": "#eb6834",  # slot 6: orange
}
_GRID_COLOR = "#e1e0d9"
_AXIS_COLOR = "#c3c2b7"
_INK = "#0b0b0b"
_MUTED = "#898781"

_METRICS = [
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1"),
    ("mean_pos_error_px", "Mean position error (px)"),
]


def _read_accuracy_csv(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    rows.sort(key=lambda r: int(r["frame"]))
    frames = [int(r["frame"]) for r in rows]
    series = {}
    for key, _ in _METRICS:
        series[key] = [float(r[key]) if r[key] != "" else float("nan") for r in rows]
    return frames, series


def _aggregate_from_csv(rows_path):
    """Aggregate precision/recall/F1/mean error from summed tp/fp/fn — matches
    benchmark.py's own overall-summary calculation, not a mean of per-frame values.
    mean_pos_error_px is tp-weighted across frames (approximates benchmark.py's
    pooled-distance mean; an unweighted mean-of-means would skew toward frames
    with few matches)."""
    tp = fp = fn = 0
    weighted_err_sum = 0.0
    err_weight = 0
    with open(rows_path) as f:
        for row in csv.DictReader(f):
            frame_tp = int(row["n_tp"])
            tp += frame_tp
            fp += int(row["n_fp"])
            fn += int(row["n_fn"])
            if row["mean_pos_error_px"] != "" and frame_tp > 0:
                weighted_err_sum += float(row["mean_pos_error_px"]) * frame_tp
                err_weight += frame_tp
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    mean_err = weighted_err_sum / err_weight if err_weight else float("nan")
    return {"precision": prec, "recall": rec, "f1": f1, "mean_pos_error_px": mean_err}


def _read_tracking_csv(path):
    with open(path) as f:
        return next(csv.DictReader(f))


def _color_for(model_type, seen_order):
    if model_type in _MODEL_COLORS:
        return _MODEL_COLORS[model_type]
    # Unknown model type: fall back to the next unused slot in fixed order,
    # never a generated/cycled hue.
    fallback_slots = ["#e87ba4", "#eda100", "#1baf7a", "#4a3aa7", "#e34948"]
    used = {c for m, c in _MODEL_COLORS.items() if m in seen_order}
    for slot in fallback_slots:
        if slot not in used:
            return slot
    return "#898781"


def main():
    parser = argparse.ArgumentParser(
        description="Plot per-frame benchmark metrics across model types"
    )
    parser.add_argument("--output-dir", default="verification_output/")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model types to include (default: autodetect from accuracy_metrics_*.csv present)",
    )
    args = parser.parse_args()

    out = Path(args.output_dir)

    if args.models:
        model_types = args.models
    else:
        model_types = sorted(
            p.stem.replace("accuracy_metrics_", "") for p in out.glob("accuracy_metrics_*.csv")
        )

    if not model_types:
        print(f"Error: no accuracy_metrics_*.csv files found in {out}")
        raise SystemExit(1)

    per_model = {}
    for model_type in model_types:
        csv_path = out / f"accuracy_metrics_{model_type}.csv"
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found — skipping '{model_type}'.")
            continue
        frames, series = _read_accuracy_csv(csv_path)
        aggregate = _aggregate_from_csv(csv_path)
        tracking = None
        tracking_path = out / f"tracking_metrics_{model_type}.csv"
        if tracking_path.exists():
            tracking = _read_tracking_csv(tracking_path)
        per_model[model_type] = {
            "frames": frames,
            "series": series,
            "aggregate": aggregate,
            "tracking": tracking,
        }

    # --- Summary table (stdout) — the table-view twin of the plot below ---
    header = f"{'model':<10} {'precision':>10} {'recall':>8} {'f1':>8} {'mean_err_px':>12}"
    print(header)
    print("-" * len(header))
    for model_type, data in per_model.items():
        a = data["aggregate"]
        print(
            f"{model_type:<10} {a['precision']:>10.4f} {a['recall']:>8.4f} "
            f"{a['f1']:>8.4f} {a['mean_pos_error_px']:>12.2f}"
        )
    if any(data["tracking"] for data in per_model.values()):
        print()
        print(f"{'model':<10} {'mota':>8} {'idf1':>8} {'fragmentations':>15}")
        for model_type, data in per_model.items():
            t = data["tracking"]
            if t:
                print(
                    f"{model_type:<10} {float(t['mota']):>8.4f} {float(t['idf1']):>8.4f} "
                    f"{int(t['num_fragmentations']):>15}"
                )

    # --- Plot: one panel per metric, one line per model, fixed color order ---
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()

    for ax, (key, label) in zip(axes, _METRICS):
        for model_type, data in per_model.items():
            ax.plot(
                data["frames"],
                data["series"][key],
                label=model_type,
                color=_color_for(model_type, per_model.keys()),
                linewidth=1.75,
                solid_capstyle="round",
            )
        ax.set_title(label, fontsize=10, color=_INK)
        ax.set_xlabel("frame", fontsize=9, color=_MUTED)
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

    png_path = out / "benchmark_comparison.png"
    fig.savefig(str(png_path), dpi=100)
    plt.close(fig)
    print(f"\nPlot -> {png_path}")


if __name__ == "__main__":
    main()
