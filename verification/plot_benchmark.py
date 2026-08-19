#!/usr/bin/env python3
"""Plot per-frame precision/recall/F1/position-error across benchmark.py's
per-model-type outputs, for comparing detector performance side by side.

Writes two figures:
- benchmark_comparison.png: the per-frame line panels above, plus a grouped
  bar panel for aggregate tracking metrics (MOTA/IDF1/fragmentations) when
  tracking_metrics_{model_type}_{tracker}.csv files are present -- one bar
  group per model_type, one bar per tracker (trackpy/bytetrack) within that
  group.
- benchmark_summary.png: one bar per model per run-level scalar (F1, MOTA,
  IDF1, fragmentations, ID switches, median inference time) -- MOTA/IDF1/
  fragmentations/ID switches use each model's trackpy result specifically
  (this figure predates --tracker bytetrack and stays one-bar-per-model; see
  benchmark_comparison.png's bar panel for the tracker-inclusive view).

CLI: uv run python plot_benchmark.py [--output-dir verification_output/]
     [--models rf-detr lodestar]
"""
import argparse
import csv
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

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

_TRACKING_METRICS = [
    ("mota", "MOTA"),
    ("idf1", "IDF1"),
    ("num_fragmentations", "Fragmentations"),
]

# Fixed tracker display order (matches benchmark.py's --tracker choices) so bar
# groups and legends are always ordered the same way regardless of glob order.
# Unknown/future tracker names sort alphabetically after these.
_TRACKER_ORDER = ["trackpy", "bytetrack"]
_HATCHES = ["", "///", "xxx", "..."]


def _tracker_sort_key(tracker):
    try:
        return (0, _TRACKER_ORDER.index(tracker))
    except ValueError:
        return (1, tracker)


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
    with few matches). inference_time_ms is a plain (unweighted) mean/median
    across frames -- it isn't a match-quality metric, so match-count weighting
    doesn't apply; missing on older CSVs (added after benchmark.py started
    recording per-frame timing), in which case both come back None."""
    tp = fp = fn = 0
    weighted_err_sum = 0.0
    err_weight = 0
    inference_times_ms = []
    with open(rows_path) as f:
        for row in csv.DictReader(f):
            frame_tp = int(row["n_tp"])
            tp += frame_tp
            fp += int(row["n_fp"])
            fn += int(row["n_fn"])
            if row["mean_pos_error_px"] != "" and frame_tp > 0:
                weighted_err_sum += float(row["mean_pos_error_px"]) * frame_tp
                err_weight += frame_tp
            if row.get("inference_time_ms", "") != "":
                inference_times_ms.append(float(row["inference_time_ms"]))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    mean_err = weighted_err_sum / err_weight if err_weight else float("nan")
    mean_inference_ms = statistics.mean(inference_times_ms) if inference_times_ms else None
    median_inference_ms = statistics.median(inference_times_ms) if inference_times_ms else None
    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "mean_pos_error_px": mean_err,
        "mean_inference_ms": mean_inference_ms,
        "median_inference_ms": median_inference_ms,
    }


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


def _style_axes(ax):
    """Apply this file's shared minimal/muted axis styling."""
    ax.grid(True, axis="y", color=_GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(_AXIS_COLOR)
    ax.tick_params(colors=_MUTED, labelsize=8)


def _plot_tracking_bars(ax, model_order, per_tracking, metric_key, label):
    """Grouped bar chart for one aggregate tracking metric: one bar group per
    model_type (in model_order, filtered to those with >=1 tracker present),
    one bar per tracker within the group. Bar color reuses _color_for's
    fixed model-type color; trackers within a group are distinguished by
    hatch pattern (and a slight alpha shade) rather than color, since color
    is reserved for model identity per this file's convention."""
    groups = [m for m in model_order if any(mt == m for (mt, _tr) in per_tracking)]
    if not groups:
        ax.text(
            0.5,
            0.5,
            "no tracking metrics",
            ha="center",
            va="center",
            color=_MUTED,
            fontsize=9,
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(label, fontsize=10, color=_INK)
        return

    bar_width = 0.32
    gap = 0.04
    for gi, model_type in enumerate(groups):
        trackers = sorted(
            (tr for (mt, tr) in per_tracking if mt == model_type),
            key=_tracker_sort_key,
        )
        color = _color_for(model_type, model_order)
        n = len(trackers)
        for idx, tracker in enumerate(trackers):
            offset = (idx - (n - 1) / 2) * (bar_width + gap)
            raw = per_tracking[(model_type, tracker)][metric_key]
            value = int(raw) if metric_key == "num_fragmentations" else float(raw)
            ax.bar(
                gi + offset,
                value,
                width=bar_width,
                color=color,
                edgecolor=_INK,
                linewidth=0.6,
                hatch=_HATCHES[idx % len(_HATCHES)],
                alpha=1.0 if idx == 0 else 0.7,
            )

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, fontsize=8, color=_MUTED)
    ax.set_title(label, fontsize=10, color=_INK)
    _style_axes(ax)


def _build_figure():
    """2x2 grid of per-frame line panels (top), grouped bar panels for
    aggregate tracking metrics below (bottom row). The bottom row's 3 panels
    each span 2 of the mosaic's 6 columns, vs. the top row's 2 panels each
    spanning 3 -- a stacked-below layout that keeps both mark types (lines
    for per-frame series, bars for single aggregate values) in the same
    figure without cramming bars into a line-panel slot."""
    mosaic = [
        ["precision", "precision", "precision", "recall", "recall", "recall"],
        [
            "f1",
            "f1",
            "f1",
            "mean_pos_error_px",
            "mean_pos_error_px",
            "mean_pos_error_px",
        ],
        ["mota", "mota", "idf1", "idf1", "num_fragmentations", "num_fragmentations"],
    ]
    return plt.subplot_mosaic(mosaic, figsize=(12, 11.5))


def _add_tracker_legend(fig, per_tracking):
    """Add a small secondary legend mapping hatch pattern -> tracker name,
    separate from the model-color legend (color already means model_type)."""
    if not per_tracking:
        return
    trackers_seen = sorted({tr for (_mt, tr) in per_tracking}, key=_tracker_sort_key)
    tracker_handles = [
        Patch(
            facecolor="white",
            edgecolor=_INK,
            hatch=_HATCHES[i % len(_HATCHES)],
            label=tracker,
        )
        for i, tracker in enumerate(trackers_seen)
    ]
    fig.legend(
        handles=tracker_handles,
        loc="upper right",
        ncol=len(tracker_handles),
        frameon=False,
        fontsize=8,
        title="tracker",
        title_fontsize=8,
    )


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
    # Keyed by (model_type, tracker) — a model can have 0, 1, or 2 tracker
    # results present (trackpy, bytetrack, or both); see U5's
    # tracking_metrics_{model_type}_{tracker}.csv naming scheme.
    per_tracking = {}
    for model_type in model_types:
        csv_path = out / f"accuracy_metrics_{model_type}.csv"
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found — skipping '{model_type}'.")
            continue
        frames, series = _read_accuracy_csv(csv_path)
        aggregate = _aggregate_from_csv(csv_path)
        per_model[model_type] = {
            "frames": frames,
            "series": series,
            "aggregate": aggregate,
        }
        prefix = f"tracking_metrics_{model_type}_"
        for tracking_path in sorted(out.glob(f"{prefix}*.csv")):
            tracker = tracking_path.stem[len(prefix) :]
            per_tracking[(model_type, tracker)] = _read_tracking_csv(tracking_path)

    # --- Summary table (stdout) — the table-view twin of the plot below ---
    header = (
        f"{'model':<10} {'precision':>10} {'recall':>8} {'f1':>8} {'mean_err_px':>12} "
        f"{'median_ms/frame':>16}"
    )
    print(header)
    print("-" * len(header))
    for model_type, data in per_model.items():
        a = data["aggregate"]
        median_ms = a["median_inference_ms"]
        median_str = f"{median_ms:.2f}" if median_ms is not None else "n/a"
        print(
            f"{model_type:<10} {a['precision']:>10.4f} {a['recall']:>8.4f} "
            f"{a['f1']:>8.4f} {a['mean_pos_error_px']:>12.2f} {median_str:>16}"
        )
    if per_tracking:
        print()
        tracking_header = (
            f"{'model':<10} {'tracker':<10} {'mota':>8} {'idf1':>8} {'fragmentations':>15} "
            f"{'id_switches':>12}"
        )
        print(tracking_header)
        print("-" * len(tracking_header))
        for model_type in per_model:
            trackers = sorted(
                (tr for (mt, tr) in per_tracking if mt == model_type),
                key=_tracker_sort_key,
            )
            for tracker in trackers:
                t = per_tracking[(model_type, tracker)]
                print(
                    f"{model_type:<10} {tracker:<10} {float(t['mota']):>8.4f} "
                    f"{float(t['idf1']):>8.4f} {int(t['num_fragmentations']):>15} "
                    f"{int(t['num_switches']):>12}"
                )

    # --- Plot: per-frame line panels plus grouped tracking-metric bar panels ---
    fig, axd = _build_figure()
    axes = [axd[key] for key, _ in _METRICS]

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

    for key, label in _TRACKING_METRICS:
        _plot_tracking_bars(axd[key], list(per_model.keys()), per_tracking, key, label)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(per_model), frameon=False)
    _add_tracker_legend(fig, per_tracking)

    fig.tight_layout(rect=[0, 0, 1, 0.94])

    png_path = out / "benchmark_comparison.png"
    fig.savefig(str(png_path), dpi=100)
    plt.close(fig)
    print(f"\nPlot -> {png_path}")

    # --- Summary bar chart: one bar per model per metric, six run-level
    # scalars (unlike the per-frame line plots above) that the U6/U7-era
    # sweeps only ever printed as text -- F1, MOTA, IDF1, fragmentations, ID
    # switches, and inference time all need to be visually comparable across
    # all four methods too, not just readable off a table.
    summary_path = _plot_summary_bars(per_model, per_tracking, out)
    if summary_path:
        print(f"Summary plot -> {summary_path}")


def _plot_summary_bars(per_model, per_tracking, out):
    """One bar chart per run-level scalar metric (F1, MOTA, IDF1,
    fragmentations, ID switches, median inference time), one bar per model,
    same fixed color order as the per-frame line plots. Tracking-metric
    panels are skipped entirely if no model has tracking data (no
    --ground-truth-tracks run); the inference-time panel is skipped if no
    model's CSV has timing (older run, before benchmark.py recorded it).

    Unlike _plot_tracking_bars (grouped by (model, tracker), showing every
    tracker present), this figure is one-bar-per-model -- for a model with
    multiple trackers' results present, it shows trackpy's specifically
    (this figure predates --tracker bytetrack; trackpy is the default
    tracker and the closest match to this figure's original one-tracker-
    per-model assumption). A model with only a bytetrack result and no
    trackpy result contributes no bar to this figure -- see
    _plot_tracking_bars for the tracker-inclusive view.
    """
    models = list(per_model.keys())

    def _trackpy_result(model):
        return per_tracking.get((model, "trackpy"))

    has_tracking = any(_trackpy_result(m) for m in models)
    has_timing = any(
        data["aggregate"]["median_inference_ms"] is not None for data in per_model.values()
    )

    panels = [("f1", "F1", lambda d, m: d["aggregate"]["f1"])]
    if has_tracking:
        panels += [
            (
                "mota",
                "MOTA",
                lambda d, m: float(t["mota"]) if (t := _trackpy_result(m)) else None,
            ),
            (
                "idf1",
                "IDF1",
                lambda d, m: float(t["idf1"]) if (t := _trackpy_result(m)) else None,
            ),
            (
                "fragmentations",
                "Fragmentations",
                lambda d, m: int(t["num_fragmentations"]) if (t := _trackpy_result(m)) else None,
            ),
            (
                "id_switches",
                "ID switches",
                lambda d, m: int(t["num_switches"]) if (t := _trackpy_result(m)) else None,
            ),
        ]
    if has_timing:
        panels.append(
            (
                "inference_ms",
                "Inference time (ms/frame, median)",
                lambda d, m: d["aggregate"]["median_inference_ms"],
            )
        )

    if len(panels) == 1 and not has_timing:
        # Nothing beyond plain F1 (already in the line-plot figure above) --
        # not worth a whole extra figure for one redundant bar chart.
        return None

    ncols = 3
    nrows = -(-len(panels) // ncols)  # ceil division
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows))
    axes = axes.flatten() if len(panels) > 1 else [axes]

    for ax, (_, label, getter) in zip(axes, panels):
        values = [getter(per_model[m], m) for m in models]
        bar_models = [m for m, v in zip(models, values) if v is not None]
        bar_values = [v for v in values if v is not None]
        bar_colors = [_color_for(m, models) for m in bar_models]
        ax.bar(bar_models, bar_values, color=bar_colors)
        ax.set_title(label, fontsize=10, color=_INK)
        ax.grid(True, axis="y", color=_GRID_COLOR, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(_AXIS_COLOR)
        ax.tick_params(colors=_MUTED, labelsize=8)

    for ax in axes[len(panels) :]:
        ax.axis("off")

    fig.tight_layout()
    summary_path = out / "benchmark_summary.png"
    fig.savefig(str(summary_path), dpi=100)
    plt.close(fig)
    return summary_path


if __name__ == "__main__":
    main()
