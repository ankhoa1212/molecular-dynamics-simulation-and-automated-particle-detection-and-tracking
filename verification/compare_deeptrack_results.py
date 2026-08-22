#!/usr/bin/env python3
"""Compare physics-grounded vs. random-placement benchmark results.

Reads both conditions' accuracy_metrics_*.csv files, computes per-column
means, builds a three-row comparison table (physics / random / delta), and
prints it to stdout in aligned columns.  Also writes comparison_table.csv
to the current working directory.

If an optional tracking-metrics CSV exists alongside each input CSV
(same directory), its mota/idf1 columns are added; otherwise those columns
are omitted silently.

Sign convention:  delta = random − physics
  positive delta → random condition produced a higher value (random "better")
  negative delta → physics condition produced a higher value (physics "better")

Usage:
    uv run python compare_deeptrack_results.py \\
        --physics verification_output/deeptrack_comparison/physics/accuracy_metrics_rf-detr.csv \\
        --random  verification_output/deeptrack_comparison/random/accuracy_metrics_rf-detr.csv
"""

import argparse
import csv
import sys
from pathlib import Path
from statistics import mean


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def summarize(csv_path: Path) -> dict:
    """Read accuracy_metrics_*.csv and return mean metrics as a dict.

    Returns a dict with keys: precision, recall, f1, and optionally
    loc_error_px (omitted when all mean_pos_error_px cells are empty).

    Uses only stdlib csv and statistics.mean — no pandas.
    """
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"No data rows in {csv_path}")

    precision = mean(float(r["precision"]) for r in rows)
    recall = mean(float(r["recall"]) for r in rows)
    f1 = mean(float(r["f1"]) for r in rows)

    loc_values = [
        float(r["mean_pos_error_px"]) for r in rows if r.get("mean_pos_error_px", "").strip() != ""
    ]

    result: dict = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    if loc_values:
        result["loc_error_px"] = mean(loc_values)

    return result


def _read_tracking_metrics(csv_dir: Path) -> dict | None:
    """Look for tracking_metrics_*.csv in csv_dir and return mean mota/idf1.

    Returns None if no such file exists (columns will be omitted silently).
    Reads the first matching file found; if multiple trackers were run, the
    first alphabetically is used (deterministic, matches the comparison
    config's single --tracker bytetrack run).
    """
    candidates = sorted(csv_dir.glob("tracking_metrics_*.csv"))
    if not candidates:
        return None

    tracking_path = candidates[0]
    with open(tracking_path, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return None

    result = {}
    mota_values = [
        float(r["mota"]) for r in rows if r.get("mota", "").strip() not in ("", "nan", "None")
    ]
    idf1_values = [
        float(r["idf1"]) for r in rows if r.get("idf1", "").strip() not in ("", "nan", "None")
    ]

    if mota_values:
        result["mota"] = mean(mota_values)
    if idf1_values:
        result["idf1"] = mean(idf1_values)

    return result if result else None


def build_comparison_table(physics: dict, random_cond: dict) -> list[dict]:
    """Build a 3-row comparison table from pre-computed metric dicts.

    Parameters
    ----------
    physics:      dict of mean metrics for the physics-grounded condition
    random_cond:  dict of mean metrics for the random-placement condition

    Returns
    -------
    A list of exactly 3 dicts:
      [0] condition="physics"
      [1] condition="random"
      [2] condition="delta"  where delta = random − physics elementwise

    All numeric keys present in both input dicts appear in the output.
    Keys present in only one dict are omitted from delta (treated as missing).
    """
    # Numeric columns: union of keys from both dicts, excluding 'condition'
    numeric_keys = [k for k in physics if k != "condition" and k in random_cond]

    physics_row = {"condition": "physics", **{k: physics[k] for k in numeric_keys}}
    random_row = {"condition": "random", **{k: random_cond[k] for k in numeric_keys}}
    delta_row = {
        "condition": "delta",
        **{k: random_cond[k] - physics[k] for k in numeric_keys},
    }

    return [physics_row, random_row, delta_row]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _format_value(v: float, col: str) -> str:
    """Format a numeric value for aligned table display."""
    if col == "loc_error_px":
        return f"{v:.2f}"
    return f"{v:.4f}"


def _print_table(table: list[dict], columns: list[str]) -> None:
    """Print the comparison table to stdout in aligned columns.

    Header includes the sign convention note: delta = random − physics.
    """
    # Build header labels
    header_labels = {"condition": "condition"}
    for col in columns:
        header_labels[col] = col

    # Compute column widths (max of header and all data values)
    col_widths: dict[str, int] = {}
    all_cols = ["condition"] + columns
    for col in all_cols:
        label = header_labels.get(col, col)
        width = len(label)
        for row in table:
            cell = row.get(col, "")
            if isinstance(cell, float):
                cell = _format_value(cell, col)
            width = max(width, len(str(cell)))
        col_widths[col] = width + 2  # padding

    # Print sign convention note
    print("# delta = random − physics  (positive = random better, negative = physics better)")
    print()

    # Header row
    header = "".join(header_labels.get(col, col).ljust(col_widths[col]) for col in all_cols)
    print(header.rstrip())

    # Separator
    print("-" * len(header.rstrip()))

    # Data rows
    for row in table:
        line = ""
        for col in all_cols:
            val = row.get(col, "")
            if isinstance(val, float):
                cell = _format_value(val, col)
            else:
                cell = str(val)
            line += cell.ljust(col_widths[col])
        print(line.rstrip())


def _write_csv(table: list[dict], columns: list[str], out_path: Path) -> None:
    """Write the comparison table to a CSV file."""
    fieldnames = ["condition"] + columns
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(table)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare physics-grounded vs. random-placement benchmark results.\n"
            "Reads accuracy_metrics_*.csv from both conditions and prints a\n"
            "3-row table: physics / random / delta (random − physics).\n"
            "Also writes comparison_table.csv to the current working directory."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--physics",
        required=True,
        metavar="PATH",
        help="Path to accuracy_metrics_*.csv for the physics-grounded condition.",
    )
    parser.add_argument(
        "--random",
        required=True,
        metavar="PATH",
        help="Path to accuracy_metrics_*.csv for the random-placement condition.",
    )
    args = parser.parse_args()

    physics_path = Path(args.physics)
    random_path = Path(args.random)

    # Validate inputs — non-critical: underlying CSVs are authoritative
    missing = []
    if not physics_path.exists():
        missing.append(f"  physics: {physics_path}")
    if not random_path.exists():
        missing.append(f"  random:  {random_path}")
    if missing:
        print("FileNotFoundError: the following input CSV(s) do not exist:")
        for m in missing:
            print(m)
        print(
            "\nThe underlying per-condition accuracy_metrics_*.csv files are the "
            "authoritative outputs.\nThe comparison table can be produced manually "
            "from them without re-running either benchmark."
        )
        sys.exit(1)

    # Summarize both conditions
    physics_metrics = summarize(physics_path)
    random_metrics = summarize(random_path)

    # Optionally add tracking metrics (mota/idf1) if CSVs exist alongside inputs
    physics_tracking = _read_tracking_metrics(physics_path.parent)
    random_tracking = _read_tracking_metrics(random_path.parent)

    if physics_tracking:
        physics_metrics.update(physics_tracking)
    if random_tracking:
        random_metrics.update(random_tracking)

    # Build the comparison table
    table = build_comparison_table(physics_metrics, random_metrics)

    # Determine the set of numeric columns (in a stable order)
    base_cols = ["precision", "recall", "f1", "loc_error_px"]
    tracking_cols = ["mota", "idf1"]
    all_possible = base_cols + tracking_cols

    # Only include columns present in both physics and random rows
    columns = [col for col in all_possible if col in physics_metrics and col in random_metrics]

    # Print to stdout
    _print_table(table, columns)

    # Write comparison_table.csv to the current working directory
    out_path = Path("comparison_table.csv")
    _write_csv(table, columns, out_path)
    print(f"\nWrote: {out_path.resolve()}")


if __name__ == "__main__":
    main()
