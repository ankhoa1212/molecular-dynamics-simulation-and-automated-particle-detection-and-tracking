#!/bin/bash
# Density/overlap ablation driver: simulates 3 new particle counts (200, 600,
# 1000) at the same epsilon=5.0/box_size=200 as the existing default
# verification trajectory (continuous_force_1500_5.0.lammpstrj, N~1446
# packed particles after boundary effects), renders + benchmarks each
# against all 4 detector/tracker arms, and files results next to a copy of
# the existing N~1446 baseline so plot_density_ablation.py can plot all 4
# density points together.
#
# Safety: benchmark.py hardcodes its output to ./verification_output/
# relative to CWD (not configurable), so every run below lands in the same
# place as the paper's real N~1446 headline numbers. This script backs up
# the real accuracy_metrics_*.csv/tracking_metrics_*.csv into
# verification_output/density_ablation/N1446/ before touching anything, and
# restores them to the top level when done (even on failure, via the trap).
# render.py's own output (synthetic_frames/, ground_truth*.json/csv) is kept
# out of the way entirely by pointing each N's render at its own scratch
# config.yaml copy with a per-N synthetic.output_dir -- it never touches the
# top-level verification_output/ directly.
#
# Usage: cd verification && ./run_density_ablation.sh

set -euo pipefail

cd "$(dirname "$0")"  # always run from verification/

VENV_PY="./.venv/bin/python"
OUT=verification_output
ABLATION_DIR="$OUT/density_ablation"
MODELS=(rf-detr yolo12m yolo12n lodestar trackpy)
DENSITIES=(200 600 1000)
FRAMES=151
LAMMPS_OUT_DIR="../lammps-scripts/results/density_ablation"
# This script never passes --tracker to benchmark.py, so every invocation
# below uses its trackpy default; benchmark.py names its tracking-metrics
# output tracking_metrics_{model}_{tracker}.csv (since the ByteTrack-support
# commit), so all tracking_metrics filenames in this script carry this
# suffix -- a bare tracking_metrics_{model}.csv is a pre-ByteTrack relic that
# benchmark.py no longer writes.
TRACKER=trackpy

mkdir -p "$ABLATION_DIR"

# --- 0. Back up the real N~1446 baseline CSVs, restore on any exit. ---
mkdir -p "$ABLATION_DIR/N1446"
for m in "${MODELS[@]}"; do
    [ -f "$OUT/accuracy_metrics_$m.csv" ] && cp "$OUT/accuracy_metrics_$m.csv" "$ABLATION_DIR/N1446/"
    [ -f "$OUT/tracking_metrics_${m}_${TRACKER}.csv" ] && cp "$OUT/tracking_metrics_${m}_${TRACKER}.csv" "$ABLATION_DIR/N1446/"
done

restore_baseline() {
    echo "Restoring the real N~1446 baseline CSVs to $OUT/ ..."
    for m in "${MODELS[@]}"; do
        [ -f "$ABLATION_DIR/N1446/accuracy_metrics_$m.csv" ] && cp "$ABLATION_DIR/N1446/accuracy_metrics_$m.csv" "$OUT/"
        [ -f "$ABLATION_DIR/N1446/tracking_metrics_${m}_${TRACKER}.csv" ] && cp "$ABLATION_DIR/N1446/tracking_metrics_${m}_${TRACKER}.csv" "$OUT/"
    done
}
trap restore_baseline EXIT

# --- 0b. Backfill N=1446 baseline CSVs for yolo12m and yolo12n. ---
# These models were not in the original MODELS array, so their CSVs may not
# exist in $OUT/ yet. Run benchmark.py against the existing production frames,
# ground_truth.json, and ground_truth_tracks.csv to produce them so the
# Phase 0 backup (already executed above) can be re-run retroactively -- but
# we must also copy them into N1446/ now, since Phase 0 already ran before
# these existed.
echo "=== Phase 0b: Backfilling N=1446 CSVs for yolo12m and yolo12n ==="
for m in yolo12m yolo12n; do
    if [ ! -f "$OUT/accuracy_metrics_$m.csv" ]; then
        echo "=== [N=1446] Benchmarking $m (backfill) ==="
        "$VENV_PY" benchmark.py \
            --frames "$OUT/synthetic_frames" \
            --ground-truth "$OUT/ground_truth.json" \
            --ground-truth-tracks "$OUT/ground_truth_tracks.csv" \
            --model-type "$m"
    else
        echo "=== [N=1446] $m CSV already exists, skipping backfill ==="
    fi
    # File into N1446/ so the restore trap can recover them on subsequent runs.
    [ -f "$OUT/accuracy_metrics_$m.csv" ] && cp "$OUT/accuracy_metrics_$m.csv" "$ABLATION_DIR/N1446/"
    [ -f "$OUT/tracking_metrics_${m}_${TRACKER}.csv" ] && cp "$OUT/tracking_metrics_${m}_${TRACKER}.csv" "$ABLATION_DIR/N1446/"
done

# --- 1. Run the LAMMPS molecule-count sweep (N=200,600,1000 @ epsilon=5.0). ---
echo "=== Running LAMMPS density sweep (N=200,600,1000) ==="
( cd ../lammps-scripts && python3 run.py --config config/density_ablation.json )

# --- 2. Render + benchmark each new density. ---
for N in "${DENSITIES[@]}"; do
    TRAJ="$LAMMPS_OUT_DIR/continuous_force_${N}_5.0.lammpstrj"
    if [ ! -f "$TRAJ" ]; then
        echo "ERROR: expected trajectory not found: $TRAJ" >&2
        exit 1
    fi

    N_DIR="$ABLATION_DIR/N$N"
    mkdir -p "$N_DIR"

    # Scratch config: identical to config.yaml except synthetic.output_dir,
    # so render.py's frames/ground_truth land under N_DIR and never touch
    # the top-level verification_output/ (ground_truth.json/tracks.csv are
    # written to output_dir.parent by render.py -- see its docstring).
    SCRATCH_CFG="$N_DIR/_render_config.yaml"
    python3 - "$SCRATCH_CFG" "$N_DIR/synthetic_frames" <<'PYEOF'
import sys, re
scratch_cfg, frames_dir = sys.argv[1], sys.argv[2]
text = open("config.yaml").read()
text, n = re.subn(
    r'(?m)^(\s*output_dir:\s*).*$',
    lambda m: f"{m.group(1)}{frames_dir}",
    text, count=1,
)
assert n == 1, "config.yaml's synthetic.output_dir line not found/replaced"
open(scratch_cfg, "w").write(text)
PYEOF

    echo "=== [N=$N] Rendering synthetic frames ==="
    "$VENV_PY" render.py --config "$SCRATCH_CFG" --lammps "$TRAJ" --frames "$FRAMES"

    for m in "${MODELS[@]}"; do
        echo "=== [N=$N] Benchmarking $m ==="
        "$VENV_PY" benchmark.py \
            --frames "$N_DIR/synthetic_frames" \
            --ground-truth "$N_DIR/ground_truth.json" \
            --ground-truth-tracks "$N_DIR/ground_truth_tracks.csv" \
            --model-type "$m"
    done

    # benchmark.py just wrote accuracy/tracking CSVs to the top-level $OUT/
    # (hardcoded) -- file them under this N before the next iteration
    # overwrites them.
    for m in "${MODELS[@]}"; do
        [ -f "$OUT/accuracy_metrics_$m.csv" ] && cp "$OUT/accuracy_metrics_$m.csv" "$N_DIR/"
        [ -f "$OUT/tracking_metrics_${m}_${TRACKER}.csv" ] && cp "$OUT/tracking_metrics_${m}_${TRACKER}.csv" "$N_DIR/"
    done

    # Drop the rendered frames after benchmarking -- large (tens of MB) and
    # not needed once the CSVs are filed.
    rm -rf "$N_DIR/synthetic_frames" "$N_DIR/ground_truth.json" "$N_DIR/ground_truth_tracks.csv"

    echo "=== [N=$N] done -> $N_DIR ==="
done

echo "=== Density ablation sweep complete. Run: $VENV_PY plot_density_ablation.py ==="
