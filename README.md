# Molecular Dynamics Simulation & Particle Tracking

End-to-end pipeline for running molecular dynamics simulations, auto-labeling microscopy data with [LodeSTAR](https://github.com/softmatterlab/DeepTrack2), training a particle detection model, and tracking particles across frames.

## Table of Contents

- [Repository Structure](#repository-structure)
- [Data & Model Availability](#data--model-availability)
- [Components](#components)
  - [1. Simulation](#1-simulation-lammps-scripts)
  - [2. Auto-Labeling with LodeSTAR](#2-auto-labeling-with-lodestar-data-setup)
  - [3. Particle Detection — RF-DETR](#3-particle-detection--rf-detr-rf-detr)
  - [4. Particle Tracking](#4-particle-tracking-particle-tracking)
  - [5. Verification](#5-verification-verification)
- [Full Pipeline Overview](#full-pipeline-overview)
- [Contributing](#contributing)
- [Resources](#resources)

## Repository Structure

```
molecular-dynamics-simulation/
├── lammps-scripts/          # LAMMPS simulation scripts and analysis tools
├── data-setup/              # LodeSTAR auto-labeling pipeline (generates YOLO labels)
├── rf-detr/                 # RF-DETR training, evaluation, weights, and venv
│   ├── checkpoints/         # Trained RF-DETR checkpoints
│   ├── rf-detr-base.pth     # Pretrained base weights
│   └── rf-detr-large-2026.pth
├── yolov12/                 # YOLOv12 training, evaluation, and weights
│   ├── runs/detect/yolo12m-particles/weights/best.pt
│   └── processed_data/      # Train/validation image splits
├── detectors-common/        # Shared detector-loading/tiling/config-merge package (rf-detr/, particle-tracking/)
├── trackers-common/         # Shared trackpy-linking/ByteTrack-tracking/tracking-tuning package (particle-tracking/, verification/, rf-detr/)
├── dataset-profiles/        # Per-dataset scale profiles (size_px/spacing_px) detection/tracking params derive from
├── particle-tracking/       # Particle tracking pipeline
│   ├── track.py             # Unified tracker (RF-DETR, YOLOv12, or LodeSTAR)
│   ├── config.yaml          # Tracking configuration
│   ├── models/              # Local model weights (optional)
│   ├── data/raw/            # Raw input TIFF files
│   └── evaluation/results/  # Tracking outputs (tracks.csv, annotated video)
└── verification/            # End-to-end verification pipeline
    ├── render.py            # LAMMPS trajectory → synthetic microscopy TIFFs
    ├── benchmark.py         # RF-DETR detection accuracy + MOTA/IDF1 tracking metrics
    ├── compare.py           # Physics observable comparison (MSD, hexatic order)
    ├── calibrate_psf.py     # Fit PSF/noise parameters from real microscopy frames
    ├── compare_renders.py   # Side-by-side SNR/PSD comparison of rendering strategies
    ├── dataset_profile_builder.py  # Builds a dataset-profiles/ YAML from a LAMMPS trajectory
    └── config.yaml          # Rendering, benchmarking, and tracking metric settings
```

---

## Data & Model Availability

None of the training data or trained checkpoints are committed to this repository (large, and machine-specific paths don't belong in tracked files). Every config across the repo resolves dataset/checkpoint paths relative to a top-level `data/` directory, which is gitignored and not created automatically — point it at wherever your data actually lives with a single symlink:

```bash
ln -s "/path/to/your/dataset/root" data
```

For example, on the original development machine this was a symlink to a separate data drive:

```bash
ln -s "/mnt/d/Particle Tracking Data" data
```

After that one symlink, every package's `../data/<name>` reference (in `rf-detr/config.yaml`, `yolov12/config.yaml`, `particle-tracking/*.yaml`, `data-setup/yolo_split.py`, `rf-detr/k8s-launch.sh`) resolves correctly regardless of where your actual data lives on disk.

`data-setup/mlflow.db` is likewise not tracked in git -- see `data-setup/README.md`'s MLflow section for details.

<!--
**Datasets and checkpoints used by this repo's reported results are also published on Hugging Face for anyone without access to the original raw microscopy data:**

Download and place these under `data/<name>` (datasets) or point the relevant `checkpoint:`/`weights:` config key at wherever you save the model files.
-->

---

## Components

### 1. Simulation (`lammps-scripts/`)

Runs LAMMPS molecular dynamics simulations and analyzes results.

**Setup:** [Install and build LAMMPS](https://docs.lammps.org/Install.html), then add the executable to `PATH`:

```bash
export PATH=/path/to/lammps/bin:$PATH
```

**Run a simulation:**

```bash
cd lammps-scripts
python3 run.py --config config/continuous_force_test.json
```

**Analyze results:**

```bash
python3 velocity_graph.py --filename results/simulation.lammpstrj   # velocity distribution plots
python3 temp_graph.py --filename results/simulation.lammpstrj        # temperature vs. time
python3 phase_diagram.py results/     # phase diagram
python3 hexatic_order_analysis.py     # hexatic order parameter
```

**Install dependencies:**

```bash
pip install -r lammps-scripts/requirements.txt
```

---

### 2. Auto-Labeling with LodeSTAR (`data-setup/`)

Cascade labeling pipeline: hand-label a few particle crops → train a LodeSTAR model → auto-label thousands of frames → human-verify in the GUI → export for RF-DETR/YOLO training.

**Install dependencies:**

```bash
cd data-setup
pip install -r requirements.txt
```

**Workflow: Crop → Train → Auto-label → Verify → Export**

**Step 1 — Extract frames from TIFF:**

```bash
python extract_frames.py video.tif frames/ --nth 5
```

**Step 2 — Draw a few training crops (GUI, Crop Mode):**

```bash
python crop_tool.py frames/
```
Draw 3–10 bounding boxes around representative particles. That's all LodeSTAR needs.

**Step 3 — Train LodeSTAR model:**

```bash
python train_lodestar.py \
  --input-dir frames/ \
  --model-path models/lodestar_model_15/
```

Training is logged to MLflow automatically. View runs:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://localhost:5000
```

**Step 4 — Batch-label images with the trained model:**

```bash
python lodestar_autolabeler.py \
  --model models/lodestar_model_15/ \
  --input data/raw_tiffs/ \
  --use-radius \
  --alpha 0.9 --cutoff 0.001 \
  --nms-distance 35 \
  --plot
```

Outputs YOLO `.txt` label files alongside images in an `images/` + `labels/` directory structure.

**Step 5 — Review & export (GUI, Label Mode):**

```bash
python crop_tool.py path/to/<name>_dataset/images/
```

Open in **Label Mode**. Click **[Accept All]** on frames where the model was accurate, fix any mistakes with **Edit Mode**, then click **[Export COCO]** or **[Export YOLO]** to build the final training set.

Pre-tuned configs are available in `data-setup/configs/`. Pass `--config configs/autolabel_2um_lodestar_model_15.json` to use one.

See [`data-setup/README.md`](data-setup/README.md) for full argument reference and configuration details.

---

### 3. Particle Detection — RF-DETR (`rf-detr/`)

Trains and evaluates an [RF-DETR](https://github.com/roboflow/rf-detr) transformer-based object detector on the labeled data produced by the auto-labeling step. Experiment tracking via MLflow.

**Requirements:** Python 3.11, [uv](https://docs.astral.sh/uv/), CUDA GPU recommended.

**Setup:**

```bash
cd rf-detr
uv sync
```

This installs all dependencies into `rf-detr/.venv`, including the `rfdetr` package that `particle-tracking/track.py` loads at runtime for inference.

Pretrained weights (`rf-detr-base.pth`, `rf-detr-large-2026.pth`) and trained checkpoints (`checkpoints/`) are stored here alongside the training code.

**Dataset format:** A directory with `images/` and a single `annotations.json` in COCO format. Edit `config.yaml` to set the dataset path and assign experiment names to train/val/test splits.

**Train:**

```bash
uv run python train.py --config config.yaml
```

**Evaluate:**

```bash
# Most recent checkpoint
uv run python evaluate.py --config config.yaml --batch-size 16

# Specific MLflow run
uv run python evaluate.py --config config.yaml --run-id <run-id> --batch-size 16
```

**View MLflow results:**

```bash
uv run mlflow ui --backend-store-uri sqlite:///../data-setup/mlflow.db
# Open http://127.0.0.1:5000
```

**Run tests:**

```bash
uv run pytest tests/ -v
```

See [`rf-detr/README.md`](rf-detr/README.md) for full configuration options.

---

### 4. Particle Tracking (`particle-tracking/`)

Runs a detection model (RF-DETR, YOLOv12, or LodeSTAR) on microscopy data and links detections into tracks. Also accepts `.lammpstrj` LAMMPS trajectories directly, bypassing detection and using atom IDs as track IDs.

**Output:** `tracks.csv` with per-frame `(track_id, x, y, w, h, conf)` and optionally an annotated `.mp4`.

**Quick start:**

```bash
cd particle-tracking
uv sync                          # install dependencies
# edit config.yaml to set your input and model, then:
uv run python track.py
```

The RF-DETR backend requires its own one-time install (run from the repo root):

```bash
cd rf-detr && uv sync
```

See [`particle-tracking/README.md`](particle-tracking/README.md) for the full setup guide, configuration reference, and CLI options.

---

### 5. Verification (`verification/`)

End-to-end pipeline for validating the simulation → detection → tracking chain with realistic synthetic rendering. Converts LAMMPS trajectories into synthetic microscopy frames and measures detection/tracking accuracy against known ground truth.

**Setup:**

```bash
cd verification
uv sync
```

**Rendering strategies** (set `render_strategy` in `config.yaml`):

| Strategy | Description |
|----------|-------------|
| `procedural` | Flat 2D Gaussian PSF + Poisson/Gaussian noise (fast; also backs the density stress-test configs) |
| `brightfield` | Coherent whole-frame optical-field solve via DeepTrack2's `Brightfield` optics; small-batch/reference-quality, not a bulk generator |
| `brightfield_fast` | FFT-based reimplementation of the same coherent optics, independent of particle count; default strategy, used for bulk/production-density rendering (see `configs/render_brightfield_fast.yaml`) |

**Calibrate from real frames (run once):**

```bash
uv run python calibrate_psf.py --real-frames /path/to/real/tifs/
# Paste the printed values into config.yaml under synthetic:
```

**Render synthetic frames:**

```bash
uv run python render.py --lammps ../lammps-scripts/results/sim.lammpstrj
```

**Benchmark detection + tracking accuracy:**

```bash
uv run python benchmark.py \
    --frames verification_output/synthetic_frames/ \
    --ground-truth verification_output/ground_truth.json \
    --ground-truth-tracks verification_output/ground_truth_tracks.csv
# Outputs: accuracy_metrics.csv (precision/recall/F1) and tracking_metrics.csv (MOTA/IDF1)
```

**Compare physics observables against real tracks:**

```bash
uv run python compare.py \
    --lammps ../lammps-scripts/results/sim.lammpstrj \
    --tracks ../particle-tracking/output/tracks.csv
```

**Run tests:**

```bash
uv run pytest tests/ -v
```

See [`verification/README.md`](verification/README.md) for the full calibration workflow, configuration reference, and acceptance criteria.

---

## Reported Results

The synthetic ground-truth benchmark (4-way detector/tracker comparison against known particle
positions, on the repo's default 151-frame trajectory) reproduces as:

| Model    | Precision | Recall | F1    | Loc. Error (px) |
|----------|-----------|--------|-------|------------------|
| Trackpy  | 100.0%    | 16.8%  | 28.8% | 3.24             |
| LodeSTAR | 43.4%     | 15.7%  | 23.1% | 2.43             |
| YOLOv12  | 98.7%     | 82.1%  | 89.7% | 0.99             |
| RF-DETR  | 93.9%     | 77.9%  | 85.2% | 0.99             |

Precision/Recall/F1 are pooled over true/false positive/negative counts across all frames (10px
greedy nearest-neighbor matching, each ground-truth particle matched at most once); localization
error is the mean center-to-center distance over matched pairs only. See
[`verification/README.md`](verification/README.md) for the full metric/matching definitions.

**Exact commands to reproduce this table**, from `verification/` (after `uv sync`, with the
pretrained checkpoints in place per [Data & Model Availability](#data--model-availability)). This
table was generated under `render_strategy: procedural`, not the current `config.yaml` default of
`brightfield_fast`, so the render step below passes `configs/render_procedural.yaml` explicitly:

```bash
uv run python render.py \
    --lammps ../lammps-scripts/single_continuous_force_test/continuous_force_1500_5.0.lammpstrj \
    --frames 151 \
    --config configs/render_procedural.yaml

for model in rf-detr yolo12m lodestar trackpy; do
  uv run python benchmark.py \
      --frames verification_output/synthetic_frames/ \
      --ground-truth verification_output/ground_truth.json \
      --ground-truth-tracks verification_output/ground_truth_tracks.csv \
      --model-type "$model"
done

uv run python plot_benchmark.py   # aggregates accuracy_metrics_*.csv into the table above + benchmark_comparison.png
```

---

## Full Pipeline Overview

```
Raw microscopy TIFFs
       │
       ▼
[data-setup — Crop Mode]  Draw 3-10 particle crops in crop_tool.py
       │
       ▼
[data-setup]              Train LodeSTAR model on those crops (train_lodestar.py)
       │
       ▼
[data-setup]              Auto-label thousands of frames (lodestar_autolabeler.py)
       │ images/ + YOLO labels/
       ▼
[data-setup — Label Mode] Review & verify in crop_tool.py (Accept All / Edit Mode)
       │ Export COCO JSON or YOLO dataset
       ▼
[rf-detr]                 Train RF-DETR detection model on verified labels
       │ Trained checkpoint (.pth)
       ▼
[particle-tracking]       Detect + track particles across frames
       │
       ▼
tracks.csv  +  annotated video
       │
       ▼
[verification]            Benchmark detector on synthetic ground-truth frames
                          Measure MOTA/IDF1/fragmentation; compare physics observables
```

`run.sh` dispatches the labeling, tracking, and verification stages into their own subproject venvs, without needing to `cd` into each one manually:

```bash
./run.sh label ...       # -> data-setup/lodestar_autolabeler.py
./run.sh track ...       # -> particle-tracking/track.py
./run.sh render ...      # -> verification/render.py
./run.sh benchmark ...   # -> verification/benchmark.py
./run.sh compare ...     # -> verification/compare.py
```

RF-DETR training and LAMMPS simulation are separate workflows and aren't covered — run them directly from `rf-detr/` and `lammps-scripts/` as shown above.

---

## Contributing

### Workflow

1. Create a branch off `main` for your changes:
   ```bash
   git checkout main
   git pull
   git checkout -b feat/your-feature-name
   ```
2. Make your changes and commit them (the pre-commit hook will run automatically).
3. Push the branch and open a pull request against `main`:
   ```bash
   git push -u origin feat/your-feature-name
   ```
4. Address any review feedback, then merge once approved.

### Pre-commit hook

This repo uses [pre-commit](https://pre-commit.com/) to auto-format Python files with [Black](https://black.readthedocs.io/) (line length 100) before every commit.

Install the hook once after cloning:

```bash
pip install pre-commit
pre-commit install
```

After that, Black runs automatically on staged files. If it reformats anything, stage the changes and commit again.

### Linting

`lint.sh` mirrors the CI lint check locally. Run it before opening a PR:

```bash
# Lint only files changed relative to origin/main (default)
./lint.sh

# Lint the entire repository
./lint.sh --full
```

Reports are written to `lint-reports/` (pylint text, JSON, and a summary). Fix any Black formatting issues with the command printed by the script, then re-run to confirm.

### Testing

CI runs the `rf-detr/`, `particle-tracking/`, `verification/`, `detectors-common/`, `data-setup/`, and `trackers-common/` test suites on every push and PR, and blocks on failure — same as Black. Run them yourself before opening a PR:

```bash
cd <subproject> && uv run pytest tests/ -v
```

`yolov12/` has no test suite yet, and `lammps-scripts/` doesn't either despite having a `test/` directory (it currently holds only fixture data).

### PR size

Raw diff line counts can be misleading here — a repo-hygiene audit found one 18,700-line PR was only 31% behavior-changing source once `uv.lock` and other generated files, docs, and test code were split out separately. When judging whether a PR is reasonably scoped:

- Exclude `uv.lock` and other generated files (see `.gitattributes`) — they're not meant to be reviewed line-by-line.
- Look at the source-vs-test split, not just the total. A large diff that's mostly tests for a small behavior change is a different shape than a large diff of new source.
- Prefer landing a feature branch's commits as one PR reasonably promptly over letting many commits accumulate across a long-lived branch before opening one.

This is guidance, not an enforced threshold — a large PR with a good reason (a real feature, thorough test coverage) is fine; a large PR that's an accident of scope creep is worth splitting.

---

## Resources

- [LAMMPS Manual](https://docs.lammps.org/Manual.html)
- [LodeSTAR / DeepTrack2](https://github.com/softmatterlab/DeepTrack2)
- [RF-DETR (Roboflow)](https://github.com/roboflow/rf-detr)
- [OVITO (simulation visualization)](https://www.ovito.org/)
- [Light-Responsive Assembly](https://pubs.acs.org/doi/10.1021/acs.jpcb.4c02301)
- [Molecular Dynamics Simulation of Active Particles Video](https://www.youtube.com/watch?v=wsM2kUB6XU4&ab_channel=SoftMatterLab)
- [Molecular Dynamics Simulation of Active Particles (Brownian Motion)](https://arxiv.org/abs/2102.10399)
