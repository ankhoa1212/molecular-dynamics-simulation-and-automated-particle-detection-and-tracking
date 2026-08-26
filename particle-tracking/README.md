# Particle Tracking

Unified particle tracking pipeline that runs a detection model on microscopy data (or reads positions directly from LAMMPS trajectories) and links detections into tracks across frames.

## Supported Inputs

| Input | Description |
|---|---|
| Video file (`.mp4`, `.avi`, …) | Decoded frame by frame with OpenCV |
| Multi-page TIFF (`.tif`, `.tiff`) | All pages loaded as frames |
| Folder of images | PNG/JPG/BMP files, sorted numerically |
| LAMMPS trajectory (`.lammpstrj`) | Atom positions read directly — no detection model needed |

## Supported Detection Models

| Backend | Weights location | Notes |
|---|---|---|
| RF-DETR | `../rf-detr/checkpoints/` | Uses the venv in `../rf-detr/.venv` |
| YOLOv12 | `../yolov12/runs/detect/yolo12m-particles/weights/best.pt` | Loaded via `ultralytics` |
| LodeSTAR | `../data-setup/models/lodestar_model_*/model.pt` | No retraining needed; uses autolabeling model |

For `.lammpstrj` input the detection step is skipped entirely — LAMMPS atom IDs are used directly as track IDs.

## Directory Structure

```
particle-tracking/
├── track.py                 # Main entry point
├── config.yaml              # Configuration file (edit this before running)
├── pyproject.toml           # Python dependencies (managed with uv)
├── models/                  # Local model weights (optional)
├── data/
│   └── raw/                 # Raw input TIFF/video files
└── evaluation/
    └── results/             # Tracking outputs (tracks.csv, annotated video)
```

Model weights and training outputs live in their respective root-level directories:

```
rf-detr/
├── .venv/                   # rfdetr package — used by track.py at runtime
├── checkpoints/             # Trained RF-DETR checkpoints
├── rf-detr-base.pth         # Pretrained base weights
└── rf-detr-large-2026.pth   # Pretrained large weights

yolov12/
├── runs/detect/yolo12m-particles/
│   └── weights/best.pt      # Trained YOLOv12 weights
└── processed_data/          # Train/validation splits used for training

data-setup/
└── models/lodestar_model_*/ # Trained LodeSTAR autolabeling models
```

---

## Setup

**1. Install particle-tracking dependencies:**

```bash
cd particle-tracking
uv sync
```

**2. Install the RF-DETR backend** (only needed for `model.type: rf-detr`):

```bash
cd rf-detr
uv sync
```

---

## Configuration

Edit `config.yaml` before running. All paths are relative to `particle-tracking/`.

```yaml
# Path to your input (video, TIFF stack, image folder, or .lammpstrj)
input: data/raw/your_video.tif

model:
  type: rf-detr          # rf-detr | yolo | lodestar
  checkpoint: ../rf-detr/checkpoints/checkpoint_best_ema.pth
  variant: large         # rf-detr only: nano | small | medium | large
  device: "0"            # "0" for first GPU, "cpu" for CPU-only

detection:
  threshold: 0.25        # confidence threshold (ignored for .lammpstrj)

tracking:
  tracker: trackpy       # trackpy (default, offline) | bytetrack (online)
  search_range: 10.0     # trackpy: max pixel distance a particle can move per frame
  memory: 3              # trackpy: frames a particle may disappear before track ends
  stub_filter: 5         # trackpy: discard tracks shorter than this (0 = keep all)

output:
  dir: evaluation/results/tracking_output
  save_video: false      # save an annotated .mp4 (not supported for .lammpstrj)
  fps: 30
```

CLI arguments always override config values.

**Dataset scale profile:** `dataset_profile` (top-level key, unset by default) points to a scale
profile YAML (`size_px`/`spacing_px` — see `../dataset-profiles/README.md`). When set, it derives
`tiling.tile_size`, `detection.nms_distance` (LodeSTAR only), and `tracking.search_range` for any
of those left unset above; an explicit value here always wins over the derived one, and today's
hardcoded defaults still apply when no profile is referenced. `tracking.memory` never derives from
the profile — it falls back to `trackers-common`'s per-model canonical tuning instead (see
`../trackers-common/README.md`).

---

## Usage

**Run with config defaults:**

```bash
cd particle-tracking
uv run python track.py
```

**Override specific values on the command line:**

```bash
# RF-DETR
uv run python track.py \
  --model-type rf-detr \
  --checkpoint ../rf-detr/checkpoints/checkpoint_best_ema.pth \
  --input data/raw/trial1.tif \
  --save-video

# YOLOv12
uv run python track.py \
  --model-type yolo \
  --checkpoint ../yolov12/runs/detect/yolo12m-particles/weights/best.pt \
  --input data/raw/trial1.tif

# LodeSTAR (no detection model training needed)
uv run python track.py \
  --model-type lodestar \
  --checkpoint ../data-setup/models/lodestar_model_15/model.pt \
  --input data/raw/trial1.tif

# LAMMPS trajectory (detection skipped — atom IDs become track IDs)
uv run python track.py \
  --input ../lammps-scripts/results/central_pair_interaction.in.lammpstrj \
  --output-dir evaluation/results/simulation/

# Use a scenario config -- each one declares `extends: config.yaml` and is
# still complete and standalone on its own, same as before
uv run python track.py --config lodestar_config.yaml

# --override merges a second file onto --config explicitly, for combining
# any two files directly (rarely needed -- extends covers the common case)
uv run python track.py --config config.yaml --override lodestar_config.yaml
```

---

## CLI Reference

| Argument | Config key | Default | Description |
|---|---|---|---|
| `--config` | — | `config.yaml` | Path to a YAML config file. If it has a top-level `extends:` key, that file is auto-merged underneath first |
| `--override` | — | none | Path to a second YAML config merged onto `--config` (override wins at any nesting depth) |
| `--model-type` | `model.type` | `rf-detr` | `rf-detr`, `yolo`, or `lodestar` |
| `--checkpoint` | `model.checkpoint` | best EMA checkpoint | Path to model weights |
| `--variant` | `model.variant` | `large` | RF-DETR size: `nano`, `small`, `medium`, `large` |
| `--device` | `model.device` | `0` | Inference device (`0` for GPU, `cpu`) |
| `--threshold` | `detection.threshold` | `0.25` | Detection confidence threshold |
| `--input` | `input` | — | Video, image folder, TIFF stack, or `.lammpstrj` |
| `--output-dir` | `output.dir` | `evaluation/results/tracking_output` | Where to write results |
| `--tracker` | `tracking.tracker` | `trackpy` | `trackpy` (offline) or `bytetrack` (online) |
| `--search-range` | `tracking.search_range` | derived from `dataset_profile`, else per-model canonical | Trackpy: max pixel distance per frame |
| `--memory` | `tracking.memory` | per-model canonical (`trackers-common/trackers_common/tracker_defaults.yaml`) | Trackpy: frames a particle may be missing |
| `--stub-filter` | `tracking.stub_filter` | `5` | Trackpy: min track length to keep |
| `--save-video` | `output.save_video` | off | Save annotated `.mp4` |
| `--fps` | `output.fps` | `30` | FPS for output video |

---

## Output

Results are written to `output.dir`:

| File | Description |
|---|---|
| `tracks.csv` | Per-detection rows: `frame, track_id, x, y, w, h, conf` |
| `tracks.csv` (LAMMPS) | Per-atom rows: `frame, timestep, track_id, x, y` |
| `tracking_visualization.mp4` | Annotated video with bounding boxes and track IDs (if `--save-video`) |

---

## Trackers

### Trackpy (default)

Offline nearest-neighbour linking using the [Crocker–Grier algorithm](http://www.physics.emory.edu/faculty/weeks/research/trackpy.html). Processes all frames before linking, which produces more accurate tracks at the cost of loading everything into memory.

- `search_range`: maximum pixel distance a particle can move between consecutive frames. Set this to roughly the particle diameter.
- `memory`: how many frames a particle may disappear before its track is terminated.
- `stub_filter`: remove tracks shorter than this many frames to reduce noise.

### ByteTrack

Online frame-by-frame tracking via [ByteTrack](https://github.com/ifzhang/ByteTrack) (wrapped by `supervision`). Lower memory usage for long sequences; track IDs may be reassigned if a particle is lost for several frames.
