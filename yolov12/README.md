# YOLOv12 Particle Detector

Trains and evaluates a YOLOv12 object detection model on particle microscopy images. Uses the [Ultralytics](https://docs.ultralytics.com) framework.

## Setup

```bash
cd yolov12
uv sync
```

## Directory Structure

```
yolov12/
├── main.py              # CLI entrypoint (process / train / evaluate)
├── train.py             # Training logic
├── evaluate.py          # Evaluation logic
├── process.py           # Data preprocessing (train/val split)
├── config.yaml          # Training configuration
├── data/                # Your labeled dataset (create this before running)
│   ├── data.yaml        # Dataset config (paths + class names)
│   └── images/          # Raw labeled images (.jpg/.png) + YOLO .txt labels
├── processed_data/      # Auto-generated train/val splits (created by process stage)
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   └── validation/
│       ├── images/
│       └── labels/
└── runs/detect/         # Training outputs (weights, metrics, plots)
```

## Data Format

Images must have corresponding YOLO-format `.txt` label files in the same directory:

```
data/images/frame_001.jpg
data/images/frame_001.txt   # one line per particle: <class> <cx> <cy> <w> <h> (normalized)
```

`data/data.yaml` must define the dataset:

```yaml
path: ../processed_data
train: train/images
val: validation/images
nc: 1
names: ["particle"]
```

## Data Setup

Before running, create the `data/` directory and populate it:

```
data/
├── data.yaml          # Dataset config (see format above)
└── images/            # Your labeled images (.jpg/.png) + matching YOLO .txt files
```

The `process` stage reads from `data/images/` and writes 80/20 splits to `processed_data/`. If `processed_data/` already contains split data, the process stage skips automatically.

## Usage

All stages are run through `main.py`:

```bash
uv run python main.py <stage>
```

### 1. Process

Splits raw data into train/validation sets (80/20). Skips automatically if splits already exist.

```bash
uv run python main.py process
```

Output written to `processed_data/`.

### 2. Train

Trains a YOLOv12 detection model. Checkpoints and metrics saved under `runs/detect/`.

```bash
uv run python main.py train
```

Best weights are saved to `runs/detect/<name>/weights/best.pt`.

### 3. Evaluate

Evaluates a trained checkpoint against a validation set.

```bash
uv run python main.py evaluate
```

## Using Trained Weights in Particle Tracking

Point `particle-tracking/config.yaml` at the best checkpoint:

```yaml
model:
  type: yolo
  checkpoint: ../yolov12/runs/detect/yolo12m-particles/weights/best.pt
```

Then run tracking:

```bash
cd particle-tracking
uv run python track.py --input /path/to/video.tif
```

## MLflow

Training metrics are logged to the shared MLflow database at `../data-setup/mlflow.db`, which is not tracked in git (see `data-setup/README.md`'s MLflow section for details).

View runs:

```bash
cd data-setup
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```
