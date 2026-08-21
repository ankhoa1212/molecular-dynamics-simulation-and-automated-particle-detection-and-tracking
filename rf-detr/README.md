# RF-DETR Particle Detection Pipeline

Training and evaluation pipeline for detecting particles in microscopy images using [RF-DETR](https://github.com/roboflow/rf-detr). Experiment results are tracked with MLflow.

## Requirements

- Python 3.11
- [uv](https://docs.astral.sh/uv/) for dependency management
- A CUDA-capable GPU (recommended; CPU-only will work but is very slow for training)

## Setup

```bash
cd rf-detr
uv sync
```

This installs all dependencies including PyTorch with CUDA 13.0 support into a local `.venv`.

---

## Dataset Format

The pipeline expects a dataset directory with this structure:

```
my-dataset/
├── images/               # all image files (PNG or JPG)
└── annotations.json      # single COCO JSON covering all images and annotations
```

That's the only input `train.py` needs -- `split/{train,valid,test}/` gets created automatically on first run (see "Training" below) from `images/` + `annotations.json`, so it's not part of the dataset you need to provide or download. The HF dataset release (see the top-level `README.md`'s "Data & Model Availability" section) accordingly ships only `images/` + `annotations.json`, not a pre-built `split/`.

The COCO JSON must have the standard structure:

```json
{
  "categories": [{"id": 1, "name": "particle"}],
  "images": [
    {"id": 1, "file_name": "trial1_frame_000.png", "width": 640, "height": 480},
    ...
  ],
  "annotations": [
    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [x, y, w, h], "area": ..., "iscrowd": 0},
    ...
  ]
}
```

### Per-Experiment Split

Images are assigned to train/val/test splits by matching substrings of their filenames against experiment lists in `config.yaml`. For example, if `train_experiments: ["trial1", "trial2"]`, then any image whose filename contains `"trial1"` or `"trial2"` goes into the training split. The first matching experiment list wins.

---

## Configuration

Edit `config.yaml` before running:

```yaml
dataset:
  path: /path/to/my-dataset       # path to your dataset directory
  train_experiments:
    - trial1
    - trial2
  val_experiments:
    - trial3
  test_experiments:
    - trial4

model:
  variant: base                   # base (~30M params) or large (~128M params)

training:
  epochs: 50
  batch_size: 4
  grad_accum_steps: 4             # effective batch = batch_size * grad_accum_steps
  learning_rate: 1.0e-4
  num_workers: 4                   # number of data loading workers
  pin_memory: true                # speeds up data transfer to GPU
  checkpoint_dir: checkpoints     # where model weights are saved

mlflow:
  experiment_name: rf-detr
```

---

## Training

```bash
uv run python train.py --config config.yaml
```

This will:
1. Split the dataset by experiment into `<dataset_path>/split/{train,valid,test}/`
2. Start an MLflow run under the configured experiment name
3. Log all config parameters to MLflow
4. Download pretrained RF-DETR weights on first run (requires internet access)
5. Train the model, logging loss and mAP metrics per epoch to MLflow
6. Save checkpoints to `checkpoints/` and log the best one as an MLflow artifact

Training progress and metrics are printed to stdout.

---

## Kubernetes Training (optional)

`k8s-launch.sh`/`k8s-retrieve.sh` run the same training on a Kubernetes cluster (`config.k8s.yaml`) instead of locally. Two things must be set before running either script -- both fail with a clear error if missing:

- **`K8S_NAMESPACE`** (env var, required): your cluster namespace.
  ```bash
  export K8S_NAMESPACE=your-namespace
  ```
- **Container registry** (required): `k8s-job.yaml`'s `image:` field is a placeholder (`your-registry/rf-detr-trainer:latest`) -- build and push your own training image, then edit that line to point at it.

Also requires the dataset symlinked at `../data/2um-coco-merged` (see the top-level `README.md`'s "Data & Model Availability" section) -- `k8s-launch.sh` checks for this and errors with instructions if it's missing.

```bash
export K8S_NAMESPACE=your-namespace
bash k8s-launch.sh      # syncs the dataset to the cluster and starts training
bash k8s-retrieve.sh    # pulls results back (works mid-training or after completion)
```

---

## Viewing Results

```bash
uv run mlflow ui --backend-store-uri sqlite:///../data-setup/mlflow.db
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser to view all runs, compare metrics across experiments, and download artifacts.

---

## Evaluation

Evaluate on the test split after training. Two options:

**Option A — use the most recent local checkpoint:**

```bash
uv run python evaluate.py --config config.yaml --batch-size 16
```

**Option B — load a checkpoint from a specific MLflow run and log metrics back to it:**

```bash
uv run python evaluate.py --config config.yaml --run-id <run-id> --batch-size 16
```

The run ID is visible in the MLflow UI or in the training output. This option downloads the checkpoint artifact from MLflow and logs the evaluation metrics (`test/mAP50`, `test/mAP50_95`, `test/precision`, `test/recall`) back to the same run for side-by-side comparison.

Evaluation results are printed to stdout:

```
=== Evaluation Results ===
  test/mAP50:     0.8712
  test/mAP50_95:  0.6340
  test/precision: 0.9100
  test/recall:    0.8450
```

---

## Running Tests

```bash
uv run pytest tests/ -v
```

---

## File Overview

| File | Purpose |
|------|---------|
| `train.py` | Training entry point |
| `evaluate.py` | Evaluation entry point |
| `dataset.py` | Loads COCO JSON and produces per-experiment train/valid/test splits |
| `mlflow_utils.py` | Thin wrappers around MLflow (start run, log metrics, log artifact) |
| `config.yaml` | All tunable parameters |
| `mlruns/` | MLflow experiment store (auto-created, gitignored) |
| `checkpoints/` | Saved model weights (auto-created, gitignored) |
