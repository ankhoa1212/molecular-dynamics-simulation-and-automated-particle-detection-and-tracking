# YOLOv12 Training MLflow Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix `yolov12/train.py` to expose a `run(config_path)` function, read hyperparameters from `config.yaml`, and log per-epoch metrics to the shared `data-setup/mlflow.db` MLflow database using `yolov12m.pt` at `imgsz=1280`.

**Architecture:** Three files: a new `config.yaml` holds all hyperparameters, a new `mlflow_utils.py` (mirroring `rf-detr/mlflow_utils.py`) owns the MLflow connection, and `train.py` is rewritten to wire them together via an Ultralytics `on_fit_epoch_end` callback. No automated tests — training requires GPU + dataset; verification is manual.

**Tech Stack:** Python 3.11+, Ultralytics (local `yolov12.ultralytics`), MLflow, PyYAML, uv.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `yolov12/pyproject.toml` | Add `mlflow` and `pyyaml` dependencies |
| Create | `yolov12/config.yaml` | All training hyperparameters and MLflow experiment name |
| Create | `yolov12/mlflow_utils.py` | MLflow setup — resolves path to shared `data-setup/mlflow.db` |
| Modify | `yolov12/train.py` | `flatten_config`, `run(config_path)`, updated `main()` |

---

## Task 1: Add dependencies and create `config.yaml`

**Files:**
- Modify: `yolov12/pyproject.toml`
- Create: `yolov12/config.yaml`

- [ ] **Step 1: Add mlflow and pyyaml to pyproject.toml**

Replace the `dependencies` list in `yolov12/pyproject.toml`:

```toml
[project]
name = "yolov12"
version = "0.2.0"
description = "machine learning for particle tracking"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "mlflow>=2.0.0",
    "pyyaml>=6.0",
]
```

- [ ] **Step 2: Sync the venv**

```bash
cd yolov12 && uv sync
```

Expected: `mlflow` and `pyyaml` resolved and installed.

- [ ] **Step 3: Create `yolov12/config.yaml`**

```yaml
model:
  weights: yolov12m.pt   # medium variant — better spatial feature depth for dense single-class detection
  imgsz: 1280            # preserve detail from 3200x2200 input images

training:
  epochs: 100
  batch: 8               # reduced from 16 to accommodate larger imgsz
  device: "0"
  name: yolov12m-particles

mlflow:
  experiment_name: yolov12
```

- [ ] **Step 4: Stage files (do NOT git commit — user commits manually)**

```bash
git add yolov12/pyproject.toml yolov12/uv.lock yolov12/config.yaml
```

---

## Task 2: Create `yolov12/mlflow_utils.py`

**Files:**
- Create: `yolov12/mlflow_utils.py`

This mirrors `rf-detr/mlflow_utils.py` exactly, with the path adjusted for the `yolov12/` directory.

- [ ] **Step 1: Create `yolov12/mlflow_utils.py`**

```python
import os
from typing import Any
import mlflow


def start_run(experiment_name: str, run_name: str, params: dict[str, Any]) -> mlflow.ActiveRun:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, "..", "data-setup", "mlflow.db")
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")

    mlflow.set_experiment(experiment_name)
    run = mlflow.start_run(run_name=run_name)
    if params:
        mlflow.log_params(params)
    return run


def end_run() -> None:
    mlflow.end_run()
```

- [ ] **Step 2: Verify the db path resolves correctly**

```bash
cd yolov12 && uv run python -c "
import os, mlflow_utils
base_dir = os.path.dirname(os.path.abspath(mlflow_utils.__file__))
db_path = os.path.normpath(os.path.join(base_dir, '..', 'data-setup', 'mlflow.db'))
print('db_path:', db_path)
print('exists:', os.path.exists(db_path))
"
```

Expected output:
```
db_path: /home/ankhoa1212/git/molecular-dynamics-simulation/data-setup/mlflow.db
exists: True
```

- [ ] **Step 3: Stage file (do NOT git commit)**

```bash
git add yolov12/mlflow_utils.py
```

---

## Task 3: Rewrite `yolov12/train.py`

**Files:**
- Modify: `yolov12/train.py`

The current `train.py` hard-codes model weights, ignores config, has no MLflow integration, and is missing the `run()` function that `main.py` calls. Replace it entirely.

- [ ] **Step 1: Replace `yolov12/train.py` with the following**

```python
import os

import mlflow
import yaml

from mlflow_utils import end_run, start_run
from yolov12.ultralytics import YOLO


def flatten_config(config: dict, prefix: str = "") -> dict[str, str]:
    flat: dict[str, str] = {}
    for key, value in config.items():
        full_key = f"{prefix}{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_config(value, prefix=f"{full_key}."))
        elif isinstance(value, list):
            flat[full_key] = ",".join(str(item) for item in value)
        else:
            flat[full_key] = str(value)
    return flat


def run(config_path: str = "config.yaml") -> None:
    config_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(config_path):
        config_path = os.path.join(config_dir, config_path)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]
    train_cfg = config["training"]
    mlflow_cfg = config["mlflow"]

    data_yaml = os.path.join(config_dir, "data", "data.yaml")
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")

    start_run(
        experiment_name=mlflow_cfg["experiment_name"],
        run_name="train-yolov12m",
        params=flatten_config(config),
    )

    model = YOLO(model_cfg["weights"])

    def _log_metrics(trainer) -> None:
        metrics = {
            key: float(value)
            for key, value in trainer.metrics.items()
            if isinstance(value, (int, float))
        }
        mlflow.log_metrics(metrics, step=trainer.epoch)

    model.add_callback("on_fit_epoch_end", _log_metrics)

    model.train(
        data=data_yaml,
        epochs=train_cfg["epochs"],
        imgsz=model_cfg["imgsz"],
        batch=train_cfg["batch"],
        device=train_cfg["device"],
        name=train_cfg["name"],
        task="detect",
    )

    end_run()


def main() -> None:
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    run(config_path)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the file parses without import errors**

```bash
cd yolov12 && uv run python -c "import train; print('OK')"
```

Expected: `OK` (no ImportError or SyntaxError). Note: `mlflow_utils` and `yolov12.ultralytics` must both be importable — if ultralytics is not installed in this venv, this check will print an ImportError for it, which is acceptable since it's loaded at runtime from a different venv. The important thing is no syntax errors.

- [ ] **Step 3: Verify `flatten_config` works correctly**

```bash
cd yolov12 && uv run python -c "
from train import flatten_config
cfg = {'model': {'weights': 'yolov12m.pt', 'imgsz': 1280}, 'training': {'epochs': 100, 'batch': 8}}
result = flatten_config(cfg)
print(result)
assert result['model.weights'] == 'yolov12m.pt'
assert result['model.imgsz'] == '1280'
assert result['training.epochs'] == '100'
print('flatten_config OK')
"
```

Expected:
```
{'model.weights': 'yolov12m.pt', 'model.imgsz': '1280', 'training.epochs': '100', 'training.batch': '8'}
flatten_config OK
```

- [ ] **Step 4: Stage file (do NOT git commit)**

```bash
git add yolov12/train.py
```

---

## Manual Verification (requires GPU + dataset)

Once `data/data.yaml` and labeled images are in place:

```bash
cd yolov12 && uv run python main.py train
```

Then check MLflow:

```bash
cd data-setup && uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Open `http://localhost:5000` → experiment `yolov12` → confirm a new run appears with per-epoch metrics (`train/box_loss`, `metrics/mAP50`, etc.) logged.
