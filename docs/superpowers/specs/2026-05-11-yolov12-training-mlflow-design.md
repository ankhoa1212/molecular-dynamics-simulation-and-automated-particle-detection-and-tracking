# YOLOv12 Training MLflow Integration — Design

## Goal

Fix `yolov12/train.py` so it:
1. Exposes a `run(config_path)` function callable from `main.py`
2. Reads training parameters from `config.yaml`
3. Logs metrics to the shared `data-setup/mlflow.db` database
4. Uses `yolov12m.pt` with `imgsz=1280` appropriate for 3200×2200 microscopy images with hundreds of small particles

---

## Files

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `yolov12/config.yaml` | Training hyperparameters and MLflow config |
| Create | `yolov12/mlflow_utils.py` | MLflow setup pointing to shared `data-setup/mlflow.db` |
| Modify | `yolov12/train.py` | Add `run(config_path)`, wire MLflow, fix model weights |

---

## `yolov12/config.yaml`

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

---

## `yolov12/mlflow_utils.py`

Mirrors `rf-detr/mlflow_utils.py`. Resolves the shared database path relative to its own file location:

```python
db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data-setup", "mlflow.db")
mlflow.set_tracking_uri(f"sqlite:///{db_path}")
```

Exports: `start_run(experiment_name, run_name, params)`, `end_run()`

---

## `yolov12/train.py`

### `run(config_path)` function

1. Load `config.yaml` via PyYAML
2. Resolve `data_yaml` as `os.path.join(os.path.dirname(__file__), "data", "data.yaml")`
3. Call `start_run(experiment_name, run_name="train-yolov12m", params)` where `params` is a flat `dict[str, str]` of all config keys (using the same `flatten_config` pattern from `rf-detr/train.py`)
4. Register Ultralytics `on_fit_epoch_end` callback to log per-epoch metrics:
   - `train/box_loss`, `train/cls_loss`, `train/dfl_loss`
   - `metrics/precision`, `metrics/recall`, `metrics/mAP50`, `metrics/mAP50-95`
5. Call `YOLO(weights).train(data=data_yaml, epochs=..., imgsz=..., batch=..., device=..., name=...)`
6. Call `end_run()`

### `main()` function (kept as direct CLI fallback)

Updated to call `run("config.yaml")` instead of inlining training logic.

### MLflow callback pattern

Ultralytics exposes trainer metrics via `add_callback("on_fit_epoch_end", fn)`. The callback receives the `trainer` object; metrics are at `trainer.metrics` (a dict of `str → float`).

---

## Error Handling

- If `config.yaml` is missing: raise `FileNotFoundError` with a clear message
- If `data/data.yaml` is missing: `train.py` already raises `FileNotFoundError` — keep this behaviour
- MLflow connection errors: let them propagate (same pattern as `rf-detr`)

---

## Testing

No automated tests — training requires GPU + dataset. Verify manually:

1. `uv run python main.py train` completes without error
2. `mlflow ui --backend-store-uri sqlite:///../data-setup/mlflow.db` shows a new run under the `yolov12` experiment with per-epoch metrics logged
