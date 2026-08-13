import os
from pathlib import Path

import mlflow
import yaml

from mlflow_utils import end_run, start_run
from ultralytics import YOLO


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
    data_dir = config["data"]["dir"]

    if not os.path.isabs(data_dir):
        data_dir = os.path.join(os.path.dirname(config_path), data_dir)
    data_yaml = os.path.join(data_dir, "data.yaml")
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")

    start_run(
        experiment_name=mlflow_cfg["experiment_name"],
        run_name=f"train-{Path(model_cfg['weights']).stem}",
        params=flatten_config(config),
    )

    model = YOLO(model_cfg["weights"])

    def _log_metrics(trainer) -> None:
        metrics = {
            key.replace("(", "_").replace(")", ""): float(value)
            for key, value in trainer.metrics.items()
            if isinstance(value, (int, float))
        }
        mlflow.log_metrics(metrics, step=trainer.epoch)

    model.add_callback("on_fit_epoch_end", _log_metrics)

    try:
        model.train(
            data=data_yaml,
            epochs=train_cfg["epochs"],
            imgsz=model_cfg["imgsz"],
            batch=train_cfg["batch"],
            device=train_cfg["device"],
            project=os.path.join(config_dir, "runs", "detect"),
            name=train_cfg["name"],
            flipud=train_cfg["flipud"],
            task="detect",
        )
    finally:
        end_run()


def main() -> None:
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    run(config_path)


if __name__ == "__main__":
    main()
