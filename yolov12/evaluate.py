import os

import mlflow
import yaml

from mlflow_utils import end_run, start_run
from ultralytics import YOLO


def run(config_path: str = "config.yaml", weights: str | None = None) -> None:
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

    if weights is None:
        weights = os.path.join(
            config_dir, "runs", "detect", train_cfg["name"], "weights", "best.pt"
        )
    if not os.path.exists(weights):
        raise FileNotFoundError(
            f"Trained weights not found: {weights}. Run 'main.py train' first, "
            "or pass an explicit weights path."
        )

    start_run(
        experiment_name=mlflow_cfg["experiment_name"],
        run_name=f"evaluate-{os.path.splitext(os.path.basename(weights))[0]}",
        params={"weights": weights, "data": data_yaml},
    )

    try:
        model = YOLO(weights)
        metrics = model.val(data=data_yaml, imgsz=model_cfg["imgsz"], split="test")
        mlflow.log_metrics(
            {
                key.replace("(", "_").replace(")", ""): float(value)
                for key, value in metrics.results_dict.items()
            }
        )
    finally:
        end_run()


if __name__ == "__main__":
    run()
