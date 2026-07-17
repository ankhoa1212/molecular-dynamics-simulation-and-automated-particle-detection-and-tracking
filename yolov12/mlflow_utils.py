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
