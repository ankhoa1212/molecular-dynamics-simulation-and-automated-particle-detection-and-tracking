import os
from typing import Any
import mlflow


def start_run(experiment_name: str, run_name: str, params: dict[str, Any]) -> mlflow.ActiveRun:
    # Respect MLFLOW_TRACKING_URI if set (e.g. in containerised/K8s runs),
    # otherwise fall back to the shared database in the data-setup directory.
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(base_dir, "..", "data-setup", "mlflow.db")
        tracking_uri = f"sqlite:///{db_path}"
    mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(experiment_name)
    run = mlflow.start_run(run_name=run_name)
    if params:
        mlflow.log_params(params)
    return run


def log_epoch_metrics(metrics: dict[str, float], step: int) -> None:
    mlflow.log_metrics(metrics, step=step)


def log_artifact(path: str) -> None:
    mlflow.log_artifact(path)


def end_run() -> None:
    mlflow.end_run()
