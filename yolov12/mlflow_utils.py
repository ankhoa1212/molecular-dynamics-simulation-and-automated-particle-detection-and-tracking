import os
from typing import Any
import mlflow


def start_run(experiment_name: str, run_name: str, params: dict[str, Any]) -> mlflow.ActiveRun:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, "..", "data-setup", "mlflow.db")
    tracking_uri = f"sqlite:///{db_path}"
    # Also set as env vars: ultralytics' own built-in MLflow autolog callback
    # (on by default, fires from model.train()) reads MLFLOW_TRACKING_URI /
    # MLFLOW_EXPERIMENT_NAME independently of the in-process mlflow client we
    # configure below. Without these, it silently redirects logging to its own
    # default store (<repo_root>/runs/mlflow, experiment "/Shared/Ultralytics")
    # instead of this project's shared data-setup/mlflow.db.
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment_name
    mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(experiment_name)
    run = mlflow.start_run(run_name=run_name)
    if params:
        mlflow.log_params(params)
    return run


def end_run() -> None:
    mlflow.end_run()
