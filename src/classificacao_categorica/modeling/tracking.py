import mlflow
import mlflow.sklearn
from ..config import MLFLOW_TRACKING_URI


def setup_mlflow():
    """Configura tracking do MLflow"""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def log_run(model, params, metrics, run_name="Default Run"):
    """Registra experimento no MLflow"""
    setup_mlflow()

    with mlflow.start_run(run_name=run_name):
        # Log parameters
        for key, value in params.items():
            mlflow.log_param(key, value)

        # Log metrics
        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        # Log model
        mlflow.sklearn.log_model(model, "model")