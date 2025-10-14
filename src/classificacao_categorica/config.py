from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_RAW_PATH = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"

# Model parameters
MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 5,
    "random_state": 42
}

# MLflow
MLFLOW_TRACKING_URI = "./mlruns"