from flaml.automl.automl import AutoML, size
from flaml.automl.logger import logger_formatter
from flaml.automl.state import AutoMLState, SearchState
from flaml.fabric.autofe import Featurization


def register_automl_pipeline(automl, model_name=None, signature=None, artifact_path="model"):
    """Register an AutoML pipeline; MLflow is required only when this function is called."""
    from flaml.fabric.mlflow import register_automl_pipeline as register_pipeline

    return register_pipeline(automl, model_name, signature, artifact_path)


__all__ = [
    "AutoML",
    "AutoMLState",
    "SearchState",
    "logger_formatter",
    "size",
    "Featurization",
    "register_automl_pipeline",
]
