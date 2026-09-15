import os


def is_pure_python_env() -> bool:
    """Return whether Fabric's notebook runtime identifies itself as pure Python."""
    return os.environ.get("MSNOTEBOOKUTILS_RUNTIME_TYPE", "").lower().startswith("jupyter")


def is_fabric_spark() -> bool:
    """Detect Fabric Spark using its runtime context file, without starting Spark."""
    return os.path.isfile("/home/trusted-service-user/.trident-context") and not is_pure_python_env()


def is_fabric_runtime() -> bool:
    """Return whether Fabric defaults should be enabled.

    By default, detect Fabric's notebook runtime marker or Spark context file.
    ``FLAML_FABRIC_RUNTIME=true`` (or ``1``) explicitly enables Fabric behavior;
    ``false`` (or ``0``) opts out, including inside Fabric. ``auto`` restores
    detection. Explicit AutoML/Tune options still take precedence over defaults.
    Detection does not import Spark, MLflow, or platform services.
    """
    override = os.environ.get("FLAML_FABRIC_RUNTIME", "auto").strip().lower()
    if override in ("true", "1"):
        return True
    if override in ("false", "0"):
        return False
    if override != "auto":
        raise ValueError("FLAML_FABRIC_RUNTIME must be 'auto', 'true', 'false', '1', or '0'.")
    return is_fabric_spark() or is_pure_python_env()
