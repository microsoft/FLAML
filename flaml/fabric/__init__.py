import os


def is_pure_python_env() -> bool:
    return os.environ.get("MSNOTEBOOKUTILS_RUNTIME_TYPE", "").lower().startswith("jupyter")


def is_fabric_spark() -> bool:
    return os.path.isfile("/home/trusted-service-user/.trident-context") and not is_pure_python_env()


def is_fabric_runtime() -> bool:
    return is_fabric_spark() or is_pure_python_env()
