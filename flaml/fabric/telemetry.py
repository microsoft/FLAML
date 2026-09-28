import logging
import sys

from flaml.automl.logger import logger_formatter
from flaml.fabric import is_fabric_runtime
from flaml.version import __version__

report_usage_telemetry = None
if is_fabric_runtime():
    try:
        from synapse.ml.fabric.telemetry_utils import report_usage_telemetry
    except ModuleNotFoundError as exc:
        if exc.name is not None and exc.name.split(".")[0] != "synapse":
            raise
        logging.getLogger(__name__).debug("Fabric telemetry is unavailable: %s", exc)


logger = logging.getLogger(__name__)
if not logger.handlers:
    # Add the console handler.
    _ch = logging.StreamHandler(stream=sys.stdout)
    _ch.setFormatter(logger_formatter)
    logger.addHandler(_ch)


def log_telemetry(activity_name: str = ""):
    if is_fabric_runtime() and report_usage_telemetry:
        report_usage_telemetry(
            "PyLibraryImport",
            activity_name,
            attributes={"version": __version__, "ImportType": "EXPLICIT_IMPORTED_BY_USER"},
        )
    else:
        logger.debug("Fabric telemetry unavailable for %s", activity_name)
