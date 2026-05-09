import logging
import sys

from flaml.automl.logger import logger_formatter
from flaml.version import __version__

try:
    from synapse.ml.fabric.telemetry_utils import report_usage_telemetry
except ImportError:
    report_usage_telemetry = None


logger = logging.getLogger(__name__)
if not logger.handlers:
    # Add the console handler.
    _ch = logging.StreamHandler(stream=sys.stdout)
    _ch.setFormatter(logger_formatter)
    logger.addHandler(_ch)


def log_telemetry(activity_name: str = ""):
    if report_usage_telemetry:
        report_usage_telemetry(
            "PyLibraryImport",
            activity_name,
            attributes={"version": __version__, "ImportType": "EXPLICIT_IMPORTED_BY_USER"},
        )
    else:
        # For unit test and robustness
        logger.info(f"log_telemetry: {activity_name}")
