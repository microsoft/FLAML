import logging

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


def log_telemetry(activity_name: str = ""):
    if is_fabric_runtime() and report_usage_telemetry:
        try:
            report_usage_telemetry(
                "PyLibraryImport",
                activity_name,
                attributes={"version": __version__, "ImportType": "EXPLICIT_IMPORTED_BY_USER"},
            )
        except Exception:
            logger.warning("Fabric usage telemetry failed for %s.", activity_name, exc_info=True)
    else:
        logger.debug("Fabric telemetry unavailable for %s", activity_name)
