import logging

from flaml.fabric import is_fabric_runtime
from flaml.version import __version__

report_usage_telemetry = None
_import_attempted = False

logger = logging.getLogger(__name__)


def log_telemetry(activity_name: str = ""):
    global report_usage_telemetry, _import_attempted
    if not is_fabric_runtime():
        return
    if report_usage_telemetry is None and not _import_attempted:
        try:
            from synapse.ml.fabric.telemetry_utils import report_usage_telemetry
        except ModuleNotFoundError as exc:
            if exc.name is not None and exc.name.split(".")[0] != "synapse":
                raise
            logger.debug("Fabric telemetry is unavailable: %s", exc)
        _import_attempted = True
    if report_usage_telemetry:
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
