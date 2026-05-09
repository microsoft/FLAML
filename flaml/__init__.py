import logging
import warnings

try:
    from flaml.automl import AutoML, logger_formatter

    has_automl = True
except ImportError:
    has_automl = False
from flaml.onlineml.autovw import AutoVW
from flaml.tune.searcher import CFO, FLOW2, BlendSearch, BlendSearchTuner, RandomSearch
from flaml.version import __version__

try:
    from flaml.fabric.telemetry import log_telemetry

    is_log_telemetry = True
except ImportError:
    is_log_telemetry = False


# Set the root logger.
logger = logging.getLogger(__name__)
if logger.level == logging.NOTSET:
    logger.setLevel(logging.INFO)

# Log telemetry.
log_telemetry(activity_name="flaml") if is_log_telemetry else None

if not has_automl:
    warnings.warn("flaml.automl is not available. Please install flaml[automl] to enable AutoML functionalities.")
