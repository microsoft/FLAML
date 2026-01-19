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

# Set the root logger.
logger = logging.getLogger(__name__)
if logger.level == logging.NOTSET:
    logger.setLevel(logging.INFO)

if not has_automl:
    warnings.warn("flaml.automl is not available. Please install flaml[automl] to enable AutoML functionalities.")
