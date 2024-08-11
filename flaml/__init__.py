import logging

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
logger.setLevel(logging.INFO)

if not has_automl:
    logger.warning("flaml.automl is not available. Please install flaml[automl] to enable AutoML functionalities.")
