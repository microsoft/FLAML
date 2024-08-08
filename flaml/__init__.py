import logging

try:
    from flaml.automl import AutoML, logger_formatter
except ImportError:
    pass
from flaml.onlineml.autovw import AutoVW
from flaml.tune.searcher import CFO, FLOW2, BlendSearch, BlendSearchTuner, RandomSearch
from flaml.version import __version__

# Set the root logger.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
