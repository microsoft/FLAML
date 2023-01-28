import logging
from flaml.automl import AutoML, logger_formatter
from flaml.tune.searcher import CFO, BlendSearch, FLOW2, BlendSearchTuner, RandomSearch
from flaml.onlineml.autovw import AutoVW
from flaml.version import __version__

try:
    from flaml.integrations import oai
except ImportError:
    pass

# Set the root logger.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
