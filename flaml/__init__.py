from flaml.searcher import CFO, BlendSearch, FLOW2, BlendSearchTuner
from flaml.automl import AutoML, logger_formatter
from flaml.onlineml.autovw import AutoVW
from flaml.version import __version__
try:
    from flaml.utils import *
except ImportError:
    # In case some import from sklearn breaks, don't make `import flaml` fail
    pass
import logging

# Set the root logger.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
