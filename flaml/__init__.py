from flaml.searcher import CFO, BlendSearch, FLOW2
from flaml.automl import AutoML
from flaml.version import __version__
import logging

# Set the root logger.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logger_formatter = logging.Formatter(
    '[%(name)s: %(asctime)s] {%(lineno)d} %(levelname)s - %(message)s',
    '%m-%d %H:%M:%S')
