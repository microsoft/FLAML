import logging
import os
import sys


class ColoredFormatter(logging.Formatter):
    # ANSI escape codes for colors
    COLORS = {
        # logging.DEBUG: "\033[36m",  # Cyan
        # logging.INFO: "\033[32m",   # Green
        logging.WARNING: "\033[33m",  # Yellow
        logging.ERROR: "\033[31m",  # Red
        logging.CRITICAL: "\033[1;31m",  # Bright Red
    }
    RESET = "\033[0m"  # Reset to default

    def __init__(self, fmt, datefmt, use_color=True):
        super().__init__(fmt, datefmt)
        self.use_color = use_color

    def format(self, record):
        formatted = super().format(record)
        if self.use_color:
            color = self.COLORS.get(record.levelno, "")
            if color:
                return f"{color}{formatted}{self.RESET}"
        return formatted


logger = logging.getLogger(__name__)
use_color = True
if os.getenv("FLAML_LOG_NO_COLOR"):
    use_color = False

logger_formatter = ColoredFormatter(
    "[%(name)s: %(asctime)s] {%(lineno)d} %(levelname)s - %(message)s", "%m-%d %H:%M:%S", use_color
)
logger.propagate = False


def init_logger(name=None):
    """Return a logger configured with the FLAML console handler and formatter.

    Idempotent: if the logger already has handlers attached, the existing
    handlers are preserved and no new handler is added. ``propagate`` is
    always set to ``False`` so the FLAML console handler is not duplicated
    by ancestor loggers.

    Args:
        name: Logger name to retrieve via ``logging.getLogger(name)``.
            Defaults to ``None``, which returns the module's existing logger
            (``flaml.automl.logger``). Typical callers pass ``__name__``.

    Returns:
        The configured ``logging.Logger`` instance.
    """
    _logger = logging.getLogger(name) if name is not None else logger
    if not _logger.handlers:
        _ch = logging.StreamHandler(stream=sys.stdout)
        _ch.setFormatter(logger_formatter)
        _logger.addHandler(_ch)
    _logger.propagate = False
    return _logger
