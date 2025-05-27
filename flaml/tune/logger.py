import logging
import os


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
