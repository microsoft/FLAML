import warnings

from .agentchat import *
from .code_utils import DEFAULT_MODEL, FAST_MODEL
from .oai import *

warnings.warn(
    "The `flaml.autogen` module is deprecated and will be removed in a future release. "
    "Please refer to `https://github.com/microsoft/autogen` for latest usage.",
    DeprecationWarning,
    stacklevel=2,
)
