import logging
import re
from threading import RLock

from flaml.fabric import is_fabric_runtime

_KUSTO_TABLE_NAME = "SynapseMLLogs"


class KustoLogger:
    def debug(self, *args, **kwargs):
        pass

    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def exception(self, *args, **kwargs):
        pass


kusto_logger = KustoLogger()


class ArtifactNameScrubber:
    pattern = re.compile("Artifact \\S+", re.IGNORECASE)
    mask = "[artifact name redacted]"

    def scrub(self, msg: str) -> str:
        return re.sub(self.pattern, self.mask, msg)


def _create_logger(logger_name):
    try:
        from synapse.ml.pymds import get_mds_logger
        from synapse.ml.pymds.handler import default_scrubbers
        from synapse.ml.pymds.scrubbers.scrubber import IScrub
    except ModuleNotFoundError as exc:
        if exc.name is not None and exc.name.split(".")[0] != "synapse":
            raise
        logging.getLogger(__name__).debug("Fabric logging is unavailable: %s", exc)
        return kusto_logger

    class MdsArtifactNameScrubber(ArtifactNameScrubber, IScrub):
        pass

    return get_mds_logger(
        logger_name,
        tableName=_KUSTO_TABLE_NAME,
        scrubbers=[*default_scrubbers, MdsArtifactNameScrubber()],
    )


class _RuntimeLogger(KustoLogger):
    def __init__(self, name):
        self.name = name
        self._backend = None
        self._lock = RLock()

    def __reduce__(self):
        return init_kusto_logger, (self.name,)

    def _log(self, level, *args, **kwargs):
        if not is_fabric_runtime():
            return
        if self._backend is None:
            with self._lock:
                if self._backend is None:
                    try:
                        self._backend = _create_logger(self.name)
                    except ModuleNotFoundError:
                        raise
                    except Exception:
                        logging.getLogger(__name__).warning(
                            "Fabric diagnostic logger initialization failed.", exc_info=True
                        )
                        self._backend = kusto_logger
        getattr(self._backend, level)(*args, **kwargs)

    def debug(self, *args, **kwargs):
        self._log("debug", *args, **kwargs)

    def info(self, *args, **kwargs):
        self._log("info", *args, **kwargs)

    def warning(self, *args, **kwargs):
        self._log("warning", *args, **kwargs)

    def error(self, *args, **kwargs):
        self._log("error", *args, **kwargs)

    def exception(self, *args, **kwargs):
        self._log("exception", *args, **kwargs)


def init_kusto_logger(logger_name: str = ""):
    return _RuntimeLogger(logger_name or __name__)
